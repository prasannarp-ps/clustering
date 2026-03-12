# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
conda activate ml
pip install -r requirements.txt
```

**Note:** `requirements.txt` covers TensorFlow/PyTorch but is missing some clustering-tier packages. If imports fail, install: `pip install scikit-learn duckdb click plotly python-pptx umap-learn dash joblib pyarrow`.

All filesystem paths and default parameters are centralized in `clustering_pipeline/config.py`. Edit this file first when adapting to a new environment.

No test suite or linting configuration exists. There is no CI/CD pipeline.

## CLI Commands

### Step 0: Extract embeddings (before clustering)

```bash
# Google Path Foundation (TensorFlow)
python embeddings_pipeline/embedding_extractor.py --img_path data/gi-registration --output data/global_embedding.parquet --batch_size 128 --modality stained

# Other supported models: conch | uni | uni2 | titan
python embeddings_pipeline/embedding_extractor.py --model uni --img_path data/gi-registration --output data/global_embedding_uni.parquet --batch_size 128 --modality stained
python embeddings_pipeline/embedding_extractor.py --model uni2 --img_path data/gi-registration --output data/global_embedding_uni2.parquet --batch_size 64 --modality stained
python embeddings_pipeline/embedding_extractor.py --model conch --img_path data/gi-registration --output data/global_embedding_conch.parquet --batch_size 128 --modality stained
python embeddings_pipeline/embedding_extractor.py --model titan --img_path data/gi-registration --output data/global_embedding_titan.parquet --batch_size 128 --modality stained
```

Tile images must be `.tiff` files. Modality is inferred from filename suffix (`-stained`, `-unstained`, `-inferred`) if `--modality` is omitted.

### Convenience: full pipeline in one command

```bash
python run_model_pipeline.py --model uni --k 16 --pca 100
# Skip individual steps with --skip-optimal-k, --skip-build-db, --skip-train, etc.
```

This orchestrates all steps below for a single model, auto-deriving paths from `--model`.

### Full pipeline sequence (manual)

```bash
# 1. Build DuckDB databases from Parquet
python -m clustering_pipeline.cli build-db --embed-parquet data/global_embedding.parquet

# 2. Stratified sampling (balanced per tissue_type)
python -m clustering_pipeline.cli stratify --per-class 50000 --sample-nb 2000000

# 3. Train MiniBatchKMeans
python -m clustering_pipeline.cli train --table-name stratified_extended --k 112 --batch-size 100000 --min-tissue-pct 20 --suffix 2000000above_20

# 4. Predict clusters for all embeddings (resumable)
python -m clustering_pipeline.cli predict --table-name embeddings --k 112 --batch-size 1000000

# 5. Merge predictions with extended metadata; relabel low-tissue tiles
python -m clustering_pipeline.cli postprocess

# 6. Generate HTML analysis reports
python -m clustering_pipeline.cli analyze --k 112 --preprocessing resize --min-tissue-pct 20

# 7. Generate PNG cluster image grids (requires S3/R2 access)
python -m clustering_pipeline.cli visualize --k 112 --rows 10 --cols 10 --num-grids 1 --min-tissue-pct 20
```

### Analysis utilities

```bash
python find_optimal_k_advanced.py --parquet data/global_embedding.parquet --silhouette --pca 100
# Outputs: results/advanced_optimal_k_metrics.csv + results/advanced_optimal_k_plot.html

python explore_k_sweep.py                                      # K-value sweep with metrics comparison → results/k_sweep/
python projection_plot.py --method umap --pca-components 100   # 2D UMAP/t-SNE
python umap_blur.py                                            # UMAP of blur/sharpness features
python umap_blur_viewer.py                                     # Interactive Dash app at localhost:8050
python cluster_image_grid.py --rows 5 --cols 5                 # Local image grid (no S3)
python compare_models_optimal_k.py                             # Unified model comparison → results/comparison_optimal_k_plot.html
python build_presentation.py                                   # PowerPoint report → results/clustering_presentation.pptx
python build_embedding_models_presentation.py                  # Embedding model comparison presentation
```

## Architecture

### Two-phase pipeline

```
TIFF tiles
    │
    ▼
embeddings_pipeline/embedding_extractor.py  →  data/global_embedding_{model}.parquet
    │
    ▼
clustering_pipeline (DuckDB + MiniBatchKMeans)
    ├── build-db   → embeddings.db + extended_data.db
    ├── stratify   → stratified / stratified_extended tables (in embeddings.db)
    ├── train      → models/kmeans_model_k{k}[_suffix].joblib
    ├── predict    → predictions.db  (resumable anti-join)
    ├── postprocess→ predictions_extended.db (tissue % joins + relabeling)
    ├── analyze    → results/*.html  (Plotly)
    └── visualize  → results/cluster_grids/*.png
```

### DuckDB as the pipeline backbone

Four DuckDB files serve as the data hub:
- `db/embeddings.db` — embedding vectors + base metadata; also holds `stratified` / `stratified_extended` tables after sampling
- `db/extended_data.db` — tile-level metadata (`tissue_percentage`, color histograms, etc.)
- `db/predictions.db` — raw cluster assignments
- `db/predictions_extended.db` — predictions joined with extended metadata; low-tissue tiles relabeled to `cluster = -1` (tissue < 20%) or `cluster = -2` (tissue < 5%)

When using `run_model_pipeline.py`, DBs are stored under `db/{model}/` to keep models isolated.

### Memory-efficient streaming

`duckdb_io.duckdb_iter_batches()` streams data in configurable chunks so tens of millions of tiles never load fully into RAM. Training uses `MiniBatchKMeans.partial_fit()` over these batches. Prediction uses an **anti-join** against the already-written predictions table, making it safely resumable after interruption.

### Embedding model backends (`embeddings_pipeline/embedding_extractor.py`)

| Model | Framework | Dim | Notes |
|---|---|---|---|
| `path_foundation` | TensorFlow | 384 | Google Path Foundation, default |
| `conch` | PyTorch | 512 | Recommended for frozen H&E |
| `uni` | PyTorch | 1024 | Strong FFPE H&E |
| `uni2` | PyTorch | 1536 | Strongest UNI variant; use `--batch_size 64` |
| `titan` | PyTorch | 512 | TITAN's internal CONCH v1.5 tile encoder; needs `transformers`, `einops`, `einops-exts`; separate HF gated access |

### Module responsibilities

| Module | Responsibility |
|---|---|
| `clustering_pipeline/config.py` | All path constants and defaults |
| `clustering_pipeline/cli.py` | Click-based CLI; one subcommand per pipeline stage |
| `clustering_pipeline/duckdb_io.py` | DB creation from Parquet, streaming batch iterator |
| `clustering_pipeline/sampling.py` | Two-pass stratified sampling by `tissue_type` |
| `clustering_pipeline/training.py` | `fit_kmeans_model_duckdb()` — streams from DuckDB into `MiniBatchKMeans` |
| `clustering_pipeline/prediction.py` | Resumable anti-join prediction loop |
| `clustering_pipeline/postprocessing.py` | Join predictions + extended metadata; relabel by tissue % |
| `clustering_pipeline/analysis.py` | Plotly HTML reports of cluster/tissue distributions |
| `clustering_pipeline/visualization.py` | Per-cluster PNG grids downloaded from S3 (Cloudflare R2) |
| `embeddings_pipeline/embedding_extractor.py` | Multi-backend tile embedding extraction to Parquet |
| `run_model_pipeline.py` | Orchestrates full pipeline for a single model via subprocess |
| `find_optimal_k_advanced.py` | Elbow/CHI/DBI/silhouette analysis to choose k |
| `projection_plot.py` | 2D UMAP/t-SNE visualization |

### S3/R2 integration

Tile images live in Cloudflare R2 (S3-compatible). Credentials are in `r2.conf`. `visualization.py` uses a `s3_path_builder(tile_key)` function to map tile keys to S3 URIs and downloads them with a thread pool (`downloader_workers=8` by default).

### Legacy code

`clustering_pipeline/legacy/` contains older implementations (`cluster_evaluator.py`, `cluster_predict.py`, `clustering_training.py`). The active pipeline is in the top-level modules.
