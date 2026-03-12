# Embedding Clustering Pipeline (DuckDB + MiniBatchKMeans)

This repository implements an end‑to‑end clustering pipeline for large‑scale pathology embeddings using:

- **DuckDB** for storage and streaming
- **MiniBatchKMeans** for scalable clustering
- **Stratified sampling** by `tissue_type`
- **Tissue percentage filtering** via an `extended_data` metadata table
- **HTML and image‑grid visualizations** for cluster quality inspection

The pipeline is designed to work on tens of millions of tiles while keeping memory usage under control.

---

## 1. High‑Level Overview

The typical workflow looks like this:

1. **Build DuckDB databases** from Parquet:
   - `embeddings.db` – contains the global embedding table (`embeddings`)
   - `extended_data.db` – contains tile‑level metadata (`extended_data`) including `tissue_percentage`

2. **Stratified sampling** from the global embedding table:
   - Balanced sample per `tissue_type`
   - Written back into `embeddings.db` as `stratified` and `stratified_extended` tables
   - Optionally saved as a Parquet file

3. **Train MiniBatchKMeans** on the stratified dataset:
   - Directly from DuckDB, streaming `vector` batches
   - Optionally filtering tiles by `tissue_percentage` (e.g. `>= 20%`)

4. **Predict clusters** for all embeddings:
   - Using KMeans model
   - Writing results into `predictions.db` (table: `predictions`)
   - Optionally exporting results to Parquet

5. **Postprocess predictions** with extended metadata:
   - Merging predictions with `extended_data` (`tissue_percentage`)
   - Re‑labeling low‑tissue tiles into dedicated clusters (e.g. `cluster = -1`, `cluster = -2`)
   - Saving into `predictions_extended.db`

6. **Analyze** cluster vs tissue distribution:
   - HTML plots summarizing per‑cluster counts
   - Tissue percentage histograms / statistics

7. **Visualize** cluster samples as image grids:
   - For each cluster, sample tiles and download images from S3
   - Build labeled PNG grids for quick qualitative inspection

All steps can be run via CLI commands defined in `cli.py`, with additional programmatic helpers in the various modules.

---

## 2. Project Layout

```text
clustering_pipeline/
├── __init__.py
├── config.py               # Global paths and defaults
├── cli.py                  # Command‑line entrypoint
├── duckdb_io.py            # DuckDB I/O utilities
├── sampling.py             # Stratified sampling
├── training.py             # KMeans training
├── prediction.py           # KMeans prediction
├── postprocessing.py       # Merge predictions + extended metadata
├── analysis.py             # Cluster / tissue stats and HTML plots
└── visualization.py        # Cluster image grids (S3 downloader, OpenCV)
```

---

## 3. Configuration (`config.py`)

`config.py` centralizes all filesystem paths and default parameters. A typical configuration looks like:

```python
import os

BASE_DIR = "/Volumes/KINGSTON/embeddings_v7"

PARQUET_FILE_PATH = os.path.join(BASE_DIR, "global_embedding.parquet")
EXTENDED_PARQUET_FILE_PATH = os.path.join(BASE_DIR, "db_tiles_extended.parquet")

DB_DIR = os.path.join(BASE_DIR, "db")
OUT_DIR_MODEL = os.path.join(BASE_DIR, "models")
OUT_DIR_DATA = os.path.join(BASE_DIR, "data")
OUT_DIR_RESULTS = os.path.join(BASE_DIR, "results")

EMBED_DB_PATH = os.path.join(DB_DIR, "embeddings.db")
EXTENDED_DB_PATH = os.path.join(DB_DIR, "extended_data.db")
PRED_DB_PATH = os.path.join(DB_DIR, "predictions.db")

SAMPLE_PARQUET_PATH = os.path.join(OUT_DIR_DATA, "sample_stratified.parquet")

DEFAULT_MODALITY = "stained"
DEFAULT_K = 112
```

You can modify these paths to match your environment. All CLI commands use `config.py` for their defaults.

---

## 4. DuckDB Utilities (`duckdb_io.py`)

### 4.1. Creating `embeddings.db` from Parquet

```python
create_embeddings_db_from_parquet(embed_db_path, parquet_path)
```

- Reads `global_embedding.parquet`
- Creates an `embeddings` table
- Adds and populates a deterministic `unique_id` column (based on pathology keys)
- Creates an index on `unique_id` for fast joins and lookups

### 4.2. Creating a generic DuckDB from Parquet

```python
create_duckdb_from_parquet_simple(
    db_path,
    parquet_path,
    table_name="data",
    generate_unique_id=False,
    index_unique_id=False,
)
```

- Creates `db_path` with a single table `table_name`
- Optionally generates a synthetic `unique_id` from `rowid`
- Optionally indexes `unique_id`

This is used to build `extended_data.db` (`table_name="extended_data"`).

### 4.3. DuckDB streaming helper

```python
duckdb_iter_batches(table, batch_size, columns=None, min_tissue_pct=0)
```

- Streams data from `embeddings.db` in batches
- If `min_tissue_pct > 0`, joins with `ext_db.extended_data` and filters `tissue_percentage >= min_tissue_pct`
- Always filters to `modality = 'stained'`

Used internally by the stratified sampler to iterate over very large tables without loading everything into RAM.

---

## 5. Stratified Sampling (`sampling.py`)

### 5.1. `create_stratified_dataset(...)`

```python
create_stratified_dataset(
    parquet_paths,
    stratify_field="tissue_type",
    modality="stained",
    per_class=25000,
    max_samples=100000,
    batch_size=1_000_000,
    stats=None,
    id_col="unique_id",
    random_state=42,
    write_to_duckdb=False,
    embed_db_path=None,
    extended_db_path=None,
    stratified_table="stratified",
    stratified_ext_table="stratified_extended",
)
```

**Inputs**

- `parquet_paths` – list of sources to sample from. Each element can be:
  - A Parquet file path, or
  - `duckdb:<table_name>` (for sampling directly from DuckDB)
- `stratify_field` – usually `tissue_type`
- `modality` – typically `"stained"`
- `per_class` – max number of samples per class (before global cap)
- `max_samples` – global cap across all classes
- `batch_size` – per‑batch rows when reading from Parquet or DuckDB
- `write_to_duckdb` – if `True`, writes results into DuckDB (see below)
- `embed_db_path`, `extended_db_path` – required if `write_to_duckdb=True`
- `stratified_table`, `stratified_ext_table` – table names to create

**What it does**

1. **Pass 1** – Counts rows per `stratify_field` per file/source
2. Excludes numeric keys and computes a target quota per class, capped by:
   - `per_class`
   - `max_samples`
3. Distributes per‑class quotas proportionally across files
4. **Pass 2** – Streams each source again, sampling rows until quotas are satisfied
   - Removes duplicates by `id_col` (`unique_id`)
   - Tracks global `seen_ids` to avoid duplicate tiles across multiple Parquet files
5. Concatenates all sampled rows into `sampled_df`

If `write_to_duckdb=True`, it calls `save_stratified_to_duckdb(...)` (see below).

### 5.2. `save_stratified_to_duckdb(...)`

```python
save_stratified_to_duckdb(
    stratified_df,
    embed_db_path,
    extended_db_path,
    stratified_table="stratified",
    stratified_ext_table="stratified_extended",
)
```

Writes:

- `stratified` – sampled rows (just embeddings + metadata)
- `stratified_extended` – sampled rows joined with `ext_db.extended_data` to attach `tissue_percentage`

Both tables get an index on `unique_id`. The `stratified_extended` table can be queried like:

```sql
SELECT *
FROM stratified_extended
WHERE tissue_percentage >= 20;
```

---

## 6. Training (`training.py`)

### 6.1. Fit from an in‑memory DataFrame

```python
fit_kmeans_model(df, k, batch_size=100_000, suffix=None, out_dir="./models")
```

- Shuffles `df`
- Streams through `vector` batches into `MiniBatchKMeans.partial_fit`
- Saves `kmeans_model_k{k}[_suffix].joblib` in `out_dir`

This is mainly useful for experiments on smaller datasets.

### 6.2. Fit directly from DuckDB

```python
fit_kmeans_model_duckdb(
    embed_db_path,
    extended_db_path,
    table_name="embeddings",
    k=112,
    batch_size=100_000,
    min_tissue_pct=None,
    modality="stained",
    suffix=None,
    out_dir="./models",
)
```

- Connects to `embeddings.db` and `extended_data.db`
- If `min_tissue_pct` is set, joins with `extended_data` and filters rows by `tissue_percentage >= min_tissue_pct`
- Streams `vector` batches from `table_name` (e.g. `stratified_extended`)
- Trains `MiniBatchKMeans` using `partial_fit`
- Saves the trained model to `out_dir`

This is the recommended training path for large datasets.

---

## 7. Prediction (`prediction.py`)

### 7.1. Predict clusters for all embeddings

```python
predict_kmeans_model_duckdb(
    model_path,
    embed_db_path,
    pred_db_path,
    table_name="embeddings",
    batch_size=100_000,
    modality="stained",
    out_parquet_path=None,
)
```

- Loads the trained KMeans model from `model_path`
- Connects to `embeddings.db` and attaches `pred_db_path` as `pred_db`
- Ensures `pred_db.predictions` exists and has an index on `unique_id`
- Repeatedly fetches batches from `{table_name}` using an **anti‑join**:
  - Only rows whose `unique_id` is **not yet present** in `pred_db.predictions`
  - Ensures idempotent, resumable prediction
- For each batch:
  - Extracts `vector` as a NumPy array
  - Performs `model.predict(vectors)`
  - Inserts `(unique_id, id, slide_key, tile_key, short_box_name, box_id, cluster)` into `pred_db.predictions`
- Optionally exports the final `pred_db.predictions` table to `out_parquet_path`

Prediction is streamed and can handle tens of millions of tiles.

---

## 8. Postprocessing (`postprocessing.py`)

### 8.1. Merge predictions with extended metadata

```python
create_predictions_extended_db(
    pred_db_path,
    extended_db_path,
    output_db_path,
    pred_table="predictions",
    ext_table="extended_data",
)
```

- Attaches `pred_db` (containing `{pred_table}`) and `ext_db` (containing `{ext_table}`)
- Prints original cluster counts from `{pred_table}`
- Creates a new `predictions` table in `output_db_path` with columns:
  - `unique_id, id, slide_key, tile_key, short_box_name, box_id, cluster, tissue_percentage`
- Relabels low‑tissue tiles using a `CASE` expression, e.g.:
  - `tissue_percentage < 5`  → `cluster = -2`
  - `5 <= tissue_percentage < 20` → `cluster = -1`
  - Otherwise: keep original `cluster`
- Prints updated cluster counts and deltas
- Adds indexes on `unique_id` and `tile_key` in the output DB

This produces a `predictions_extended.db` that can be used for analysis and visualization with low‑tissue tiles explicitly separated.

---

## 9. Analysis (`analysis.py`)

### 9.1. Cluster / tissue distribution plots

```python
plot_cluster_tissue_info_duckdb(
    embed_db_path,
    pred_db_path,
    extended_db_path,
    output_dir,
    preprocessing="resize",
    k=None,
    table_name="embeddings",
    min_tissue_pct=None,
)
```

**What it does**

- Attaches:
  - `embeddings.db` (for `{table_name}`)
  - `pred_db` (for `predictions`)
  - `ext_db` (for `extended_data`)
- Joins embeddings, predictions, and extended metadata
- Optionally filters by `tissue_percentage >= min_tissue_pct`
- Computes:
  - Per‑cluster counts
  - Low‑tissue counts per threshold (e.g. `<5`, `<10`, `<15`, `<20`)
  - Percentages per cluster
- Writes one or more **HTML bar plots** (via Plotly) into `output_dir`, e.g.:
  - `k112_cluster_tissue_stats_resize.html`
  - `k112_low_tissue_clusters_resize.html`

You can open these HTML files in a browser to inspect cluster quality and low‑content clusters.

---

## 10. Visualization (`visualization.py`)

### 10.1. Cluster sample grids

```python
visualize_cluster_samples_duckdb(
    embed_db_path,
    pred_db_path,
    extended_db_path,
    table_name="predictions",
    out_dir="./cluster_grids",
    rows=2,
    cols=5,
    resize_dim=(256, 256),
    font_scale=0.4,
    font_thickness=1,
    padding=2,
    cluster_id=None,
    num_grids=1,
    k=None,
    s3_path_builder=None,
    downloader_workers=8,
    min_tissue_pct=20,
)
```

**Inputs**

- `embed_db_path` – path to `embeddings.db`
- `pred_db_path` – path to `predictions.db` (or `predictions_extended.db`)
- `extended_db_path` – path to `extended_data.db`
- `table_name` – predictions table to query (usually `"predictions"`)
- `out_dir` – where to write PNG grids
- `rows`, `cols` – grid layout per PNG
- `cluster_id` – if set, only visualize this cluster; otherwise iterate over all clusters
- `num_grids` – how many grids per cluster
- `k` – optional, used for naming subdirectories
- `s3_path_builder(tile_key: str)` – user‑provided function that maps a `tile_key` to an S3 URI
- `downloader_workers` – concurrency for S3 downloads
- `min_tissue_pct` – only tiles with `tissue_percentage >= min_tissue_pct` are used

**What it does**

For each cluster:

1. Counts how many "clean" tiles exist with `tissue_percentage >= min_tissue_pct`
2. For each requested grid:
   - Samples up to `rows * cols` tiles
   - Builds S3 paths via `s3_path_builder`
   - Downloads images into a temporary directory
   - Loads TIFFs with OpenCV, resizes and labels them with `tile_key`
   - Creates a single NumPy image grid and writes it to `out_png`

The result is a directory of labeled PNG grids, one or more per cluster, that you can inspect visually.

---

## 11. Command‑Line Interface (`cli.py`)

The CLI provides subcommands for each stage of the pipeline, including analysis and visualization.

### 11.1. Build databases

```bash
python -m clustering_pipeline.cli build-db \
    --embed-parquet /path/to/global_embedding.parquet \
    --ext-parquet /path/to/db_tiles_extended.parquet
```

Arguments:

- `--embed-parquet` – path to global embedding Parquet (default: `config.PARQUET_FILE_PATH`)
- `--ext-parquet` – path to extended metadata Parquet (default: `config.EXTENDED_PARQUET_FILE_PATH`)

### 11.2. Stratified sampling

```bash
python -m clustering_pipeline.cli stratify \
    --stratify-field tissue_type \
    --modality stained \
    --per-class 50000 \
    --sample-nb 2000000 \
    --stratified-table stratified \
    --stratified-ext-table stratified_extended \
    --sample-parquet /path/to/sample_stratified.parquet
```

Key arguments:

- `--stratify-field` – field used for stratification (default: `tissue_type`)
- `--per-class` – max samples per class (default: `50_000`)
- `--sample-nb` – global sample cap (default: `2_000_000`)
- `--stratified-table` – name for sampled table in DuckDB (default: `stratified`)
- `--stratified-ext-table` – name for sampled+extended table (default: `stratified_extended`)
- `--sample-parquet` – optional Parquet output path

### 11.3. Train KMeans

```bash
python -m clustering_pipeline.cli train \
    --table-name stratified_extended \
    --k 112 \
    --batch-size 100000 \
    --min-tissue-pct 20 \
    --modality stained \
    --suffix 2000000above_20
```

Key arguments:

- `--table-name` – DuckDB table to train on (default: `stratified_extended`)
- `--k` – number of clusters (default: `config.DEFAULT_K`)
- `--batch-size` – training batch size (default: `100_000`)
- `--min-tissue-pct` – optional lower bound on `tissue_percentage` (default: `None`)
- `--suffix` – optional model filename suffix
- `--sample-nb` – used only for building default suffix (optional)

### 11.4. Predict clusters

```bash
python -m clustering_pipeline.cli predict \
    --table-name embeddings \
    --k 112 \
    --batch-size 1000000 \
    --modality stained \
    --model-path /path/to/kmeans_model_k112_2000000above_20.joblib \
    --out-parquet /path/to/clustering_results_all_k112above_20.parquet
```

Key arguments:

- `--table-name` – embeddings table (default: `embeddings`)
- `--k` – used only for default model path / naming
- `--batch-size` – prediction batch size (default: `100_000`)
- `--model-path` – explicit path to trained `.joblib` model (optional)
- `--out-parquet` – Parquet output for predictions (optional; default in `config.OUT_DIR_RESULTS`)

### 11.5. Postprocess predictions

```bash
python -m clustering_pipeline.cli postprocess \
    --pred-db-path /path/to/predictions.db \
    --output-db-path /path/to/predictions_extended.db \
    --pred-table predictions \
    --ext-table extended_data
```

Key arguments:

- `--pred-db-path` – input predictions DB (default: `config.PRED_DB_PATH`)
- `--output-db-path` – where to write `predictions_extended.db`
- `--pred-table` – predictions table name (default: `predictions`)
- `--ext-table` – extended metadata table name (default: `extended_data`)

### 11.6. Analyze (cluster vs tissue statistics)

The `analyze` subcommand wraps `plot_cluster_tissue_info_duckdb` and writes HTML reports.

```bash
python -m clustering_pipeline.cli analyze \
    --k 112 \
    --preprocessing resize \
    --table-name embeddings \
    --min-tissue-pct 20 \
    --output-dir /path/to/results \
    --embed-db-path /path/to/embeddings.db \
    --pred-db-path /path/to/predictions.db \
    --extended-db-path /path/to/extended_data.db
```

Key arguments:

- `--embed-db-path` – path to `embeddings.db` (default: `config.EMBED_DB_PATH`)
- `--pred-db-path` – path to `predictions.db` (default: `config.PRED_DB_PATH`)
- `--extended-db-path` – path to `extended_data.db` (default: `config.EXTENDED_DB_PATH`)
- `--table-name` – embeddings table name (default: `embeddings`)
- `--k` – number of clusters (for naming/plot titles)
- `--preprocessing` – preprocessing label (for naming/plot titles, default: `resize`)
- `--min-tissue-pct` – optional lower bound on `tissue_percentage` used for analysis (default: `None`)
- `--output-dir` – where to write HTML reports (default: `config.OUT_DIR_RESULTS`)

The command produces one or more HTML files visualizing per‑cluster counts and low‑tissue distributions.

### 11.7. Visualize (cluster image grids)

The `visualize` subcommand wraps `visualize_cluster_samples_duckdb` and generates PNG grids per cluster.

```bash
python -m clustering_pipeline.cli visualize \
    --k 112 \
    --table-name predictions \
    --rows 10 \
    --cols 10 \
    --num-grids 1 \
    --min-tissue-pct 20 \
    --embed-db-path /path/to/embeddings.db \
    --pred-db-path /path/to/predictions_extended.db \
    --extended-db-path /path/to/extended_data.db \
    --out-dir /path/to/results/cluster_grids
```

Additional arguments:

- `--cluster-id` – if set, only visualize this specific cluster (e.g. `--cluster-id 2`)
- `--rows`, `--cols` – grid shape per image (defaults: 10x10)
- `--num-grids` – how many grids per cluster (default: 1)
- `--min-tissue-pct` – minimum tissue percentage for tiles to be included (default: 20)
- `--out-dir` – output directory for PNG grids (default: `OUT_DIR_RESULTS/cluster_grids`)

**Note:** `s3_path_builder` is defined in `visualization.py` (or a helper module) and used internally by the CLI. Make sure it is implemented to map your `tile_key` to the correct S3 URI.

---

## 12. Example End‑to‑End Run (CLI‑only)

Assuming `config.py` is correctly set up:

```bash
# 1) Build DuckDB databases from Parquet
python -m clustering_pipeline.cli build-db

# 2) Create a stratified dataset (2M samples, up to 50k per tissue_type)
python -m clustering_pipeline.cli stratify  --per-class 50000  --sample-nb 2000000

# 3) Train MiniBatchKMeans on stratified_extended, with tissue_percentage >= 20
python -m clustering_pipeline.cli train  --table-name stratified_extended  --k 112  --batch-size 100000  --min-tissue-pct 20  --suffix 2000000above_20

# 4) Predict clusters for all embeddings
python -m clustering_pipeline.cli predict  --table-name embeddings  --k 112  --batch-size 1000000

# 5) Merge predictions with extended metadata and relabel low-tissue tiles
python -m clustering_pipeline.cli postprocess

# 6) Analyze per-cluster tissue statistics and low-tissue distributions
python -m clustering_pipeline.cli analyze  --k 112  --preprocessing resize  --min-tissue-pct 20

# 7) Generate clean cluster grids (tiles with tissue_percentage >= 20)
python -m clustering_pipeline.cli visualize  --k 112  --rows 10  --cols 10  --num-grids 1  --min-tissue-pct 20
```

After this sequence, you will have:

- `embeddings.db`, `extended_data.db`, `predictions.db`, `predictions_extended.db`
- A trained KMeans model in `models/`
- A stratified sample Parquet in `data/`
- Prediction Parquet(s) in `results/`
- HTML analysis reports in `results/`
- Cluster grid PNGs in `results/cluster_grids`
