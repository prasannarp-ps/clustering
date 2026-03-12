"""
End-to-end clustering pipeline for a single embedding model.

Paths are automatically derived from the model name:
  Parquet  : data/global_embedding_{model}.parquet
  DB       : db/{model}/embeddings.db  |  predictions.db
  Models   : models/{model}/
  Results  : results/{model}/

Usage:
    python run_model_pipeline.py --model uni
    python run_model_pipeline.py --model uni2 --k 16 --pca 0.9
    python run_model_pipeline.py --model path_foundation --parquet data/global_embedding.parquet
    python run_model_pipeline.py --model uni --skip-optimal-k --skip-train
"""

import argparse
import os
import subprocess
import sys


def run(cmd: list[str], step: str):
    print(f"\n{'='*60}")
    print(f"  STEP: {step}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Full clustering pipeline for one embedding model"
    )
    parser.add_argument("--model", required=True,
                        help="Model name: uni | uni2 | conch | path_foundation (used for paths)")
    parser.add_argument("--parquet", default=None,
                        help="Override input parquet path (default: data/global_embedding_{model}.parquet)")
    parser.add_argument("--k", type=int, default=16,
                        help="Number of clusters (default: 16)")
    parser.add_argument("--pca", type=float, default=0.9,
                        help="PCA: int for fixed dims (e.g. 100), float 0-1 for variance threshold "
                             "(e.g. 0.9 = 90%%). Default: 0.9. Use 0 or --no-pca to skip.")
    parser.add_argument("--umap-method", default="umap", choices=["umap", "tsne"],
                        help="Projection method (default: umap)")
    # Skip flags
    parser.add_argument("--min-tissue-pct", type=float, default=None,
                        help="Filter low-tissue tiles: < 5%% → cluster -2, 5-20%% → cluster -1 "
                             "(passed to train and predict)")
    parser.add_argument("--skip-optimal-k",  action="store_true", help="Skip find_optimal_k_advanced step")
    parser.add_argument("--skip-build-db",   action="store_true", help="Skip build-db step")
    parser.add_argument("--skip-train",      action="store_true", help="Skip train step")
    parser.add_argument("--skip-predict",    action="store_true", help="Skip predict step")
    parser.add_argument("--skip-postprocess",action="store_true", help="Skip postprocess step")
    parser.add_argument("--skip-analyze",    action="store_true", help="Skip analyze step")
    parser.add_argument("--skip-projection", action="store_true", help="Skip projection_plot step")
    parser.add_argument("--skip-grids",      action="store_true", help="Skip cluster image grid step")
    parser.add_argument("--grid-rows", type=int, default=5, help="Grid rows (default: 5)")
    parser.add_argument("--grid-cols", type=int, default=5, help="Grid cols (default: 5)")
    parser.add_argument("--tile-size", type=int, default=224, help="Tile size in pixels (default: 224)")
    args = parser.parse_args()

    model      = args.model
    k          = args.k
    pca        = args.pca

    parquet    = args.parquet or f"data/global_embedding_{model}.parquet"
    db_dir     = f"db/{model}"
    embed_db   = f"{db_dir}/embeddings.db"
    pred_db    = f"{db_dir}/predictions.db"
    ext_db     = f"{db_dir}/predictions_extended.db"
    model_dir  = f"models/{model}"
    results_dir= f"results/{model}"
    tissue_suffix = f"_above_{args.min_tissue_pct}" if args.min_tissue_pct is not None else "_above_all"
    model_path = f"{model_dir}/kmeans_model_k{k}_{tissue_suffix}.joblib"
    pred_parquet = f"{results_dir}/clustering_results_all_k{k}.parquet"
    proj_out   = f"{results_dir}/{args.umap_method}_k{k}_stratified.html"

    for d in (db_dir, model_dir, results_dir):
        os.makedirs(d, exist_ok=True)

    py = sys.executable

    # ── 1. Find optimal k ──────────────────────────────────────────────────────
    if not args.skip_optimal_k:
        cmd = [py, "find_optimal_k_advanced.py",
               "--parquet", parquet,
               "--silhouette",
               "--output-dir", results_dir]
        if pca > 0:
            cmd += ["--pca", str(pca)]
        else:
            cmd += ["--no-pca"]
        run(cmd, "find_optimal_k_advanced")

    # ── 2. Build DuckDB ────────────────────────────────────────────────────────
    if not args.skip_build_db:
        run([py, "-m", "clustering_pipeline.cli", "build-db",
             "--embed-parquet", parquet,
             "--embed-db-path", embed_db],
            "build-db")

    # ── 3. Train ───────────────────────────────────────────────────────────────
    if not args.skip_train:
        train_cmd = [py, "-m", "clustering_pipeline.cli", "train",
                     "--table-name", "embeddings",
                     "--k", str(k),
                     "--embed-db-path", embed_db,
                     "--model-dir", model_dir]
        if args.min_tissue_pct is not None:
            train_cmd += ["--min-tissue-pct", str(args.min_tissue_pct)]
        run(train_cmd, "train")

    # ── 4. Predict ─────────────────────────────────────────────────────────────
    if not args.skip_predict:
        predict_cmd = [py, "-m", "clustering_pipeline.cli", "predict",
                       "--k", str(k),
                       "--model-path", model_path,
                       "--embed-db-path", embed_db,
                       "--pred-db-path", pred_db,
                       "--results-dir", results_dir]
        if args.min_tissue_pct is not None:
            predict_cmd += ["--min-tissue-pct", str(args.min_tissue_pct)]
        run(predict_cmd, "predict")

    # ── 5. Postprocess ─────────────────────────────────────────────────────────
    # Skip when --min-tissue-pct is set: relabeling (-2/-1) already happened
    # during predict, so postprocess would double-relabel.
    if not args.skip_postprocess and args.min_tissue_pct is None:
        run([py, "-m", "clustering_pipeline.cli", "postprocess",
             "--pred-db-path", pred_db,
             "--output-db-path", ext_db],
            "postprocess")
    elif args.min_tissue_pct is not None and not args.skip_postprocess:
        print("\n⏭  Skipping postprocess (tissue relabeling already applied in predict)")

    # ── 6. Analyze ─────────────────────────────────────────────────────────────
    if not args.skip_analyze:
        run([py, "-m", "clustering_pipeline.cli", "analyze",
             "--k", str(k),
             "--embed-db-path", embed_db,
             "--pred-db-path", pred_db,
             "--output-dir", results_dir],
            "analyze")

    # ── 7. Projection plot ─────────────────────────────────────────────────────
    if not args.skip_projection:
        cmd = [py, "projection_plot.py",
               "--method", args.umap_method,
               "--model", model,
               "--embed-parquet", parquet,
               "--pred-parquet", pred_parquet,
               "--out", proj_out]
        if pca > 0:
            cmd += ["--pca-components", str(pca)]
        else:
            cmd += ["--no-pca"]
        run(cmd, "projection_plot")

    # ── 8. Cluster image grids ─────────────────────────────────────────────────
    grids_dir = f"{results_dir}/cluster_grids"
    if not args.skip_grids:
        run([py, "cluster_image_grid.py",
             "--embed-parquet", parquet,
             "--pred-parquet", pred_parquet,
             "--out-dir", grids_dir,
             "--rows", str(args.grid_rows),
             "--cols", str(args.grid_cols),
             "--tile-size", str(args.tile_size)],
            "cluster_image_grid")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete for model: {model}")
    print(f"  Results : {results_dir}/")
    print(f"  Model   : {model_path}")
    print(f"  Plot    : {proj_out}")
    print(f"  Grids   : {grids_dir}/overview.html")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
