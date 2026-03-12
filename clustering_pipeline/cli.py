"""
Command-line interface for the embedding clustering pipeline.

Usage examples:

    # Build DuckDB databases from Parquet
    python -m embeddings_pipeline.cli build-db

    # Create a stratified sample and write it into embeddings.db
    python -m embeddings_pipeline.cli stratify --sample-nb 2000000 --per-class 50000

    # Train MiniBatchKMeans on stratified_extended table
    python -m embeddings_pipeline.cli train --k 112 --min-tissue-pct 20

    # Predict clusters for all embeddings
    python -m embeddings_pipeline.cli predict --batch-size 1000000

    # Merge predictions with extended metadata and relabel low-tissue tiles
    python -m embeddings_pipeline.cli postprocess

    # Analyze cluster/tissue composition and write Plotly HTML reports
    python -m embeddings_pipeline.cli analyze --k 112 --preprocessing resize

    # Visualize sample tiles per cluster as PNG grids
    python -m embeddings_pipeline.cli visualize --k 112 --num-grids 3 --rows 10 --cols 10
"""

import argparse
import os
import duckdb
from . import config
from .duckdb_io import (
    create_embeddings_db_from_parquet,
    create_duckdb_from_parquet_simple,
)
from .sampling import create_stratified_dataset
from .training import fit_kmeans_model_duckdb
from .prediction import predict_kmeans_model_duckdb
from .postprocessing import create_predictions_extended_db
from .analysis import plot_cluster_tissue_info_duckdb
from .visualization import visualize_cluster_samples_duckdb, s3_path_builder


def cmd_build_db(args: argparse.Namespace) -> None:
    """Create embeddings.db and extended_data.db from Parquet files."""
    embed_parquet = args.embed_parquet or config.PARQUET_FILE_PATH
    ext_parquet = args.ext_parquet or config.EXTENDED_PARQUET_FILE_PATH
    embed_db = args.embed_db_path or config.EMBED_DB_PATH

    print(f"Creating embeddings DB from: {embed_parquet}")
    create_embeddings_db_from_parquet(
        embed_db_path=embed_db,
        parquet_path=embed_parquet,
    )

    print(f"Creating extended metadata DB from: {ext_parquet}")
    con = create_duckdb_from_parquet_simple(
        db_path=config.EXTENDED_DB_PATH,
        parquet_path=ext_parquet,
        table_name="extended_data",
        generate_unique_id=False,
        index_unique_id=False,
    )

    # Enrich extended_data with tissue_percentage from blur features
    blur_parquet = args.blur_parquet or config.BLUR_PARQUET_FILE_PATH
    if os.path.exists(blur_parquet):
        print(f"Adding tissue_percentage from: {blur_parquet}")
        blur_sql = blur_parquet.replace("'", "''")
        con.execute("ALTER TABLE extended_data ADD COLUMN IF NOT EXISTS tissue_percentage DOUBLE;")
        con.execute(f"""
            UPDATE extended_data AS e
            SET tissue_percentage = b.tissue_percentage
            FROM (
                SELECT
                    tissue_type || '_' || block_id || '_' || slice_id || '_' ||
                    scan_date || '_' || box_id || '_' || tile_filename AS tile_key,
                    tissue_percentage
                FROM read_parquet('{blur_sql}')
                WHERE stained = true
            ) b
            WHERE {config.norm_tile_key_sql('e.tile_key')} = {config.norm_tile_key_sql('b.tile_key')};
        """)
        filled = con.execute(
            "SELECT COUNT(*) FROM extended_data WHERE tissue_percentage IS NOT NULL"
        ).fetchone()[0]
        total = con.execute("SELECT COUNT(*) FROM extended_data").fetchone()[0]
        print(f"  tissue_percentage populated: {filled:,} / {total:,} rows")
    else:
        print(f"WARNING: blur parquet not found at {blur_parquet}, skipping tissue_percentage")


def cmd_stratify(args: argparse.Namespace) -> None:
    """Create a stratified training dataset and write it into DuckDB."""
    global DUCKDB_CONN
    DUCKDB_CONN = duckdb.connect(config.EMBED_DB_PATH)

    paths = ["duckdb:embeddings"]

    sample_nb = args.sample_nb
    per_class = args.per_class

    df_sampled = create_stratified_dataset(
        parquet_paths=paths,
        stratify_field=args.stratify_field,
        modality=args.modality,
        per_class=per_class,
        max_samples=sample_nb,
        write_to_duckdb=True,
        embed_db_path=config.EMBED_DB_PATH,
        extended_db_path=config.EXTENDED_DB_PATH,
        stratified_table=args.stratified_table,
        stratified_ext_table=args.stratified_ext_table,
    )

    if args.sample_parquet:
        out_path = args.sample_parquet
    else:
        out_path = config.SAMPLE_PARQUET_PATH

    df_sampled.to_parquet(out_path, index=False)
    print(f"Saved stratified sample to: {out_path}")


def cmd_train(args: argparse.Namespace) -> None:
    """Train MiniBatchKMeans model from a DuckDB table."""
    table_name = args.table_name
    k = args.k
    min_tissue_pct = args.min_tissue_pct
    batch_size = args.batch_size
    suffix = args.suffix or f"{args.sample_nb or ''}_above_{min_tissue_pct or 'all'}"

    model_path = fit_kmeans_model_duckdb(
        embed_db_path=args.embed_db_path or config.EMBED_DB_PATH,
        extended_db_path=config.EXTENDED_DB_PATH,
        table_name=table_name,
        k=k,
        batch_size=batch_size,
        min_tissue_pct=min_tissue_pct,
        modality=args.modality,
        suffix=suffix,
        out_dir=args.model_dir or config.OUT_DIR_MODEL,
    )

    print(f"Model saved to: {model_path}")


def cmd_predict(args: argparse.Namespace) -> None:
    """Predict clusters for all embeddings and write them to predictions.db."""
    model_path = args.model_path
    if model_path is None:
        model_dir = args.model_dir or config.OUT_DIR_MODEL
        model_path = os.path.join(
            model_dir,
            f"kmeans_model_k{args.k or config.DEFAULT_K}.joblib",
        )

    results_dir = args.results_dir or config.OUT_DIR_RESULTS
    out_parquet = args.out_parquet or os.path.join(
        results_dir,
        f"clustering_results_all_k{args.k or config.DEFAULT_K}.parquet",
    )

    predict_kmeans_model_duckdb(
        model_path=model_path,
        embed_db_path=args.embed_db_path or config.EMBED_DB_PATH,
        pred_db_path=args.pred_db_path or config.PRED_DB_PATH,
        table_name=args.table_name,
        batch_size=args.batch_size,
        modality=args.modality,
        out_parquet_path=out_parquet,
        extended_db_path=args.extended_db_path or config.EXTENDED_DB_PATH if args.min_tissue_pct is not None else None,
        min_tissue_pct=args.min_tissue_pct,
        tissue_low=args.tissue_low,
        tissue_mid=args.tissue_mid,
    )


def cmd_postprocess(args: argparse.Namespace) -> None:
    """Merge predictions with extended metadata and relabel low-tissue tiles."""
    pred_db_path = args.pred_db_path or config.PRED_DB_PATH
    output_db_path = args.output_db_path or os.path.join(
        config.DB_DIR, "predictions_extended.db"
    )

    create_predictions_extended_db(
        pred_db_path=pred_db_path,
        extended_db_path=config.EXTENDED_DB_PATH,
        output_db_path=output_db_path,
        pred_table=args.pred_table,
        ext_table=args.ext_table,
    )

    print(f"Extended predictions DB created at: {output_db_path}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """
    Run cluster/tissue analysis and generate Plotly HTML reports.

    This calls plot_cluster_tissue_info_duckdb, which joins:
        - embeddings.db
        - predictions.db
        - extended_data.db
    and writes multiple HTML files into the chosen output directory.
    """
    embed_db_path = args.embed_db_path or config.EMBED_DB_PATH
    pred_db_path = args.pred_db_path or config.PRED_DB_PATH
    extended_db_path = args.extended_db_path or config.EXTENDED_DB_PATH
    output_dir = args.output_dir or config.OUT_DIR_RESULTS

    plot_cluster_tissue_info_duckdb(
        embed_db_path=embed_db_path,
        pred_db_path=pred_db_path,
        extended_db_path=extended_db_path,
        output_dir=output_dir,
        preprocessing=args.preprocessing,
        k=args.k,
        table_name=args.table_name,
        min_tissue_pct=args.min_tissue_pct,
    )


def cmd_visualize(args: argparse.Namespace) -> None:
    """
    Visualize cluster samples as image grids.

    This downloads tiles (via s3_path_builder) and saves PNG grids per cluster.
    Uses predictions + extended_data to optionally filter by tissue_percentage.
    """
    embed_db_path = args.embed_db_path or config.EMBED_DB_PATH
    pred_db_path = args.pred_db_path or config.PRED_DB_PATH
    extended_db_path = args.extended_db_path or config.EXTENDED_DB_PATH

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(config.OUT_DIR_RESULTS, "cluster_grids")

    visualize_cluster_samples_duckdb(
        embed_db_path=embed_db_path,
        pred_db_path=pred_db_path,
        extended_db_path=extended_db_path,
        table_name=args.table_name,
        out_dir=out_dir,
        rows=args.rows,
        cols=args.cols,
        resize_dim=(args.tile_size, args.tile_size),
        font_scale=args.font_scale,
        font_thickness=args.font_thickness,
        padding=args.padding,
        cluster_id=args.cluster_id,
        num_grids=args.num_grids,
        k=args.k,
        s3_path_builder=s3_path_builder,
        downloader_workers=args.downloader_workers,
        min_tissue_pct=args.min_tissue_pct,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embedding clustering pipeline (DuckDB + MiniBatchKMeans)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-db
    p_build = subparsers.add_parser(
        "build-db",
        help="Create embeddings.db and extended_data.db from Parquet files.",
    )
    p_build.add_argument("--embed-parquet", type=str, default=None,
                         help="Path to global embedding Parquet (default: config.PARQUET_FILE_PATH)")
    p_build.add_argument("--ext-parquet", type=str, default=None,
                         help="Path to extended metadata Parquet (default: config.EXTENDED_PARQUET_FILE_PATH)")
    p_build.add_argument("--blur-parquet", type=str, default=None,
                         help="Path to blur features Parquet for tissue_percentage (default: config.BLUR_PARQUET_FILE_PATH)")
    p_build.add_argument("--embed-db-path", type=str, default=None,
                         help="Output path for embeddings.db (default: config.EMBED_DB_PATH)")
    p_build.set_defaults(func=cmd_build_db)

    # stratify
    p_strat = subparsers.add_parser(
        "stratify",
        help="Create a stratified training dataset and write it to DuckDB.",
    )
    p_strat.add_argument("--stratify-field", type=str, default="tissue_type")
    p_strat.add_argument("--modality", type=str, default=config.DEFAULT_MODALITY)
    p_strat.add_argument("--per-class", type=int, default=50_000)
    p_strat.add_argument("--sample-nb", type=int, default=2_000_000)
    p_strat.add_argument("--stratified-table", type=str, default="stratified")
    p_strat.add_argument("--stratified-ext-table", type=str, default="stratified_extended")
    p_strat.add_argument("--sample-parquet", type=str, default=None,
                         help="Optional path to save sampled parquet (default: config.SAMPLE_PARQUET_PATH)")
    p_strat.set_defaults(func=cmd_stratify)

    # train
    p_train = subparsers.add_parser(
        "train",
        help="Train MiniBatchKMeans on data from a DuckDB table.",
    )
    p_train.add_argument("--table-name", type=str, default="stratified_extended")
    p_train.add_argument("--k", type=int, default=config.DEFAULT_K)
    p_train.add_argument("--batch-size", type=int, default=100_000)
    p_train.add_argument("--min-tissue-pct", type=float, default=None)
    p_train.add_argument("--modality", type=str, default=config.DEFAULT_MODALITY)
    p_train.add_argument("--suffix", type=str, default=None,
                         help="Optional suffix for model filename.")
    p_train.add_argument("--sample-nb", type=int, default=None,
                         help="Optional: used only for building a default suffix.")
    p_train.add_argument("--embed-db-path", type=str, default=None,
                         help="Path to embeddings.db (default: config.EMBED_DB_PATH)")
    p_train.add_argument("--model-dir", type=str, default=None,
                         help="Directory to save trained model (default: config.OUT_DIR_MODEL)")
    p_train.set_defaults(func=cmd_train)

    # predict
    p_pred = subparsers.add_parser(
        "predict",
        help="Predict clusters for all embeddings and write them into predictions.db.",
    )
    p_pred.add_argument("--table-name", type=str, default="embeddings")
    p_pred.add_argument("--k", type=int, default=config.DEFAULT_K)
    p_pred.add_argument("--batch-size", type=int, default=100_000)
    p_pred.add_argument("--modality", type=str, default=config.DEFAULT_MODALITY)
    p_pred.add_argument("--model-path", type=str, default=None,
                        help="Path to a trained MiniBatchKMeans .joblib model.")
    p_pred.add_argument("--out-parquet", type=str, default=None,
                        help="Where to write predictions Parquet.")
    p_pred.add_argument("--embed-db-path", type=str, default=None,
                        help="Path to embeddings.db (default: config.EMBED_DB_PATH)")
    p_pred.add_argument("--pred-db-path", type=str, default=None,
                        help="Path to predictions.db (default: config.PRED_DB_PATH)")
    p_pred.add_argument("--model-dir", type=str, default=None,
                        help="Directory containing the model (default: config.OUT_DIR_MODEL)")
    p_pred.add_argument("--results-dir", type=str, default=None,
                        help="Directory for output parquet (default: config.OUT_DIR_RESULTS)")
    p_pred.add_argument("--extended-db-path", type=str, default=None,
                        help="Path to extended_data.db (default: config.EXTENDED_DB_PATH)")
    p_pred.add_argument("--min-tissue-pct", type=float, default=None,
                        help="Enable tissue filtering: tiles below this threshold are relabeled "
                             "(< tissue-low → cluster -2, tissue-low to tissue-mid → cluster -1).")
    p_pred.add_argument("--tissue-low", type=float, default=5.0,
                        help="Tissue %% below which tiles become cluster -2 (default: 5).")
    p_pred.add_argument("--tissue-mid", type=float, default=20.0,
                        help="Tissue %% below which tiles become cluster -1 (default: 20).")
    p_pred.set_defaults(func=cmd_predict)

    # postprocess
    p_post = subparsers.add_parser(
        "postprocess",
        help="Merge predictions with extended metadata and relabel low-tissue tiles.",
    )
    p_post.add_argument("--pred-db-path", type=str, default=None,
                        help="Path to predictions.db (default: config.PRED_DB_PATH)")
    p_post.add_argument("--output-db-path", type=str, default=None,
                        help="Path to write predictions_extended.db")
    p_post.add_argument("--pred-table", type=str, default="predictions")
    p_post.add_argument("--ext-table", type=str, default="extended_data")
    p_post.set_defaults(func=cmd_postprocess)

    # analyze
    p_an = subparsers.add_parser(
        "analyze",
        help="Run cluster/tissue analysis and generate Plotly HTML reports.",
    )
    p_an.add_argument("--embed-db-path", type=str, default=None,
                      help="Path to embeddings.db (default: config.EMBED_DB_PATH)")
    p_an.add_argument("--pred-db-path", type=str, default=None,
                      help="Path to predictions.db (default: config.PRED_DB_PATH)")
    p_an.add_argument("--extended-db-path", type=str, default=None,
                      help="Path to extended_data.db (default: config.EXTENDED_DB_PATH)")
    p_an.add_argument("--table-name", type=str, default="embeddings",
                      help="Name of embeddings table used in the join.")
    p_an.add_argument("--k", type=int, required=True,
                      help="Number of clusters (used only for naming/plots).")
    p_an.add_argument("--preprocessing", type=str, default="resize",
                      help="Preprocessing label (used only for naming/plots).")
    p_an.add_argument("--min-tissue-pct", type=float, default=None,
                      help="Optional: filter rows with tissue_percentage >= this value.")
    p_an.add_argument("--output-dir", type=str, default=None,
                      help="Directory to write HTML reports (default: config.OUT_DIR_RESULTS).")
    p_an.set_defaults(func=cmd_analyze)

    # visualize
    p_vis = subparsers.add_parser(
        "visualize",
        help="Visualize cluster samples as image grids (downloads tiles and writes PNGs).",
    )
    p_vis.add_argument("--embed-db-path", type=str, default=None,
                       help="Path to embeddings.db (default: config.EMBED_DB_PATH)")
    p_vis.add_argument("--pred-db-path", type=str, default=None,
                       help="Path to predictions.db (default: config.PRED_DB_PATH)")
    p_vis.add_argument("--extended-db-path", type=str, default=None,
                       help="Path to extended_data.db (default: config.EXTENDED_DB_PATH)")
    p_vis.add_argument("--table-name", type=str, default="predictions",
                       help="Predictions table name inside predictions DB.")
    p_vis.add_argument("--out-dir", type=str, default=None,
                       help="Output directory for cluster grids (default: OUT_DIR_RESULTS/cluster_grids).")
    p_vis.add_argument("--rows", type=int, default=10,
                       help="Number of rows per grid.")
    p_vis.add_argument("--cols", type=int, default=10,
                       help="Number of columns per grid.")
    p_vis.add_argument("--tile-size", type=int, default=256,
                       help="Tile resize dimension (tile_size x tile_size).")
    p_vis.add_argument("--font-scale", type=float, default=0.4,
                       help="Font scale for tile labels.")
    p_vis.add_argument("--font-thickness", type=int, default=1,
                       help="Font thickness for tile labels.")
    p_vis.add_argument("--padding", type=int, default=2,
                       help="Padding in pixels around each tile.")
    p_vis.add_argument("--cluster-id", type=int, default=None,
                       help="If set, visualize only this cluster. Otherwise, all clusters.")
    p_vis.add_argument("--num-grids", type=int, default=1,
                       help="Number of grids to generate per cluster.")
    p_vis.add_argument("--k", type=int, default=None,
                       help="Optional: k value for subfolder naming.")
    p_vis.add_argument("--min-tissue-pct", type=float, default=20.0,
                       help="Minimum tissue_percentage to include (default: 20).")
    p_vis.add_argument("--downloader-workers", type=int, default=8,
                       help="Number of parallel workers for S3 downloads.")
    p_vis.set_defaults(func=cmd_visualize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
