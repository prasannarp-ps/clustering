from collections import Counter
import numpy as np
import pandas as pd
from collections import defaultdict
import duckdb
import pyarrow.parquet as pq

from . import config
from .duckdb_io import (
    duckdb_iter_batches,
    duckdb_connect_if_needed,
    generate_unique_id,
)

DUCKDB_PREFIX = "duckdb:"
DUCKDB_CONN = None  # used by stratified sampler for embeddings.db


# ---------------------------------------------------------
# 1. STRATIFIED SAMPLING FROM PARQUET OR DUCKDB
# ---------------------------------------------------------
def create_stratified_dataset(
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
):
    # -----------------------------------------
    # PASS 1 — COUNT VALUES PER CLASS PER FILE
    # -----------------------------------------
    print("PASS 1 — Counting values per class per file...")
    per_file_counts = []
    total_per_class = Counter()

    for path in parquet_paths:
        file_counter = Counter()

        if path.startswith(DUCKDB_PREFIX):
            table = path[len(DUCKDB_PREFIX):]
            batches = duckdb_iter_batches(
                table=table,
                batch_size=batch_size,
                columns=[stratify_field, "modality"]
            )
        else:
            pf = pq.ParquetFile(path)
            batches = (
                batch.to_pandas()
                for batch in pf.iter_batches(
                    batch_size=batch_size,
                    columns=[stratify_field, "modality"]
                )
            )

        for df in batches:
            df = df[df["modality"] == modality]
            vc = df[stratify_field].value_counts(dropna=False)

            for cls, cnt in vc.items():
                file_counter[cls] += int(cnt)
                total_per_class[cls] += int(cnt)

        per_file_counts.append(file_counter)

    # -----------------------------------------
    # FILTER CLASSES + COMPUTE TARGETS
    # -----------------------------------------
    classes = [
        c for c in total_per_class.keys()
        if total_per_class[c] > 0 and not str(c).isdigit()
    ]
    print("Classes:", classes)

    raw_targets = {c: min(per_class, total_per_class[c]) for c in classes}
    total_target = sum(raw_targets.values())

    if total_target > max_samples:
        scale = max_samples / total_target
        floored = {c: int(np.floor(raw_targets[c] * scale)) for c in classes}
        remaining = max_samples - sum(floored.values())

        remainders = {
            c: (raw_targets[c] * scale - floored[c]) for c in classes
        }
        for c in sorted(classes, key=lambda x: remainders[x], reverse=True)[:remaining]:
            floored[c] += 1

        targets = floored
    else:
        targets = raw_targets

    print("Computed per-class target quotas:", targets)

    # -----------------------------------------
    # DISTRIBUTE QUOTAS PER FILE
    # -----------------------------------------
    quotas = [defaultdict(int) for _ in parquet_paths]

    for c in classes:
        total_count = total_per_class[c]
        if total_count == 0 or targets[c] == 0:
            continue

        proportional = [
            (i, per_file_counts[i][c] / total_count * targets[c])
            for i in range(len(parquet_paths))
        ]

        floors = {i: int(np.floor(val)) for i, val in proportional}
        assigned = sum(floors.values())
        need = targets[c] - assigned

        rema = sorted(
            [(i, proportional[i][1] - floors[i]) for i in range(len(parquet_paths))],
            key=lambda x: x[1],
            reverse=True
        )

        for i, _ in rema[:need]:
            floors[i] += 1

        for i, q in floors.items():
            if q > 0:
                quotas[i][c] = q

    # -----------------------------------------
    # PASS 2 — SAMPLING WITH PROGRESS BAR
    # -----------------------------------------
    print("\nPASS 2 — Sampling according to quotas...\n")

    collected = []
    seen_ids = set()

    total_samples_needed = sum(sum(q.values()) for q in quotas)
    samples_collected = 0

    for file_idx, path in enumerate(parquet_paths):
        file_quota = quotas[file_idx]
        if not file_quota:
            continue

        print(f"Source {file_idx+1}/{len(parquet_paths)}: {path}")
        print("Initial quotas:", dict(file_quota))

        # Select batch iterator
        if path.startswith(DUCKDB_PREFIX):
            table = path[len(DUCKDB_PREFIX):]

            con = duckdb_connect_if_needed()
            total_rows = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE modality='{modality}'"
            ).fetchone()[0]

            batches = duckdb_iter_batches(
                table=table,
                batch_size=batch_size,
                columns=None
            )
        else:
            pf = pq.ParquetFile(path)
            total_rows = pf.metadata.num_rows

            batches = (
                batch.to_pandas()
                for batch in pf.iter_batches(batch_size=batch_size)
            )

        scanned_rows = 0

        for df in batches:
            scanned_rows += len(df)

            df = df[df["modality"] == modality]

            if stratify_field not in df.columns:
                continue

            if id_col not in df.columns:
                df[id_col] = df.apply(generate_unique_id, axis=1)

            df = df.drop_duplicates(subset=[id_col])
            df = df[~df[id_col].isin(seen_ids)]

            pending = {c: q for c, q in file_quota.items() if q > 0}
            if not pending:
                break

            for c, q in list(pending.items()):
                grp = df[df[stratify_field] == c]
                if len(grp) == 0:
                    continue

                take_n = min(q, len(grp))
                sample = grp.sample(n=take_n, random_state=random_state)

                collected.append(sample)
                seen_ids.update(sample[id_col].tolist())
                file_quota[c] -= take_n
                samples_collected += take_n

            # --- PROGRESS BAR ---
            pct = scanned_rows / total_rows if total_rows else 1.0
            bar_len = 30
            filled = int(pct * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)

            print(
                f"\r   [{bar}] {pct*100:5.1f}%  "
                f"rows {scanned_rows:,}/{total_rows:,}  "
                f" | samples {samples_collected:,}/{total_samples_needed:,}",
                end=""
            )

        print("\nCompleted:", path)
        print("Remaining quotas:", dict(file_quota))
        print()

    # -----------------------------------------
    # FINAL DATAFRAME
    # -----------------------------------------
    sampled_df = pd.concat(collected, ignore_index=True)
    print(f"\nFinal sampled dataset size: {len(sampled_df):,}")
    # ---------------------------------------------------------
    # OPTIONAL: Store stratified dataset into DuckDB
    # ---------------------------------------------------------
    if write_to_duckdb:
        if embed_db_path is None:
            raise ValueError("embed_db_path must be provided when write_to_duckdb=True")
        if extended_db_path is None:
            raise ValueError("extended_db_path must be provided when write_to_duckdb=True")

        save_stratified_to_duckdb(
            stratified_df=sampled_df,
            embed_db_path=embed_db_path,
            extended_db_path=extended_db_path,
            stratified_table=stratified_table,
            stratified_ext_table=stratified_ext_table,
        )

    if stats:
        for col in stats:
            if col in sampled_df.columns:
                print(f"\nStats for {col}:")
                print(sampled_df[col].value_counts())

    return sampled_df


def save_stratified_to_duckdb(
    stratified_df,
    embed_db_path,
    extended_db_path,
    stratified_table="stratified",
    stratified_ext_table="stratified_extended"
):
    """
    Saves stratified_df into embeddings.db as:
        stratified               (just sampled metadata + embeddings)
        stratified_extended      (same + tissue_percentage from extended_data.db)

    After saving, you can filter the extended table:
        SELECT * FROM stratified_extended WHERE tissue_percentage >= 20;
    """

    embed_db_sql = embed_db_path.replace("'", "''")
    ext_db_sql = extended_db_path.replace("'", "''")

    print(f"\n🔄 Writing stratified dataset into {embed_db_path} ...")

    con = duckdb.connect(embed_db_path)

    # Make sure extended DB is available for joining
    con.execute(f"ATTACH '{ext_db_sql}' AS ext_db;")

    # Register stratified_df as DuckDB view
    con.register("strat_df", stratified_df)

    # Drop existing tables if needed
    con.execute(f"DROP TABLE IF EXISTS {stratified_table};")
    con.execute(f"DROP TABLE IF EXISTS {stratified_ext_table};")

    # ---------------------------
    # 1) CREATE stratified TABLE
    # ---------------------------
    con.execute(f"""
        CREATE TABLE {stratified_table} AS
        SELECT * FROM strat_df;
    """)
    print(f"✅ Wrote table: {stratified_table}")

    # Add index for speed
    con.execute(f"CREATE INDEX idx_{stratified_table}_uid ON {stratified_table}(unique_id);")

    # ----------------------------------------------------------
    # 2) CREATE stratified_extended TABLE WITH TISSUE PERCENTAGE
    # ----------------------------------------------------------
    con.execute(f"""
        CREATE TABLE {stratified_ext_table} AS
        SELECT
            s.*,
            x.tissue_percentage
        FROM strat_df s
        LEFT JOIN (
                SELECT {config.norm_tile_key_sql('tile_key')} AS norm_key,
                       MAX(tissue_percentage) AS tissue_percentage
                FROM ext_db.extended_data
                GROUP BY {config.norm_tile_key_sql('tile_key')}
            ) x ON {config.norm_tile_key_sql('s.tile_key')} = x.norm_key;
    """)
    print(f"✅ Wrote table with extended metadata: {stratified_ext_table}")

    con.execute(f"CREATE INDEX idx_{stratified_ext_table}_uid ON {stratified_ext_table}(unique_id);")

    print("🎉 Stratified tables saved successfully.\n")


#
#
#
# if __name__ == "__main__":
#     # Base paths
#     BASE_DIR = "/Volumes/KINGSTON/embeddings_v7"
#     PARQUET_FILE_PATH = os.path.join(BASE_DIR, "global_embedding.parquet")
#     EXTENDED_PARQUET_FILE_PATH = os.path.join(BASE_DIR, "db_tiles_extended.parquet")
#     DB_DIR = os.path.join(BASE_DIR, "db")
#     OUT_DIR_MODEL = os.path.join(BASE_DIR, "models")
#     OUT_DIR_DATA = os.path.join(BASE_DIR, "data")
#     OUT_DIR_RESULTS = os.path.join(BASE_DIR, "results")
#
#     EMBED_DB_PATH = os.path.join(DB_DIR, "embeddings.db")
#     EXTENDED_DB_PATH = os.path.join(DB_DIR, "extended_data.db")
#     PRED_DB_PATH = os.path.join(DB_DIR, "predictions_above_20_new.db")
#
#     SAMPLE_NB = 2_000_000
#
#     os.makedirs(BASE_DIR, exist_ok=True)
#     os.makedirs(DB_DIR, exist_ok=True)
#     os.makedirs(OUT_DIR_MODEL, exist_ok=True)
#     os.makedirs(OUT_DIR_DATA, exist_ok=True)
#     os.makedirs(OUT_DIR_RESULTS, exist_ok=True)
#
#
#
#     # 1) Create or reuse embeddings DB from Parquet
#     embeddings_con = create_embeddings_db_from_parquet(embed_db_path=EMBED_DB_PATH, parquet_path=PARQUET_FILE_PATH)
#
#     create_duckdb_from_parquet_simple(EXTENDED_DB_PATH, EXTENDED_PARQUET_FILE_PATH, table_name="extended_data", generate_unique_id=False, index_unique_id=False)
#
#
#     DUCKDB_CONN = duckdb.connect(EMBED_DB_PATH)
#     paths = ["duckdb:embeddings"]
#     df_sampled = create_stratified_dataset(
#         paths,
#         stratify_field="tissue_type",
#         modality="stained",
#         per_class=50_000,
#         max_samples=SAMPLE_NB,
#         write_to_duckdb=True,
#         embed_db_path=EMBED_DB_PATH,
#         extended_db_path=EXTENDED_DB_PATH,
#     )
#     df_sampled.to_parquet(
#         os.path.join(OUT_DIR_DATA, "sample_stratified_above_20.parquet"),
#         index=False
#     )
#     # exit()
#
#
#     # Example: training KMeans model on sampled data
#     # df_sampled = pd.read_parquet(
#     #     os.path.join(OUT_DIR_DATA, "sample_stratified_above_20.parquet")
#     # )
#     fit_kmeans_model_duckdb(embed_db_path=EMBED_DB_PATH,
#                             extended_db_path=EXTENDED_DB_PATH,
#                             table_name="stratified_extended",
#                             min_tissue_pct=20,
#                             k=112,
#                             suffix=str(SAMPLE_NB) + "above_20_new",
#                             out_dir=OUT_DIR_MODEL)
#     exit()
#     # fit_kmeans_model(df_sampled, k=112, suffix=str(SAMPLE_NB) + "above_20", out_dir=OUT_DIR_MODEL)
#     # exit()
#     con = duckdb.connect(EMBED_DB_PATH)
#
#     # 2) Predict clusters for all embeddings into predictions.db and export to Parquet
#     model_path = os.path.join(OUT_DIR_MODEL, "kmeans_model_k112_2000000above_20_new.joblib")
#     out_parquet_path = os.path.join(OUT_DIR_RESULTS, "clustering_results_all_k112above_20_new.parquet")
#
#     predict_kmeans_model_duckdb(
#         model_path=model_path,
#         embed_db_path=EMBED_DB_PATH,
#         pred_db_path=PRED_DB_PATH,
#         table_name="embeddings",
#         out_parquet_path=out_parquet_path,
#         batch_size=1_000_000,
#         modality="stained",
#     )
#
#     create_predictions_extended_db(
#         pred_db_path=PRED_DB_PATH,
#         extended_db_path=EXTENDED_DB_PATH,
#         output_db_path="/Volumes/KINGSTON/embeddings_v7/db/predictions_above_20_extended.db"
#     )
#
#     export_predictions_table_to_parquet(
#         db_path=PRED_DB_PATH,
#         table_name="predictions",
#         out_parquet_path="/Volumes/KINGSTON/embeddings_v7/results/2025_12_12_clustering_results_all_k112above_20.parquet"
#     )
#
#
#     plot_cluster_tissue_info_duckdb(
#         embed_db_path=EMBED_DB_PATH,
#         pred_db_path=PRED_DB_PATH,
#         extended_db_path=EXTENDED_DB_PATH,
#         output_dir=OUT_DIR_RESULTS,
#         preprocessing="resize",
#         k=112,
#         table_name="embeddings",
#         min_tissue_pct=None,  # None = no filter
#     )
#
#     visualize_cluster_samples_duckdb(
#         embed_db_path=EMBED_DB_PATH,
#         pred_db_path="/Volumes/KINGSTON/embeddings_v7/db/predictions_above_20_extended.db",
#         extended_db_path=EXTENDED_DB_PATH,
#         table_name="predictions",
#         out_dir="/Volumes/KINGSTON/embeddings_v7/results/cluster_grids_new_above_20",
#         rows=10,
#         cols=10,
#         s3_path_builder=s3_path_builder,
#         min_tissue_pct=0
#     )