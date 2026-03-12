"""
Prediction utilities for assigning cluster IDs to all embeddings.

This module is responsible for streaming from DuckDB, running
MiniBatchKMeans predictions in batches, and writing predictions into
a separate DuckDB database (predictions.db).
"""
import time

import duckdb
import joblib
import numpy as np
from tqdm import tqdm
from . import config


def predict_kmeans_model_duckdb(
    model_path,
    embed_db_path,
    pred_db_path,
    table_name="embeddings",
    batch_size=100_000,
    modality="stained",
    out_parquet_path=None,
    extended_db_path=None,
    min_tissue_pct=None,
    tissue_low=5.0,
    tissue_mid=20.0,
):
    """
    Fast single-process KMeans prediction using:
      - DuckDB streaming + anti-join
      - tqdm progress bar
      - optimized vector extraction
      - minimized pandas overhead

    Tissue filtering (optional, requires extended_db_path + min_tissue_pct):
      - tiles with tissue_percentage < tissue_low (default 5%)  → cluster = -2
      - tiles with tissue_percentage < tissue_mid (default 20%) → cluster = -1
      - remaining tiles get their normal KMeans cluster assignment
    """

    print("\n🚀 Loading KMeans model...")
    model = joblib.load(model_path)

    print(f"🔌 Connecting to embeddings DB: {embed_db_path}")
    con = duckdb.connect(embed_db_path, read_only=False)

    # Attach predictions.db
    pred_sql = pred_db_path.replace("'", "''")
    con.execute(f"ATTACH '{pred_sql}' AS pred_db;")

    # Attach extended_data.db for tissue filtering
    do_tissue_filter = min_tissue_pct is not None and extended_db_path is not None
    if do_tissue_filter:
        ext_sql = extended_db_path.replace("'", "''")
        con.execute(f"ATTACH '{ext_sql}' AS ext_db;")
        print(f"🔎 Tissue filtering enabled: < {tissue_low}% → cluster -2, "
              f"{tissue_low}-{tissue_mid}% → cluster -1")

    # Ensure predictions table exists
    con.execute("""
        CREATE TABLE IF NOT EXISTS pred_db.predictions (
            unique_id VARCHAR,
            id VARCHAR,
            slide_key VARCHAR,
            tile_key VARCHAR,
            short_box_name VARCHAR,
            box_id VARCHAR,
            cluster INTEGER
        );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS pred_uid_idx ON pred_db.predictions(unique_id);")
    print("📌 Index ensured.")

    # Count remaining rows (ANTI-JOIN)
    total_to_predict = con.execute(f"""
        SELECT COUNT(*)
        FROM {table_name} e
        LEFT JOIN pred_db.predictions p USING(unique_id)
        WHERE e.modality = '{modality}' AND p.unique_id IS NULL;
    """).fetchone()[0]

    print(f"\n📊 Rows remaining to predict: {total_to_predict:,}")

    if total_to_predict == 0:
        print("Nothing to do.")
        if out_parquet_path:
            con.execute(f"COPY pred_db.predictions TO '{out_parquet_path}' (FORMAT PARQUET)")
        return 0

    processed = 0
    start_time = time.time()

    # ----------------------------------------
    # tqdm PROGRESS BAR
    # ----------------------------------------
    pbar = tqdm(total=total_to_predict, unit="rows", dynamic_ncols=True)

    # ----------------------------------------
    # MAIN STREAMING LOOP
    # ----------------------------------------
    while processed < total_to_predict:

        # Fast batch retrieval
        if do_tissue_filter:
            nk = config.norm_tile_key_sql
            # Deduplicate extended_data by normalized tile_key + modality to
            # avoid many-to-many join (multiple box_ids share the same norm key).
            df = con.execute(f"""
                SELECT
                    e.unique_id,
                    e.id,
                    e.slide_key,
                    e.tile_key,
                    e.short_box_name,
                    e.box_id,
                    e.vector,
                    xd.tissue_percentage
                FROM {table_name} e
                LEFT JOIN pred_db.predictions p USING(unique_id)
                LEFT JOIN (
                    SELECT {nk('tile_key')} AS norm_key,
                           MAX(tissue_percentage) AS tissue_percentage
                    FROM ext_db.extended_data
                    WHERE modality = '{modality}'
                    GROUP BY {nk('tile_key')}
                ) xd ON {nk('e.tile_key')} = xd.norm_key
                WHERE e.modality = '{modality}'
                  AND p.unique_id IS NULL
                LIMIT {batch_size};
            """).df()
        else:
            df = con.execute(f"""
                SELECT
                    e.unique_id,
                    e.id,
                    e.slide_key,
                    e.tile_key,
                    e.short_box_name,
                    e.box_id,
                    e.vector
                FROM {table_name} e
                LEFT JOIN pred_db.predictions p USING(unique_id)
                WHERE e.modality = '{modality}'
                  AND p.unique_id IS NULL
                LIMIT {batch_size};
            """).df()

        if df.empty:
            break

        n = len(df)

        # Convert vectors → np.ndarray
        vectors = np.stack(df["vector"].values)

        # Predict KMeans
        clusters = model.predict(vectors)

        # Apply tissue-based relabeling: < tissue_low → -2, < tissue_mid → -1
        if do_tissue_filter:
            tissue_pct = df["tissue_percentage"].fillna(100.0).values.astype(np.float32)
            clusters[tissue_pct < tissue_low] = -2
            clusters[(tissue_pct >= tissue_low) & (tissue_pct < tissue_mid)] = -1

        # Prepare for insertion (no heavy transforms)
        df_out = df[["unique_id", "id", "slide_key", "tile_key", "short_box_name", "box_id"]].copy()
        df_out["cluster"] = clusters

        # Insert into predictions.db
        con.register("batch_df", df_out)
        con.execute("INSERT INTO pred_db.predictions SELECT * FROM batch_df;")
        con.unregister("batch_df")

        processed += n
        pbar.update(n)

        # ETA printed by tqdm automatically

    pbar.close()

    # ----------------------------------------
    # Export predictions
    # ----------------------------------------
    if out_parquet_path:
        out_sql = out_parquet_path.replace("'", "''")
        con.execute(f"COPY pred_db.predictions TO '{out_sql}' (FORMAT PARQUET)")
        print(f"\n💾 Exported predictions to {out_parquet_path}")

    total_minutes = (time.time() - start_time) / 60
    print(f"\n🎉 Done! Total time: {total_minutes:.2f} minutes")

    return processed


def export_predictions_table_to_parquet(
    db_path,
    table_name="predictions",
    out_parquet_path="predictions_export.parquet"
):
    """
    Export a DuckDB table to Parquet.
    Useful for exporting corrected predictions (e.g., with -1/-2 clusters).
    """
    print(f"\n📂 Connecting to DB: {db_path}")
    con = duckdb.connect(db_path, read_only=True)

    out_sql = out_parquet_path.replace("'", "''")
    print(f"💾 Exporting table '{table_name}' → {out_parquet_path}")

    con.execute(
        f"COPY {table_name} TO '{out_sql}' (FORMAT PARQUET)"
    )

    print("✅ Export complete.")
