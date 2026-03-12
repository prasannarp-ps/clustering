"""
Model training utilities (MiniBatchKMeans).

This module handles training on either:
- an in-memory pandas DataFrame, or
- data streamed directly from DuckDB.
"""
import os
import numpy as np
import duckdb
import joblib
from sklearn.cluster import MiniBatchKMeans
from . import config


# Functions to move here from the big script:
#
# - fit_kmeans_model
# - fit_kmeans_model_duckdb
#
# These functions depend on:
#   - numpy
#   - joblib
#   - sklearn.cluster.MiniBatchKMeans
#   - duckdb (for the DuckDB-version)
#
# and, for the DuckDB version, `extended_data.db` and tissue filters.


def fit_kmeans_model(df, k, batch_size=100_000, suffix=None, out_dir="./models"):
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"kmeans_model_k{k}"
    if suffix:
        out_name += f"_{suffix}"

    mbk = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    total = len(df)

    done = 0
    for start in range(0, total, batch_size):
        batch = df.iloc[start:start + batch_size]
        vectors = np.vstack(batch["vector"].tolist())
        mbk.partial_fit(vectors)
        done += len(batch)
        print(f"Progress: {done}/{total}")

    model_path = os.path.join(out_dir, out_name + ".joblib")
    joblib.dump(mbk, model_path)
    print("Saved model to:", model_path)
    return model_path


def fit_kmeans_model_duckdb(
    embed_db_path,
    extended_db_path,
    table_name="embeddings",
    k=112,
    batch_size=100_000,
    min_tissue_pct=None, # (None = no filtering)
    modality="stained",
    suffix=None,
    out_dir="./models",
):
    """
    Train MiniBatchKMeans ON DUCKDB DATA directly, streaming batches and
    optionally filtering by tissue_percentage >= min_tissue_pct.
    """

    os.makedirs(out_dir, exist_ok=True)
    out_name = f"kmeans_model_k{k}"
    if suffix:
        out_name += f"_{suffix}"
    model_path = os.path.join(out_dir, out_name + ".joblib")

    # -------------------------------------------
    # CONNECT TO DATABASE
    # -------------------------------------------
    con = duckdb.connect(embed_db_path)
    ext_sql = extended_db_path.replace("'", "''")
    con.execute(f"ATTACH '{ext_sql}' AS ext_db;")

    print("📂 Connected to embeddings DB and extended DB")

    # -------------------------------------------
    # BUILD SQL CLAUSES
    # -------------------------------------------
    join_clause = ""
    tissue_filter = ""

    if min_tissue_pct is not None:
        nk = config.norm_tile_key_sql
        # Deduplicate extended_data by normalized tile_key + modality to avoid
        # many-to-many join explosion (multiple box_ids share the same norm key).
        dedup_subquery = (
            f"(SELECT {nk('tile_key')} AS norm_key, "
            f"MAX(tissue_percentage) AS tissue_percentage "
            f"FROM ext_db.extended_data "
            f"WHERE modality = '{modality}' "
            f"GROUP BY {nk('tile_key')}) x"
        )
        join_clause = f"JOIN {dedup_subquery} ON {nk('e.tile_key')} = x.norm_key"
        tissue_filter = f"AND x.tissue_percentage >= {min_tissue_pct}"
        print(f"🔎 Using filter: tissue_percentage >= {min_tissue_pct}%")

    # -------------------------------------------
    # COUNT TRAINING ROWS
    # -------------------------------------------
    total_rows = con.execute(f"""
        SELECT COUNT(*)
        FROM {table_name} e
        {join_clause}
        WHERE e.modality = '{modality}'
        {tissue_filter}
    """).fetchone()[0]

    print(f"🧮 Total rows for training: {total_rows:,}")

    if total_rows == 0:
        raise ValueError("❌ No data available for training after filtering.")

    # -------------------------------------------
    # INITIALIZE MiniBatchKMeans
    # -------------------------------------------
    mbk = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        random_state=42
    )

    # -------------------------------------------
    # STREAM BATCHES FROM DUCKDB (SHUFFLED)
    # -------------------------------------------
    print("\n🚀 Starting MiniBatchKMeans training...\n")

    offset = 0
    processed = 0

    while offset < total_rows:

        df = con.execute(f"""
            SELECT e.vector
            FROM {table_name} e
            {join_clause}
            WHERE e.modality = '{modality}'
            {tissue_filter}
            ORDER BY RANDOM()      -- shuffle inside DB
            LIMIT {batch_size}
        """).df()

        if df.empty:
            break

        vectors = np.vstack(df["vector"].to_numpy())
        mbk.partial_fit(vectors)

        processed += len(df)
        pct = processed / total_rows * 100
        print(f"Progress: {processed:,}/{total_rows:,} ({pct:5.2f}%)")

        offset += batch_size

    # -------------------------------------------
    # SAVE MODEL
    # -------------------------------------------
    joblib.dump(mbk, model_path)
    print(f"\n🎉 Saved model to: {model_path}")

    return model_path
