"""
Postprocessing of predictions and extended metadata.

This includes merging predictions with extended metadata and
relabeling low-content tiles into special clusters.
"""
import os
import duckdb
from . import config


def create_predictions_extended_db(
    pred_db_path,
    extended_db_path,
    output_db_path,
    pred_table="predictions",
    ext_table="extended_data",
):
    """
    Creates predictions_extended.db by merging predictions.db and extended_data.db
    and assigning new cluster IDs for tiles with low tissue percentage.

    Also prints cluster counts BEFORE and AFTER modification.
    """

    pred_sql = pred_db_path.replace("'", "''")
    ext_sql = extended_db_path.replace("'", "''")
    out_sql = output_db_path.replace("'", "''")

    # Remove old DB if exists
    if os.path.exists(output_db_path):
        os.remove(output_db_path)

    print("Creating:", output_db_path)
    con = duckdb.connect(output_db_path)

    # ------------------------------------------------------------
    # ATTACH SOURCE DATABASES
    # ------------------------------------------------------------
    con.execute(f"ATTACH '{pred_sql}' AS pred_db;")
    con.execute(f"ATTACH '{ext_sql}' AS ext_db;")

    # ------------------------------------------------------------
    # PRINT ORIGINAL CLUSTER COUNTS
    # ------------------------------------------------------------
    print("\n=== ORIGINAL CLUSTER COUNTS (pred_db) ===")
    before_df = con.execute(f"""
        SELECT cluster, COUNT(*) AS count
        FROM pred_db.{pred_table}
        GROUP BY cluster
        ORDER BY cluster
    """).df()

    print(before_df)

    # ------------------------------------------------------------
    # DETECT WHETHER tissue_percentage IS AVAILABLE
    # ------------------------------------------------------------
    has_tissue_pct = False
    try:
        con.execute(f"SELECT tissue_percentage FROM ext_db.{ext_table} LIMIT 1")
        has_tissue_pct = True
    except duckdb.Error:
        print("WARNING: tissue_percentage not found in extended_data; skipping low-tissue relabeling.")

    # ------------------------------------------------------------
    # CREATE NEW TABLE WITH EXTENDED CLUSTERS
    # ------------------------------------------------------------
    if has_tissue_pct:
        cluster_expr = f"""
            CASE
                WHEN x.tissue_percentage < 5 THEN -2
                WHEN x.tissue_percentage < 20 THEN -1
                ELSE p.cluster
            END"""
        tissue_col = "x.tissue_percentage,"
    else:
        cluster_expr = "p.cluster"
        tissue_col = ""

    con.execute(f"""
        CREATE TABLE predictions AS
        SELECT
            p.unique_id,
            p.id,
            p.slide_key,
            p.tile_key,
            p.short_box_name,
            p.box_id,
            {cluster_expr} AS cluster{', ' + tissue_col.rstrip(',') if tissue_col else ''}
        FROM pred_db.{pred_table} p
        LEFT JOIN (
            SELECT DISTINCT tile_key{', tissue_percentage' if has_tissue_pct else ''}
            FROM ext_db.{ext_table}
            WHERE modality = 'stained'
        ) x ON {config.norm_tile_key_sql('p.tile_key')} = {config.norm_tile_key_sql('x.tile_key')};
    """)

    # ------------------------------------------------------------
    # PRINT UPDATED COUNTS
    # ------------------------------------------------------------
    print("\n=== UPDATED CLUSTER COUNTS (predictions_extended) ===")
    after_df = con.execute("""
        SELECT cluster, COUNT(*) AS count
        FROM predictions
        GROUP BY cluster
        ORDER BY cluster
    """).df()

    print(after_df)

    # ------------------------------------------------------------
    # PRINT DELTA (DIFFERENCE)
    # ------------------------------------------------------------
    print("\n=== DELTA (after - before) ===")
    merged = before_df.merge(after_df, on="cluster", how="outer", suffixes=("_before", "_after"))
    merged = merged.fillna(0)
    merged["delta"] = merged["count_after"] - merged["count_before"]
    print(merged)

    # ------------------------------------------------------------
    # INDEXES FOR FAST QUERYING
    # ------------------------------------------------------------
    con.execute("CREATE INDEX idx_pred_ext_uid ON predictions(unique_id);")
    con.execute("CREATE INDEX idx_pred_ext_tilekey ON predictions(tile_key);")

    print("\nDone. New DB saved at:", output_db_path)
    return output_db_path
