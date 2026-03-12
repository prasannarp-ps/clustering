"""
DuckDB I/O helpers and streaming utilities.
"""
import os
import uuid

import duckdb
from . import config

# Global connection used by the stratified sampler
DUCKDB_PREFIX = "duckdb:"
DUCKDB_CONN = None


def duckdb_connect_if_needed():
    global DUCKDB_CONN
    if DUCKDB_CONN is None:
        raise RuntimeError("DUCKDB_CONN is not initialized. Set it before proceeding.")
    return DUCKDB_CONN


def duckdb_iter_batches(
    table,
    batch_size,
    columns=None,
    min_tissue_pct=0,
):
    """
    Stream batches from embeddings.db.

    If filter_tissue_percentage=True, only rows with
    ext_db.extended_data.tissue_percentage > min_tissue_pct are returned.
    """
    con = duckdb_connect_if_needed()

    if columns is None:
        col_expr = "e.*"
    else:
        col_expr = ", ".join([f"e.{c}" for c in columns])

    base_from = f"{table} e"

    join_clause = ""
    where_extra = ""

    if min_tissue_pct:
        nk = config.norm_tile_key_sql
        # Deduplicate extended_data by normalized tile_key to avoid
        # many-to-many join (multiple box_ids share the same norm key).
        join_clause = (
            f"JOIN (SELECT {nk('tile_key')} AS norm_key, "
            f"MAX(tissue_percentage) AS tissue_percentage "
            f"FROM ext_db.extended_data GROUP BY {nk('tile_key')}) x "
            f"ON {nk('e.tile_key')} = x.norm_key"
        )
        where_extra = f"AND x.tissue_percentage > {min_tissue_pct}"

    total = con.execute(f"""
        SELECT COUNT(*)
        FROM {base_from}
        {join_clause}
        WHERE e.modality = 'stained'
        {where_extra}
    """).fetchone()[0]

    offset = 0
    while offset < total:
        df = con.execute(f"""
            SELECT {col_expr}
            FROM {base_from}
            {join_clause}
            WHERE e.modality = 'stained'
            {where_extra}
            LIMIT {batch_size} OFFSET {offset}
        """).df()
        yield df
        offset += batch_size


def generate_unique_id(row):
    key = (
        f"{row['tissue_type']}/"
        f"{row['block_id']}/"
        f"{row['slice_id']}/"
        f"{row['scan_date']}/"
        f"{row['box_id']}/"
        f"{row['filename']}/"
        f"{row['tile_type']}/"
        f"{row['preprocessing']}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def create_embeddings_db_from_parquet(
    embed_db_path,
    parquet_path
):
    """
    Creates embeddings.db from a Parquet file AND automatically generates unique_id
    and an index on unique_id. If the DB already exists, it is reused.
    """
    os.makedirs(os.path.dirname(embed_db_path), exist_ok=True)

    if os.path.exists(embed_db_path):
        print(f"Embeddings DB already exists at: {embed_db_path}")
        con = duckdb.connect(embed_db_path)
        return con

    con = duckdb.connect(embed_db_path)
    parquet_path_sql = parquet_path.replace("'", "''")

    print(f"Creating embeddings DB from Parquet: {parquet_path}")

    # 1. Create empty table schema
    con.execute(f"""
        CREATE OR REPLACE TABLE embeddings AS
        SELECT *
        FROM read_parquet('{parquet_path_sql}', hive_partitioning=false)
        WHERE FALSE
    """)
    print("Embeddings table schema created.")

    # 2. Insert data streaming from Parquet
    con.execute(f"""
        INSERT INTO embeddings
        SELECT *
        FROM read_parquet('{parquet_path_sql}', union_by_name=true)
    """)
    print("Data inserted into embeddings table.")

    # 3. Add unique_id column if not exists
    con.execute("""
        ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS unique_id VARCHAR;
    """)
    print("unique_id column added to embeddings table.")

    # 4. Populate unique_id using deterministic md5 hash
    con.execute("""
        UPDATE embeddings
        SET unique_id = md5(
            tissue_type || '/' ||
            block_id || '/' ||
            slice_id || '/' ||
            scan_date || '/' ||
            box_id || '/' ||
            filename || '/' ||
            tile_type || '/' ||
            preprocessing
        )
        WHERE unique_id IS NULL OR unique_id = '';
    """)
    print("unique_id populated for all rows in embeddings table.")

    # 5. Create index on unique_id
    try:
        con.execute("CREATE INDEX embeddings_unique_id_idx ON embeddings(unique_id);")
        print("Index on embeddings.unique_id created.")
    except Exception as exc:
        print(f"Index creation on embeddings.unique_id failed or exists: {exc}")

    print(f"Embeddings DB ready at: {embed_db_path}")
    return con

def create_duckdb_from_parquet_simple(
    db_path,
    parquet_path,
    table_name="data",
    generate_unique_id=False,
    index_unique_id=False
):
    """
    Create a DuckDB database from a Parquet file.
    - db_path:            path to .db file to create
    - parquet_path:       path to source parquet file
    - table_name:         name of table inside DB (default: data)
    - generate_unique_id: create unique_id column using md5 hash
    - index_unique_id:    create index on unique_id
    """

    # Ensure parent folder exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    con = duckdb.connect(db_path)

    parquet_sql = parquet_path.replace("'", "''")

    print(f"Creating database: {db_path}")
    print(f"Reading parquet: {parquet_path}")

    # 1. Create table
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT *
        FROM read_parquet('{parquet_sql}', union_by_name=true)
    """)
    print(f"Table '{table_name}' created.")

    # 2. Generate unique_id if requested
    if generate_unique_id:
        print("Generating unique_id column...")

        con.execute(f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS unique_id VARCHAR;
        """)

        con.execute(f"""
            UPDATE {table_name}
            SET unique_id = md5(CAST(rowid AS VARCHAR))
            WHERE unique_id IS NULL OR unique_id = '';
        """)

        print("unique_id generated.")

    # 3. Index unique_id (optional)
    if index_unique_id:
        try:
            con.execute(f"CREATE INDEX {table_name}_unique_id_idx ON {table_name}(unique_id)")
            print("Index created on unique_id.")
        except:
            print("Index already exists (skipped).")

    print(f"Database ready: {db_path}")
    return con