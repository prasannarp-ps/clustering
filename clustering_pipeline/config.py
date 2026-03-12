"""
Central configuration for paths and defaults.
Adjust these to match your environment.
"""

import os

# Base root for everything
BASE_DIR = "./"

# Parquet sources
PARQUET_FILE_PATH = os.path.join(BASE_DIR, "data/global_embedding.parquet")
EXTENDED_PARQUET_FILE_PATH = os.path.join(BASE_DIR, "data/global_color_histograms.parquet")
BLUR_PARQUET_FILE_PATH = os.path.join(BASE_DIR, "data/global_blur_features.parquet")

# Database directory and paths
DB_DIR = os.path.join(BASE_DIR, "db")
os.makedirs(DB_DIR, exist_ok=True)

EMBED_DB_PATH = os.path.join(DB_DIR, "embeddings.db")
EXTENDED_DB_PATH = os.path.join(DB_DIR, "extended_data.db")
PRED_DB_PATH = os.path.join(DB_DIR, "predictions.db")

# Output directories
OUT_DIR_MODEL = os.path.join(BASE_DIR, "models")
OUT_DIR_DATA = os.path.join(BASE_DIR, "data")
OUT_DIR_RESULTS = os.path.join(BASE_DIR, "results")

for d in (OUT_DIR_MODEL, OUT_DIR_DATA, OUT_DIR_RESULTS):
    os.makedirs(d, exist_ok=True)

# Default clustering configuration
DEFAULT_K = 112
DEFAULT_MODALITY = "stained"


def norm_tile_key_sql(column: str) -> str:
    """Return a DuckDB SQL expression that normalizes a tile_key by stripping
    the box_id / short_box_name segment so keys from different sources match.

    Input:  skin_25m5043_1_2025-10-20_64551_tile_0-0-512-512
         or skin_25m5043_1_2025-10-20_16400-0_tile_0-0-512-512
    Output: skin_25m5043_1_2025-10-20_tile_0-0-512-512
    """
    return f"regexp_replace({column}, '_[^_]+_tile_', '_tile_')"

# Example default for stratified sample parquet
SAMPLE_PARQUET_PATH = os.path.join(OUT_DIR_DATA, "sample_stratified_above_20.parquet")
