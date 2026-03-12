"""
Generate image grids for each cluster using locally available tiles.

Path formula:
  {tiles_root}/{tissue_type}/{block_id}/{slice_id}/{scan_date}/{short_box_name}/{filename}-stained.tiff

Outputs:
  results/cluster_grids/cluster_{id}.png   — grid PNG per cluster
  results/cluster_grids/overview.html       — single HTML page with all grids

Usage:
    python cluster_image_grid.py
    python cluster_image_grid.py --rows 4 --cols 6 --tile-size 224
"""

import argparse
import base64
import io
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ── Paths ─────────────────────────────────────────────────────────────────────
EMBED_PARQUET = "data/global_embedding.parquet"
PRED_PARQUET  = "results/clustering_results_all_k16.parquet"
TILES_ROOT    = "data/gi-registration/slide-registration-staging/tiles_origin/nanozoomers20/40x/frozen/he/512"
OUT_DIR       = "results/cluster_grids"

DEFAULT_ROWS      = 5
DEFAULT_COLS      = 5
DEFAULT_TILE_SIZE = 224   # pixels each tile is resized to
# ──────────────────────────────────────────────────────────────────────────────


def build_local_path(row, tiles_root: str) -> str:
    return os.path.join(
        tiles_root,
        str(row["tissue_type"]),
        str(row["block_id"]),
        str(row["slice_id"]),
        str(row["scan_date"]),
        str(row["short_box_name"]),
        f"{row['filename']}-stained.tiff",
    )


def load_and_resize(path: str, size: int) -> Image.Image | None:
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        return img
    except Exception:
        return None


def make_label(text: str, width: int, height: int = 20) -> Image.Image:
    """Small black banner with white text to sit below each tile."""
    banner = Image.new("RGB", (width, height), color=(20, 20, 20))
    draw = ImageDraw.Draw(banner)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    draw.text((3, 2), text, fill=(220, 220, 220), font=font)
    return banner


def build_grid(images: list[Image.Image], labels: list[str],
               rows: int, cols: int, tile_size: int, pad: int = 4) -> Image.Image:
    label_h = 20
    cell_w  = tile_size + pad * 2
    cell_h  = tile_size + label_h + pad * 2

    grid_w = cols * cell_w
    grid_h = rows * cell_h
    canvas = Image.new("RGB", (grid_w, grid_h), color=(40, 40, 40))

    for idx in range(rows * cols):
        row_i = idx // cols
        col_i = idx % cols
        x = col_i * cell_w + pad
        y = row_i * cell_h + pad

        if idx < len(images) and images[idx] is not None:
            canvas.paste(images[idx], (x, y))
            label_img = make_label(labels[idx], tile_size, label_h)
            canvas.paste(label_img, (x, y + tile_size))
        else:
            # blank tile
            blank = Image.new("RGB", (tile_size, tile_size), (60, 60, 60))
            canvas.paste(blank, (x, y))

    return canvas


def img_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_overview_html(cluster_imgs: dict[int, Image.Image],
                        cluster_counts: dict[int, int]) -> str:
    parts = ["<html><head><meta charset='utf-8'>",
             "<title>Cluster Image Grids</title>",
             "<style>",
             "body{background:#1a1a1a;color:#eee;font-family:sans-serif;margin:20px}",
             "h1{color:#ddd} .cluster{display:inline-block;margin:12px;vertical-align:top}",
             ".cluster h3{margin:4px 0;font-size:13px;color:#aaa}",
             "img{border:2px solid #444;border-radius:4px}",
             "</style></head><body>",
             "<h1>Cluster Image Grids — k=16 (stained)</h1>"]

    for cl_id in sorted(cluster_imgs):
        b64 = img_to_base64(cluster_imgs[cl_id])
        local_n = cluster_counts.get(cl_id, 0)
        parts.append(
            f"<div class='cluster'>"
            f"<h3>Cluster {cl_id} &nbsp;·&nbsp; {local_n} local tiles</h3>"
            f"<img src='data:image/png;base64,{b64}'>"
            f"</div>"
        )

    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Cluster image grids from local tiles")
    parser.add_argument("--embed-parquet", default=EMBED_PARQUET)
    parser.add_argument("--pred-parquet",  default=PRED_PARQUET)
    parser.add_argument("--tiles-root",    default=TILES_ROOT)
    parser.add_argument("--out-dir",       default=OUT_DIR)
    parser.add_argument("--rows",     type=int, default=DEFAULT_ROWS)
    parser.add_argument("--cols",     type=int, default=DEFAULT_COLS)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    images_per_grid = args.rows * args.cols

    # ── 1. Load & merge ───────────────────────────────────────────────────────
    print("Loading embeddings metadata …")
    meta_cols = ["tile_key", "modality", "tissue_type", "block_id",
                 "slice_id", "scan_date", "short_box_name", "filename"]
    emb = pd.read_parquet(args.embed_parquet, columns=meta_cols)
    emb = emb[emb["modality"] == "stained"].reset_index(drop=True)
    print(f"  Stained tiles: {len(emb):,}")

    print("Loading predictions …")
    pred = pd.read_parquet(args.pred_parquet, columns=["tile_key", "cluster"])

    df = emb.merge(pred, on="tile_key", how="inner")
    print(f"  Merged rows:   {len(df):,}")

    # ── 2. Build & check local paths ──────────────────────────────────────────
    print("Resolving local paths …")
    df["local_path"] = df.apply(
        lambda r: build_local_path(r, args.tiles_root), axis=1
    )
    df["exists"] = df["local_path"].map(os.path.exists)

    total_local = df["exists"].sum()
    print(f"  Tiles found locally: {total_local:,} / {len(df):,}")

    local_df = df[df["exists"]].copy()
    if local_df.empty:
        print("ERROR: No local tiles found. Check --tiles-root path.")
        return

    # Per-cluster availability
    avail = local_df.groupby("cluster").size().sort_index()
    print("\nLocal tiles per cluster:")
    for cl, n in avail.items():
        print(f"  Cluster {cl:>2d}: {n:>4d} tiles")

    # ── 3. Build grids ────────────────────────────────────────────────────────
    clusters = sorted(local_df["cluster"].unique())
    cluster_imgs: dict[int, Image.Image] = {}
    cluster_counts: dict[int, int] = {}

    for cl_id in clusters:
        cl_df = local_df[local_df["cluster"] == cl_id]
        n_avail = len(cl_df)
        cluster_counts[cl_id] = n_avail

        sample = cl_df.sample(
            n=min(images_per_grid, n_avail), random_state=42
        ).reset_index(drop=True)

        print(f"\nCluster {cl_id}: loading {len(sample)} images …")
        imgs, labels = [], []
        for _, row in sample.iterrows():
            img = load_and_resize(row["local_path"], args.tile_size)
            imgs.append(img)
            labels.append(f"{row['tissue_type']} · {row['filename']}")

        grid = build_grid(imgs, labels, args.rows, args.cols, args.tile_size)
        cluster_imgs[cl_id] = grid

        out_png = os.path.join(args.out_dir, f"cluster_{cl_id}.png")
        grid.save(out_png)
        print(f"  Saved: {out_png}")

    # ── 4. HTML overview ──────────────────────────────────────────────────────
    print("\nBuilding HTML overview …")
    html = build_overview_html(cluster_imgs, cluster_counts)
    html_path = os.path.join(args.out_dir, "overview.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {html_path}")

    print(f"\nDone. Open {html_path} to see all clusters.")


if __name__ == "__main__":
    main()
