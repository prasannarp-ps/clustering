# -*- coding: utf-8 -*-
"""
K-sweep explorer for skin fixed sections - path_foundation embeddings.

For each K in K_VALUES, trains MiniBatchKMeans on:
  (a) unfiltered — all stained tiles
  (b) filtered   — only tiles with tissue_percentage >= 20%
                   (tiles < 5% get cluster=-2, 5-20% get cluster=-1)

Outputs in results/k_sweep/:
  metrics_comparison.html           — CHI / DBI / inertia for all K × both conditions
  k{K}_unfiltered/overview.html     — cluster image grid
  k{K}_filtered/overview.html       — cluster image grid

Usage:
  python explore_k_sweep.py
  python explore_k_sweep.py --k-values 16 32 64 --no-unfiltered
"""

import argparse
import base64
import io
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "path_foundation"
BLUR_PARQUET   = "data/global_blur_features.parquet"
TILES_ROOT     = "data/gi-registration/slide-registration-staging/tiles_origin/nanozoomers20/40x/frozen/he/512"

K_VALUES       = [16, 32, 48, 64, 80, 96, 112]
PCA_COMPONENTS = 0.9
BATCH_SIZE    = 50_000
GRID_ROWS     = 5
GRID_COLS     = 5
TILE_SIZE     = 160   # px per tile in grid
RANDOM_STATE  = 42

TISSUE_EXCLUDE_LOW  = 5.0   # < 5%  -> cluster -2
TISSUE_EXCLUDE_MID  = 20.0  # < 20% -> cluster -1
# ──────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(parquet_path=None):
    """Load embeddings + tissue_percentage, stained tiles only."""
    path = parquet_path or EMBED_PARQUET
    print(f"Loading embeddings from {path} ...")
    emb = pd.read_parquet(path)
    emb = emb[emb["tile_type"] == "stained"].reset_index(drop=True)
    print(f"  Stained tiles: {len(emb):,}")

    print("Loading blur/tissue features ...")
    blur_cols = ["tile_name", "stained", "tissue_type", "block_id", "slice_id",
                 "scan_date", "box_id", "tile_filename", "tissue_percentage"]
    blur = pd.read_parquet(BLUR_PARQUET, columns=blur_cols)
    blur = blur[blur["stained"] == True].reset_index(drop=True)

    # Build join key matching tile_key format: {tissue_type}_{block_id}_{slice_id}_{scan_date}_{box_id}_{tile_filename}
    blur["tile_key"] = (
        blur["tissue_type"].astype(str) + "_" +
        blur["block_id"].astype(str)    + "_" +
        blur["slice_id"].astype(str)    + "_" +
        blur["scan_date"].astype(str)   + "_" +
        blur["box_id"].astype(str)      + "_" +
        blur["tile_filename"].astype(str)
    )

    print(f"  Stained blur rows: {len(blur):,}")

    merged = emb.merge(blur[["tile_key", "tissue_percentage"]], on="tile_key", how="left")
    print(f"  Merged: {len(merged):,}  (tissue_pct null: {merged['tissue_percentage'].isna().sum():,})")

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# Clustering
# ═══════════════════════════════════════════════════════════════════════════════

def extract_vectors(df):
    return np.vstack(df["vector"].values).astype(np.float32)


def apply_pca(X, n_components=PCA_COMPONENTS):
    # Float 0-1 -> variance threshold (e.g. 0.9 = 90%); int -> fixed dims
    if isinstance(n_components, float) and 0 < n_components < 1:
        print(f"  PCA {X.shape[1]}d -> {n_components:.0%} variance threshold ...")
    else:
        n_components = int(n_components)
        n_components = min(n_components, X.shape[0], X.shape[1])
        print(f"  PCA {X.shape[1]}d -> {n_components}d ...")
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_r = pca.fit_transform(X)
    print(f"  Components selected: {X_r.shape[1]}  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    return X_r, pca


def train_kmeans(X_pca, k):
    model = MiniBatchKMeans(
        n_clusters=k, random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE, n_init=3,
    )
    for i in range(0, len(X_pca), BATCH_SIZE):
        batch = X_pca[i:i + BATCH_SIZE]
        if len(batch) >= 2:
            model.partial_fit(batch)
    return model


def predict_all(model, X_pca):
    labels = np.empty(len(X_pca), dtype=np.int32)
    for i in range(0, len(X_pca), BATCH_SIZE):
        batch = X_pca[i:i + BATCH_SIZE]
        labels[i:i + len(batch)] = model.predict(batch)
    return labels


def apply_tissue_relabeling(labels, tissue_pct):
    """Relabel low-tissue tiles: < 5% -> -2, 5-20% -> -1."""
    labels = labels.copy()
    pct = np.array(tissue_pct, dtype=np.float32)
    labels[pct < TISSUE_EXCLUDE_LOW] = -2
    labels[(pct >= TISSUE_EXCLUDE_LOW) & (pct < TISSUE_EXCLUDE_MID)] = -1
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# Image grid
# ═══════════════════════════════════════════════════════════════════════════════

def build_local_path(row):
    return os.path.join(
        TILES_ROOT,
        str(row["tissue_type"]),
        str(row["block_id"]),
        str(row["slice_id"]),
        str(row["scan_date"]),
        str(row["short_box_name"]),
        f"{row['filename']}-stained.tiff",
    )


def load_tile(path, size=TILE_SIZE):
    try:
        return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    except Exception:
        return None


def make_label_banner(text, width, height=18):
    banner = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(banner)
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
    draw.text((3, 2), text, fill=(200, 200, 200), font=font)
    return banner


def build_grid(imgs, labels, rows, cols, tile_size, pad=3):
    label_h = 18
    cw = tile_size + pad * 2
    ch = tile_size + label_h + pad * 2
    canvas = Image.new("RGB", (cols * cw, rows * ch), (40, 40, 40))
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        x, y = c * cw + pad, r * ch + pad
        if idx < len(imgs) and imgs[idx] is not None:
            canvas.paste(imgs[idx], (x, y))
            canvas.paste(make_label_banner(labels[idx], tile_size, label_h), (x, y + tile_size))
        else:
            canvas.paste(Image.new("RGB", (tile_size, tile_size), (60, 60, 60)), (x, y))
    return canvas


def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_grids_html(df_labeled, k, condition_label):
    """Build overview HTML with one grid per cluster."""
    real_clusters = sorted(c for c in df_labeled["cluster"].unique() if c >= 0)
    special = {-2: "< 5% tissue", -1: "5-20% tissue"}

    cluster_imgs = {}
    cluster_counts = {}
    n_per_grid = GRID_ROWS * GRID_COLS

    for cl_id in real_clusters:
        cl_df = df_labeled[df_labeled["cluster"] == cl_id]
        cluster_counts[cl_id] = len(cl_df)
        sample = cl_df.sample(n=min(n_per_grid, len(cl_df)), random_state=RANDOM_STATE)
        imgs, lbls = [], []
        for _, row in sample.iterrows():
            path = build_local_path(row)
            imgs.append(load_tile(path))
            lbls.append(f"{row['block_id']} · {row['filename'][:20]}")
        cluster_imgs[cl_id] = build_grid(imgs, lbls, GRID_ROWS, GRID_COLS, TILE_SIZE)

    # Count special clusters
    special_counts = {c: (df_labeled["cluster"] == c).sum() for c in [-2, -1]}

    parts = [
        "<html><head><meta charset='utf-8'>",
        f"<title>k={k} {condition_label}</title>",
        "<style>",
        "body{background:#1a1a1a;color:#eee;font-family:sans-serif;margin:20px}",
        "h1,h2{color:#ddd} .cluster{display:inline-block;margin:10px;vertical-align:top}",
        ".cluster h3{margin:4px 0;font-size:12px;color:#aaa}",
        ".special{background:#2a1a1a;padding:8px 12px;border-radius:6px;",
        "display:inline-block;margin:10px;color:#e07070;font-size:13px}",
        "img{border:2px solid #444;border-radius:4px}",
        "</style></head><body>",
        f"<h1>k={k} — {condition_label}</h1>",
        f"<h2>Total tiles: {len(df_labeled):,} | Clusters: {len(real_clusters)}</h2>",
    ]

    # Special cluster summary
    for c, desc in special.items():
        n = special_counts.get(c, 0)
        if n > 0:
            parts.append(f"<div class='special'>cluster {c} ({desc}): {n:,} tiles</div>")

    parts.append("<hr style='border-color:#333;margin:16px 0'>")

    for cl_id in real_clusters:
        b64 = img_to_b64(cluster_imgs[cl_id])
        n = cluster_counts[cl_id]
        pct = n / len(df_labeled) * 100
        parts.append(
            f"<div class='cluster'>"
            f"<h3>Cluster {cl_id} &nbsp;·&nbsp; {n:,} tiles ({pct:.1f}%)</h3>"
            f"<img src='data:image/png;base64,{b64}'>"
            f"</div>"
        )

    parts.append("</body></html>")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics comparison chart
# ═══════════════════════════════════════════════════════════════════════════════

def build_metrics_html(all_metrics, model_name="path_foundation"):
    """all_metrics: list of dicts with keys: k, condition, inertia, chi, dbi, n_tiles"""
    df = pd.DataFrame(all_metrics)
    conditions = df["condition"].unique()
    colors = {"unfiltered": "#4a9eff", "filtered": "#ff7f50"}

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Inertia (lower=better)", "Calinski-Harabasz (higher=better)", "Davies-Bouldin (lower=better)"],
    )

    for cond in conditions:
        sub = df[df["condition"] == cond].sort_values("k")
        c = colors.get(cond, "#aaa")
        hover = [f"k={row.k}<br>{cond}<br>tiles={row.n_tiles:,}" for _, row in sub.iterrows()]
        common = dict(x=sub["k"], mode="lines+markers",
                      name=cond, legendgroup=cond,
                      marker=dict(color=c, size=8), line=dict(color=c),
                      hovertext=hover, hoverinfo="text+y")
        fig.add_trace(go.Scatter(y=sub["inertia"],  **common),             row=1, col=1)
        fig.add_trace(go.Scatter(y=sub["chi"],      **{**common, "showlegend": False}), row=1, col=2)
        fig.add_trace(go.Scatter(y=sub["dbi"],      **{**common, "showlegend": False}), row=1, col=3)

    fig.update_layout(
        title=f"K-sweep metrics — filtered vs unfiltered (stained, {model_name})",
        height=450, hovermode="x unified",
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font_color="white",
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
    )
    fig.update_xaxes(title_text="k", gridcolor="#333")
    fig.update_yaxes(gridcolor="#333")
    return fig.to_html(full_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_condition(df, k_values, condition, out_base, pca_components=PCA_COMPONENTS):
    """Train + grid + metrics for one condition (filtered or unfiltered)."""
    print(f"\n{'='*60}")
    print(f"  Condition: {condition.upper()}  ({len(df):,} tiles for training)")
    print(f"{'='*60}")

    X = extract_vectors(df)
    if pca_components and pca_components > 0:
        X_pca, _ = apply_pca(X, n_components=pca_components)
    else:
        X_pca = X
        print("  Skipping PCA.")

    all_metrics = []

    for k in k_values:
        print(f"\n-- k={k} --")
        model = train_kmeans(X_pca, k)
        labels = predict_all(model, X_pca)

        # Metrics (before tissue relabeling so all tiles are included)
        chi = calinski_harabasz_score(X_pca[:100_000], labels[:100_000])
        dbi = davies_bouldin_score(X_pca[:100_000], labels[:100_000])
        print(f"  inertia={model.inertia_:,.0f}  CHI={chi:.2f}  DBI={dbi:.4f}")

        all_metrics.append({
            "k": k, "condition": condition,
            "inertia": float(model.inertia_), "chi": chi, "dbi": dbi,
            "n_tiles": len(df),
        })

        # Attach labels to df for grid building
        df_pred = df.copy()
        df_pred["cluster"] = labels

        # Apply tissue relabeling after prediction
        if "tissue_percentage" in df_pred.columns:
            df_pred["cluster"] = apply_tissue_relabeling(
                df_pred["cluster"].values,
                df_pred["tissue_percentage"].fillna(100).values,
            )

        # Build grid HTML
        out_dir = os.path.join(out_base, f"k{k}_{condition}")
        os.makedirs(out_dir, exist_ok=True)
        html = build_grids_html(df_pred, k, condition)
        html_path = os.path.join(out_dir, "overview.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Grid saved: {html_path}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Embedding model name (e.g. path_foundation, uni, uni2, titan). "
                             "Auto-derives parquet path and output dir.")
    parser.add_argument("--parquet", default=None,
                        help="Override embedding parquet path (default: data/global_embedding_{model}.parquet)")
    parser.add_argument("--k-values", nargs="+", type=int, default=K_VALUES)
    parser.add_argument("--no-unfiltered", action="store_true",
                        help="Skip the unfiltered condition (saves time)")
    parser.add_argument("--no-filtered", action="store_true",
                        help="Skip the filtered condition")
    parser.add_argument("--pca", type=float, default=PCA_COMPONENTS,
                        help="PCA: int for fixed dims (e.g. 100), float 0-1 for variance threshold (e.g. 0.9). "
                             "Use 0 to skip PCA.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: results/{model}/k_sweep)")
    args = parser.parse_args()

    global EMBED_PARQUET
    EMBED_PARQUET = args.parquet or f"data/global_embedding_{args.model}.parquet"
    out_dir = args.out_dir or f"results/{args.model}/k_sweep"

    os.makedirs(out_dir, exist_ok=True)
    print(f"Model: {args.model}")
    print(f"Parquet: {EMBED_PARQUET}")
    print(f"Output: {out_dir}")

    df = load_data()

    all_metrics = []

    # ── Unfiltered ─────────────────────────────────────────────────────────────
    if not args.no_unfiltered:
        metrics = run_condition(df, args.k_values, "unfiltered", out_dir, pca_components=args.pca)
        all_metrics.extend(metrics)

    # ── Filtered (tissue >= 20%) ───────────────────────────────────────────────
    if not args.no_filtered:
        pct = df["tissue_percentage"].fillna(100)
        df_filtered = df[pct >= TISSUE_EXCLUDE_MID].reset_index(drop=True)
        print(f"\nFiltered dataset: {len(df_filtered):,} tiles "
              f"({len(df_filtered)/len(df)*100:.1f}% of total)")
        metrics = run_condition(df_filtered, args.k_values, "filtered", out_dir, pca_components=args.pca)
        all_metrics.extend(metrics)

    # ── Metrics comparison chart ───────────────────────────────────────────────
    if all_metrics:
        metrics_html = build_metrics_html(all_metrics, model_name=args.model)
        metrics_path = os.path.join(out_dir, "metrics_comparison.html")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(metrics_html)
        print(f"\nMetrics chart saved: {metrics_path}")

        # Also print summary table
        summary = pd.DataFrame(all_metrics)
        print("\n" + summary.to_string(index=False))

    print(f"\nAll outputs in: {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
