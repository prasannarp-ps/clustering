"""
2D embedding projection (t-SNE or UMAP) of cluster assignments.

Key improvements over v1:
  - Stratified sampling: equal tiles per cluster so every cluster is visible
  - UMAP option: much faster + preserves global structure better than t-SNE
  - Higher default perplexity when using t-SNE
  - Centroids plotted as large stars on top of the scatter

Usage:
    # UMAP (recommended, faster, better separation)
    python tsne_plot.py --method umap

    # t-SNE with stratified sampling
    python tsne_plot.py --method tsne --per-cluster 600 --perplexity 50

    # Custom output name
    python tsne_plot.py --method umap --out results/umap_k16.html
"""

import argparse
import base64
import io
import os

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

EMBED_PARQUET    = "data/global_embedding.parquet"
PRED_PARQUET     = "results/clustering_results_all_k16.parquet"
RESULTS_DIR      = "results"
TILES_ROOT       = "data/gi-registration/slide-registration-staging/tiles_origin/nanozoomers20/40x/frozen/he/512"
DEFAULT_PER_CLUSTER = 600   # stratified: this many tiles per cluster
PCA_COMPONENTS   = 0.95   # retain enough components to explain 95% of variance
CENTROID_IMG_SIZE = 48    # thumbnail size in pixels
RANDOM_STATE     = 42


# ── Centroid image helpers ─────────────────────────────────────────────────────

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


def load_centroid_images(df: pd.DataFrame, tiles_root: str,
                         size: int = CENTROID_IMG_SIZE) -> dict[int, str | None]:
    """For each cluster find the locally available tile closest to its 2D centroid.
    Returns {cluster_id: 'data:image/png;base64,...'} or None if no local tile found."""
    results = {}
    for cluster_id, group in df.groupby("cluster"):
        cx, cy = group["x"].mean(), group["y"].mean()
        dists = np.sqrt((group["x"] - cx) ** 2 + (group["y"] - cy) ** 2)
        results[cluster_id] = None
        for idx in dists.sort_values().index:
            row = group.loc[idx]
            path = build_local_path(row, tiles_root)
            if os.path.exists(path):
                try:
                    img = PILImage.open(path).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    results[cluster_id] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
                    break
                except Exception:
                    continue
    return results


# ── Sampling ──────────────────────────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, per_cluster: int) -> pd.DataFrame:
    """Sample up to `per_cluster` rows from each cluster."""
    return (
        df.groupby("cluster", group_keys=False)
        .apply(lambda g: g.sample(n=min(per_cluster, len(g)), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )


# ── Dimensionality reduction ───────────────────────────────────────────────────

def run_pca(vectors: np.ndarray, n_components) -> tuple[np.ndarray, float]:
    from sklearn.decomposition import PCA
    # n_components may be an int/float>=1 (fixed dims) or float 0–1 (variance threshold)
    if isinstance(n_components, float) and n_components >= 1:
        n_components = int(n_components)
    if isinstance(n_components, int):
        n_components = min(n_components, vectors.shape[1], vectors.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(vectors)
    var = pca.explained_variance_ratio_.sum() * 100
    return reduced, var


def run_tsne(reduced: np.ndarray, perplexity: float) -> np.ndarray:
    from sklearn.manifold import TSNE
    print(f"t-SNE: {reduced.shape[1]}D → 2D  (perplexity={perplexity}, n={len(reduced):,}) …")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    return tsne.fit_transform(reduced)


def run_umap(reduced: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    import umap
    print(f"UMAP: {reduced.shape[1]}D → 2D  (n_neighbors={n_neighbors}, min_dist={min_dist}, n={len(reduced):,}) …")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=RANDOM_STATE,
        verbose=True,
    )
    return reducer.fit_transform(reduced)


# ── Plotting ───────────────────────────────────────────────────────────────────

def build_figure(df: pd.DataFrame, method: str, var_explained: float,
                 orig_dims: int, pca_dims: int,
                 centroid_images: dict | None = None,
                 model_name: str | None = None) -> go.Figure:

    df = df.copy()
    df["cluster_label"] = "Cluster " + df["cluster"].astype(str)

    # Compute centroids in 2D projection space
    centroids = df.groupby("cluster_label")[["x", "y"]].mean().reset_index()

    # Main scatter (no tissue_type symbol — redundant)
    fig = px.scatter(
        df,
        x="x", y="y",
        color="cluster_label",
        hover_data={"tile_key": True, "tissue_type": True,
                    "cluster": True, "x": False, "y": False},
        labels={"cluster_label": "Cluster"},
        opacity=0.65,
    )
    fig.update_traces(marker=dict(size=5))

    # Centroid stars
    for _, row in centroids.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["x"]], y=[row["y"]],
            mode="markers+text",
            marker=dict(symbol="star", size=18, color="black",
                        line=dict(color="white", width=1)),
            text=[row["cluster_label"].replace("Cluster ", "C")],
            textposition="top center",
            textfont=dict(size=11, color="black"),
            showlegend=False,
            hoverinfo="skip",
        ))

    n_total = len(df)
    n_clusters = df["cluster"].nunique()
    method_label = method.upper()
    model_label = model_name.upper().replace("_", " ") if model_name else "embeddings"

    has_imgs = bool(centroid_images and any(v is not None for v in centroid_images.values()))

    # Dimensions for ~15" laptop screen in a browser
    fig_w   = 1800 if has_imgs else 1400
    fig_h   = 950
    mar_l, mar_r = 60, (380 if has_imgs else 60)
    mar_t, mar_b = 90, 60
    # Actual plot-area dimensions in pixels (used to calibrate paper-coord sizes)
    plot_w = fig_w - mar_l - mar_r
    plot_h = fig_h - mar_t - mar_b

    fig.update_layout(
        title=dict(
            text=(
                f"{method_label} — {model_label}, k={n_clusters} clusters "
                f"(stratified {n_total:,} tiles, ~{n_total//n_clusters} per cluster)<br>"
                + (f"<sub>PCA {orig_dims}D→{pca_dims}D ({var_explained:.1f}% var) → {method_label} 2D"
                   if var_explained is not None
                   else f"<sub>No PCA ({orig_dims}D) → {method_label} 2D")
                + " · stars = cluster centroids</sub>"
            ),
            font=dict(size=18),
        ),
        legend=dict(
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc", borderwidth=1,
            font=dict(size=13),
            itemsizing="constant",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#111111", size=14),
        width=fig_w,
        height=fig_h,
        margin=dict(l=mar_l, r=mar_r, t=mar_t, b=mar_b),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(size=12))
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(size=12))

    # ── Centroid thumbnail grid on the right (2 cols × dynamic rows) ────────────
    if has_imgs:
        sorted_clusters = sorted(centroid_images.keys())
        n_clusters = len(sorted_clusters)
        COLS = 2
        ROWS = (n_clusters + COLS - 1) // COLS   # ceil(n_clusters / 2)

        # Slot height spans the full plot area divided by number of rows
        slot_h = 1.0 / ROWS
        img_h  = slot_h * 0.88
        # Square thumbnails: adjust width for plot aspect ratio
        img_w  = img_h * (plot_h / plot_w)

        # Column step = image width + label clearance + inter-column gap
        label_gap = 0.035
        col_step  = img_w + label_gap

        x0 = 1.01   # just right of the plot area

        for i, cl_id in enumerate(sorted_clusters):
            col = i // ROWS
            row = i % ROWS

            x_img = x0 + col * col_step
            y_top = 1.0 - row * slot_h
            y_mid = y_top - img_h / 2

            # Label: special names for -2 and -1
            if cl_id == -2:
                label = "C-2 (<5%)"
            elif cl_id == -1:
                label = "C-1 (5-20%)"
            else:
                label = f"C{cl_id}"

            if centroid_images[cl_id] is not None:
                fig.add_layout_image(
                    source=centroid_images[cl_id],
                    xref="paper", yref="paper",
                    x=x_img, y=y_top,
                    sizex=img_w, sizey=img_h,
                    xanchor="left", yanchor="top",
                    layer="above",
                )

            fig.add_annotation(
                xref="paper", yref="paper",
                x=x_img + img_w + 0.005,
                y=y_mid,
                text=label,
                showarrow=False,
                font=dict(size=11, color="#333333"),
                xanchor="left",
                yanchor="middle",
            )

    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stratified 2D projection (t-SNE or UMAP) of cluster assignments"
    )
    parser.add_argument("--embed-parquet",  default=EMBED_PARQUET)
    parser.add_argument("--pred-parquet",   default=PRED_PARQUET)
    parser.add_argument("--method",         default="umap", choices=["tsne", "umap"],
                        help="Projection method: 'umap' (default) or 'tsne'")
    parser.add_argument("--per-cluster",    type=int, default=DEFAULT_PER_CLUSTER,
                        help="Tiles per cluster for stratified sampling (default: 600)")
    parser.add_argument("--pca-components", type=float, default=PCA_COMPONENTS,
                        help="PCA components: int for fixed dims, float 0-1 for variance threshold (default: 0.95)")
    # t-SNE options
    parser.add_argument("--perplexity",     type=float, default=50,
                        help="t-SNE perplexity (default: 50)")
    # UMAP options
    parser.add_argument("--n-neighbors",    type=int,   default=30,
                        help="UMAP n_neighbors (default: 30)")
    parser.add_argument("--min-dist",       type=float, default=0.1,
                        help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--no-pca",         action="store_true",
                        help="Skip PCA and feed raw vectors directly to UMAP/t-SNE")
    parser.add_argument("--model",          default=None,
                        help="Model name for the title (e.g. path_foundation, uni, conch)")
    parser.add_argument("--out",            default=None)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = args.out or os.path.join(
        RESULTS_DIR, f"{args.method}_k16_stratified.html"
    )

    # 1. Load & merge
    print("Loading embeddings …")
    emb = pd.read_parquet(
        args.embed_parquet,
        columns=["tile_key", "modality", "tissue_type", "vector",
                 "block_id", "slice_id", "scan_date", "short_box_name", "filename"],
    )
    emb = emb[emb["modality"] == "stained"].reset_index(drop=True)
    print(f"  Stained tiles: {len(emb):,}")

    print("Loading predictions …")
    pred = pd.read_parquet(args.pred_parquet, columns=["tile_key", "cluster"])

    df = emb.merge(pred, on="tile_key", how="inner")
    print(f"  Merged:        {len(df):,}")

    # 2. Stratified sample
    df = stratified_sample(df, args.per_cluster)
    print(f"  After stratified sample: {len(df):,} "
          f"(~{args.per_cluster} per cluster)")

    vectors = np.stack(df["vector"].values).astype(np.float32)

    # 3. PCA (optional)
    if args.no_pca:
        print("\nSkipping PCA — using raw vectors.")
        reduced = vectors
        var_explained = None
        pca_dims = vectors.shape[1]
    else:
        print(f"\nPCA: {vectors.shape[1]}D → {args.pca_components} (variance threshold) …")
        reduced, var_explained = run_pca(vectors, args.pca_components)
        pca_dims = reduced.shape[1]
        print(f"  Components selected: {pca_dims}  Variance explained: {var_explained:.1f}%")

    # 4. Project
    if args.method == "tsne":
        coords = run_tsne(reduced, args.perplexity)
    else:
        coords = run_umap(reduced, args.n_neighbors, args.min_dist)
    print("  Done.")

    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    # 5. Load centroid images
    print("\nLoading centroid images …")
    centroid_images = load_centroid_images(df, TILES_ROOT)
    found = sum(1 for v in centroid_images.values() if v is not None)
    print(f"  Images found: {found} / {len(centroid_images)}")

    # 6. Plot
    print("\nBuilding figure …")
    fig = build_figure(df, args.method, var_explained, vectors.shape[1], pca_dims,
                       centroid_images=centroid_images, model_name=args.model)
    fig.write_html(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
