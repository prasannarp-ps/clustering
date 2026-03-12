"""
Advanced optimal-k finder using the same metrics as embeddings_analyzer.py:
  - Inertia  (elbow via KneeLocator — more robust than second derivative)
  - Calinski-Harabasz Index (CHI) — higher is better
  - Davies-Bouldin Index (DBI)   — lower is better
  - Silhouette score             — optional, slow, higher is better

Optional PCA dimensionality reduction before clustering (same as analyzer).

Usage:
    python find_optimal_k_advanced.py
    python find_optimal_k_advanced.py --parquet data/global_embedding.parquet --k-values 8 16 24 32 --pca 100
    python find_optimal_k_advanced.py --silhouette
    python find_optimal_k_advanced.py --no-pca

Outputs:
    results/advanced_optimal_k_metrics.csv   — raw numbers per metric and k
    results/advanced_optimal_k_plot.html     — interactive Plotly chart
"""

import argparse
import os

import numpy as np
import pandas as pd
from kneed import KneeLocator
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.utils import shuffle
from tqdm import tqdm

RESULTS_DIR = "results"
RANDOM_STATE = 42


def load_vectors(parquet_path, sample_size, modality=None):
    print(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  Total rows: {len(df):,}")

    if modality and "modality" in df.columns:
        df = df[df["modality"] == modality]
        print(f"  After modality filter ({modality!r}): {len(df):,} rows")

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)
        print(f"  Sampled: {len(df):,} rows")

    vectors = np.vstack(df["vector"].values).astype(np.float32)
    print(f"  Vector shape: {vectors.shape}")
    return vectors


def apply_pca(vectors, n_components):
    # Float 0-1 → variance threshold (e.g. 0.9 = 90%); int → fixed dims
    if isinstance(n_components, float) and 0 < n_components < 1:
        print(f"Applying PCA: {vectors.shape[1]}d -> {n_components:.0%} variance threshold ...")
    else:
        n_components = int(n_components)
        n_components = min(n_components, vectors.shape[0], vectors.shape[1])
        print(f"Applying PCA: {vectors.shape[1]}d -> {n_components}d ...")
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(vectors)
    print(f"  Components selected: {reduced.shape[1]}  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    return reduced


def run_analysis(X, k_values, batch_size, silhouette_sample_size, calculate_silhouette):
    X_shuffled = shuffle(X, random_state=RANDOM_STATE)
    inertias, ch_scores, db_scores, sil_scores = [], [], [], []

    for k in k_values:
        print(f"\n  k={k} — fitting ...")
        model = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, batch_size=batch_size, n_init=3)

        for i in tqdm(range(0, X_shuffled.shape[0], batch_size), desc=f"    partial_fit k={k}", leave=False):
            batch = X_shuffled[i:i + batch_size]
            if len(batch) < 2:
                continue
            model.partial_fit(batch)

        labels = np.empty(X.shape[0], dtype=np.int32)
        for i in tqdm(range(0, X.shape[0], batch_size), desc=f"    predict   k={k}", leave=False):
            batch = X[i:i + batch_size]
            labels[i:i + len(batch)] = model.predict(batch)

        inertia = float(model.inertia_)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        inertias.append(inertia)
        ch_scores.append(ch)
        db_scores.append(db)

        if calculate_silhouette:
            sil = silhouette_score(X[:silhouette_sample_size], labels[:silhouette_sample_size])
            sil_scores.append(sil)
            print(f"  k={k}  inertia={inertia:,.0f}  CHI={ch:.2f}  DBI={db:.4f}  silhouette={sil:.4f}")
        else:
            sil_scores.append(None)
            print(f"  k={k}  inertia={inertia:,.0f}  CHI={ch:.2f}  DBI={db:.4f}")

    return inertias, ch_scores, db_scores, sil_scores


def build_plot(k_values, inertias, ch_scores, db_scores, sil_scores, elbow_k, best_ch_k, best_db_k, best_sil_k):
    has_sil = any(s is not None for s in sil_scores)
    rows, cols = (2, 2) if has_sil else (1, 3)

    titles = ["Inertia (elbow)", "Calinski-Harabasz (higher=better)", "Davies-Bouldin (lower=better)"]
    if has_sil:
        titles.append("Silhouette (higher=better)")

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)

    positions = [(1, 1), (1, 2), (1, 3)] if not has_sil else [(1, 1), (1, 2), (2, 1), (2, 2)]

    traces = [
        ("Inertia", inertias, "steelblue", elbow_k),
        ("CHI", ch_scores, "green", best_ch_k),
        ("DBI", db_scores, "crimson", best_db_k),
    ]
    if has_sil:
        valid_sil = [s if s is not None else float("nan") for s in sil_scores]
        traces.append(("Silhouette", valid_sil, "orange", best_sil_k))

    for (name, values, color, best_k), (row, col) in zip(traces, positions):
        fig.add_trace(go.Scatter(x=k_values, y=values, mode="lines+markers",
                                 name=name, marker=dict(color=color, size=8),
                                 line=dict(color=color)), row=row, col=col)
        if best_k:
            fig.add_vline(x=best_k, line_dash="dot", line_color=color,
                          annotation_text=f"k={best_k}", row=row, col=col)

    fig.update_layout(title="Advanced Optimal-k Analysis", height=500 * rows,
                      hovermode="x unified")
    fig.update_xaxes(title_text="k")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="data/global_embedding.parquet")
    parser.add_argument("--modality", default=None,
                        help="Filter by modality column (e.g. stained). Omit if your parquet has no modality column.")
    parser.add_argument("--sample", type=int, default=200_000)
    parser.add_argument("--k-values", nargs="+", type=int, default=[16, 32, 48, 64, 80, 96, 112, 128])
    parser.add_argument("--pca", type=float, default=0.9,
                        help="PCA: int for fixed dims (e.g. 100), float 0-1 for variance threshold (e.g. 0.9 = 90%%). "
                             "Use 0 or --no-pca to skip.")
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--silhouette", action="store_true", help="Also compute silhouette (slow, O(n^2))")
    parser.add_argument("--silhouette-sample", type=int, default=5_000)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for CSV and HTML output (default: results/)")
    args = parser.parse_args()

    out_dir = args.output_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    vectors = load_vectors(args.parquet, args.sample, modality=args.modality)

    if args.no_pca or args.pca == 0:
        X = vectors
        print("Skipping PCA.")
    else:
        X = apply_pca(vectors, args.pca)

    print(f"\nTesting k values: {args.k_values}")
    inertias, ch_scores, db_scores, sil_scores = run_analysis(
        X, args.k_values, args.batch_size, args.silhouette_sample, args.silhouette
    )

    kl = KneeLocator(args.k_values, inertias, curve="convex", direction="decreasing")
    elbow_k = kl.elbow
    best_ch_k = args.k_values[int(np.argmax(ch_scores))]
    best_db_k = args.k_values[int(np.argmin(db_scores))]
    valid_sil = [(k, s) for k, s in zip(args.k_values, sil_scores) if s is not None]
    best_sil_k = max(valid_sil, key=lambda x: x[1])[0] if valid_sil else None

    print(f"\n{'='*50}")
    print(f"  Elbow (inertia):         k = {elbow_k}")
    print(f"  Best Calinski-Harabasz:  k = {best_ch_k}")
    print(f"  Best Davies-Bouldin:     k = {best_db_k}")
    if best_sil_k:
        print(f"  Best Silhouette:         k = {best_sil_k}")
    print(f"{'='*50}\n")

    results_df = pd.DataFrame({
        "k": args.k_values,
        "inertia": inertias,
        "calinski_harabasz": ch_scores,
        "davies_bouldin": db_scores,
        "silhouette": sil_scores,
    })
    csv_path = os.path.join(out_dir, "advanced_optimal_k_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to: {csv_path}")
    print(results_df.to_string(index=False))

    fig = build_plot(args.k_values, inertias, ch_scores, db_scores, sil_scores,
                     elbow_k, best_ch_k, best_db_k, best_sil_k)
    html_path = os.path.join(out_dir, "advanced_optimal_k_plot.html")
    fig.write_html(html_path)
    print(f"\nInteractive plot saved to: {html_path}")


if __name__ == "__main__":
    main()
