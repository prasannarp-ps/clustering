"""
Find the optimal number of clusters (k) for MiniBatchKMeans via:
  - Elbow method (inertia vs k)
  - Silhouette score (on a small subset)

Usage:
    python find_optimal_k.py
    python find_optimal_k.py --parquet data/global_embedding.parquet --sample 50000

Outputs:
    results/optimal_k_analysis.html   — Interactive Plotly chart
    results/optimal_k_results.csv     — Raw inertia + silhouette numbers
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_PARQUET = "data/global_embedding.parquet"
DEFAULT_SAMPLE = 200_000          # rows used for k-finding (all modalities)
DEFAULT_SILHOUETTE_SAMPLE = 5_000  # subset for silhouette (O(n²), must be small)
DEFAULT_K_VALUES = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96 ]
RESULTS_DIR = "results"
RANDOM_STATE = 42
# ─────────────────────────────────────────────────────────────────────────────


def load_vectors(parquet_path: str, sample_size: int, modality: str = "stained") -> np.ndarray:
    print(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path, columns=["vector", "modality"])
    print(f"  Total rows: {len(df):,}")

    df = df[df["modality"] == modality]
    print(f"  After modality filter ({modality!r}): {len(df):,} rows")

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)
        print(f"  Sampled:    {len(df):,} rows")

    vectors = np.stack(df["vector"].values).astype(np.float32)
    print(f"  Vector shape: {vectors.shape}")
    return vectors


def find_elbow(inertias: list[float], k_values: list[int]) -> int:
    """Return the k at the 'elbow' using maximum second derivative."""
    if len(inertias) < 3:
        return k_values[0]
    arr = np.array(inertias, dtype=float)
    second_deriv = np.diff(np.diff(arr))
    idx = int(np.argmax(second_deriv)) + 1  # +1 to offset double diff
    return k_values[idx]


def run_analysis(
    vectors: np.ndarray,
    k_values: list[int],
    silhouette_sample: int,
) -> tuple[list[float], list[float | None]]:
    sil_vecs = vectors[:silhouette_sample]

    inertias: list[float] = []
    silhouettes: list[float | None] = []

    for k in k_values:
        print(f"  k={k:>3d} ...", end=" ", flush=True)
        mbk = MiniBatchKMeans(
            n_clusters=k,
            batch_size=10_000,
            random_state=RANDOM_STATE,
            n_init=3,
        )
        mbk.fit(vectors)
        inertias.append(float(mbk.inertia_))

        if k < silhouette_sample:
            labels = mbk.predict(sil_vecs)
            sil = float(silhouette_score(sil_vecs, labels))
            silhouettes.append(sil)
            print(f"inertia={mbk.inertia_:,.0f}  silhouette={sil:.4f}")
        else:
            silhouettes.append(None)
            print(f"inertia={mbk.inertia_:,.0f}  silhouette=N/A (k >= sample size)")

    return inertias, silhouettes


def build_plot(
    k_values: list[int],
    inertias: list[float],
    silhouettes: list[float | None],
    elbow_k: int,
    best_sil_k: int | None,
    sample_size: int,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Inertia trace
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=inertias,
            mode="lines+markers",
            name="Inertia (elbow)",
            marker=dict(size=8, color="steelblue"),
            line=dict(color="steelblue"),
        ),
        secondary_y=False,
    )

    # Silhouette trace
    valid_k = [k for k, s in zip(k_values, silhouettes) if s is not None]
    valid_sil = [s for s in silhouettes if s is not None]
    if valid_sil:
        fig.add_trace(
            go.Scatter(
                x=valid_k,
                y=valid_sil,
                mode="lines+markers",
                name="Silhouette score",
                marker=dict(size=8, color="crimson"),
                line=dict(color="crimson", dash="dash"),
            ),
            secondary_y=True,
        )

    # Vertical markers
    fig.add_vline(
        x=elbow_k,
        line_dash="dot",
        line_color="steelblue",
        annotation_text=f"Elbow k={elbow_k}",
        annotation_position="top right",
    )
    if best_sil_k and best_sil_k != elbow_k:
        fig.add_vline(
            x=best_sil_k,
            line_dash="dot",
            line_color="crimson",
            annotation_text=f"Best silhouette k={best_sil_k}",
            annotation_position="top left",
        )

    fig.update_layout(
        title=(
            f"Optimal k Analysis — sample size {sample_size:,}<br>"
            f"<sub>Elbow: k={elbow_k}"
            + (f" | Best silhouette: k={best_sil_k}" if best_sil_k else "")
            + "</sub>"
        ),
        xaxis_title="Number of clusters (k)",
        legend=dict(x=0.72, y=0.95),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Inertia (lower = better)", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette score (higher = better)", secondary_y=True)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Find optimal k for MiniBatchKMeans")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET, help="Path to embeddings parquet")
    parser.add_argument("--modality", default="stained", help="Modality to cluster (default: stained)")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="Sample size for k search")
    parser.add_argument("--silhouette-sample", type=int, default=DEFAULT_SILHOUETTE_SAMPLE)
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        help="List of k values to test (e.g. --k-values 8 16 32 64 112)",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load data
    vectors = load_vectors(args.parquet, args.sample, modality=args.modality)

    # 2. Run analysis
    print(f"\nTesting k values: {args.k_values}")
    inertias, silhouettes = run_analysis(vectors, args.k_values, args.silhouette_sample)

    # 3. Determine recommendations
    elbow_k = find_elbow(inertias, args.k_values)

    valid_sil = [(k, s) for k, s in zip(args.k_values, silhouettes) if s is not None]
    best_sil_k = max(valid_sil, key=lambda x: x[1])[0] if valid_sil else None

    print(f"\n{'='*50}")
    print(f"  Elbow method recommends:      k = {elbow_k}")
    if best_sil_k:
        print(f"  Best silhouette score at:     k = {best_sil_k}")
    print(f"{'='*50}\n")

    # 4. Save CSV
    results_df = pd.DataFrame(
        {"k": args.k_values, "inertia": inertias, "silhouette": silhouettes}
    )
    csv_path = os.path.join(RESULTS_DIR, "optimal_k_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results table saved to: {csv_path}")
    print(results_df.to_string(index=False))

    # 5. Save HTML plot
    fig = build_plot(args.k_values, inertias, silhouettes, elbow_k, best_sil_k, len(vectors))
    html_path = os.path.join(RESULTS_DIR, "optimal_k_analysis.html")
    fig.write_html(html_path)
    print(f"\nInteractive plot saved to: {html_path}")

    print(f"\nNext step: run the full pipeline with your chosen k, e.g.:")
    print(f"  python -m clustering_pipeline.cli build-db --embed-parquet {args.parquet}")
    print(f"  python -m clustering_pipeline.cli train --table-name embeddings --k {elbow_k}")
    print(f"  python -m clustering_pipeline.cli predict --k {elbow_k}")
    print(f"  python -m clustering_pipeline.cli postprocess")
    print(f"  python -m clustering_pipeline.cli analyze --k {elbow_k}")


if __name__ == "__main__":
    main()
