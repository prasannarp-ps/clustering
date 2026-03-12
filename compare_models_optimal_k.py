"""
Unified comparison plot of optimal-k metrics across all embedding models.

Reads pre-computed CSVs from results/{model}/advanced_optimal_k_metrics.csv
and produces a single interactive HTML with all models overlaid per metric.

Usage:
    python compare_models_optimal_k.py
    python compare_models_optimal_k.py --models path_foundation uni uni2 titan
    python compare_models_optimal_k.py --output results/comparison_optimal_k_plot.html
"""

import argparse
import os

from kneed import KneeLocator
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_MODELS = ["path_foundation", "uni", "uni2", "titan"]

MODEL_COLORS = {
    "path_foundation": "#2196F3",   # blue
    "uni":             "#4CAF50",   # green
    "uni2":            "#FF9800",   # orange
    "titan":           "#E91E63",   # pink-red
    "conch":           "#9C27B0",   # purple
}

MODEL_LABELS = {
    "path_foundation": "Path Foundation (384-dim)",
    "uni":             "UNI (1024-dim)",
    "uni2":            "UNI2 (1536-dim)",
    "titan":           "TITAN / CONCH v1.5 (512-dim)",
    "conch":           "CONCH (512-dim)",
}


def load_metrics(models: list[str], results_dir: str) -> dict[str, pd.DataFrame]:
    loaded = {}
    for m in models:
        path = os.path.join(results_dir, m, "advanced_optimal_k_metrics.csv")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        df = pd.read_csv(path)
        # Normalise inertia to k_min value so curves are comparable across embedding spaces
        df["inertia_norm"] = df["inertia"] / df.loc[df["k"] == df["k"].min(), "inertia"].values[0]
        # Rate of change: % drop in inertia per unit k
        df["inertia_roc"] = df["inertia"].pct_change().abs() / df["k"].diff() * 100
        loaded[m] = df
        print(f"  Loaded {m}: {len(df)} k-values  ({df['k'].min()}–{df['k'].max()})")
    return loaded


def build_comparison_plot(data: dict[str, pd.DataFrame]) -> go.Figure:
    subplot_titles = [
        "Inertia — Elbow Method (lower = tighter clusters)",
        "Calinski-Harabasz Index (higher = better)",
        "Davies-Bouldin Index (lower = better)",
        "Silhouette Score (higher = better)",
    ]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
    )

    metrics = [
        ("inertia_norm",        1, 1),
        ("calinski_harabasz",   1, 2),
        ("davies_bouldin",      2, 1),
        ("silhouette",          2, 2),
    ]

    for model, df in data.items():
        color = MODEL_COLORS.get(model, "#607D8B")
        label = MODEL_LABELS.get(model, model)
        show_legend = True   # show once; Plotly deduplicates by name

        for metric, row, col in metrics:
            if metric not in df.columns:
                continue
            y_vals = df[metric]
            if metric == "silhouette" and y_vals.isna().all():
                continue

            fig.add_trace(
                go.Scatter(
                    x=df["k"],
                    y=y_vals,
                    mode="lines+markers",
                    name=label,
                    legendgroup=model,
                    showlegend=show_legend,
                    marker=dict(color=color, size=7),
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{label}</b><br>k=%{{x}}<br>{metric}=%{{y:.4f}}<extra></extra>",
                ),
                row=row, col=col,
            )
            show_legend = False  # only first subplot adds to legend

    # Mark best k per metric per model with a star marker
    best_markers = {
        "calinski_harabasz": max,
        "davies_bouldin":    min,
        "silhouette":        max,
    }
    metric_positions = {
        "calinski_harabasz": (1, 2),
        "davies_bouldin":    (2, 1),
        "silhouette":        (2, 2),
    }

    for model, df in data.items():
        color = MODEL_COLORS.get(model, "#607D8B")
        label = MODEL_LABELS.get(model, model)

        # Elbow marker on inertia subplot
        kl = KneeLocator(
            df["k"].tolist(), df["inertia"].tolist(),
            curve="convex", direction="decreasing",
        )
        if kl.elbow is not None:
            elbow_row = df[df["k"] == kl.elbow].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[elbow_row["k"]],
                    y=[elbow_row["inertia_norm"]],
                    mode="markers",
                    name=f"{label} elbow",
                    legendgroup=model,
                    showlegend=False,
                    marker=dict(color=color, size=14, symbol="star",
                                line=dict(color="white", width=1)),
                    hovertemplate=(
                        f"<b>{label}</b><br>Elbow k=%{{x}}<br>inertia_norm=%{{y:.4f}}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

        for metric, fn in best_markers.items():
            if metric not in df.columns or df[metric].isna().all():
                continue
            best_idx = df[metric].agg(fn.__name__) if hasattr(fn, "__name__") else fn(df[metric])
            best_row = df[df[metric] == best_idx].iloc[0]
            row, col = metric_positions[metric]
            fig.add_trace(
                go.Scatter(
                    x=[best_row["k"]],
                    y=[best_row[metric]],
                    mode="markers",
                    name=f"{label} best",
                    legendgroup=model,
                    showlegend=False,
                    marker=dict(color=color, size=14, symbol="star",
                                line=dict(color="white", width=1)),
                    hovertemplate=(
                        f"<b>{label}</b><br>Best k=%{{x}}<br>{metric}=%{{y:.4f}}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )

    fig.update_layout(
        title=dict(
            text="Embedding Model Comparison — Optimal-k Metrics",
            font=dict(size=20),
        ),
        height=800,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        margin=dict(t=80, b=140, l=60, r=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Axis labels and grid styling
    for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.update_xaxes(
            title_text="Number of clusters (k)",
            showgrid=True, gridcolor="#E0E0E0", gridwidth=1,
            zeroline=False,
            row=row, col=col,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="#E0E0E0", gridwidth=1,
            zeroline=False,
            row=row, col=col,
        )

    fig.update_yaxes(title_text="Normalised inertia", row=1, col=1)
    fig.update_yaxes(title_text="CHI",                row=1, col=2)
    fig.update_yaxes(title_text="DBI",                row=2, col=1)
    fig.update_yaxes(title_text="Silhouette",         row=2, col=2)

    # Annotation explaining normalised inertia
    fig.add_annotation(
        text="★ = best k for that model",
        xref="paper", yref="paper",
        x=1.0, y=1.03,
        showarrow=False,
        font=dict(size=11, color="#555"),
        align="right",
    )
    fig.add_annotation(
        text="Inertia normalised to each model's k_min value (raw values not comparable across embedding dims)",
        xref="paper", yref="paper",
        x=0.0, y=-0.25,
        showarrow=False,
        font=dict(size=10, color="#777"),
        align="left",
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Unified model comparison plot for optimal-k metrics")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Models to include (must have results/{model}/advanced_optimal_k_metrics.csv)")
    parser.add_argument("--results-dir", default="results",
                        help="Root results directory (default: results)")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: results/comparison_optimal_k_plot.html)")
    args = parser.parse_args()

    out_path = args.output or os.path.join(args.results_dir, "comparison_optimal_k_plot.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Loading metrics for: {args.models}")
    data = load_metrics(args.models, args.results_dir)

    if not data:
        print("No data found. Check that results/{model}/advanced_optimal_k_metrics.csv files exist.")
        return

    print(f"\nBuilding comparison plot for {list(data.keys())} ...")
    fig = build_comparison_plot(data)

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
