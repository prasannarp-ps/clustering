"""
Cluster-level analysis helpers.

This module is for numeric / tabular analysis of clusters, especially
with respect to tissue_percentage and other extended metadata
stored in DuckDB.

Typical usage:
    - Plot distributions of tissue_percentage per cluster
    - Compute counts of tiles below/above thresholds per cluster
    - Export summary tables for reporting
"""


import os
from collections import defaultdict

import duckdb
from . import config
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
from plotly.colors import qualitative


def plot_cluster_tissue_info_duckdb(
    embed_db_path,
    pred_db_path,
    extended_db_path,
    output_dir,
    preprocessing,
    k,
    table_name="embeddings",
    min_tissue_pct=None,  # None = no filter
):
    """
    Plot various cluster/tissue distributions.

    If ext_db.extended_data.tissue_percentage does not exist, this function will:
      - default tissue_percentage to 100 for all rows
      - ignore min_tissue_pct filtering (since everything is treated as 100% tissue)
    """

    os.makedirs(output_dir, exist_ok=True)

    pred_db_path_sql = pred_db_path.replace("'", "''")
    extended_db_path_sql = extended_db_path.replace("'", "''")

    con = duckdb.connect(embed_db_path)
    con.execute(f"ATTACH '{pred_db_path_sql}' AS pred_db;")
    con.execute(f"ATTACH '{extended_db_path_sql}' AS ext_db;")

    # ------------------------------------------------------------------
    # Detect whether ext_db.extended_data and tissue_percentage exist
    # ------------------------------------------------------------------
    has_extended = False
    has_tissue_pct = False
    try:
        # If this succeeds, the table exists
        con.execute("SELECT 1 FROM ext_db.extended_data LIMIT 1")
        has_extended = True

        # Now check for column
        con.execute("SELECT tissue_percentage FROM ext_db.extended_data LIMIT 1")
        has_tissue_pct = True
    except duckdb.Error:
        has_extended = False
        has_tissue_pct = False

    if not has_extended or not has_tissue_pct:
        print(
            "WARNING: ext_db.extended_data.tissue_percentage not found; "
            "defaulting tissue_percentage to 100 and ignoring min_tissue_pct filter."
        )

    # Build SQL fragments depending on availability
    if has_extended and has_tissue_pct:
        nk = config.norm_tile_key_sql
        # Deduplicate extended_data by normalized tile_key to avoid
        # many-to-many join explosion (multiple box_ids share the same norm key).
        join_clause = (
            f"LEFT JOIN ("
            f"SELECT {nk('tile_key')} AS norm_key, "
            f"MAX(tissue_percentage) AS tissue_percentage "
            f"FROM ext_db.extended_data "
            f"GROUP BY {nk('tile_key')}"
            f") x ON {nk('e.tile_key')} = x.norm_key"
        )
        tissue_col_sql = "COALESCE(x.tissue_percentage, 100.0) AS tissue_percentage"
        tissue_filter_sql = ""
        if min_tissue_pct is not None:
            tissue_filter_sql = (
                f"AND COALESCE(x.tissue_percentage, 100.0) >= {min_tissue_pct}"
            )
    else:
        # No usable tissue information → constant 100, no join, no filter
        join_clause = ""
        tissue_col_sql = "CAST(100.0 AS DOUBLE) AS tissue_percentage"
        tissue_filter_sql = ""

    # ------------------------------------------------------------------
    # Main query
    # ------------------------------------------------------------------
    df = con.execute(f"""
        SELECT
            p.cluster,
            e.tissue_type,
            {tissue_col_sql}
        FROM pred_db.predictions p
        JOIN {table_name} e ON p.unique_id = e.unique_id
        {join_clause}
        WHERE TRUE
        {tissue_filter_sql}
    """).df()

    if df.empty:
        print("No data returned. Check filters / DB contents.")
        return

    # ------------------------------------------------------------------
    # Standard tissue / cluster analysis
    # ------------------------------------------------------------------
    # Remove numeric-only labels
    df = df[df["tissue_type"].astype(str).str.isalpha()]

    df_counts = (
        df.groupby(["cluster", "tissue_type"])
        .size()
        .reset_index(name="count")
    )

    cluster_totals = (
        df_counts.groupby("cluster")["count"]
        .sum()
        .reset_index(name="total_count")
        .sort_values("total_count", ascending=False)
    )

    print("\n=== Cluster totals (sorted by size) ===")
    print(cluster_totals.to_string(index=False))


    tissue_types = sorted(df_counts["tissue_type"].unique())
    clusters = sorted(df_counts["cluster"].unique())

    tissue_counts = defaultdict(dict)
    for _, row in df_counts.iterrows():
        tissue_counts[int(row["cluster"])][row["tissue_type"]] = row["count"]

    # Cluster → Tissue pivot
    pivot_df = df_counts.pivot(
        index="cluster",
        columns="tissue_type",
        values="count",
    ).fillna(0)

    pivot_norm = pivot_df.div(pivot_df.sum(axis=1), axis=0)

    # Tissue → Cluster pivot
    inv_pivot_df = df_counts.pivot(
        index="tissue_type",
        columns="cluster",
        values="count",
    ).fillna(0)

    inv_pivot_norm = inv_pivot_df.div(inv_pivot_df.sum(axis=1), axis=0)

    # Colors per tissue
    color_map = {
        t: qualitative.Plotly[i % len(qualitative.Plotly)]
        for i, t in enumerate(tissue_types)
    }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # 1) Stacked tissue count per cluster
    stacked_fig = go.Figure()
    for t in pivot_df.columns:
        stacked_fig.add_trace(
            go.Bar(
                name=t,
                x=pivot_df.index.astype(str),
                y=pivot_df[t],
                marker_color=color_map[t],
            )
        )
    stacked_fig.update_layout(
        barmode="stack",
        title=f"Stacked Tissue Count per Cluster (k={k})",
    )

    # 2) Normalized stacked bar (cluster → tissue distribution)
    stacked_fig_norm = go.Figure()
    for t in pivot_norm.columns:
        stacked_fig_norm.add_trace(
            go.Bar(
                name=t,
                x=pivot_norm.index.astype(str),
                y=pivot_norm[t],
                marker_color=color_map[t],
            )
        )
    stacked_fig_norm.update_layout(
        barmode="stack",
        title=f"Normalized Tissue Distribution per Cluster (k={k})",
    )

    # 3) Polar plot of tissue distributions for each cluster
    polar_fig = go.Figure()
    for c in clusters:
        cd = df_counts[df_counts["cluster"] == c]
        polar_fig.add_trace(
            go.Barpolar(
                r=cd["count"],
                theta=cd["tissue_type"],
                name=f"Cluster {c}",
                marker_color=[color_map[t] for t in cd["tissue_type"]],
                opacity=0.7,
            )
        )
    polar_fig.update_layout(
        title="Polar Tissue Distribution Across Clusters",
    )

    # 4) Cluster size distribution (number of tiles per cluster)
    cluster_counts = df_counts.groupby("cluster")["count"].sum()
    min_id, max_id = cluster_counts.idxmin(), cluster_counts.idxmax()
    fig_size = go.Figure()
    fig_size.add_trace(
        go.Bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            marker_color=[
                "crimson" if cid == min_id else
                "darkgreen" if cid == max_id else
                "steelblue"
                for cid in cluster_counts.index
            ],
        )
    )
    fig_size.update_layout(
        title="Cluster Size Distribution",
        xaxis_title="Cluster",
        yaxis_title="Count",
    )

    # 5) Tissue → Cluster distribution (normalized)
    fig_inv_stacked = go.Figure()
    for c in inv_pivot_norm.columns:
        fig_inv_stacked.add_trace(
            go.Bar(
                x=inv_pivot_norm.index,
                y=inv_pivot_norm[c],
                name=f"Cluster {c}",
            )
        )
    fig_inv_stacked.update_layout(
        barmode="stack",
        title="Normalized Tissue → Cluster Distribution",
    )

    # 6) Tissue → Cluster distribution (raw counts)
    fig_inv_stacked_raw = go.Figure()
    for c in inv_pivot_df.columns:
        fig_inv_stacked_raw.add_trace(
            go.Bar(
                x=inv_pivot_df.index,
                y=inv_pivot_df[c],
                name=f"Cluster {c}",
            )
        )
    fig_inv_stacked_raw.update_layout(
        barmode="stack",
        title="Raw Tissue → Cluster Distribution (Counts)",
    )

    # 7) Low-tissue analysis (only if tissue_percentage is meaningful)
    if "tissue_percentage" in df.columns:
        df_low = df[df["tissue_percentage"] < 20]
        cluster_totals = (
            df.groupby("cluster")
            .size()
            .reset_index(name="total_count")
        )

        if not df_low.empty:
            low_counts = (
                df_low.groupby("cluster")
                .size()
                .reset_index(name="low_count")
            )
        else:
            low_counts = pd.DataFrame(columns=["cluster", "low_count"])

        low_summary = cluster_totals.merge(
            low_counts, on="cluster", how="left"
        )
        low_summary["low_count"] = (
            low_summary["low_count"].fillna(0).astype(int)
        )
        low_summary["low_pct"] = (
            low_summary["low_count"] / low_summary["total_count"] * 100.0
        )

        fig_low = px.bar(
            low_summary,
            x="cluster",
            y="low_pct",
            hover_data=["low_count", "total_count"],
            title="Percentage of Low-Tissue Tiles (<20%) per Cluster",
            labels={"low_pct": "% Low-Tissue Tiles"},
        )

        fig_low_counts = px.bar(
            low_summary,
            x="cluster",
            y="low_count",
            title="Low-Tissue Tile Counts (<20%) per Cluster",
        )

        fig_low.write_html(
            os.path.join(
                output_dir,
                f"{k}_low_tissue_percentage_{preprocessing}.html",
            )
        )
        fig_low_counts.write_html(
            os.path.join(
                output_dir,
                f"{k}_low_tissue_clusters_{preprocessing}.html",
            )
        )

    # ------------------------------------------------------------------
    # Save all figures
    # ------------------------------------------------------------------
    stacked_fig.write_html(
        os.path.join(output_dir, f"{k}_stacked_bar_{preprocessing}.html")
    )
    stacked_fig_norm.write_html(
        os.path.join(
            output_dir, f"{k}_normalized_stacked_bar_{preprocessing}.html"
        )
    )
    fig_inv_stacked.write_html(
        os.path.join(
            output_dir, f"{k}_inverse_normalized_stacked_bar_{preprocessing}.html"
        )
    )
    fig_inv_stacked_raw.write_html(
        os.path.join(
            output_dir, f"{k}_inverse_stacked_bar_{preprocessing}.html"
        )
    )
    polar_fig.write_html(
        os.path.join(output_dir, f"{k}_polar_overlap_clusters_{preprocessing}.html")
    )
    fig_size.write_html(
        os.path.join(output_dir, f"{k}_cluster_size_{preprocessing}.html")
    )

    print(f"All cluster/tissue plots saved to {output_dir}")
    return df_counts
