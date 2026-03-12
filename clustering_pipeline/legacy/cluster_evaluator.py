import glob
import os
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
from embeddings.data_handling.get_data import S3Downloader
BASE_DIR = "/Volumes/KINGSTON/v6_2_comparison"
os.makedirs(BASE_DIR, exist_ok=True)


def find_matching_inferred_and_stained():
    # Extract same samples from both chemical and virtual stain embeddings
    df_inferr = pd.read_parquet(f"{BASE_DIR}/infered_resize.parquet")
    infer_slide_keys = set(df_inferr['slide_key'])

    stained = []
    matched_slide_keys = set()

    for x in glob.glob("/Volumes/KINGSTON/v4_flat_stained/*.parquet"):
        df = pd.read_parquet(x)

        # Find intersection of slide_keys in current file
        df_keys = set(df['slide_key'])
        common_keys = infer_slide_keys.intersection(df_keys)

        if common_keys:
            filtered_df = df[df['slide_key'].isin(common_keys)]
            stained.append(filtered_df)
            matched_slide_keys.update(common_keys)
            print(f"{x} => matched {len(filtered_df)} rows")

        print(f"Current total is: {sum([len(xx) for xx in stained])}")

    # After processing all files
    df_stained = pd.concat(stained, ignore_index=True)
    print(f"Total stained embeddings: {len(df_stained)}")
    df_stained.to_parquet(f"{BASE_DIR}/stained_resize.parquet")

    # Compute unmatched slide_keys
    unmatched_slide_keys = infer_slide_keys - matched_slide_keys
    print(f"Total unmatched slide_keys: {len(unmatched_slide_keys)}")

    # Save unmatched slide_keys to a file
    unmatched_path = f"{BASE_DIR}/unmatched_slide_keys.txt"
    with open(unmatched_path, "w") as f:
        for slide_key in sorted(unmatched_slide_keys):
            f.write(f"{slide_key}\n")

    print(f"Unmatched slide_keys saved to {unmatched_path}")


from pathlib import Path
import tempfile
import shutil
import cv2

def get_image_paths(samples_df, image_type, local_dir=None):
    """
    image_type: 'stained' or 'virtual'
    local_dir: optional path to look for images locally
    """
    paths = []

    for _, row in samples_df.iterrows():
        filename = f"{row['filename']}-stained.tiff" if image_type == "stained" else f"{row['filename']}-stained-sr.tiff"
        subdir = os.path.join(row["tissue_type"], row["block_id"], row["slice_id"],
                              row["scan_date"], row["box_id"])
        relative_path = os.path.join(subdir, filename)

        if local_dir:
            full_path = os.path.join(local_dir, relative_path)
            if os.path.exists(full_path):
                paths.append(full_path)
            else:
                print(f"[WARN] Missing local {image_type} image: {full_path}")
        else:
            s3_path = f"s3://gi-registration/slide-registration-production/tiles_nanozoomers360md/512/{relative_path}"
            paths.append(s3_path)

    return paths



class ClusterDistanceEvaluator:
    def __init__(self, cluster_dir: str, chem_path: str, virtual_path: str, out_dir: str = None):
        """
        cluster_dir: directory where your cluster assignment parquet files are stored (per model)
        chem_path: parquet file containing chemical stain embeddings
        virtual_path: parquet file containing virtual stain embeddings
        """
        self.cluster_dir = cluster_dir
        self.chem_path = chem_path
        self.virtual_path = virtual_path

        self.chem_df = None
        self.virtual_df = None
        self.models = {}
        self.distance_df = {}
        self.stats_df = {}

        self.cluster_ranking_df = None
        self.summary_stats_df = None

        self.color_map = {"stained": "#1f77b4", "virtual": "#ff7f0e"}

    def load_embeddings(self):
        print("Loading chemical stain embeddings...")
        self.chem_df = pd.read_parquet(self.chem_path)
        # If "-stained" string is present in the value of the row in slide_key, remove it
        self.chem_df['slide_key'] = self.chem_df['slide_key'].str.replace('-stained', '', regex=False)
        print("Loading virtual stain embeddings...")
        self.virtual_df = pd.read_parquet(self.virtual_path)

    def align_pairs(self):
        """
        Align the chemical and virtual stain embeddings by slide_key
        """
        print("Aligning pairs...")
        merged = pd.merge(self.chem_df, self.virtual_df, on="slide_key", suffixes=("_chem", "_virt"))
        print(f"Aligned {len(merged)} pairs.")
        self.paired_df = merged

    def visualize_slide_key_stats(self, output_path: str):
        """
        Analyze and visualize the distribution of parts extracted from slide_key.
        The expected format is: tissue_type_block_id_slice_id_scan_date_box_id_filename
        where filename may contain underscores.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if self.paired_df is None:
            raise ValueError("Paired dataframe not loaded. Run align_pairs() first.")

        # Split slide_key by "_" with filename as remainder
        split_all = self.paired_df["slide_key"].str.split("_", expand=False)

        # Ensure each split has at least 6 parts
        valid = [x for x in split_all if len(x) >= 6]
        if not valid:
            print("No valid slide_keys found.")
            return

        # Extract first 6 components and group remaining as filename
        extracted_parts = {
            "tissue_type": [],
            "block_id": [],
            "slice_id": [],
            "scan_date": [],
            "box_id": [],
            "filename": []
        }

        for parts in split_all:
            if len(parts) >= 6:
                extracted_parts["tissue_type"].append(parts[0])
                extracted_parts["block_id"].append(parts[1])
                extracted_parts["slice_id"].append(parts[2])
                extracted_parts["scan_date"].append(parts[3])
                extracted_parts["box_id"].append(parts[4])
                extracted_parts["filename"].append("_".join(parts[5:]))

        # Create DataFrame from parsed components
        parts_df = pd.DataFrame(extracted_parts)

        # Count frequencies for each column
        value_counts = {col: parts_df[col].value_counts() for col in parts_df.columns}
        fig = make_subplots(
            rows=len(value_counts), cols=1,
            subplot_titles=[f"Value Counts for {col}" for col in value_counts],
            vertical_spacing=0.05
        )

        for i, (col, counts) in enumerate(value_counts.items(), start=1):
            fig.add_trace(go.Bar(
                x=counts.index.astype(str),
                y=counts.values,
                name=col,
                text=counts.values,
                textposition="outside"
            ), row=i, col=1)

        fig.update_layout(
            height=350 * len(value_counts),
            width=1200,
            title="Sample distribution",
            showlegend=False,
            uniformtext_minsize=8,
            uniformtext_mode="hide"
        )

        out_path = os.path.join(output_dir, "data_description.html")
        fig.write_html(out_path)
        print(f"Saved slide_key stats to {out_path}")

    def load_cluster_model(self, k: int):
        # path_stained = os.path.join(self.cluster_dir, f"clustering_results_stained_k{k}.parquet")
        path_stained = os.path.join(self.cluster_dir, f"clustering_results_stained_center_crop_k{k}.parquet")
        path_virtual = os.path.join(self.cluster_dir, f"clustering_results_inferred_k{k}.parquet")

        if not os.path.exists(path_stained):
            raise FileNotFoundError(f"Stained cluster file for k={k} not found: {path_stained}")
        if not os.path.exists(path_virtual):
            raise FileNotFoundError(f"Virtual cluster file for k={k} not found: {path_virtual}")

        print(f"Loading cluster models for k={k}...")
        df_stained = pd.read_parquet(path_stained)
        df_stained['slide_key'] = df_stained['slide_key'].str.replace('-stained', '', regex=False)
        df_virtual = pd.read_parquet(path_virtual)

        print(f"[DEBUG] k={k} stained.columns = {df_stained.columns.tolist()}")
        print(f"[DEBUG] k={k} virtual.columns = {df_virtual.columns.tolist()}")

        self.models[k] = {"stained": df_stained, "virtual": df_virtual}

    def get_cluster_pair_dataframe(self, k: int) -> pd.DataFrame:
        """
        Create a DataFrame with slide_key, stained cluster ID, and virtual cluster ID for model k.
        Returns a DataFrame with columns: ['slide_key', 'cluster_stained', 'cluster_virtual']
        """
        if k not in self.models:
            self.load_cluster_model(k)

        df_stained = self.models[k]["stained"][["slide_key", "cluster"]].rename(columns={"cluster": "cluster_stained"})
        df_virtual = self.models[k]["virtual"][["slide_key", "cluster"]].rename(columns={"cluster": "cluster_virtual"})

        df_merged = pd.merge(df_stained, df_virtual, on="slide_key", how="inner")
        print(f"Generated cluster pair DataFrame with {len(df_merged)} matched slide_keys for k={k}")
        if not hasattr(self, "cluster_df"):
            self.cluster_df = df_merged

    def export_summary_data(self, output_dir: str):
        """
        Save summary, rankings, and consolidated per-cluster stats (stained + virtual) to disk.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save overall summary
        if self.summary_stats_df is not None:
            self.summary_stats_df.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)

        # Save top cluster rankings
        if self.cluster_ranking_df is not None:
            self.cluster_ranking_df.to_csv(os.path.join(output_dir, "top_cluster_ranking.csv"), index=False)

        # Save per-model cluster stats, consolidated
        for k, stats in self.stats_df.items():
            df_stained = stats["stained"].copy()
            df_virtual = stats["virtual"].copy()

            df_stained["cluster_id"] = df_stained["cluster"]
            df_virtual["cluster_id"] = df_virtual["cluster_virtual"]

            df_stained["type"] = "stained"
            df_virtual["type"] = "virtual"

            combined = pd.concat([
                df_stained.drop(columns=["cluster"], errors="ignore"),
                df_virtual.drop(columns=["cluster_virtual"], errors="ignore")
            ], ignore_index=True)

            combined.to_csv(os.path.join(output_dir, f"cluster_stats_k{k}.csv"), index=False)

        print(f"✅ Exported consolidated per-model stats to {output_dir}")

    def load_cached_stats(self, summary_path: str, ranking_path: str):
        """
        Load previously saved summary and ranking dataframes.
        """
        self.summary_stats_df = pd.read_csv(summary_path)
        self.cluster_ranking_df = pd.read_csv(ranking_path)
        print("Loaded cached summary and ranking data.")

    def compute_distances_for_model(self, k: int):
        """
        Compute per-tile vector distances for model k.
        """
        cluster_stained_df = self.models[k]["stained"]
        cluster_virtual_df = self.models[k]["virtual"]

        # Merge both cluster assignments
        merged = pd.merge(self.paired_df, cluster_stained_df, on="slide_key", suffixes=("", "_stained"))
        merged = pd.merge(merged, cluster_virtual_df, on="slide_key", suffixes=("", "_virtual"))

        print(f"Computing distances for {len(merged)} samples for model k={k}...")

        # Ensure proper array conversion before stacking
        vectors_chem = np.vstack(merged['vector_chem'].apply(np.array).to_numpy())
        vectors_virt = np.vstack(merged['vector_virt'].apply(np.array).to_numpy())
        distances = np.linalg.norm(vectors_chem - vectors_virt, axis=1)

        merged['distance'] = distances
        self.distance_df[k] = merged

    def compute_stats_for_model(self, k: int):
        """
        Compute summary stats for each cluster in model k.
        You can choose to group by stained, virtual, or both clusters.
        """
        df = self.distance_df[k]

        # Grouping example: group by stained clusters
        stats_stained = df.groupby("cluster")["distance"].agg([
            ("mean", np.mean),
            ("median", np.median),
            ("p75", lambda x: np.percentile(x, 75)),
            ("p90", lambda x: np.percentile(x, 90)),
            ("p99", lambda x: np.percentile(x, 99)),
            ("std", np.std)
        ]).reset_index()

        # Grouping example: group by virtual clusters
        stats_virtual = df.groupby("cluster_virtual")["distance"].agg([
            ("mean", np.mean),
            ("median", np.median),
            ("p75", lambda x: np.percentile(x, 75)),
            ("p90", lambda x: np.percentile(x, 90)),
            ("p99", lambda x: np.percentile(x, 99)),
            ("std", np.std)
        ]).reset_index()

        self.stats_df[k] = {
            "stained": stats_stained,
            "virtual": stats_virtual
        }

    def evaluate_models(self, k_list: List[int], output_dir: str = None):
        """
        Full pipeline for multiple cluster models.
        If output_dir is provided, tries to load consolidated cluster stats from CSV.
        """
        for k in k_list:
            stats_exist = False
            combined_path = None

            if output_dir:
                combined_path = os.path.join(output_dir, f"cluster_stats_k{k}.csv")
                stats_exist = os.path.exists(combined_path)

            if stats_exist:
                print(f"Loading cached consolidated cluster stats for k={k}...")
                df = pd.read_csv(combined_path)

                # Separate back into stained and virtual
                df_stained = df[df["type"] == "stained"].copy()
                df_virtual = df[df["type"] == "virtual"].copy()

                # Restore expected column names
                df_stained["cluster"] = df_stained["cluster_id"]
                df_virtual["cluster_virtual"] = df_virtual["cluster_id"]

                self.stats_df[k] = {
                    "stained": df_stained.drop(columns=["cluster_id"]),
                    "virtual": df_virtual.drop(columns=["cluster_id"])
                }

            else:
                self.load_cluster_model(k)
                self.compute_distances_for_model(k)
                self.compute_stats_for_model(k)

                if output_dir:
                    self.export_summary_data(output_dir)  # Save full set after computation

    def summarize_results(self, summary_path: str = None):
        """
        Compare models based on average std deviation (for both clusterings).
        If summary_path is provided and exists, loads from CSV instead of recomputing.
        """
        if summary_path and os.path.exists(summary_path):
            print(f"Loading cached summary from {summary_path}")
            self.summary_stats_df = pd.read_csv(summary_path)
            return self.summary_stats_df

        summary = []
        for k, stats in self.stats_df.items():
            avg_std_stained = stats['stained']['std'].mean()
            avg_std_virtual = stats['virtual']['std'].mean()

            summary.append({
                "k": k,
                "avg_std_stained": avg_std_stained,
                "avg_std_virtual": avg_std_virtual
            })

        self.summary_stats_df = pd.DataFrame(summary).sort_values("avg_std_stained")
        print("Model ranking (sorted by stained avg std dev):")
        print(self.summary_stats_df)

        if summary_path:
            self.summary_stats_df.to_csv(summary_path, index=False)
            print(f"Saved summary stats to {summary_path}")

        return self.summary_stats_df

    def export_results(self, output_dir: str):
        """
        Save full cluster stats for each model.
        """
        os.makedirs(output_dir, exist_ok=True)
        for k, stats in self.stats_df.items():
            stats['stained'].to_csv(os.path.join(output_dir, f"cluster_stats_stained_k{k}.csv"), index=False)
            stats['virtual'].to_csv(os.path.join(output_dir, f"cluster_stats_virtual_k{k}.csv"), index=False)

    def visualize_summary(self, output_dir: str):
        """
        Generate a single summary plot comparing avg cluster stats across models.
        Each stat (mean, median, p75, p90, p99, std) is a subplot with stained and virtual.
        """
        os.makedirs(output_dir, exist_ok=True)

        metrics = ["mean", "median", "p75", "p90", "p99", "std"]

        # Build summary_df
        summary_records = []
        for k, stats in self.stats_df.items():
            record = {"k": k}
            for metric in metrics:
                for t in ["stained", "virtual"]:
                    try:
                        record[f"{metric}_{t}"] = stats[t][metric].mean()
                    except KeyError:
                        record[f"{metric}_{t}"] = None  # skip if metric missing
            summary_records.append(record)

        summary_df = pd.DataFrame(summary_records).sort_values("k")

        # Plot
        fig = sp.make_subplots(
            rows=3, cols=2,
            subplot_titles=[f"{m.capitalize()} vs Cluster Size" for m in metrics],
            vertical_spacing=0.1
        )

        for idx, metric in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1

            for cluster_type in ["stained", "virtual"]:
                y_col = f"{metric}_{cluster_type}"
                show_legend = idx == 0  # show legend only once

                fig.add_trace(go.Scatter(
                    x=summary_df["k"],
                    y=summary_df[y_col],
                    name=cluster_type if show_legend else None,
                    mode="lines+markers",
                    marker=dict(color=self.color_map[cluster_type]),
                    legendgroup=cluster_type,
                    showlegend=show_legend
                ), row=row, col=col)

        fig.update_layout(
            height=900,
            width=1200,
            title="Summary of Per-Cluster Embedding Distances Across Models",
            xaxis_title="Number of Clusters",
            showlegend=True
        )

        out_path = os.path.join(output_dir, "summary_all.html")
        fig.write_html(out_path)
        print(f"Saved {out_path}")

    def visualize_std_comparison_per_model(self, output_dir: str, top_n: int = 5):
        """
        For each model, generate a grouped bar plot comparing stained and virtual std per cluster.
        Adds markers to top N clusters with the highest absolute difference in std.
        Uses consistent coloring and legend entries.
        """
        os.makedirs(output_dir, exist_ok=True)

        for k, stats in self.stats_df.items():
            df_stained = stats["stained"].copy()
            df_virtual = stats["virtual"].copy()

            df_stained["cluster_id"] = df_stained["cluster"]
            df_virtual["cluster_id"] = df_virtual["cluster_virtual"]

            df_stained["type"] = "stained"
            df_virtual["type"] = "virtual"

            # Merge to compute differences
            merged = pd.merge(
                df_stained[["cluster_id", "std"]],
                df_virtual[["cluster_id", "std"]],
                on="cluster_id",
                suffixes=("_stained", "_virtual")
            )
            merged["diff"] = np.abs(merged["std_stained"] - merged["std_virtual"])
            top_clusters = merged.sort_values("diff", ascending=False).head(top_n)["cluster_id"].tolist()

            # Combine data for plotting
            combined = pd.concat([
                df_stained[["cluster_id", "std", "type"]],
                df_virtual[["cluster_id", "std", "type"]]
            ])

            fig = go.Figure()

            for cluster_type in ["stained", "virtual"]:
                sub_df = combined[combined["type"] == cluster_type]
                show_legend = True  # only add legend once per type

                fig.add_trace(go.Bar(
                    x=sub_df["cluster_id"].astype(str),
                    y=sub_df["std"],
                    name=cluster_type,
                    marker_color=self.color_map[cluster_type],
                    legendgroup=cluster_type,
                    showlegend=show_legend
                ))

            # Add red marker above bars with highest diff
            for cluster_id in top_clusters:
                max_std = combined[combined["cluster_id"] == cluster_id]["std"].max()
                fig.add_trace(go.Scatter(
                    x=[str(cluster_id)],
                    y=[max_std + 0.01],
                    mode="markers",
                    marker=dict(color="red", size=10, symbol="circle"),
                    name="High Δ std" if cluster_id == top_clusters[0] else None,
                    showlegend=(cluster_id == top_clusters[0])
                ))

            fig.update_layout(
                title=f"Per-Cluster Std Comparison | k={k}",
                xaxis_title="Cluster ID",
                yaxis_title="Standard Deviation",
                barmode="group",
                xaxis_type="category",
                width=1000,
                height=600
            )

            out_path = os.path.join(output_dir, f"std_comparison_k{k}.html")
            fig.write_html(out_path)
            print(f"Saved {out_path}")

    def summarize_cluster_ranking(self, top_n: int = 10, sort_by: str = "std_ratio", ranking_path: str = None):
        """
        Print and return a summary of top N clusters across all models ranked by variability.
        If ranking_path exists, loads it from disk instead of recomputing.
        """
        if ranking_path and os.path.exists(ranking_path):
            print(f"Loading cached cluster ranking from {ranking_path}")
            self.cluster_ranking_df = pd.read_csv(ranking_path)
            return self.cluster_ranking_df

        ranking_records = []

        for k, stats in self.stats_df.items():
            for cluster_type in ["stained", "virtual"]:
                df = stats[cluster_type].copy()
                cluster_col = "cluster" if cluster_type == "stained" else "cluster_virtual"

                df["std_ratio"] = df.apply(
                    lambda row: row["std"] / row["mean"] if row["mean"] > 1e-6 else np.nan,
                    axis=1
                )
                df = df.dropna(subset=["std_ratio"])

                df["model_k"] = k
                df["type"] = cluster_type
                df["cluster_id"] = df[cluster_col]

                selected = df[["model_k", "type", "cluster_id", "mean", "std", "std_ratio"]]
                selected = selected.sort_values(sort_by, ascending=False)

                ranking_records.extend(selected.head(top_n).to_dict("records"))

        self.cluster_ranking_df = pd.DataFrame(ranking_records)
        print(f"\nTop {top_n} clusters across all models sorted by '{sort_by}':")
        print(self.cluster_ranking_df.to_string(index=False))

        if ranking_path:
            self.cluster_ranking_df.to_csv(ranking_path, index=False)
            print(f"Saved cluster ranking to {ranking_path}")

        return self.cluster_ranking_df

    def visualize_cluster_ranking(self, output_dir: str, top_n: int = 10, sort_by: str = "std_ratio"):
        """
        Generate one HTML with vertically stacked scatter plots of top N clusters per model.
        Each subplot compares stained vs virtual for the selected sort_by stat.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        os.makedirs(output_dir, exist_ok=True)
        ranking_df = self.summarize_cluster_ranking(top_n=top_n, sort_by=sort_by)

        color_map = {"stained": "#1f77b4", "virtual": "#ff7f0e"}
        model_ks = sorted(ranking_df["model_k"].unique())

        fig = make_subplots(
            rows=len(model_ks),
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.1,
            subplot_titles=[f"k = {k}" for k in model_ks]
        )

        for i, k in enumerate(model_ks):
            df_k = ranking_df[ranking_df["model_k"] == k]

            for cluster_type in ["stained", "virtual"]:
                sub_df = df_k[df_k["type"] == cluster_type]
                fig.add_trace(go.Scatter(
                    x=sub_df["cluster_id"].astype(str),
                    y=sub_df[sort_by],
                    mode="markers",
                    marker=dict(color=color_map[cluster_type]),
                    name=cluster_type if i == 0 else None,  # show legend only in first plot
                    legendgroup=cluster_type,
                    showlegend=(i == 0)
                ), row=i + 1, col=1)

            fig.update_xaxes(
                title_text="Cluster ID",
                row=i + 1, col=1
            )
            fig.update_yaxes(
                title_text=sort_by,
                row=i + 1, col=1
            )

        fig.update_layout(
            height=300 * len(model_ks),
            width=1000,
            title_text=f"Top {top_n} Clusters by {sort_by.upper()} (Stained vs Virtual)",
            showlegend=True
        )

        out_path = os.path.join(output_dir, f"top_clusters_by_{sort_by}.html")
        fig.write_html(out_path)
        print(f"Saved {out_path}")

    def visualize_full_stats_per_model(self, output_dir: str):
        """
        For each model, generate one HTML with all full per-cluster stats
        (mean, median, p75, p90, p99, std) in subplots.
        Both stained and virtual are plotted together with grouped bars.
        Uses consistent coloring and only two legend entries.
        """
        os.makedirs(output_dir, exist_ok=True)
        metrics = ["mean", "median", "p75", "p90", "p99", "std"]

        for k, stats in self.stats_df.items():
            df_stained = stats["stained"].copy()
            df_virtual = stats["virtual"].copy()

            df_stained["cluster_id"] = df_stained["cluster"]
            df_virtual["cluster_id"] = df_virtual["cluster_virtual"]
            df_stained["type"] = "stained"
            df_virtual["type"] = "virtual"

            combined = pd.concat([df_stained, df_virtual], ignore_index=True)

            fig = sp.make_subplots(
                rows=3, cols=2,
                subplot_titles=[m.capitalize() for m in metrics],
                shared_xaxes=False,
                vertical_spacing=0.12
            )

            for i, metric in enumerate(metrics):
                row = i // 2 + 1
                col = i % 2 + 1

                for cluster_type in ["stained", "virtual"]:
                    sub_df = combined[combined["type"] == cluster_type]

                    # Only show legend once
                    show_legend = True if i == 0 else False

                    fig.add_trace(
                        go.Bar(
                            x=sub_df["cluster_id"],
                            y=sub_df[metric],
                            name=cluster_type if show_legend else None,
                            marker_color=self.color_map[cluster_type],
                            legendgroup=cluster_type,
                            showlegend=show_legend
                        ),
                        row=row, col=col
                    )

            fig.update_layout(
                height=1000,
                width=1200,
                title_text=f"Full Cluster Stats (Stained vs Virtual) | k={k}",
                barmode="group"
            )

            out_path = os.path.join(output_dir, f"full_cluster_stats_combined_k{k}.html")
            fig.write_html(out_path)
            print(f"Saved {out_path}")

    def visualize_all(self, output_dir: str, top_n: int = 20):
        self.visualize_summary(output_dir)
        self.visualize_std_comparison_per_model(output_dir)
        self.visualize_cluster_ranking(output_dir, top_n=top_n, sort_by="std_ratio")
        self.visualize_cluster_ranking(output_dir, top_n=top_n, sort_by="std")
        self.visualize_full_stats_per_model(output_dir)

    def visualize_cluster_samples(self, rows=2, cols=5, resize_dim=(256, 256), font_scale=0.4,
                                  font_thickness=1, padding=2, cluster_id=None, num_grids=1,
                                  k=None, out_dir=None, show_cluster_mismatch=False):

        df = self.cluster_df

        if "cluster_stained" not in df.columns:
            print("'cluster_stained' column missing.")
            return
        if "cluster_virtual" not in df.columns:
            print("'cluster_virtual' column missing.")
            return

        # Determine which clusters to process
        if cluster_id is not None:
            clusters_to_process = [cluster_id]
            images_per_cluster = rows * cols * num_grids
            out_dir = f"{out_dir}/cluster_{cluster_id}_grids"
            os.makedirs(out_dir, exist_ok=True)
        else:
            clusters_to_process = list(range(0, max([df["cluster_stained"].max(), df["cluster_stained"].max()]) + 1))
            images_per_cluster = rows * cols
            num_grids = 1
            out_dir = f"{out_dir}/cluster_grids" if k is None else f"{out_dir}/cluster_grids/{k}"
            os.makedirs(out_dir, exist_ok=True)

        print(f"Found {len(clusters_to_process)} clusters to process. Generating grids...")

        def build_s3_path(slide_key, stain_type="s"):
            if stain_type == "s":
                return f"s3://gi-registration/slide-registration-production/tiles_nanozoomers360md/512/{slide_key.replace('_', '/').replace('tile/', 'tile_')}-stained.tiff"
            elif stain_type == "v":
                return f"s3://gi-tmp/milos-inferred-tiles/{slide_key.replace('_', '/').replace('tile/', 'tile_')}-inferred.tiff"
            elif stain_type == "u":
                return f"s3://gi-registration/slide-registration-production/tiles_nanozoomers360md/512/{slide_key.replace('_', '/').replace('tile/', 'tile_')}-unstained.tiff"
            return ""

        for cl_id in sorted(clusters_to_process):
            if os.path.exists(f"{out_dir}/cluster_{cl_id}_grid.png"):
                print(f"\nSkipping existing grid cluster {cl_id}...")
                continue
            else:
                print(f"\nProcessing cluster {cl_id}...")

            # Get samples for this cluster
            cluster_samples = df[df["cluster_stained"] == cl_id]
            total_samples = images_per_cluster

            if len(cluster_samples) < total_samples:
                print(f"Cluster {cl_id} has only {len(cluster_samples)} samples. Using all available.")
                samples = cluster_samples
                total_samples = len(cluster_samples)
            else:
                samples = cluster_samples.sample(total_samples, random_state=42)

            slide_keys = samples["slide_key"].tolist()

            # Build S3 paths
            s3_paths_stained = [build_s3_path(doc, "s") for doc in slide_keys]
            s3_paths_virtual = [build_s3_path(doc, "v") for doc in slide_keys]

            # Download images
            temp_dir_stained = tempfile.mkdtemp("stained")
            temp_dir_virtual = tempfile.mkdtemp("virtual")
            downloader_stained = S3Downloader(s3_paths=s3_paths_stained, dl_dir=temp_dir_stained, num_workers=8)
            downloader_virtual = S3Downloader(s3_paths=s3_paths_virtual, dl_dir=temp_dir_virtual, num_workers=8)
            file_list_stained = downloader_stained.build_s3_paths(suffixess=["-stained.tiff"])
            file_list_virtual = downloader_virtual.build_s3_paths(suffixess=["-inferred.tiff"])
            downloader_stained.run(file_list_stained)
            downloader_virtual.run(file_list_virtual)

            for temp_dir, stain_type in zip([temp_dir_stained, temp_dir_virtual], ["stained", "virtual"]):
                tiff_files = sorted(Path(temp_dir).rglob("*.tiff"))
                if not tiff_files:
                    print(f"No images downloaded for cluster {cl_id}. Skipping.")
                    shutil.rmtree(temp_dir)
                    continue

                # Load images and prepare labels
                loaded_images = []
                labels = []

                for idx, path in enumerate(tiff_files):
                    img = cv2.imread(str(path))
                    if img is None:
                        print(f"Failed to load image {path}")
                        continue
                    # do a center crop of the image so that the image is 224x224
                    if stain_type == "stained":
                        img = img[(img.shape[0] - 224) // 2: (img.shape[0] + 224) // 2,
                                (img.shape[1] - 224) // 2: (img.shape[1] + 224) // 2]

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    loaded_images.append(img)

                    # Get label
                    if idx < len(slide_keys):
                        labels.append(str(slide_keys[idx]))
                    else:
                        labels.append("N/A")

                # Ensure correct number of images by padding if needed
                while len(loaded_images) < total_samples:
                    placeholder = np.zeros((resize_dim[1], resize_dim[0], 3), dtype=np.uint8)
                    loaded_images.append(placeholder)
                    labels.append("N/A")

                # Resize and add labels to images
                labeled_images = []
                font = cv2.FONT_HERSHEY_SIMPLEX
                max_width = resize_dim[0] - 10  # Leave margin

                for img, label in zip(loaded_images, labels):
                    img_resized = cv2.resize(img, resize_dim)
                    labeled_img = img_resized.copy()

                    # Estimate character width for text wrapping
                    (char_width, _), _ = cv2.getTextSize("A", font, font_scale, font_thickness)
                    max_chars_per_line = max(1, max_width // char_width)

                    # Split label into multiple lines
                    lines = [label[i:i + max_chars_per_line] for i in range(0, len(label), max_chars_per_line)]

                    # Add label in top left
                    y0 = 20
                    dy = int(20 * font_scale) + 5
                    for idx_line, line in enumerate(lines[:3]):
                        y = y0 + idx_line * dy
                        cv2.putText(labeled_img, line, (5, y), font, font_scale,
                                    (255, 255, 255), font_thickness, cv2.LINE_AA)

                    if show_cluster_mismatch and stain_type == "virtual":
                        row_match = df[df["slide_key"] == label]
                        if not row_match.empty:
                            stained_id = row_match["cluster_stained"].values[0]
                            virtual_id = row_match["cluster_virtual"].values[0]

                            # Show predicted cluster ID in lower-right corner
                            cluster_text = f"{virtual_id}"
                            (tw, th), _ = cv2.getTextSize(cluster_text, font, font_scale + 0.1, font_thickness + 1)
                            x_pos = resize_dim[0] - tw - 5
                            y_pos = resize_dim[1] - 5
                            cv2.putText(labeled_img, cluster_text, (x_pos, y_pos), font,
                                        font_scale + 0.1, (255, 255, 0), font_thickness + 1, cv2.LINE_AA)

                            # Highlight with red border if mismatch
                            if stained_id != virtual_id:
                                cv2.rectangle(labeled_img, (0, 0), (resize_dim[0] - 1, resize_dim[1] - 1),
                                              (255, 0, 0), 3)
                            else:
                                cv2.rectangle(labeled_img, (0, 0), (resize_dim[0] - 1, resize_dim[1] - 1),
                                              (0, 255, 0), 3)

                    # Add padding
                    padded_img = cv2.copyMakeBorder(labeled_img, padding, padding, padding, padding,
                                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    labeled_images.append(padded_img)

                # Create and save grids
                for g in range(num_grids):
                    start_idx = g * rows * cols
                    end_idx = min(start_idx + rows * cols, len(labeled_images))
                    grid_images = labeled_images[start_idx:end_idx]

                    # If we don't have enough images for this grid, skip
                    if len(grid_images) < rows * cols:
                        if cluster_id is not None:
                            print(f"Not enough images for grid {g + 1}. Skipping.")
                        break

                    # Create grid
                    row_imgs = []
                    for r in range(rows):
                        start = r * cols
                        end = (r + 1) * cols
                        row = np.hstack(grid_images[start:end])
                        row_imgs.append(row)
                    grid = np.vstack(row_imgs)

                    # Save to specific cluster folder
                    save_path = f"{out_dir}/cluster_{cl_id}_grid_{g + 1}_{stain_type}.png" if k is None else f"{out_dir}/{k}/cluster_{cl_id}_{stain_type}_grid_{g + 1}_{stain_type}.png"
                    cv2.imwrite(save_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                    print(f"Grid {g + 1} saved to {save_path}")


                # Clean temp directory
                shutil.rmtree(temp_dir)


            print("\nAll cluster grids generated!")

    def visualize_boxplot_per_cluster_combined(self, k: int, output_dir: str, top_n=None):
        """
        For a given model k, generate a single box plot figure showing per-cluster
        distance distributions, using cluster ID on x-axis. Annotates the number of
        outliers above each box. Highlights top N clusters with highest outlier count in red.
        """
        os.makedirs(output_dir, exist_ok=True)
        if top_n is None:
            top_n = int(0.2 * k)

        if k not in self.distance_df:
            print(f"Computing distances for model k={k} before visualization...")
            self.load_cluster_model(k)
            self.compute_distances_for_model(k)

        df = self.distance_df[k]

        outlier_info = []
        for cluster_id, group in df.groupby("cluster"):
            distances = group["distance"]
            q1 = np.percentile(distances, 25)
            q3 = np.percentile(distances, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = ((distances < lower_bound) | (distances > upper_bound)).sum()
            outlier_info.append((cluster_id, distances, outlier_count))

        # Sort by outlier count to identify top N
        sorted_outliers = sorted(outlier_info, key=lambda x: x[2], reverse=True)
        top_clusters = set(x[0] for x in sorted_outliers[:top_n])

        fig = go.Figure()
        annotations = []

        for cluster_id, distances, outlier_count in outlier_info:
            color = "red" if cluster_id in top_clusters else self.color_map["stained"]

            fig.add_trace(go.Box(
                y=distances,
                name=str(cluster_id),
                boxpoints="outliers",
                marker_color=color,
                showlegend=False
            ))

            # Add annotation above box
            max_y = distances.max()
            annotations.append(dict(
                x=str(cluster_id),
                y=max_y + 0.01,
                text=str(outlier_count),
                showarrow=False,
                font=dict(size=10, color="black")
            ))

        fig.update_layout(
            title=f"Per-Cluster Distance Distributions with Outlier Counts | k={k}",
            xaxis_title="Cluster ID",
            yaxis_title="Vector Distance (stained ↔ virtual)",
            xaxis_type="category",
            height=600,
            width=1400,
            annotations=annotations
        )

        out_path = os.path.join(output_dir, f"boxplots_combined_k{k}_top{top_n}_highlighted.html")
        fig.write_html(out_path)
        print(f"Saved boxplots to {out_path}")


if __name__ == "__main__":
    output_dir = f"{BASE_DIR}/plots_resize"
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary_stats.csv")
    ranking_path = os.path.join(output_dir, "top_cluster_ranking.csv")
    k_list = [28, 31, 41, 56, 112, 140]

    evaluator = ClusterDistanceEvaluator(
        cluster_dir=f"/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_resize/data",
        chem_path=f"{BASE_DIR}/stained_resize.parquet",
        virtual_path=f"{BASE_DIR}/infered_resize.parquet",
        out_dir=output_dir
    )

    # output_dir = f"{BASE_DIR}/plots_center_crop"
    # os.makedirs(output_dir, exist_ok=True)
    # summary_path = os.path.join(output_dir, "summary_stats.csv")
    # ranking_path = os.path.join(output_dir, "top_cluster_ranking.csv")
    # k_list = [28, 31, 41, 56, 112, 140]
    #
    # evaluator = ClusterDistanceEvaluator(
    #     cluster_dir=f"/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_resize/data",
    #     chem_path=f"{BASE_DIR}/stained_center_crop.parquet",
    #     virtual_path=f"{BASE_DIR}/infered_resize.parquet",
    #     out_dir=output_dir
    # )

    evaluator.load_embeddings()
    evaluator.align_pairs()
    evaluator.visualize_slide_key_stats(output_dir)
    for k in k_list:
        evaluator.visualize_boxplot_per_cluster_combined(k=k, output_dir=output_dir)
    # Will load from CSVs if they exist, else compute
    # evaluator.evaluate_models(k_list, output_dir=output_dir)
    #
    # summary_df = evaluator.summarize_results(summary_path=summary_path)
    # evaluator.summarize_cluster_ranking(top_n=20, sort_by="std_ratio", ranking_path=ranking_path)
    #
    # evaluator.visualize_all(output_dir)
    #
    # evaluator.load_cluster_model(112)
    # evaluator.get_cluster_pair_dataframe(k=112)
    # evaluator.visualize_cluster_samples(rows=5, cols=5,out_dir=output_dir,
    #                                     show_cluster_mismatch=True)

