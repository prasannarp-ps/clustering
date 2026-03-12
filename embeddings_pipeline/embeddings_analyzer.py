import argparse
import glob
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from elasticsearch import ConnectionTimeout
from pprint import pprint

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import tifffile as tiff
from elasticsearch import helpers
from kneed import KneeLocator
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.utils import shuffle
from tqdm import tqdm

from embeddings.data_handling.get_data import S3Downloader
from embeddings.embeddings_pipeline.utils import EmbeddingIO, ElasticEmbeddingFetcher, ElasticsearchFilterBuilder, retry_with_backoff, generate_unique_id


class ElasticEmbeddingAnalyzer:
    def __init__(self, fetcher: ElasticEmbeddingFetcher, out_dir, keyword_field=False, batch_size=10000):
        self.batch_size = batch_size
        self.keyword_field = keyword_field
        self.fetcher = fetcher
        self.io = EmbeddingIO(out_dir)

    def visualize_clusters(self, df, vectors,
                           sample_size=100000, name="embedding_clusters",
                           figsize=(18, 14), cmap='rainbow', num_colors=10,
                           pca_result=None, explained_variance=None,
                           skip_internal_sampling=False):

        colors = px.colors.sample_colorscale("Turbo", [i / (num_colors - 1) for i in range(num_colors)])

        df_query = df[df["slide_key"] == "__query__"]
        df_normal = df[df["slide_key"] != "__query__"]

        if not skip_internal_sampling:
            if len(df_normal) > sample_size:
                df_normal = df_normal.sample(sample_size, random_state=42)
            df = pd.concat([df_normal, df_query]).reset_index(drop=True)
            vectors = vectors[df.index]
        else:
            vectors = vectors[df.index]

        # Now create query_mask after final df is assembled
        query_mask = df["slide_key"] == "__query__"

        X = np.vstack(vectors.values)
        cluster_labels = df["cluster"].values
        filenames = df["slide_key"].values

        if pca_result is None or explained_variance is None:
            print("Performing PCA...")
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(X)
            explained_variance = pca.explained_variance_ratio_
        else:
            print("Using precomputed PCA...")

        print("Creating 2D scatter plots...")
        fig = plt.figure(figsize=figsize)

        ax1 = fig.add_subplot(1, 3, 1)
        scatter = ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap=cmap, alpha=0.8)
        ax1.set_title(f'{name.capitalize()} PCA - 1st, 2nd')
        ax1.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
        ax1.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.scatter(pca_result[:, 1], pca_result[:, 2], c=cluster_labels, cmap=cmap, alpha=0.8)
        ax2.set_title(f'{name.capitalize()} PCA - 2nd, 3rd')
        ax2.set_xlabel(f'PC2 ({explained_variance[1]:.2%})')
        ax2.set_ylabel(f'PC3 ({explained_variance[2]:.2%})')

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(pca_result[:, 0], pca_result[:, 2], c=cluster_labels, cmap=cmap, alpha=0.8)
        ax3.set_title(f'{name.capitalize()} PCA - 1st, 3rd')
        ax3.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
        ax3.set_ylabel(f'PC3 ({explained_variance[2]:.2%})')

        plt.colorbar(scatter, ax=ax3, label="Cluster")
        plt.tight_layout()
        plt.savefig(f"{self.io.results_dir}/{name.replace(' ', '_')}_2d")
        plt.close()

        print("Creating interactive 3D Plotly scatter plot...")
        fig_3d = go.Figure()
        unique_clusters = np.unique(cluster_labels[~query_mask.values])
        for i, cluster_id in enumerate(unique_clusters):
            color = colors[i % len(colors)]
            cluster_df = df[(df["cluster"] == cluster_id) & (~query_mask)]
            fig_3d.add_trace(go.Scatter3d(
                x=pca_result[cluster_df.index, 0],
                y=pca_result[cluster_df.index, 1],
                z=pca_result[cluster_df.index, 2],
                mode='markers',
                name=f"Cluster {cluster_id}",
                marker=dict(size=4, opacity=0.8, color=color),
                text=[
                    f"Slide Key: {row['tissue_type']}/{row.get('block_id', '')}/{row.get('slice_id', '')}/{row.get('scan_date', '')}/{row.get('box_id', '')}<br>"
                    f"Tile name: {row.get('filename', '')}<br>"
                    f"Modality: {row.get('modality', '')}<br>"
                    f"Preprocessing: {row.get('preprocessing', '')}<br>"
                    for _, row in cluster_df.iterrows()
                ],
                hovertemplate="<b>%{text}</b><br>Cluster: " + str(cluster_id) + "<br>"
                                                                                "PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>"
            ))

        if not df_query.empty:
            idx = df_query.index[0]
            row = df_query.loc[idx]
            fig_3d.add_trace(go.Scatter3d(
                x=[pca_result[idx, 0]],
                y=[pca_result[idx, 1]],
                z=[pca_result[idx, 2]],
                mode='markers+text',
                name='Query Sample',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                text=[row.get("filename", "")],
                hoverinfo='text'
            ))

        fig_3d.update_layout(
            title=f'{name.capitalize()} PCA - 3D Visualization - {len(df)} samples',
            scene=dict(
                xaxis_title=f'PC1 ({explained_variance[0]:.2%})',
                yaxis_title=f'PC2 ({explained_variance[1]:.2%})',
                zaxis_title=f'PC3 ({explained_variance[2]:.2%})',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            )
        )
        fig_3d.update_layout(showlegend=True)
        fig_3d.write_html(f"{self.io.results_dir}/{name.replace(' ', '_')}_3d.html", include_plotlyjs='cdn')

    def visualize_similar_images_grid(self, target_doc, matches, N=10, name=""):

        assert isinstance(target_doc, dict) and "slide_key" in target_doc, "Target document must include metadata."
        assert len(matches) >= N, "Not enough matches returned from kNN search."

        def build_s3_path(doc):
            return (
                f"s3://gi-registration/slide-registration-production/tiles/512/"
                f"{doc['tissue_type']}/{doc['block_id']}/{doc['slice_id']}/"
                f"{doc['scan_date']}/{doc['box_id']}/{doc['filename']}-stained-sr.tiff"
            )
        s3_paths = [build_s3_path(target_doc)]  # query image first
        output_info = [{
            "score": "1.000",
            "modality": target_doc.get("modality"),
            "preprocessing": target_doc.get("preprocessing"),
            "tissue_type": target_doc.get("tissue_type"),
            "slide_key": target_doc.get("slide_key"),
            "title": "Query Image"
        }]

        for match in matches[:N+1]:
            src = match["_source"]
            s3_paths.append(build_s3_path(src))
            output_info.append({
                "score": f"{match.get('_score', 0):.4f}",
                "modality": src.get("modality"),
                "preprocessing": src.get("preprocessing"),
                "tissue_type": src.get("tissue_type"),
                "slide_key": src.get("slide_key"),
            })

        # Download images from S3
        temp_dir = tempfile.mkdtemp()
        downloader = S3Downloader(s3_paths=s3_paths, dl_dir=temp_dir, num_workers=8)
        file_list = downloader.build_s3_paths()
        downloader.run(file_list)

        tiff_files = sorted(Path(temp_dir).rglob("*.tiff"))
        loaded_images = [tiff.imread(str(p)) for p in tiff_files]
        fig = make_subplots(
            rows=5, cols=3,
            specs=[[{"rowspan": 5}, {}, {}]] + [[None, {}, {}] for _ in range(4)],
            subplot_titles=["Query Image"] + [f"{info['score']}" for info in output_info[1:]],
            column_widths=[0.4, 0.3, 0.3],
            vertical_spacing=0.03,
            horizontal_spacing=0.02
        )

        for i, (img, info) in enumerate(zip(loaded_images, output_info)):
            row, col = (1, 1) if i == 0 else ((i - 1) % 5 + 1, 2 if i <= 5 else 3)
            hover_text = "<br>".join([f"{k}: {v}" for k, v in info.items()])
            if img.ndim == 2:
                fig.add_trace(go.Heatmap(
                    z=img,
                    colorscale="gray",
                    showscale=False,
                    hoverinfo="text",
                    hovertext=hover_text
                ), row=row, col=col)
            else:
                fig.add_trace(go.Image(z=img), row=row, col=col)
                h, w = img.shape[:2]

                fig.add_trace(go.Scatter(
                    x=[w // 2],
                    y=[h // 2],
                    mode="markers",
                    marker=dict(size=0.1, opacity=0),
                    hoverinfo="text",
                    hovertext=hover_text,
                    showlegend=False
                ), row=row, col=col)

        fig.update_layout(
            title="Query Image vs. Top Similar Tiles",
            height=1000,
            width=1200,
            margin=dict(l=20, r=20, t=60, b=20),
            hoverlabel=dict(bgcolor="white", font_size=12)
        )

        for i in range(1, 16):
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)

        out_name = "similar_images_grid" if not name else "similar_images_grid_" + name
        self.io.save_html(fig, out_name)
        shutil.rmtree(temp_dir)

    def visualize_grid_search(self):
        base_dir = Path(self.io.out_dir)
        metrics_files = list(base_dir.rglob("results/cluster_metrics.csv"))

        # Load and annotate all metric files
        all_metrics = []
        for file in metrics_files:
            run_name = file.parts[-3]
            df = pd.read_csv(file)
            df["run_name"] = run_name
            df["preprocessing"] = run_name.split("_")[0]
            df["apply_pca"] = "True" if "_pca1_" in run_name else "False"
            df["min_samples"] = int(re.search(r"_min(\d+)", run_name).group(1))
            all_metrics.append(df)

        # Combine all data
        combined_df = pd.concat(all_metrics, ignore_index=True)
        self.io.save_csv(combined_df, "combined_metrics", subfolder="results/grid_search", index=False)
        metrics_to_plot = ["inertia", "calinski_harabasz", "davies_bouldin"]

        for metric in metrics_to_plot:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("PCA: OFF (pca0)", "PCA: ON (pca1)"),
                shared_yaxes=False,
                horizontal_spacing=0.1
            )

            for i, pca_setting in enumerate(["False", "True"], start=1):
                df_pca = combined_df[combined_df["apply_pca"] == pca_setting]

                for run_name, group in df_pca.groupby("run_name"):
                    x = group["k"]
                    y = group[metric]

                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode="lines+markers",
                        name=run_name,
                        legendgroup=run_name,
                        showlegend=(i == 1),
                        text=[f"{run_name}<br>k={k}<br>{metric}={val:.2f}" for k, val in zip(x, y)],
                        hoverinfo="text"
                    ), row=1, col=i)

                    # Best-k marker
                    best_k = None
                    if not y.isna().all() and not y.empty:
                        try:
                            if metric == "inertia":
                                kl = KneeLocator(x, y, curve="convex", direction="decreasing")
                                best_k = kl.elbow
                            elif metric == "davies_bouldin":
                                best_k = x.loc[y.idxmin()]
                            else:
                                best_k = x.loc[y.idxmax()]
                        except Exception as e:
                            print(f"Skipping best-k for {run_name} on {metric}: {e}")
                            best_k = None

                    if best_k is not None and best_k in group["k"].values:
                        best_y = group[group["k"] == best_k][metric].values[0]
                        fig.add_trace(go.Scatter(
                            x=[best_k],
                            y=[best_y],
                            mode="markers+text",
                            text=[f"Best k = {best_k}"],
                            textposition="top center",
                            marker=dict(size=10, symbol="star", color="red"),
                            name=f"Best k ({run_name})",
                            legendgroup=run_name,
                            showlegend=False
                        ), row=1, col=i)

            fig.update_layout(
                title=f"{metric.capitalize()} vs Number of Clusters (k)",
                height=600,
                width=1200,
                xaxis_title="Number of Clusters (k)",
                yaxis_title=metric.capitalize(),
                legend_title="Run Name"
            )

            self.io.save_html(fig, f"grid_search/{metric}_grid_search_split_by_pca", out_dir=self.io.results_dir)

    def visualize_cluster_samples(self, result_file_name="clustering_results.parquet",
                                  rows=2, cols=5, resize_dim=(256, 256),
                                  text_field="slide_key", font_scale=0.4, font_thickness=1, padding=2,
                                  cluster_id=None, num_grids=1, k=None, local=False):

        # Load clustering results
        result_path = f"{self.io.data_dir}/{result_file_name}"
        if not os.path.exists(result_path):
            print(f"Result file {result_path} not found.")
            return

        print(f"Loading clustering results from {result_path}...")
        df = pd.read_parquet(result_path)

        if "cluster" not in df.columns:
            print("'cluster' column missing.")
            return

        # Determine which clusters to process
        if cluster_id is not None:
            clusters_to_process = [cluster_id]
            images_per_cluster = rows * cols * num_grids
            out_dir = f"{self.io.results_dir}/cluster_{cluster_id}_grids"
            os.makedirs(out_dir, exist_ok=True)
        else:
            clusters_to_process = df["cluster"].unique()
            images_per_cluster = rows * cols
            num_grids = 1
            out_dir = f"{self.io.results_dir}/cluster_grids" if k is None else f"{self.io.results_dir}/cluster_grids/{k}"
            os.makedirs(out_dir, exist_ok=True)

        print(f"Found {len(clusters_to_process)} clusters to process. Generating grids...")

        def build_s3_path(doc, local=False):
            if not local:
                return (
                    # f"s3://gi-registration/slide-registration-production/tiles/512/"
                    # f"{doc['tissue_type']}/{doc['block_id']}/{doc['slice_id']}/"
                    # f"{doc['scan_date']}/{doc['box_id']}/{doc['filename']}-stained-sr.tiff"
                    f"s3://gi-registration/slide-registration/tiles/nanozoomers360md/40x/fixed/he/512"
                    f"{doc['tissue_type']}/{doc['block_id']}/{doc['slice_id']}/"
                    f"{doc['scan_date']}/{doc['box_id']}/{doc['filename']}-stained.tiff"
                )
            else:
                # Local path
                return f"s3://gi-registration/slide-registration/tiles/nanozoomers360md/40x/fixed/he/512/{doc}-stained.tiff"

        for cl_id in sorted(clusters_to_process):
            if os.path.exists(f"{out_dir}/cluster_{cl_id}_grid.png"):
                print(f"\nSkipping existing grid cluster {cl_id}...")
                continue
            else:
                print(f"\nProcessing cluster {cl_id}...")

            # Get samples for this cluster
            cluster_samples = df[df["cluster"] == cl_id]
            total_samples = images_per_cluster

            if len(cluster_samples) < total_samples:
                print(f"Cluster {cl_id} has only {len(cluster_samples)} samples. Using all available.")
                samples = cluster_samples
                total_samples = len(cluster_samples)
            else:
                samples = cluster_samples.sample(total_samples, random_state=42)

            if not local:
                unique_ids = samples["unique_id"].tolist()
                # Fetch documents for these IDs
                docs = []
                for i in range(0, len(unique_ids), 1000):
                    sub_batch = unique_ids[i:i + 1000]
                    res = self.fetcher.mget(batch=sub_batch)
                    docs.extend(doc["_source"] for doc in res["docs"] if doc.get("found"))
            else:
                docs = samples["slide_key"].apply(lambda x: f"{x.replace('_', '/').replace('tile/', 'tile_')}").tolist()

            # Build S3 paths
            s3_paths = [build_s3_path(doc, local=local) for doc in docs]

            # Download images
            temp_dir = tempfile.mkdtemp()
            downloader = S3Downloader(s3_paths=s3_paths, dl_dir=temp_dir, num_workers=8)
            file_list = downloader.build_s3_paths_v2()
            downloader.run(file_list)

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
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                loaded_images.append(img)

                # Get label
                if idx < len(docs):
                    if not local:
                        labels.append(str(docs[idx].get(text_field, "N/A")))
                    else:
                        labels.append(str(docs[idx]))
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

                # Add text to image
                y0 = 20  # Starting y position
                dy = int(20 * font_scale) + 5  # Line spacing

                for idx, line in enumerate(lines[:3]):  # Limit to 3 lines max
                    y = y0 + idx * dy
                    cv2.putText(labeled_img, line, (5, y), font, font_scale, (255, 255, 255), font_thickness,
                                cv2.LINE_AA)

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

                # Save the grid
                if cluster_id is not None:
                    # Save to specific cluster folder
                    save_path = f"{out_dir}/cluster_{cl_id}_grid_{g + 1}.png" if k is None else f"{out_dir}/{k}/cluster_{cl_id}_grid_{g + 1}.png"
                    cv2.imwrite(save_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                    print(f"Grid {g + 1} saved to {save_path}")
                else:
                    # Save to general output using the IO interface
                    self.io.save_grid_image(grid, cl_id, k=k)

            # Clean temp directory
            shutil.rmtree(temp_dir)

        print("\nAll cluster grids generated!")

    def get_stratified_cluster_dataset(self, filter_query, all_strata, max_docs=5_000_000, stratify_field="tissue_type",
                                             min_per_class=1000, max_samples=500_000):

        seen_ids = set()
        total = 0
        max_per_class = min_per_class
        filled_classes = set()
        collected = defaultdict(list)

        for df_batch in self.fetcher.fetch_filtered_vectors_generator(filter_query, max_docs=max_docs):
            df_batch = df_batch[~df_batch["unique_id"].isin(seen_ids)]
            seen_ids.update(df_batch["unique_id"])

            for _, group in df_batch.groupby(stratify_field):
                key = group[stratify_field].iloc[0]
                current = len(collected[key])
                if current >= max_per_class:
                    continue
                needed = min(max_per_class - current, len(group))
                if needed == 0:
                    continue
                sample = group.sample(n=needed, random_state=42)
                collected[key].extend(sample.to_dict("records"))
                total += len(sample)

                if len(collected[key]) >= max_per_class and key not in filled_classes:
                    filled_classes.add(key)
                    print(f"Class '{key}' reached max_per_class ({max_per_class})")

            class_counts = {k: len(v) for k, v in collected.items()}
            print(f"Total samples: {total} | Classes filled: {len(class_counts)}")
            pprint(dict(sorted(class_counts.items(), key=lambda x: x[0])))

            if total >= max_samples or all_strata.issubset(filled_classes):
                print(f"Got all samples for {all_strata}")
                break
        return collected

    def get_stratified_dataset_quota_streaming(
            self,
            parquet_paths,
            stratify_field="tissue_type",
            per_class=10_000,
            max_samples=200_000,
            id_col="unique_id",
            columns=None,  # None => read all columns in pass 2
            batch_size=200_000,
            random_state=42,
            drop_duplicates_on="unique_id",
            stats=None,  # e.g. ["tissue_type","block_id","slice_id"]
    ):

        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
        from collections import Counter, defaultdict

        rng = np.random.RandomState(random_state)

        # ---------- PASS 1: per-file, per-class counts ----------
        per_file_counts = []  # list[Counter], one per file
        total_per_class = Counter()
        print("Doing first pass to count classes in each file...")
        for path in parquet_paths:
            pf = pq.ParquetFile(path)
            file_counter = Counter()
            for batch in pf.iter_batches(batch_size=batch_size, columns=[stratify_field, "modality"]):
                df = batch.to_pandas()
                df = df[df["modality"] == "stained"]
                vc = df[stratify_field].value_counts(dropna=False)
                for cls, cnt in vc.items():
                    file_counter[cls] += int(cnt)
                    total_per_class[cls] += int(cnt)
            per_file_counts.append(file_counter)

        # classes present
        classes = [c for c in total_per_class.keys() if total_per_class[c] > 0 and not c.isdigit()]
        if not classes:
            print("No classes found.")
            return pd.DataFrame()
        print("Classes found:", classes)
        # Target per class (cap by available; then scale to respect global max_samples)
        raw_targets = {c: int(min(per_class, total_per_class[c])) for c in classes}
        total_target = sum(raw_targets.values())
        if total_target == 0:
            print("No targets to sample.")
            return pd.DataFrame()
        print("Raw targets per class:", raw_targets)
        if total_target > max_samples:
            scale = max_samples / float(total_target)
            floored = {c: int(np.floor(raw_targets[c] * scale)) for c in classes}
            remainders = {c: (raw_targets[c] * scale) - floored[c] for c in classes}
            remaining = max_samples - sum(floored.values())
            for c in sorted(classes, key=lambda x: remainders[x], reverse=True)[:remaining]:
                floored[c] += 1
            targets = floored
        else:
            targets = raw_targets
        print("Building per-class targets:", targets)
        # Build per-file quotas per class using proportional split + remainder distribution
        quotas = [defaultdict(int) for _ in parquet_paths]  # list[dict[class->quota]]
        for c in classes:
            tot = total_per_class[c]
            if tot == 0 or targets[c] == 0:
                continue
            proportional = []
            for i, fc in enumerate(per_file_counts):
                share = fc[c] / tot if tot > 0 else 0.0
                proportional.append((i, share * targets[c]))
            # floors
            floors = {i: int(np.floor(val)) for i, val in proportional}
            assigned = sum(floors.values())
            need = targets[c] - assigned
            # assign remaining to largest remainders
            rema = sorted(((i, proportional[i][1] - floors[i]) for i in range(len(parquet_paths))),
                          key=lambda x: x[1], reverse=True)
            for i, _ in rema[:need]:
                floors[i] += 1
            # write quotas
            for i, q in floors.items():
                if q > 0:
                    quotas[i][c] = q
        print("Doing second pass to sample according to quotas...")
        # ---------- PASS 2: sample according to quotas ----------
        need_any = any(sum(q.values()) > 0 for q in quotas)
        if not need_any:
            print("Nothing to sample after quota computation.")
            return pd.DataFrame()

        # Ensure we read at least needed cols in pass 2
        need_cols = set(columns or [])
        need_cols.update([stratify_field, id_col])
        if stats:
            need_cols.update(stats)
        need_cols = list(need_cols)

        writer = None
        collected = []
        seen_ids = set()

        for file_idx, path in enumerate(parquet_paths):
            file_quota = quotas[file_idx]
            if not file_quota:
                continue  # this file contributes nothing

            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=batch_size):
                df = batch.to_pandas()
                df = df[df["modality"] == "stained"]

                # hygiene
                if stratify_field not in df.columns:
                    continue
                if drop_duplicates_on in df.columns:
                    df = df.drop_duplicates(subset=[drop_duplicates_on])

                # process only classes that still need rows from this file
                pending_classes = {c: q for c, q in file_quota.items() if q > 0}
                if not pending_classes:
                    break

                # filter out IDs already taken
                if id_col in df.columns:
                    df = df[~df[id_col].isin(seen_ids)]

                for c, q in list(pending_classes.items()):
                    if q <= 0:
                        continue
                    grp = df[df[stratify_field] == c]
                    if grp.empty:
                        continue
                    take_n = min(q, len(grp))
                    sample = grp.sample(n=take_n, random_state=random_state, replace=False)

                    collected.append(sample)

                    # bookkeeping
                    if id_col in sample.columns:
                        seen_ids.update(sample[id_col].tolist())
                    file_quota[c] -= take_n
            print("Done with file:", path)

        sampled_df = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()

        if stats and not sampled_df.empty:
            print(f"\nTotal sampled rows: {len(sampled_df)}")
            for col in stats:
                if col in sampled_df.columns:
                    print(f"\nStats for '{col}':")
                    print(sampled_df[col].value_counts(dropna=False))

        return sampled_df

    def evaluate_optimal_clusters_stratified(self, filter_builder: ElasticsearchFilterBuilder,
                                             n_clusters_list=[10, 20, 30], max_docs=5_000_000,
                                             stratify_field="tissue_type", apply_pca=True, pca_components=100,
                                             min_per_class=1000, max_samples=500_000, sub_sample=None,
                                             calculate_silhouette_score=False, calculate_light_scores=True, legacy=False):

        silhouette_results = []
        inertia_values = []
        ch_scores = []
        db_scores = []

        all_strata = set(self.fetcher.get_unique_field_values(stratify_field))
        print(f"Expecting {len(all_strata)} unique {stratify_field} categories: {sorted(all_strata)}\n")
        filter_query = filter_builder.build()
        if os.path.exists(f"{self.io.data_dir}/sample_stratified.parquet"):
            # return
            sample_df = pd.read_parquet(f"{self.io.data_dir}/sample_stratified.parquet")
            if sub_sample is not None:
                sample_df = sample_df.sample(n=min(len(sample_df), 10_000), random_state=42).reset_index(drop=True)
        else:
            if legacy:
                collected = self.get_stratified_cluster_dataset(
                    filter_query=filter_query,
                    all_strata=all_strata,
                    max_docs=max_docs,
                    stratify_field=stratify_field,
                    min_per_class=min_per_class,
                    max_samples=max_samples
                )
                print("Stratification complete")
            else:
                collected = {}
                for tt in all_strata:
                    filter_builder.add_term_filter(stratify_field, tt)
                    filter_query = filter_builder.build()

                    try:
                        result = retry_with_backoff(
                            self.get_stratified_cluster_dataset,
                            filter_query=filter_query,
                            all_strata={tt},
                            max_docs=max_docs,
                            stratify_field=stratify_field,
                            min_per_class=min_per_class,
                            max_samples=max_samples,
                            max_retries=5,
                            initial_wait=5,
                            exceptions=(ConnectionTimeout,)
                        )
                        collected.update(result)
                        class_counts = {k: len(v) for k, v in collected.items()}
                        print(f"Total samples: {sum(class_counts.values())} | Classes filled: {len(class_counts)}")
                        pprint(dict(sorted(class_counts.items(), key=lambda x: x[0])))
                        print("Stratification complete")
                    except Exception as e:
                        print(f"[Error] Skipping '{tt}' due to repeated failure: {e}")
            sample_df = pd.DataFrame([item for sublist in collected.values() for item in sublist])
            sample_df.to_parquet(f"{self.io.data_dir}/sample_stratified.parquet", index=False)

            print(f"Final stratified sample shape: {sample_df.shape}, covering {len(collected)} classes\n")
        print("Converting vectors...")
        #
        # Test on smaller sample
        # sample_df = sample_df.sample(n=min(len(sample_df), 10000), random_state=42).copy()
        # sample_df = sample_df.reset_index(drop=True)
        #

        if os.path.exists(f"{self.io.results_dir}/cluster_metrics.csv"):
            print("Already generated.. skipping")
            return
        vectors = np.vstack(sample_df["vector"].values)

        if apply_pca:
            print("Performing PCA...")
            pca = PCA(n_components=pca_components, random_state=42)
            X = pca.fit_transform(vectors)
            explained_variance = pca.explained_variance_ratio_
        else:
            X = vectors


        for k in n_clusters_list:
            print(f"\nClustering with k={k} using streaming batches...")
            model = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=self.batch_size)

            X_shuffled = shuffle(X, random_state=42)

            for i in tqdm(range(0, X_shuffled.shape[0], self.batch_size), desc=f"Partial fitting k={k}"):
                batch = X_shuffled[i:i + self.batch_size]
                if len(batch) < model.batch_size:
                    break
                model.partial_fit(batch)

            print("Fitting complete.")
            # Predict in batches (instead of model.fit_predict(X))
            print("Predicting in batches...")
            labels = np.empty(X.shape[0], dtype=np.int32)
            for i in tqdm(range(0, X.shape[0], self.batch_size), desc=f"Predicting k={k}"):
                batch = X[i:i + self.batch_size]
                batch_labels = model.predict(batch)
                labels[i:i + len(batch_labels)] = batch_labels

            print("Predicting complete.")
            if calculate_silhouette_score:
                print("Calculating silhouette score...")
                score = silhouette_score(X, labels)
                silhouette_results.append((k, score))
                print(f"k={k} silhouette score: {score:.4f}")

            if calculate_light_scores:
                print("Calculating CHI and DBI...")
                ch = calinski_harabasz_score(X, labels)
                db = davies_bouldin_score(X, labels)
                ch_scores.append(ch)
                db_scores.append(db)
                inertia_values.append(model.inertia_)
                print(f"k={k} CHI: {ch:.2f}, DBI: {db:.4f}, Inertia: {model.inertia_:.2f}")

            if len(os.listdir(self.io.results_dir)) != 190:
                sample_df["cluster"] = labels
                print("Visualizing clusters...")
                sampled_df = sample_df.sample(n=min(len(sample_df), 200_000), random_state=42).reset_index(drop=True)
                self.visualize_clusters(
                    df=sampled_df,
                    vectors=sampled_df["vector"],
                    name=f"stratified_k{k}",
                    pca_result=X[sampled_df.index],
                    num_colors=k,
                    sample_size=200_000,
                    explained_variance=explained_variance if apply_pca else None,
                    skip_internal_sampling=True if apply_pca else False,
                )

        if calculate_silhouette_score:
            # Save silhouette scores
            summary_df = pd.DataFrame(silhouette_results, columns=["k", "silhouette_score"])
            self.io.save_csv(summary_df, "silhouette_report", subfolder="results", index=False)
            print("\nSilhouette Report:")
            pprint(summary_df)
            print("\nGenerating interactive silhouette plot...")
            try:
                best_row = summary_df.loc[summary_df["silhouette_score"].idxmax()]
                fig = px.line(summary_df, x="k", y="silhouette_score", markers=True,
                              title="Silhouette Score vs Number of Clusters (k)",
                              labels={"k": "Number of Clusters", "silhouette_score": "Silhouette Score"})
                fig.add_scatter(
                    x=[best_row["k"]],
                    y=[best_row["silhouette_score"]],
                    mode="markers+text",
                    text=[f"Best k = {int(best_row['k'])}"],
                    textposition="top center",
                    marker=dict(size=10, color="red", symbol="star"),
                    name="Best k"
                )
                self.io.save_html(fig, "silhouette_plot.html")
            except Exception as e:
                print(f"Failed to generate silhouette plot: {e}")

        if calculate_light_scores:
            df_metrics = pd.DataFrame({
                "k": n_clusters_list,
                "inertia": inertia_values,
                "calinski_harabasz": ch_scores,
                "davies_bouldin": db_scores
            })
            if calculate_silhouette_score:
                df_metrics["silhouette_score"] = [v for _, v in silhouette_results]
            kl = KneeLocator(df_metrics["k"], df_metrics["inertia"], curve="convex", direction="decreasing")
            optimal_k = kl.elbow
            best_k = {
                "inertia": df_metrics[df_metrics["k"] == optimal_k].index[0],
                "calinski_harabasz": df_metrics["calinski_harabasz"].idxmax(),
                "davies_bouldin": df_metrics["davies_bouldin"].idxmin()
            }
            if calculate_silhouette_score:
                best_k["silhouette_score"] = df_metrics["silhouette_score"].idxmax()

            df_melted = df_metrics.melt(id_vars=["k"], var_name="metric", value_name="score")
            direction = {
                "inertia": "elbow location",
                "calinski_harabasz": "higher is better",
                "davies_bouldin": "lower is better",
                "silhouette_score": "higher is better"
            }
            df_melted["direction"] = df_melted["metric"].map(direction)

            fig = px.line(df_melted, x="k", y="score", color="metric", markers=True,
                          title="Cluster Evaluation Metrics",
                          labels={"k": "Number of Clusters", "score": "Metric Value", "metric": "Metric"})

            for metric, idx in best_k.items():
                best_row = df_metrics.iloc[idx]
                fig.add_scatter(
                    x=[best_row["k"]],
                    y=[best_row[metric]],
                    mode="markers+text",
                    marker=dict(size=10, symbol="star", color="red"),
                    text=[f"Best k = {int(best_row['k'])} ({direction[metric]})"],
                    textposition="top center",
                    name=f"Best {metric}"
                )
            self.io.save_html(fig, "cluster_metrics_plot.html", out_dir=self.io.results_dir)
        self.io.save_csv(df_metrics, "cluster_metrics", subfolder="results", index=False, out_dir=self.io.results_dir)
        return df_metrics if calculate_light_scores else pd.DataFrame()


    def find_optimal_k(self, df, run_name=None):
        if run_name:
            df = df[df['run_name'] == run_name]
        results = {}
        # 1. Elbow method for inertia
        try:
            k_values = df['k'].values
            inertia_values = df['inertia'].values
            # Using kneed package to find the elbow point
            kneedle = KneeLocator(k_values, inertia_values, curve='convex', direction='decreasing')
            results['inertia_elbow'] = kneedle.elbow
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, inertia_values, 'bo-')
            plt.xlabel('Number of clusters (K)')
            plt.ylabel('Inertia')
            plt.title(f'Elbow Method for Inertia{" - " + run_name if run_name else ""}')
            if kneedle.elbow:
                plt.axvline(x=kneedle.elbow, color='r', linestyle='--',
                            label=f'Elbow point: K={kneedle.elbow}')
                plt.legend()
            self.io.save_plot(plt, f'grid_search/inertia_plot{"_" + run_name if run_name else ""}', override_default=True)
        except Exception as e:
            print(f"Error finding elbow for inertia: {e}")
            results['inertia_elbow'] = None
        # 2. Calinski-Harabasz index
        ch_values = df['calinski_harabasz'].values
        results['ch_max'] = df.loc[df['calinski_harabasz'].idxmax(), 'k']
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, ch_values, 'go-')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Calinski-Harabasz Index')
        plt.title(f'Calinski-Harabasz Index{" - " + run_name if run_name else ""}')
        plt.axvline(x=results['ch_max'], color='r', linestyle='--',
                    label=f'Max value: K={results["ch_max"]}')
        plt.legend()
        self.io.save_plot(plt, f'grid_search/ch_plot{"_" + run_name if run_name else ""}', override_default=True)
        # 3. Davies-Bouldin index
        db_values = df['davies_bouldin'].values
        results['db_min'] = df.loc[df['davies_bouldin'].idxmin(), 'k']
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, db_values, 'ro-')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Davies-Bouldin Index')
        plt.title(f'Davies-Bouldin Index{" - " + run_name if run_name else ""}')
        plt.axvline(x=results['db_min'], color='g', linestyle='--',
                    label=f'Min value: K={results["db_min"]}')
        plt.legend()
        self.io.save_plot(plt, f'grid_search/db_plot{"_" + run_name if run_name else ""}', override_default=True)
        return results

    def results_summary(self):
        data = self.io.load_csv("combined_metrics", subfolder="results/grid_search")
        run_names = data['run_name'].unique()

        run_summaries = []

        # Analyze each run separately
        for run_name in run_names:
            print(f"\n--- Analysis for run: {run_name} ---")
            results = self.find_optimal_k(data, run_name)
            print(f"Optimal K based on Elbow Method (Inertia): {results['inertia_elbow']}")
            print(f"Optimal K based on Calinski-Harabasz Index (max): {results['ch_max']}")
            print(f"Optimal K based on Davies-Bouldin Index (min): {results['db_min']}")

            recommendations = []
            if results['inertia_elbow']:
                recommendations.append(results['inertia_elbow'])
            recommendations.append(results['ch_max'])
            recommendations.append(results['db_min'])

            if len(recommendations) > 0:
                counter = Counter(recommendations)
                most_common = counter.most_common(1)[0][0]
                median = int(np.median(recommendations))
                print(f"Most frequently recommended K: {most_common}")
                print(f"Median of recommended K values: {median}")
                if most_common == median:
                    print(f"Final recommendation: K = {most_common}")
                else:
                    print(f"Final recommendations: K = {most_common} (most common) or K = {median} (median)")
            else:
                print("No clear recommendation available.")

            run_summaries.append({
                "run_name": run_name,
                "elbow": results['inertia_elbow'],
                "ch": results['ch_max'],
                "db": results['db_min'],
                "elbow_db_diff": abs((results['inertia_elbow'] or 0) - (results['db_min'] or 0))  # Handle possible None
            })

        print("\n--- Overall Analysis ---")
        combined_results = self.find_optimal_k(data)
        print(f"Overall optimal K based on Elbow Method (Inertia): {combined_results['inertia_elbow']}")
        print(f"Overall optimal K based on Calinski-Harabasz Index (max): {combined_results['ch_max']}")
        print(f"Overall optimal K based on Davies-Bouldin Index (min): {combined_results['db_min']}")

        df_summary = pd.DataFrame(run_summaries)

        # Rank runs by smallest elbow-db distance
        ranked = df_summary.sort_values(by="elbow_db_diff", ascending=True)

        print("\n--- Runs Ranked by Strongest Elbow-DB Consensus ---")
        print(ranked.to_string(index=False))

        # --- PLOT distribution ---
        plt.figure(figsize=(12, 6))
        sns.histplot(df_summary["elbow"], bins=10, color="blue", label="Elbow (Inertia)", kde=True)
        sns.histplot(df_summary["db"], bins=10, color="green", label="Davies-Bouldin (DB)", kde=True, alpha=0.6)
        plt.axvline(x=28, color="red", linestyle="--", label="Suggested K = 28")
        plt.xlabel("Recommended Number of Clusters (K)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Optimal K across Runs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        self.io.save_plot(plt, "grid_search/optimal_k_distribution", override_default=True)
        plt.close()

    def load_ids(self, output_path="unique_ids.parquet"):
        print(f"Loading IDs from {self.io.data_dir}/{output_path}...")
        return pd.read_parquet(f"{self.io.data_dir}/{output_path}").squeeze().tolist()

    def batch_generator(self, id_list):
        np.random.shuffle(id_list)
        for i in range(0, len(id_list), self.batch_size):
            yield id_list[i:i + self.batch_size]

    def partial_fit_clustering(self, k):
        done = 0
        ids = self.load_ids()
        mbk = MiniBatchKMeans(n_clusters=k, batch_size=self.batch_size, random_state=42)

        for batch_ids in self.batch_generator(ids):
            batch_idds, batch_vectors = self.fetcher.fetch_vectors_by_ids(batch_ids)
            if batch_vectors:
                X = np.vstack(batch_vectors)
                mbk.partial_fit(X)
            done += len(batch_idds)
            print(f"Done: {done}/{len(ids)}")

        self.io.save_model(mbk, f"kmeans_model_k{k}")

    def partial_fit_clustering_from_df(self, k: int, df: pd.DataFrame, suffix=None):
        out_name = f"kmeans_model_k{k}"
        mbk = MiniBatchKMeans(n_clusters=k, batch_size=self.batch_size, random_state=42)

        total = len(df)
        done = 0

        # Shuffle to avoid bias
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = df_shuffled.iloc[start:end]
            vectors = batch["vector"].tolist()
            if vectors:
                X = np.vstack(vectors)
                mbk.partial_fit(X)
            done += len(vectors)
            print(f"Done: {done}/{total}")
        if suffix:
            out_name = f"kmeans_model_k{k}_{suffix}"
        self.io.save_model(mbk, out_name)

    def predict_from_model(self, model_path, result_file_name="clustering_results.parquet"):

        ids = self.load_ids()
        model = joblib.load(model_path)

        result_path = f"{self.io.data_dir}/{result_file_name}"

        # If result file already exists, load existing results
        if os.path.exists(result_path):
            print(f"🔄 Found existing results at {result_path}. Resuming...")
            existing_df = pd.read_parquet(result_path)
            predicted_ids = set(existing_df["unique_id"].tolist())
        else:
            print(f"🆕 No previous results found. Starting fresh...")
            existing_df = pd.DataFrame(columns=["unique_id", "cluster"])
            predicted_ids = set()

        remaining_ids = [uid for uid in ids if uid not in predicted_ids]
        print(f"Remaining samples to predict: {len(remaining_ids)} / {len(ids)} total.")

        batch_counter = 0

        for batch_ids in self.batch_generator(remaining_ids):
            batch_ids_fetched, batch_vectors = self.fetcher.fetch_vectors_by_ids(batch_ids)
            if batch_vectors:
                X = np.vstack(batch_vectors)
                labels = model.predict(X)

                batch_df = pd.DataFrame({
                    "unique_id": batch_ids_fetched,
                    "cluster": labels
                })

                # Append new batch to existing results
                existing_df = pd.concat([existing_df, batch_df], ignore_index=True)

                # Save after each batch
                existing_df.to_parquet(result_path, index=False)

                batch_counter += 1
                print(f"Batch {batch_counter}: Saved {len(existing_df)} total predictions so far.")

        print(f"\n🎉 Done predicting all samples. Final saved file: {result_path}")

        return existing_df


    def predict_from_model_from_local_file(self, model_path, parquet_paths,
                                           result_file_name="clustering_results.parquet",
                                           batch_size=10000, data_type="stained"):
        # Load full dataframe
        for parquet_path in parquet_paths:
            print(f"Running on: {parquet_path}")
            df = pd.read_parquet(parquet_path)
            df = df[df["modality"] == "stained"]
            df["unique_id"] = df.apply(generate_unique_id, axis=1)
            all_ids = df["unique_id"].tolist()

            model = joblib.load(model_path)
            result_path = os.path.join(self.io.data_dir, result_file_name)

            # Load existing predictions if available
            if os.path.exists(result_path):
                print(f"🔄 Found existing results at {result_path}. Resuming...")
                existing_df = pd.read_parquet(result_path)
                predicted_ids = set(existing_df["unique_id"].tolist())
            else:
                print(f"🆕 No previous results found. Starting fresh...")
                existing_df = pd.DataFrame(columns=["unique_id", "cluster"])
                predicted_ids = set()

            # Filter out already predicted IDs
            remaining_df = df[~df["unique_id"].isin(predicted_ids)]
            print(f"Remaining samples to predict: {len(remaining_df)} / {len(all_ids)} total.")

            # Predict in batches
            batch_counter = 0
            for start in range(0, len(remaining_df), batch_size):
                batch_df = remaining_df.iloc[start:start + batch_size]
                vectors = np.vstack(batch_df["vector"].to_numpy())
                labels = model.predict(vectors)

                batch_result = pd.DataFrame({
                    "unique_id": batch_df["unique_id"].tolist(),
                    "slide_key": batch_df["slide_key"].tolist(),
                    "cluster": labels
                })

                existing_df = pd.concat([existing_df, batch_result], ignore_index=True)
                existing_df.to_parquet(result_path, index=False)

                batch_counter += 1
                print(f"Batch {batch_counter}: Saved {len(existing_df)} total predictions so far.")
            del df
            print(f"\n🎉 Done predicting all samples. Final saved file: {result_path}")
        return existing_df

    def hierarchical_subclustering_from_local_file(self, base_cluster_file, parquet_paths, n_subclusters=5,
                                                   pca_components=50):
        model_df = pd.read_parquet(base_cluster_file)

        all_results = []

        for path in parquet_paths:
            print(f"Processing {path}")
            df = pd.read_parquet(path)
            df["unique_id"] = df.apply(generate_unique_id, axis=1)
            df = df[df["unique_id"].isin(model_df["unique_id"])]

            merged = df.merge(model_df, on="unique_id", how="inner")
            print(f"→ Merged {len(merged)} samples")

            for cluster_id in sorted(merged["cluster"].unique()):
                cluster_df = merged[merged["cluster"] == cluster_id]

                if len(cluster_df) < n_subclusters:
                    print(f"Skipping cluster {cluster_id}, only {len(cluster_df)} samples")
                    continue

                vectors = np.vstack(cluster_df["vector"].to_numpy())

                if pca_components and vectors.shape[1] > pca_components:
                    vectors = PCA(n_components=pca_components, random_state=42).fit_transform(vectors)

                sub_model = MiniBatchKMeans(n_clusters=n_subclusters, random_state=42).fit(vectors)
                cluster_df["subcluster"] = sub_model.labels_
                all_results.append(cluster_df)

        final_df = pd.concat(all_results, ignore_index=True)
        out_path = os.path.join(self.io.data_dir, "hierarchical_clusters.parquet")
        final_df.to_parquet(out_path, index=False)
        print(f"✅ Saved hierarchical subclusters to {out_path}")
        return final_df

    def hierarchical_subclustering_from_es(self, base_cluster_file, n_subclusters=5, pca_components=50,
                                           batch_size=10000):
        model_df = pd.read_parquet(base_cluster_file)
        unique_ids = model_df["unique_id"].tolist()

        subclustered = []

        for batch_ids in self.batch_generator(unique_ids):
            _, vectors = self.fetcher.fetch_vectors_by_ids(batch_ids)

            if not vectors:
                continue

            batch_vectors = np.vstack(vectors)
            batch_df = model_df[model_df["unique_id"].isin(batch_ids)].copy()

            batch_df["vector"] = list(vectors)

            for cluster_id in sorted(batch_df["cluster"].unique()):
                sub_df = batch_df[batch_df["cluster"] == cluster_id]

                if len(sub_df) < n_subclusters:
                    print(f"Skipping cluster {cluster_id}, only {len(sub_df)} samples")
                    continue

                vectors = np.vstack(sub_df["vector"].to_numpy())
                if pca_components and vectors.shape[1] > pca_components:
                    vectors = PCA(n_components=pca_components, random_state=42).fit_transform(vectors)

                sub_model = MiniBatchKMeans(n_clusters=n_subclusters, random_state=42).fit(vectors)
                sub_df["subcluster"] = sub_model.labels_
                subclustered.append(sub_df)

        final_df = pd.concat(subclustered, ignore_index=True)
        out_path = os.path.join(self.io.data_dir, "hierarchical_clusters_es.parquet")
        final_df.to_parquet(out_path, index=False)
        print(f"✅ Saved hierarchical subclusters from ES to {out_path}")
        return final_df

    def assign_clusters_to_documents(self, result_file_name="clustering_results.parquet"):

        result_path = f"{self.io.data_dir}/{result_file_name}"

        if not os.path.exists(result_path):
            print(f"Result file {result_path} not found.")
            return

        print(f"Loading clustering results from {result_path}...")
        df = pd.read_parquet(result_path)

        if "unique_id" not in df.columns or "cluster" not in df.columns:
            print("Required columns 'unique_id' and 'cluster' not found.")
            return

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Assigning clusters"):
            uid = row["unique_id"]
            cluster_value = int(row["cluster"])

            try:
                doc = self.fetcher.es.get(index=self.fetcher.index, id=uid)
                current_cluster = doc["_source"].get("cluster", None)

                # If cluster field exists, prompt user
                if current_cluster is not None:
                    print(f"\n🟡 Document {uid} already has cluster = {current_cluster}")
                    decision = input("Overwrite? (y = yes, s = skip, q = quit): ").strip().lower()
                    if decision == "s":
                        continue
                    elif decision == "q":
                        print("Aborted")
                        break

                # Update (or assign) cluster value
                self.fetcher.es.update(index=self.fetcher.index, id=uid, body={"doc": {"cluster": cluster_value}})
                print(f"Assigned cluster {cluster_value} to {uid}")

            except Exception as e:
                print(f"Error updating document {uid}: {e}")

    def assign_clusters_to_documents_batch(self, result_file_name="clustering_results.parquet",
                                           batch_size=1000, force=False, dry_run=False):

        result_path = f"{self.io.data_dir}/{result_file_name}"
        if not os.path.exists(result_path):
            print(f"Result file {result_path} not found.")
            return

        print(f"Loading clustering results from {result_path}...")
        df = pd.read_parquet(result_path)

        if "unique_id" not in df.columns or "cluster" not in df.columns:
            print("Required columns 'unique_id' and 'cluster' not found.")
            return

        df = df[["unique_id", "cluster"]].copy()
        uid_to_cluster = dict(zip(df["unique_id"], df["cluster"]))
        all_ids = list(uid_to_cluster.keys())

        for i in tqdm(range(0, len(all_ids), batch_size), desc="Processing batches"):
            batch_ids = all_ids[i:i + batch_size]

            try:
                res = self.fetcher.es.mget(index=self.fetcher.index, body={"ids": batch_ids})
            except Exception as e:
                print(f"Error during mget: {e}")
                continue

            actions = []
            for doc in res["docs"]:
                if not doc.get("found"):
                    continue

                uid = doc["_id"]
                src = doc["_source"]
                predicted_cluster = int(uid_to_cluster[uid])
                current_cluster = src.get("cluster")

                if current_cluster is not None and not force:
                    print(f"\nDocument {uid} already has cluster = {current_cluster}")
                    decision = input("Overwrite? (y = yes, s = skip, q = quit, a = overwrite all): ").strip().lower()
                    if decision == "s":
                        continue
                    elif decision == "q":
                        print("Aborted")
                        return
                    elif decision == "a":
                        force = True

                if dry_run:
                    print(f"[DRY-RUN] Would assign cluster={predicted_cluster} to document {uid}")
                    continue

                actions.append({
                    "_op_type": "update",
                    "_index": self.fetcher.index,
                    "_id": uid,
                    "doc": {"cluster": predicted_cluster}
                })

            if not dry_run and actions:
                try:
                    success, failed = helpers.bulk(self.fetcher.es, actions, raise_on_error=False, stats_only=True)
                    print(f"Updated {success}, Failed {failed}")
                except Exception as e:
                    print(f"Bulk update failed: {e}")

    def describe_cluster_local(self, cluster_id: int, parquet_paths,
                               result_file_name="clustering_results.parquet",
                               top_k_cats=20):

        result_path = f"{self.io.data_dir}/{result_file_name}"
        if not os.path.exists(result_path):
            print(f"Result file {result_path} not found.")
            return

        print(f"Loading clustering results from {result_path}...")
        df = pd.read_parquet(result_path)

        if "cluster" not in df.columns:
            print("'cluster' column missing.")
            return

        cluster_df = df[df["cluster"] == cluster_id]
        if cluster_df.empty:
            print(f"No samples found for cluster {cluster_id}")
            return

        unique_ids = cluster_df["unique_id"].tolist()
        print(f"Fetching {len(unique_ids)} documents for cluster {cluster_id}...")
        valids = []
        for x in parquet_paths:
            dff = pd.read_parquet(x)
            dff["unique_id"] = dff.apply(generate_unique_id, axis=1)
            # keep only rows where its unique_id is in unique_ids list
            dff = dff[dff["unique_id"].isin(unique_ids)]
            valids.append(dff)

        doc_df = pd.concat(valids, ignore_index=True)
        print(f"Stats for Cluster {cluster_id} — {len(doc_df)} documents\n")

        figs = []

        for col in sorted(doc_df.columns):
            values = doc_df[col].dropna()
            if values.empty or col == "vector":
                continue

            print(f"\n— {col} —")
            if pd.api.types.is_numeric_dtype(values):
                stats = values.describe()
                print(stats.to_string())

                fig = px.histogram(doc_df, x=col, nbins=40, title=f"{col} Distribution (Cluster {cluster_id})")
                figs.append((fig, f"{col}_histogram_cluster_{cluster_id}"))

            else:
                vc = values.value_counts().nlargest(top_k_cats)
                print(vc.to_string())
                if len(values.unique()) > top_k_cats:
                    print(f"... ({len(values.unique())} unique values total)")

                fig = go.Figure(data=[go.Bar(
                    x=vc.index.astype(str),
                    y=vc.values,
                    text=vc.values,
                    textposition='outside'
                )])
                fig.update_layout(
                    title=f"{col} Top {top_k_cats} Values (Cluster {cluster_id})",
                    xaxis_title=col,
                    yaxis_title="Count"
                )
                figs.append((fig, f"{col}_barchart_cluster_{cluster_id}"))

        # Save all charts
        for fig, name in figs:
            self.io.save_html(fig, name, cluster_id)
        self.io.save_csv(doc_df, f"{cluster_id}_described")
        return doc_df

    def describe_cluster(self, cluster_id: int, result_file_name="clustering_results.parquet",
                         batch_size=10000, top_k_cats=20):

        result_path = f"{self.io.data_dir}/{result_file_name}"
        if not os.path.exists(result_path):
            print(f"Result file {result_path} not found.")
            return

        print(f"Loading clustering results from {result_path}...")
        df = pd.read_parquet(result_path)

        if "cluster" not in df.columns:
            print("'cluster' column missing.")
            return

        cluster_df = df[df["cluster"] == cluster_id]
        if cluster_df.empty:
            print(f"No samples found for cluster {cluster_id}")
            return

        unique_ids = cluster_df["unique_id"].tolist()
        print(f"Fetching {len(unique_ids)} documents for cluster {cluster_id}...")

        docs = []
        for i in range(0, len(unique_ids), batch_size):
            sub_ids = unique_ids[i:i + batch_size]
            try:
                res = self.fetcher.mget(sub_ids)
                docs.extend(doc["_source"] for doc in res["docs"] if doc.get("found"))
            except Exception as e:
                print(f"Error during mget: {e}")

        if not docs:
            print("No documents were fetched.")
            return

        doc_df = pd.DataFrame(docs)
        print(f"Stats for Cluster {cluster_id} — {len(doc_df)} documents\n")

        figs = []

        for col in sorted(doc_df.columns):
            values = doc_df[col].dropna()
            if values.empty or col == "vector":
                continue

            print(f"\n— {col} —")
            if pd.api.types.is_numeric_dtype(values):
                stats = values.describe()
                print(stats.to_string())

                fig = px.histogram(doc_df, x=col, nbins=40, title=f"{col} Distribution (Cluster {cluster_id})")
                figs.append((fig, f"{col}_histogram_cluster_{cluster_id}"))

            else:
                vc = values.value_counts().nlargest(top_k_cats)
                print(vc.to_string())
                if len(values.unique()) > top_k_cats:
                    print(f"... ({len(values.unique())} unique values total)")

                fig = go.Figure(data=[go.Bar(
                    x=vc.index.astype(str),
                    y=vc.values,
                    text=vc.values,
                    textposition='outside'
                )])
                fig.update_layout(
                    title=f"{col} Top {top_k_cats} Values (Cluster {cluster_id})",
                    xaxis_title=col,
                    yaxis_title="Count"
                )
                figs.append((fig, f"{col}_barchart_cluster_{cluster_id}"))

        # Save all charts
        for fig, name in figs:
            self.io.save_html(fig, name, cluster_id)
        self.io.save_csv(doc_df, f"{cluster_id}_described")
        return doc_df

    def plot_cluster_tissue_info(self, preprocessing="resize", max_vals=100):

        df_counts = self.io.load_csv(f"cluster_tissue_distribution_{preprocessing}", subfolder="data")
        if df_counts is None:
            print("Fetching unique tissue types from Elasticsearch...")
            tissue_types = self.fetcher.get_unique_field_values(field="tissue_type", size=max_vals)
            clusters = sorted(map(int, self.fetcher.get_unique_field_values(field="cluster", size=max_vals)))

            print(f"Found {len(tissue_types)} tissue types: {tissue_types}")
            print(f"Found {len(clusters)} clusters: {clusters}")

            tissue_counts = defaultdict(dict)

            print("Counting samples per (cluster, tissue_type)...")
            for cluster_id in tqdm(clusters, desc="Clusters"):
                for tissue in tissue_types:
                    query = {
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"cluster": cluster_id}},
                                    {"term": {"preprocessing": preprocessing}},
                                    {"term": {"tissue_type": tissue}}
                                ]
                            }
                        }
                    }
                    try:
                        res = self.fetcher.es.count(index=self.fetcher.index, body=query)
                        count = res["count"]
                        tissue_counts[cluster_id][tissue] = count
                    except Exception as e:
                        print(f"Count failed for cluster {cluster_id}, tissue {tissue}: {e}")
                        tissue_counts[cluster_id][tissue] = 0

            records = [
                {"cluster": c, "tissue_type": t, "count": tissue_counts[c].get(t, 0)}
                for c in clusters for t in tissue_types
            ]

            df_counts = pd.DataFrame.from_records(records)
            self.io.save_csv(df_counts, f"cluster_tissue_distribution_{preprocessing}", subfolder="data", index=False)
        else:
            print("Loading from cached CSV... Reconstructing metadata.")
            tissue_types = sorted(df_counts["tissue_type"].unique().tolist())
            clusters = sorted(df_counts["cluster"].unique().tolist())

            tissue_counts = defaultdict(dict)
            for _, row in df_counts.iterrows():
                tissue_counts[int(row["cluster"])][row["tissue_type"]] = row["count"]

        color_map = {t: qualitative.D3[i % len(qualitative.Set3)] for i, t in enumerate(tissue_types)}
        cluster_totals = df_counts.groupby("cluster")["count"].sum().to_dict()
        subplot_titles = [f"Cluster {c} (n={cluster_totals.get(c, 0)})" for c in clusters]
        total_samples = df_counts['count'].sum()
        pivot_df = df_counts.pivot(index="cluster", columns="tissue_type", values="count").fillna(0)
        pivot_df_norm = pivot_df.div(pivot_df.sum(axis=1), axis=0)
        inv_pivot_df = df_counts.pivot(index="tissue_type", columns="cluster", values="count").fillna(0)
        inv_pivot_df_norm = inv_pivot_df.div(inv_pivot_df.sum(axis=1), axis=0)

        fig = go.Figure()
        for tissue in tissue_types:
            y_vals = [tissue_counts[c].get(tissue, 0) for c in clusters]
            fig.add_trace(go.Bar(
                x=[str(c) for c in clusters],
                y=y_vals,
                name=tissue,
                marker_color=color_map[tissue],
                text=y_vals,
                textposition="outside"
            ))

        fig.update_layout(
            barmode="group",
            title=f"Cluster-wise Tissue Type Distribution (total samples = {total_samples})",
            xaxis_title="Cluster ID",
            yaxis_title="Number of Samples",
        )

        stacked_fig = go.Figure()
        for tissue in pivot_df.columns:
            stacked_fig.add_trace(go.Bar(
                name=tissue,
                x=pivot_df.index.astype(str),
                y=pivot_df[tissue],
                marker_color=color_map[tissue]
            ))

        stacked_fig.update_layout(
            barmode='stack',
            title=f"Tissue Type per Cluster (total samples = {total_samples})",
            xaxis_title="Cluster ID",
            yaxis_title="Proportion",
        )

        stacked_fig_norm = go.Figure()
        for tissue in pivot_df_norm.columns:
            stacked_fig_norm.add_trace(go.Bar(
                name=tissue,
                x=pivot_df_norm.index.astype(str),
                y=pivot_df_norm[tissue],
                marker_color=color_map[tissue]
            ))

        stacked_fig_norm.update_layout(
            barmode='stack',
            title=f"Normalized Tissue Type per Cluster (total samples = {total_samples})",
            xaxis_title="Cluster ID",
            yaxis_title="Proportion",
        )

        polar_fig = go.Figure()

        for cluster_id in clusters:
            cluster_data = df_counts[df_counts["cluster"] == cluster_id]
            cluster_data = cluster_data[cluster_data["count"] > 0]
            polar_fig.add_trace(go.Barpolar(
                r=cluster_data["count"],
                theta=cluster_data["tissue_type"],
                name=f"Cluster {cluster_id}",
                marker_color=[color_map[t] for t in cluster_data["tissue_type"]],
                opacity=0.6,
            ))

        polar_fig.update_layout(
            title=f"Tissue Distribution Across Clusters (total samples = {total_samples})",
            polar=dict(
                radialaxis=dict(showticklabels=True, ticks=''),
                angularaxis=dict(direction="clockwise")
            ),
        )

        rows = (len(clusters) + 2) // 3
        cols = 3

        pie = make_subplots(
            rows=rows, cols=cols,
            specs=[[{"type": "domain"}] * cols for _ in range(rows)],
            subplot_titles=subplot_titles
        )

        for i, cluster in enumerate(clusters):
            row = i // cols + 1
            col = i % cols + 1

            cluster_data = df_counts[df_counts["cluster"] == cluster]
            pie.add_trace(go.Pie(
                labels=cluster_data["tissue_type"],
                values=cluster_data["count"],
                name=f"Cluster {cluster}",
                marker=dict(colors=[color_map[t] for t in cluster_data["tissue_type"]])
            ), row=row, col=col)

        pie.update_layout(
            title_text=f"Tissue Distribution per Cluster (total samples = {total_samples})",
            height=rows * 400,
            showlegend=False
        )

        cluster_counts = df_counts.groupby("cluster")["count"].sum().sort_index()
        cluster_ids = cluster_counts.index.tolist()
        counts = cluster_counts.values.tolist()

        min_val = cluster_counts.min()
        max_val = cluster_counts.max()
        min_id = cluster_counts.idxmin()
        max_id = cluster_counts.idxmax()
        avg_val = cluster_counts.mean()
        median_val = cluster_counts.median()

        # Bar colors: highlight min/max clusters
        bar_colors = [
            'crimson' if cid == min_id else
            'darkgreen' if cid == max_id else
            'steelblue'
            for cid in cluster_ids
        ]

        # Create main bar plot
        fig_size = go.Figure(data=[
            go.Bar(
                x=cluster_ids,
                y=counts,
                text=counts,
                textposition='outside',
                marker=dict(color=bar_colors),
                name="Cluster Size"
            ),
            # Average line
            go.Scatter(
                x=cluster_ids,
                y=[avg_val] * len(cluster_ids),
                mode='lines',
                line=dict(color='orange', dash='dash'),
                name=f"Average ({avg_val:.1f})"
            ),
            # Median line
            go.Scatter(
                x=cluster_ids,
                y=[median_val] * len(cluster_ids),
                mode='lines',
                line=dict(color='purple', dash='dot'),
                name=f"Median ({median_val:.1f})"
            )
        ])

        fig_size.update_layout(
            title="Cluster Sizes",
            xaxis_title="Cluster ID",
            yaxis_title="Number of Samples",
        )

        fig_inv_grouped = go.Figure()
        for cluster in inv_pivot_df.columns:
            y_vals = inv_pivot_df[cluster].tolist()
            fig_inv_grouped.add_trace(go.Bar(
                x=tissue_types,
                y=y_vals,
                name=f"Cluster {cluster}",
                text=y_vals,
                textposition="outside"
            ))

        fig_inv_grouped.update_layout(
            barmode="group",
            title=f"Tissue Type → Cluster Distribution (total samples = {total_samples})",
            xaxis_title="Tissue Type",
            yaxis_title="Number of Samples",
        )
        fig_inv_stacked = go.Figure()
        for cluster in inv_pivot_df_norm.columns:
            fig_inv_stacked.add_trace(go.Bar(
                name=f"Cluster {cluster}",
                x=inv_pivot_df_norm.index,
                y=inv_pivot_df_norm[cluster]
            ))

        fig_inv_stacked.update_layout(
            barmode='stack',
            title=f"Normalized Cluster Distribution per Tissue Type",
            xaxis_title="Tissue Type",
            yaxis_title="Proportion",
        )

        fig_inv_stacked_norm = go.Figure()
        for cluster in inv_pivot_df.columns:
            fig_inv_stacked_norm.add_trace(go.Bar(
                name=f"Cluster {cluster}",
                x=inv_pivot_df.index,
                y=inv_pivot_df[cluster]
            ))

        fig_inv_stacked_norm.update_layout(
            barmode='stack',
            title=f"Cluster Distribution per Tissue Type",
            xaxis_title="Tissue Type",
            yaxis_title="Proportion",
        )

        self.io.save_html(fig_size, f"cluster_size_barplot_{preprocessing}")
        self.io.save_html(fig, f"cluster_tissue_distribution_counts_{preprocessing}")
        self.io.save_html(stacked_fig_norm, f"normalized_stacked_bar_{preprocessing}")
        self.io.save_html(stacked_fig, f"stacked_bar_{preprocessing}")
        self.io.save_html(polar_fig, f"polar_overlap_clusters_{preprocessing}")
        self.io.save_html(pie, f"pie_charts_per_cluster_{preprocessing}")
        self.io.save_html(fig_inv_grouped, f"inverse_grouped_bar_{preprocessing}")
        self.io.save_html(fig_inv_stacked, f"inverse_stacked_bar_{preprocessing}")
        self.io.save_html(fig_inv_stacked_norm, f"inverse_normalized_stacked_bar_{preprocessing}")

    def plot_cluster_tissue_info_local(self, parquet_paths, result_file_name, preprocessing, k):
        df_counts = self.io.load_csv(f"cluster_tissue_distribution_{preprocessing}_{k}", subfolder="data")

        result_path = f"{self.io.data_dir}/{result_file_name}"
        if df_counts is None:
            if not os.path.exists(result_path):
                print(f"Result file {result_path} not found.")
                return None

            df_results = pd.read_parquet(result_path)
            if "cluster" not in df_results.columns or "unique_id" not in df_results.columns:
                print("Missing 'cluster' or 'unique_id' columns.")
                return None

            local_dfs = []
            for path in parquet_paths:
                print(f"Getting: {path}")
                df = pd.read_parquet(path)
                df["unique_id"] = df.apply(generate_unique_id, axis=1)
                local_dfs.append(df)

            df_meta = pd.concat(local_dfs, ignore_index=True)
            df_merged = df_results.merge(df_meta, on="unique_id", how="inner")
            df_counts = df_merged.groupby(["cluster", "tissue_type"]).size().reset_index(name="count")
            df_counts = df_counts[~df_counts["tissue_type"].astype(str).str.isdigit()]

            tissue_types = sorted(df_counts["tissue_type"].unique().tolist())
            clusters = sorted(df_counts["cluster"].unique().tolist())

            tissue_counts = defaultdict(dict)
            for _, row in df_counts.iterrows():
                tissue_counts[int(row["cluster"])][row["tissue_type"]] = row["count"]

            df_counts = df_counts[~df_counts["tissue_type"].astype(str).str.isdigit()]
            self.io.save_csv(df_counts, f"cluster_tissue_distribution_{preprocessing}_{k}", subfolder="data", index=False)
        else:

            print("Loading from cached CSV... Reconstructing metadata.")
            df_counts = df_counts[~df_counts["tissue_type"].astype(str).str.isdigit()]
            tissue_types = sorted([x for x in df_counts["tissue_type"].unique().tolist() if not x.isdigit()])
            clusters = sorted(df_counts["cluster"].unique().tolist())

            tissue_counts = defaultdict(dict)
            for _, row in df_counts.iterrows():
                tissue_counts[int(row["cluster"])][row["tissue_type"]] = row["count"]

        color_map = {t: qualitative.Plotly[i % len(qualitative.Plotly)] for i, t in enumerate(tissue_types)}
        cluster_totals = df_counts.groupby("cluster")["count"].sum().to_dict()
        subplot_titles = [f"Cluster {c} (n={cluster_totals.get(c, 0)})" for c in clusters]
        total_samples = df_counts['count'].sum()
        pivot_df = df_counts.pivot(index="cluster", columns="tissue_type", values="count").fillna(0)
        pivot_df_norm = pivot_df.div(pivot_df.sum(axis=1), axis=0)
        inv_pivot_df = df_counts.pivot(index="tissue_type", columns="cluster", values="count").fillna(0)
        inv_pivot_df_norm = inv_pivot_df.div(inv_pivot_df.sum(axis=1), axis=0)

        fig = go.Figure()
        for tissue in tissue_types:
            y_vals = [tissue_counts[c].get(tissue, 0) for c in clusters]
            fig.add_trace(go.Bar(
                x=[str(c) for c in clusters],
                y=y_vals,
                name=tissue,
                marker_color=color_map[tissue],
                text=y_vals,
                textposition="outside"
            ))

        fig.update_layout(
            barmode="group",
            title=f"Cluster-wise Tissue Type Distribution (total samples = {total_samples})",
            xaxis_title="Cluster ID",
            yaxis_title="Number of Samples",
        )

        stacked_fig = go.Figure()
        for tissue in pivot_df.columns:
            stacked_fig.add_trace(go.Bar(
                name=tissue,
                x=pivot_df.index.astype(str),
                y=pivot_df[tissue],
                marker_color=color_map[tissue]
            ))

        stacked_fig.update_layout(
            barmode='stack',
            title=f"Tissue Type per Cluster (total samples = {total_samples})",
            xaxis_title="Cluster ID",
            yaxis_title="Proportion",
        )

        stacked_fig_norm = go.Figure()
        for tissue in pivot_df_norm.columns:
            stacked_fig_norm.add_trace(go.Bar(
                name=tissue,
                x=pivot_df_norm.index.astype(str),
                y=pivot_df_norm[tissue],
                marker_color=color_map[tissue]
            ))

        stacked_fig_norm.update_layout(
            barmode='stack',
            title=f"Normalized Tissue Type per Cluster (total samples = {total_samples})",
            xaxis_title="Cluster ID",
            yaxis_title="Proportion",
        )

        polar_fig = go.Figure()

        for cluster_id in clusters:
            cluster_data = df_counts[df_counts["cluster"] == cluster_id]
            cluster_data = cluster_data[cluster_data["count"] > 0]
            polar_fig.add_trace(go.Barpolar(
                r=cluster_data["count"],
                theta=cluster_data["tissue_type"],
                name=f"Cluster {cluster_id}",
                marker_color=[color_map[t] for t in cluster_data["tissue_type"]],
                opacity=0.6,
            ))

        polar_fig.update_layout(
            title=f"Tissue Distribution Across Clusters (total samples = {total_samples})",
            polar=dict(
                radialaxis=dict(showticklabels=True, ticks=''),
                angularaxis=dict(direction="clockwise")
            ),
        )

        rows = (len(clusters) + 2) // 3
        cols = 3

        pie = make_subplots(
            rows=rows, cols=cols,
            specs=[[{"type": "domain"}] * cols for _ in range(rows)],
            subplot_titles=subplot_titles
        )

        for i, cluster in enumerate(clusters):
            row = i // cols + 1
            col = i % cols + 1

            cluster_data = df_counts[df_counts["cluster"] == cluster]
            pie.add_trace(go.Pie(
                labels=cluster_data["tissue_type"],
                values=cluster_data["count"],
                name=f"Cluster {cluster}",
                marker=dict(colors=[color_map[t] for t in cluster_data["tissue_type"]])
            ), row=row, col=col)

        pie.update_layout(
            title_text=f"Tissue Distribution per Cluster (total samples = {total_samples})",
            height=rows * 400,
            showlegend=False
        )

        cluster_counts = df_counts.groupby("cluster")["count"].sum().sort_index()
        cluster_ids = cluster_counts.index.tolist()
        counts = cluster_counts.values.tolist()

        min_val = cluster_counts.min()
        max_val = cluster_counts.max()
        min_id = cluster_counts.idxmin()
        max_id = cluster_counts.idxmax()
        avg_val = cluster_counts.mean()
        median_val = cluster_counts.median()

        # Bar colors: highlight min/max clusters
        bar_colors = [
            'crimson' if cid == min_id else
            'darkgreen' if cid == max_id else
            'steelblue'
            for cid in cluster_ids
        ]

        # Create main bar plot
        fig_size = go.Figure(data=[
            go.Bar(
                x=cluster_ids,
                y=counts,
                text=counts,
                textposition='outside',
                marker=dict(color=bar_colors),
                name="Cluster Size"
            ),
            # Average line
            go.Scatter(
                x=cluster_ids,
                y=[avg_val] * len(cluster_ids),
                mode='lines',
                line=dict(color='orange', dash='dash'),
                name=f"Average ({avg_val:.1f})"
            ),
            # Median line
            go.Scatter(
                x=cluster_ids,
                y=[median_val] * len(cluster_ids),
                mode='lines',
                line=dict(color='purple', dash='dot'),
                name=f"Median ({median_val:.1f})"
            )
        ])

        fig_size.update_layout(
            title="Cluster Sizes",
            xaxis_title="Cluster ID",
            yaxis_title="Number of Samples",
        )

        fig_inv_grouped = go.Figure()
        for cluster in inv_pivot_df.columns:
            y_vals = inv_pivot_df[cluster].tolist()
            fig_inv_grouped.add_trace(go.Bar(
                x=tissue_types,
                y=y_vals,
                name=f"Cluster {cluster}",
                text=y_vals,
                textposition="outside"
            ))

        fig_inv_grouped.update_layout(
            barmode="group",
            title=f"Tissue Type → Cluster Distribution (total samples = {total_samples})",
            xaxis_title="Tissue Type",
            yaxis_title="Number of Samples",
        )
        fig_inv_stacked = go.Figure()
        for cluster in inv_pivot_df_norm.columns:
            fig_inv_stacked.add_trace(go.Bar(
                name=f"Cluster {cluster}",
                x=inv_pivot_df_norm.index,
                y=inv_pivot_df_norm[cluster]
            ))

        fig_inv_stacked.update_layout(
            barmode='stack',
            title=f"Normalized Cluster Distribution per Tissue Type",
            xaxis_title="Tissue Type",
            yaxis_title="Proportion",
        )

        fig_inv_stacked_norm = go.Figure()
        for cluster in inv_pivot_df.columns:
            fig_inv_stacked_norm.add_trace(go.Bar(
                name=f"Cluster {cluster}",
                x=inv_pivot_df.index,
                y=inv_pivot_df[cluster]
            ))

        fig_inv_stacked_norm.update_layout(
            barmode='stack',
            title=f"Cluster Distribution per Tissue Type",
            xaxis_title="Tissue Type",
            yaxis_title="Proportion",
        )

        self.io.save_html(fig_size, f"{k}_cluster_size_barplot_{preprocessing}")
        self.io.save_html(fig, f"{k}_cluster_tissue_distribution_counts_{preprocessing}")
        self.io.save_html(stacked_fig_norm, f"{k}_normalized_stacked_bar_{preprocessing}")
        self.io.save_html(stacked_fig, f"{k}_stacked_bar_{preprocessing}")
        self.io.save_html(polar_fig, f"{k}_polar_overlap_clusters_{preprocessing}")
        self.io.save_html(pie, f"{k}_pie_charts_per_cluster_{preprocessing}")
        self.io.save_html(fig_inv_grouped, f"{k}_inverse_grouped_bar_{preprocessing}")
        self.io.save_html(fig_inv_stacked, f"{k}_inverse_stacked_bar_{preprocessing}")
        self.io.save_html(fig_inv_stacked_norm, f"{k}_inverse_normalized_stacked_bar_{preprocessing}")

        from embeddings.embeddings_pipeline.utils import generate_per_plot_type_dashboards

        generate_per_plot_type_dashboards(self.io.plots_dir,
                                          os.path.join(self.io.results_dir, "plot_dashboards"))


    def export_s3_paths_for_cluster(self, cluster_id, result_file_name="clustering_results.parquet", output_txt=None):

        result_path = f"{self.io.data_dir}/{result_file_name}"
        if not os.path.exists(result_path):
            print(f"Result file {result_path} not found.")
            return
        print(f"Loading clustering results from {result_path}...")
        df = pd.read_parquet(result_path)

        if "cluster" not in df.columns:
            print("'cluster' column missing.")
            return

        cluster_df = df[df["cluster"] == cluster_id]
        if cluster_df.empty:
            print(f"No samples found for cluster {cluster_id}")
            return

        unique_ids = cluster_df["unique_id"].tolist()

        # Fetch docs by IDs in chunks
        docs = []
        for i in range(0, len(unique_ids), 1000):
            sub_batch = unique_ids[i:i + 1000]
            res = self.fetcher.mget(sub_batch)
            docs.extend(doc["_source"] for doc in res["docs"] if doc.get("found"))
            print(f"Fetched {len(docs)}/{len(unique_ids)}")

        def build_s3_path(doc):
            return (
                f"s3://gi-registration/slide-registration-production/tiles/512/"
                f"{doc['tissue_type']}/{doc['block_id']}/{doc['slice_id']}/"
                f"{doc['scan_date']}/{doc['box_id']}/{doc['filename']}-stained-sr.tiff"
            )

        s3_paths = [build_s3_path(doc) for doc in docs]

        # Save to file
        out_dir = f"{self.io.results_dir}/s3_paths"
        os.makedirs(out_dir, exist_ok=True)

        output_path = output_txt or f"{out_dir}/cluster_{cluster_id}_s3_paths.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(s3_paths))

        print(f"Saved {len(s3_paths)} S3 paths to: {output_path}")
        return output_path

    def grid_search(self, preprocess_options, apply_pca_options,
                    min_samples_options, n_clusters_list,
                    calculate_light_scores=True, calculate_silhouette_score=False,
                    max_samples=10_000_000, max_docs=2_000_000, modality="stained-sr"):
        # Build filter
        filter_builder = ElasticsearchFilterBuilder(keyword_suffix=self.fetcher.keyword_field)
        # Grid search
        for preprocessing, apply_pca, min_per_class in product(preprocess_options, apply_pca_options,
                                                               min_samples_options):
            print(f"\n🔍 Running config: preprocessing={preprocessing}, PCA={apply_pca}, min_class={min_per_class}")

            filter_builder.set_filter(modality=modality, preprocessing=preprocessing)

            # Output naming
            run_name = f"{modality.replace('-', '_')}_{preprocessing}_pca{int(apply_pca)}_min{min_per_class}"
            self.io.data_dir = f"{self.io.grid_search_dir}/{run_name}/data"
            self.io.results_dir = f"{self.io.grid_search_dir}/{run_name}/results"
            self.io.ensure_dirs()

            try:
                self.evaluate_optimal_clusters_stratified(
                    filter_builder=filter_builder,
                    n_clusters_list=n_clusters_list,
                    max_docs=max_docs,
                    apply_pca=apply_pca,
                    min_per_class=min_per_class,
                    max_samples=max_samples,
                    calculate_light_scores=calculate_light_scores,
                    calculate_silhouette_score=calculate_silhouette_score
                )
            except Exception as e:
                print(f"Failed on config {run_name}: {e}")

        # Visualize grid search
        self.visualize_grid_search()
        self.results_summary()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elasticsearch Bulk Ingest Tool")
    parser.add_argument("--es_host", type=str, default="https://10.88.0.3:9200", help="Elasticsearch host URL")
    parser.add_argument("--username", type=str, default="milos_z", help="Elasticsearch username")
    parser.add_argument("--password", type=str, default="wcV4!w^eYQSuqzRK", help="Elasticsearch password")
    parser.add_argument("--index", type=str, default="app_proto_data", help="Index name to write to")
    args = parser.parse_args()


    # ######################################################################################
    # # Init
    # ######################################################################################

    # analyzer = ElasticEmbeddingAnalyzer(
    #     es_host=args.es_host,
    #     index=args.index,
    #     username=args.username,
    #     password=args.password,
    #     batch_size=10000,
    #     out_dir="data/results_st_rs_no_pca"
    # )

    filter_builder = ElasticsearchFilterBuilder()

    # ######################################################################################
    # # Filter builder
    # ######################################################################################

    filter_builder.set_filter(
        # slice_id=None,
        # block_id=None,
        # scan_date=None,
        # box_id=None,
        # tile_type=None,
        # created_at=None,
        # dimension=None,
        # id=None,
        # model=None,
        # slide_key=None,
        # unique_id=None,
        # filename="tile_3_level0_2048-7424-2560-7936",
        modality="stained-sr",
        preprocessing="resize",
        # tissue_type="prostate",
    )
    filter_query = filter_builder.build()

    # ######################################################################################
    # # Similarity search
    # ######################################################################################

    # target_doc = analyzer.get_doc_by_id(target_id)
    # vector = target_doc["vector"]
    #
    # print("Target sample:")
    # target_doc.pop("vector", None)
    # pprint(target_doc)
    #
    # start_time = time.time()
    #
    #
    # matches = analyzer.knn_vector_search(vector, top_k=50, num_candidates=1000, filter_query={"bool": filter_query["bool"]},
    #                                      # exclude_id=target_id, exclude_block_id=target_doc["block_id"])
    #                                      exclude_id=target_id)
    #
    # analyzer.visualize_similar_images_grid(target_doc, matches, N=10)
    #
    # # Target vector as match-style dict
    # target_entry = {
    #     "_id": "__query_vector__",
    #     "_source": {
    #         "vector": vector,
    #         "slide_key": "__query__",
    #         "modality": "query",
    #         "preprocessing": "query",
    #         "tissue_type": "query"
    #     }
    # }
    # matches.append(target_entry)
    #
    # print(f"Done in {time.time() - start_time:.2f} seconds")
    # for k, v in matches[0]["_source"].items():
    #     if k not in ["block_id", "box_id", "created_at", "dimension", "filename", "id", "model", "scan_date", "slice_id", "slide_key", "tile_type", "unique_id", "vector"]:
    #         print(f"{k}: {Counter([x['_source'][k] for x in matches])}")
    #
    # result_df = analyzer.run_clustering(matches=matches, n_clusters=10, max_docs=10_000)
    # analyzer.visualize_clusters(result_df, vectors=result_df["vector"])

    # ######################################################################################
    # # Clustering
    # ######################################################################################

    # slide_key = "smallbowel_24a121_1_2024-06-11_41000-16400"
    # tissue_type, block_id, slice_id, scan_date, box_id = slide_key.split("_")
    # # filename = "tile_3_level0_2048-7424-2560-7936"
    # filename = None
    # # box_id = None
    # test = analyzer.get_docs_by_tile_group(
    #     tissue_type=tissue_type,
    #     block_id=block_id,
    #     slice_id=slice_id,
    #     scan_date=scan_date,
    #     box_id=box_id,
    #     filename=filename,
    #     size=10000
    # )
    #
    # result_df = analyzer.run_clustering(matches=test, n_clusters=4, max_docs=10_000)
    # analyzer.visualize_clusters(result_df, vectors=result_df["vector"])

    # ######################################################################################
    # # Cleaning
    # ######################################################################################

    # approved = ["colon", "endometrium", "falltube", "ovary", "placenta", "prostate",
    #             "skin", "smallbowel", "thyroid", "uterus"]
    # fetcher.review_and_clean_field(approved_types=approved)

    # ######################################################################################
    # # Grid Search
    # ######################################################################################

    fetcher = ElasticEmbeddingFetcher(
        es_host=args.es_host,
        index=args.index,
        username=args.username,
        password=args.password,
    )
    # Setup analyzer
    analyzer = ElasticEmbeddingAnalyzer(
        fetcher=fetcher,
        out_dir="data/grid_search_fine_resize",
        batch_size=10000,
    )

    # ######################################################################################
    # # Grid Search
    # ######################################################################################

    # preprocess_options = ["resize", "center_crop"]
    # # preprocess_options = ["center_crop"]
    # apply_pca_options = [True, False]
    # # apply_pca_options = [False]
    # min_samples_options = [10_000, 50_000, 100_000, 200_000]
    # # min_samples_options = [200_000]
    # n_clusters_list = list(range(5, 100, 1))
    #
    # analyzer.grid_search(preprocess_options=preprocess_options,
    #                      apply_pca_options=apply_pca_options,
    #                      min_samples_options=min_samples_options,
    #                      n_clusters_list=n_clusters_list)

    # ######################################################################################
    # # Results summary
    # ######################################################################################

    # ids = fetcher.fetch_and_save_ids(filter_query)
    # analyzer.io.save_parquet(ids, name="unique_ids")

    # analyzer.partial_fit_clustering(k=28)

    # analyzer.predict_from_model(model_path="/Users/miloszivkovic/GIT/minimodels/embeddings/data/grid_search_fine_resize/results/models_zoo/kmeans_model_k28.joblib")
    # analyzer.predict_from_model(model_path="/Users/miloszivkovic/GIT/minimodels/embeddings/data/grid_search_fine_center_crop/results/models_zoo/kmeans_model_k28.joblib")

    # analyzer.visualize_cluster_samples(rows=10, cols=10)

    # analyzer.export_s3_paths_for_cluster(cluster_id=14)
    # analyzer.visualize_cluster_samples(cluster_id=14, rows=3, cols=3, num_grids=5)
    
    # for cl_id in range(0, 28):
    #     analyzer.describe_cluster(cluster_id=cl_id, batch_size=10000)
        
    # analyzer.describe_cluster(cluster_id=14, batch_size=10000)
    
    # analyzer.plot_cluster_tissue_info(preprocessing="center_crop")
    analyzer.plot_cluster_tissue_info(preprocessing="resize")

    # analyzer.assign_clusters_to_documents()
    # analyzer.assign_clusters_to_documents_batch(batch_size=10000)