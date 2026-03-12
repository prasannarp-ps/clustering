import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from embeddings.embeddings_pipeline.embedding_extractor import get_all_image_paths_big

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Optional: LPIPS if you want deep perceptual metric
# pip install lpips
try:
    import lpips
    import torch
    LPIPS_AVAILABLE = True
    loss_fn_alex = lpips.LPIPS(net='alex')
except ImportError:
    LPIPS_AVAILABLE = False

from embeddings.embeddings_pipeline.embedding_extractor import center_crop

def calculate_metrics(inferred_path, stained_path):
    img1 = cv2.imread(inferred_path)
    img2 = np.array(center_crop(stained_path))

    if img1 is None or img2 is None or img1.shape != img2.shape:
        return None  # skip invalid pairs


    # Convert BGR to RGB for perceptual metrics
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # MSE (per-pixel average across channels)
    mse_val = np.mean((img1_rgb.astype(np.float32) - img2_rgb.astype(np.float32)) ** 2)

    # SSIM (multichannel)
    ssim_val = ssim(img1_rgb, img2_rgb, data_range=255, channel_axis=-1)

    # PSNR (color-aware)
    psnr_val = psnr(img1_rgb, img2_rgb, data_range=255)

    result = {
        "mse": mse_val,
        "ssim": ssim_val,
        "psnr": psnr_val,
    }

    if LPIPS_AVAILABLE:
        img1_tensor = torch.tensor(img1 / 255.0).permute(2, 0, 1).unsqueeze(0).float()
        img2_tensor = torch.tensor(img2 / 255.0).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            lpips_val = loss_fn_alex(img1_tensor, img2_tensor).item()
        result["lpips"] = lpips_val

    return result

def evaluate_all(slide_keys, inferred_dir, stained_dir, output_path):
    results = []

    for key in tqdm(slide_keys):
        inferred_path = os.path.join(inferred_dir, f"{key}-inferred.tiff")
        stained_path = os.path.join(stained_dir, f"{key}-stained.tiff")

        metrics = calculate_metrics(inferred_path, stained_path)
        if metrics:
            metrics["slide_key"] = key.replace("/", "_")
            results.append(metrics)

    df = pd.DataFrame(results)
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

def calculate_metrics_for_key(key, inferred_dir, stained_dir):
    inferred_path = os.path.join(inferred_dir, f"{key}-inferred.tiff")
    stained_path = os.path.join(stained_dir, f"{key}-stained.tiff")

    img1 = cv2.imread(inferred_path)
    img2 = np.array(center_crop(stained_path))

    if img1 is None or img2 is None or img1.shape != img2.shape:
        return None

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    mse_val = np.mean((img1_rgb.astype(np.float32) - img2_rgb.astype(np.float32)) ** 2)
    ssim_val = ssim(img1_rgb, img2_rgb, data_range=255, channel_axis=-1)
    psnr_val = psnr(img1_rgb, img2_rgb, data_range=255)

    result = {
        "slide_key": key.replace("/", "_"),
        "mse": mse_val,
        "ssim": ssim_val,
        "psnr": psnr_val,
    }

    if LPIPS_AVAILABLE:
        img1_tensor = torch.tensor(img1_rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float()
        img2_tensor = torch.tensor(img2_rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            lpips_val = loss_fn_alex(img1_tensor, img2_tensor).item()
        result["lpips"] = lpips_val

    return result

def evaluate_all_parallel(slide_keys, inferred_dir, stained_dir, output_path, max_workers=8):
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(calculate_metrics_for_key, key, inferred_dir, stained_dir): key
            for key in slide_keys
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                results.append(result)

    df = pd.DataFrame(results)
    df.to_parquet(output_path, index=False)
    print(f"Saved metrics to {output_path}")

def cluster_metrics_vertical_plot(df_sampling, df_perf, output_path):
    # Ensure 112 clusters
    all_clusters = pd.DataFrame({"cluster": list(range(112))})
    df_sampling = all_clusters.merge(df_sampling, on="cluster", how="left").fillna(0)
    df_perf = all_clusters.merge(df_perf, on="cluster", how="left").fillna(0)
    merged = df_sampling.merge(df_perf, on="cluster")

    metrics = ["ssim", "psnr", "mse"]
    y_titles = {"ssim": "SSIM", "psnr": "PSNR", "mse": "MSE"}

    fig = sp.make_subplots(
        rows=3, cols=1,
        subplot_titles=[f"{m.upper()} vs Sample Count" for m in metrics],
        shared_xaxes=False,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
        vertical_spacing=0.15
    )

    for i, metric in enumerate(metrics):
        # Metric bars (left y-axis)
        fig.add_trace(go.Bar(
            x=merged["cluster"],
            y=merged[metric],
            name=metric.upper(),
            marker_color="royalblue",
            showlegend=True
        ), row=i+1, col=1, secondary_y=False)

        # Sample count bars (right y-axis)
        fig.add_trace(go.Bar(
            x=merged["cluster"],
            y=merged["sample_count"],
            name="Sample Count",
            marker_color="dimgray",
            opacity=0.6,
            showlegend=True
        ), row=i+1, col=1, secondary_y=True)

        # Axis labels
        fig.update_yaxes(title_text=y_titles[metric], row=i+1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Sample Count", row=i+1, col=1, secondary_y=True)

    fig.update_layout(
        title="Cluster-wise Metrics and Sample Count",
        width=1400,
        height=1200,
        barmode="group",
        legend=dict(x=1.01, y=1),
        margin=dict(t=80)
    )

    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"Saved to: {output_path}")


def sampling_category_vs_metrics_plotly(df_sampling, df_perf, output_path):
    # Merge and classify sampling category
    df = df_sampling.merge(df_perf, on="cluster")
    mean_count = df["sample_count"].mean()
    std_count = df["sample_count"].std()

    def classify(x):
        if x < mean_count - std_count:
            return "Undersampled"
        elif x > mean_count + std_count:
            return "Oversampled"
        else:
            return "(Mean ± Std) Range"

    df["sampling_category"] = df["sample_count"].apply(classify)
    # Aggregate mean metrics per category
    agg = df.groupby("sampling_category")[["ssim", "psnr", "mse"]].mean().reset_index()

    # Create Plotly subplot
    fig = sp.make_subplots(rows=1, cols=3, subplot_titles=["SSIM", "PSNR", "MSE"])

    metrics = ["ssim", "psnr", "mse"]
    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(x=agg["sampling_category"], y=agg[metric], name=metric.upper()),
            row=1, col=i + 1
        )
        fig.update_yaxes(title_text=metric.upper(), row=1, col=i + 1)

    fig.update_layout(
        title_text="Performance Metrics vs Sampling Category",
        showlegend=False,
        width=1000,
        height=400
    )

    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"Saved to: {output_path}")

def calculate_cluster_metrics(df_metrics, df_clusters):
    df_clusters["slide_key"] = df_clusters["slide_key"].str.replace("-stained", "", regex=False)
    merged = df_metrics.merge(df_clusters, on="slide_key")
    cluster_metrics = merged.groupby("cluster")[["ssim", "psnr", "mse"]].mean().reset_index()
    cluster_metrics.to_parquet("cluster_average_metrics.parquet")
    cluster_metrics.to_csv("cluster_average_metrics.csv")

import pandas as pd
import plotly.express as px

def plot_sample_count_metric_correlation_plotly(df_sampling, df_perf, output_path="sample_count_vs_metric_correlation.html"):
    # Normalize cluster column name
    df_perf = df_perf.rename(columns={"cluster_stained": "cluster"})

    # Merge and compute correlations
    df = df_sampling.merge(df_perf, on="cluster")
    correlations = df[["sample_count", "ssim", "psnr", "mse"]].corr().loc["sample_count"].drop("sample_count")

    # Prepare dataframe for plotting
    corr_df = correlations.reset_index()
    corr_df.columns = ["Metric", "Correlation"]

    # Plot
    fig = px.bar(
        corr_df,
        x="Metric",
        y="Correlation",
        color="Metric",
        title="Correlation between Sample Count and Model Metrics",
        text="Correlation",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        yaxis_title="Pearson Correlation",
        xaxis_title="Metric",
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        showlegend=False,
        height=500,
        width=700
    )

    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"Saved plot to {output_path}")

# Example usage
if __name__ == "__main__":
    # inferred_dir = "/Volumes/KINGSTON/v6_2_infered/"
    # stained_dir = "/Volumes/KINGSTON/v6_2_stained/"
    # output_parquet = "/Volumes/KINGSTON/v6_2_comparison/image_metrics_all.parquet"
    #
    # slide_keys = get_all_image_paths_big(inferred_dir)
    # slide_keys = [x.replace(inferred_dir, "").replace("-inferred.tiff", "") for x in slide_keys]
    # evaluate_all_parallel(slide_keys, inferred_dir, stained_dir, output_parquet, max_workers=16)

    df_metrics = pd.read_parquet("/Volumes/KINGSTON/v6_2_comparison/image_metrics_all.parquet")
    df_clusters = pd.read_parquet("/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_resize/data/clustering_results_stained_center_crop_k112.parquet")
    # df_clusters = pd.read_parquet("/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_resize/data/clustering_results_inferred_k112.parquet")
    calculate_cluster_metrics(df_metrics, df_clusters)

    # Load sampling info
    df_sampling = pd.read_csv("/Users/miloszivkovic/GIT/dataserver/output_sampling_analysis.csv")  # assumes 'cluster' and 'count'
    # Load cluster performance metrics
    df_perf = pd.read_parquet("cluster_average_metrics.parquet")  # has 'cluster_stained', 'ssim', 'psnr', etc.

    sampling_category_vs_metrics_plotly(df_sampling, df_perf, output_path="/Volumes/KINGSTON/v6_2_comparison/sampling_category_vs_metrics_plotly.html")
    cluster_metrics_vertical_plot(df_sampling, df_perf,"/Volumes/KINGSTON/v6_2_comparison/cluster_metrics_combined_plot.html")
    plot_sample_count_metric_correlation_plotly(df_sampling, df_perf,
                                                output_path="/Volumes/KINGSTON/v6_2_comparison/sample_count_vs_metric_correlation.html")

