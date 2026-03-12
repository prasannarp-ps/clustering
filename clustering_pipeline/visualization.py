import configparser
import os
import shutil
import tempfile
import threading
from pathlib import Path
from queue import Queue, Empty

import boto3
import cv2
import duckdb
import numpy as np
from tqdm import tqdm

from . import config


class S3Downloader:
    def __init__(self, s3_paths, dl_dir, num_workers=16):
        self.s3_paths = s3_paths
        self.dl_dir = Path(dl_dir)
        self.num_workers = num_workers
        self.shutdown = False

        # Create queues
        self.download_queue = Queue()
        self.failed_queue = Queue()

        # Setup tracking
        self.processed_files = set()
        self.download_threads = []
        self.pbar = None

        # Initialize S3 client
        self._setup_s3_client()

    def _setup_s3_client(self):
        """Initialize S3 client from r2.conf"""
        # Load R2 configuration
        config = configparser.ConfigParser()
        # config.read('/root/minimodels/embeddings/r2.conf')
        config.read('/Users/miloszivkovic/GIT/minimodels/r2.conf')

        self.r2_config = {
            'endpoint_url': config['r2']['endpoint_url'],
            'access_key': config['r2']['access_key'],
            'secret_key': config['r2']['secret_key'],
            'bucket_name': config['r2']['bucket_name'],
            'artefact_folder_name': config['r2']['artefact_folder_name']
        }

        # We don't create the client here - each worker will create its own

    def build_s3_paths(self, suffixess=None):
        """Parse S3 paths and build file list"""
        tiff_files = []
        # suffixes = ["-stained-sr.tiff"]
        if suffixess is None:
            suffixes = ["-stained.tiff"]
        else:
            suffixes = suffixess
        print("Building file list...")
        # Extract bucket and prefix from s3_path
        for s3_path in self.s3_paths:
            if not s3_path.startswith('s3://'):
                raise ValueError(f"Invalid S3 path format: {s3_path}")

            bucket = s3_path.split('/')[2]
            prefix = '/'.join(s3_path.split('/')[3:])
            key = s3_path.replace("s3://", "").replace(f"{bucket}/", "")
            for suffix in suffixes:
                if suffix not in key:
                    key += suffix
                tiff_files.append({
                    'bucket': bucket,
                    'key': key
                })

        return tiff_files

    def build_s3_paths_v2(self):
        tiff_files = []
        for s3_path in self.s3_paths:
            if not s3_path.startswith('s3://'):
                raise ValueError(f"Invalid S3 path format: {s3_path}")

            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
            tiff_files.append({'bucket': bucket, 'key': key})
        return tiff_files

    def _download_worker(self, worker_id):
        """Worker thread for downloading files"""
        # Create a dedicated S3 client for this worker
        s3_client = boto3.client(
            's3',
            endpoint_url=self.r2_config['endpoint_url'],
            aws_access_key_id=self.r2_config['access_key'],
            aws_secret_access_key=self.r2_config['secret_key']
        )

        worker_name = f"Worker-{worker_id}"
        files_downloaded = 0

        while not self.shutdown:
            try:
                # Get a file to download
                try:
                    file_info = self.download_queue.get(timeout=1.0)
                except Empty:
                    continue

                if file_info is None:  # Poison pill
                    break

                # local_path = self.dl_dir / file_info["key"].replace("slide-registration-production/tiles/512/", "")
                # local_path = self.dl_dir / file_info["key"].replace("slide-registration-production/tiles_nanozoomers360md/512/", "")

                # local_path = self.dl_dir / file_info["key"].replace("slide-registration/tiles/nanozoomers360md/40x/fixed/he/512/", "")
                local_path = self.dl_dir / file_info["key"]
                os.makedirs(local_path.parent, exist_ok=True)

                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)

                    # Skip if file already exists
                    if not os.path.exists(local_path):
                        # Download the file
                        s3_client.download_file(
                            file_info["bucket"],
                            file_info["key"],
                            str(local_path)
                        )
                        files_downloaded += 1

                        # Update progress bar
                        if self.pbar:
                            self.pbar.update(1)
                    else:
                        # File already exists, just update the progress bar
                        if self.pbar:
                            self.pbar.update(1)

                except Exception as e:
                    self.failed_queue.put({
                        'key': file_info["key"],
                        'error': str(e)
                    })
                    if self.pbar:
                        self.pbar.update(1)

                # Mark as done
                self.download_queue.task_done()

            except Exception as e:
                print(f"{worker_name} error: {str(e)}")
                continue

        print(f"{worker_name} shutting down. Downloaded {files_downloaded} files.")

    def start_workers(self):
        """Start download worker threads"""
        print(f"Starting {self.num_workers} download workers...")

        # Create and start worker threads
        for i in range(self.num_workers):
            thread = threading.Thread(
                target=self._download_worker,
                args=(i,),
                daemon=True
            )
            thread.name = f"DownloadWorker-{i}"
            self.download_threads.append(thread)
            thread.start()

    def queue_files(self, file_list):
        """Queue files for download"""
        for file_info in file_list:
            self.download_queue.put(file_info)

    def wait_for_completion(self):
        """Wait for all downloads to complete"""
        self.download_queue.join()

    def shutdown_workers(self):
        """Shut down all worker threads"""
        self.shutdown = True

        # Send poison pills to all workers
        for _ in range(self.num_workers):
            self.download_queue.put(None)

        # Wait for all workers to finish
        for thread in self.download_threads:
            thread.join(timeout=2.0)

        print("All worker threads shut down.")

    def report_failures(self):
        """Report any failed downloads"""
        failed_count = self.failed_queue.qsize()
        if failed_count > 0:
            print(f"\n{failed_count} files failed to download:")
            for _ in range(min(10, failed_count)):
                failure = self.failed_queue.get()
                print(f"  {failure['key']}: {failure['error']}")

            if failed_count > 10:
                print(f"  ... and {failed_count - 10} more failures")

    def run(self, file_list):
        """Run the downloader with the given file list"""
        # Create output directory if it doesn't exist
        os.makedirs(self.dl_dir, exist_ok=True)

        # Start worker threads
        self.start_workers()

        # Initialize progress bar
        total_files = len(file_list)
        self.pbar = tqdm(total=total_files, desc="Downloading", unit="file")

        # Queue files for download
        self.queue_files(file_list)

        try:
            # Wait for all downloads to complete
            self.wait_for_completion()
        except KeyboardInterrupt:
            print("\nInterrupted by user. Shutting down...")
        finally:
            # Close progress bar
            if self.pbar:
                self.pbar.close()

            # Shutdown workers
            self.shutdown_workers()

            # Report failures
            self.report_failures()



def s3_path_builder(slide_key):
    parts = slide_key.replace("tile_", "").split("_")
    tissue, block, slice_id, scan_date, box, x = parts
    return (
        f"s3://gi-registration/slide-registration/tiles_origin/nanozoomers360md/40x/frozen/he/512/"
        f"{tissue}/{block}/{slice_id}/{scan_date}/{box}/tile_{x}-stained.tiff"
    )


def print_prediction_stats(con, pred_table="predictions"):
    """
    Prints:
      - Total number of samples in predictions table
      - Cluster counts sorted by cluster ID
    """

    print("\n📌 Checking prediction database statistics...\n")

    # --- TOTAL COUNT ---
    total = con.execute(f"""
        SELECT COUNT(*) AS total_samples
        FROM pred_db.{pred_table}
    """).fetchone()[0]

    print(f"🔢 Total samples in predictions table: {total:,}\n")

    # --- PER-CLUSTER COUNTS ---
    cluster_df = con.execute(f"""
        SELECT cluster, COUNT(*) AS count
        FROM pred_db.{pred_table}
        GROUP BY cluster
        ORDER BY cluster
    """).df()

    print("📊 Samples per cluster:")
    print(cluster_df.to_string(index=False))

    return total, cluster_df


def visualize_cluster_samples_duckdb( embed_db_path, pred_db_path, extended_db_path,
                                      table_name="predictions", out_dir="./cluster_grids",
                                      rows=2, cols=5, resize_dim=(256, 256), font_scale=0.4,
                                      font_thickness=1, padding=2, cluster_id=None,
                                      num_grids=1, k=None, s3_path_builder=None,
                                      downloader_workers=8, min_tissue_pct=20):
    """
    Visualize sample tiles for each cluster.

    If ext_db.extended_data.tissue_percentage does not exist, this function
    falls back to:
      - NOT filtering by tissue percentage
      - Still working for grid visualization
    """

    if s3_path_builder is None:
        raise ValueError("s3_path_builder(tile_key) must be provided")

    os.makedirs(out_dir, exist_ok=True)

    con = duckdb.connect(embed_db_path, read_only=False)

    pred_sql = pred_db_path.replace("'", "''")
    ext_sql = extended_db_path.replace("'", "''")

    con.execute(f"ATTACH '{pred_sql}' AS pred_db;")
    con.execute(f"ATTACH '{ext_sql}' AS ext_db;")

    # Optional: print basic stats
    try:
        print_prediction_stats(con, table_name)
    except NameError:
        # If you don't have this helper defined, you can remove this block
        pass

    # -----------------------------------------------------------------
    # Detect whether ext_db.extended_data.tissue_percentage exists
    # -----------------------------------------------------------------
    has_tissue_pct = False
    try:
        con.execute("SELECT tissue_percentage FROM ext_db.extended_data LIMIT 1")
        has_tissue_pct = True
    except duckdb.Error:
        has_tissue_pct = False
        print(
            "WARNING: ext_db.extended_data.tissue_percentage not found; "
            "visualization will ignore min_tissue_pct and use all tiles."
        )

    use_tissue_filter = has_tissue_pct and (min_tissue_pct is not None)

    # -----------------------------------------------------------------
    # Determine clusters
    # -----------------------------------------------------------------
    print("Querying clusters...")

    if cluster_id is not None:
        clusters = [cluster_id]
        out_dir = f"{out_dir}/cluster_{cluster_id}_grids"
        os.makedirs(out_dir, exist_ok=True)
    else:
        clusters = [
            c[0]
            for c in con.execute(
                f"SELECT DISTINCT cluster FROM pred_db.{table_name} ORDER BY cluster"
            ).fetchall()
        ]
        if k is not None:
            out_dir = f"{out_dir}/{k}"
            os.makedirs(out_dir, exist_ok=True)

    print(f"Found {len(clusters)} clusters.")

    images_per_grid = rows * cols

    # -----------------------------------------------------------------
    # Process each cluster
    # -----------------------------------------------------------------
    for cl_id in clusters:
        print(f"\n=== Cluster {cl_id} ===")

        # -------------------------------------------------------------
        # Count tiles satisfying the tissue filter (if applicable)
        # -------------------------------------------------------------
        if use_tissue_filter:
            clean_count_query = f"""
                SELECT COUNT(*)
                FROM pred_db.{table_name} p
                JOIN (
                    SELECT {config.norm_tile_key_sql('tile_key')} AS norm_key,
                           MAX(tissue_percentage) AS tissue_percentage
                    FROM ext_db.extended_data
                    GROUP BY {config.norm_tile_key_sql('tile_key')}
                ) x ON {config.norm_tile_key_sql('p.tile_key')} = x.norm_key
                WHERE p.cluster = {cl_id}
                  AND x.tissue_percentage >= {min_tissue_pct}
            """
        else:
            clean_count_query = f"""
                SELECT COUNT(*)
                FROM pred_db.{table_name} p
                WHERE p.cluster = {cl_id}
            """

        clean_count = con.execute(clean_count_query).fetchone()[0]

        if clean_count == 0:
            print(
                f"No tiles matching filters for cluster {cl_id}, skipping."
            )
            continue

        print(f"Clean samples available: {clean_count}")

        # -----------------------------------------------------------------
        # Generate multiple grids
        # -----------------------------------------------------------------
        for g in range(num_grids):
            out_png = f"{out_dir}/cluster_{cl_id}_grid_{g+1}.png"

            if os.path.exists(out_png):
                print(f"Grid {g+1} exists, skipping.")
                continue

            print(f"Sampling grid {g+1}/{num_grids}...")

            limit_n = min(images_per_grid, clean_count)

            # -------------------------------------------------------------
            # SAMPLE TILES (filtered if tissue info exists)
            # -------------------------------------------------------------
            if use_tissue_filter:
                sample_query = f"""
                    SELECT p.tile_key, p.box_id, p.short_box_name
                    FROM pred_db.{table_name} p
                    JOIN (
                        SELECT {config.norm_tile_key_sql('tile_key')} AS norm_key,
                               MAX(tissue_percentage) AS tissue_percentage
                        FROM ext_db.extended_data
                        GROUP BY {config.norm_tile_key_sql('tile_key')}
                    ) x ON {config.norm_tile_key_sql('p.tile_key')} = x.norm_key
                    WHERE p.cluster = {cl_id}
                      AND x.tissue_percentage >= {min_tissue_pct}
                    ORDER BY RANDOM()
                    LIMIT {limit_n}
                """
            else:
                sample_query = f"""
                    SELECT p.tile_key, p.box_id, p.short_box_name
                    FROM pred_db.{table_name} p
                    WHERE p.cluster = {cl_id}
                    ORDER BY RANDOM()
                    LIMIT {limit_n}
                """

            df = con.execute(sample_query).df()

            if df.empty:
                print(f"Cluster {cl_id} grid {g+1}: no samples.")
                continue

            # Fix tile_key naming
            tile_keys = [
                tk.replace(f"_{bid}_", f"_{sbn}_") if f"_{bid}_" in tk else tk
                for tk, bid, sbn in zip(
                    df["tile_key"].astype(str),
                    df["box_id"].astype(str),
                    df["short_box_name"].astype(str),
                )
            ]

            # -------------------------------------------------------------
            # Download images
            # -------------------------------------------------------------
            s3_paths = [s3_path_builder(tk) for tk in tile_keys]

            temp_dir = tempfile.mkdtemp()
            print(f"Downloading {len(s3_paths)} images...")

            downloader = S3Downloader(
                s3_paths=s3_paths,
                dl_dir=temp_dir,
                num_workers=downloader_workers,
            )
            downloader.run(downloader.build_s3_paths_v2())

            tiff_files = sorted(Path(temp_dir).rglob("*.tiff"))
            if not tiff_files:
                print(f"No TIFFs downloaded for cluster {cl_id} grid {g+1}")
                shutil.rmtree(temp_dir)
                continue
            # -------------------------------------------------------------
            # Load images
            # -------------------------------------------------------------
            loaded = []
            for p in tiff_files[:images_per_grid]:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                loaded.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Ensure exactly images_per_grid images (pad with black tiles)
            while len(loaded) < images_per_grid:
                blank = np.zeros(
                    (resize_dim[1], resize_dim[0], 3), dtype=np.uint8
                )
                loaded.append(blank)
            if len(loaded) > images_per_grid:
                loaded = loaded[:images_per_grid]

            # Ensure tile_keys length matches images_per_grid
            if len(tile_keys) < images_per_grid:
                tile_keys += [""] * (images_per_grid - len(tile_keys))
            elif len(tile_keys) > images_per_grid:
                tile_keys = tile_keys[:images_per_grid]

            # -------------------------------------------------------------
            # Label and resize
            # -------------------------------------------------------------
            labeled = []
            font = cv2.FONT_HERSHEY_SIMPLEX

            for img, label in zip(loaded, tile_keys):
                img_r = cv2.resize(img, resize_dim)
                canvas = img_r.copy()

                (cw, _), _ = cv2.getTextSize(
                    "A", font, font_scale, font_thickness
                )
                max_chars = max(1, (resize_dim[0] - 10) // cw)
                lines = [
                    label[i : i + max_chars]
                    for i in range(0, len(label), max_chars)
                ]

                y = 20
                dy = int(20 * font_scale) + 5
                for line in lines[:3]:
                    cv2.putText(
                        canvas,
                        line,
                        (5, y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        font_thickness,
                        cv2.LINE_AA,
                    )
                    y += dy

                canvas = cv2.copyMakeBorder(
                    canvas,
                    padding,
                    padding,
                    padding,
                    padding,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
                labeled.append(canvas)

            # -------------------------------------------------------------
            # Build grid (labeled now guaranteed to have images_per_grid tiles)
            # -------------------------------------------------------------
            idx = 0
            rows_list = []
            for _ in range(rows):
                rows_list.append(np.hstack(labeled[idx : idx + cols]))
                idx += cols

            grid = np.vstack(rows_list)


            cv2.imwrite(out_png, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"Saved: {out_png}")

            shutil.rmtree(temp_dir)

    print("Done generating cluster grids.")

# if __name__ == "__main__":
    # visualize_cluster_samples_duckdb(
    #     embed_db_path="/Volumes/KINGSTON/embeddings_v7_purple/db/embeddings.db",
    #     pred_db_path="/Volumes/KINGSTON/embeddings_v7_purple/db/predictions.db",
    #     extended_db_path="/Volumes/KINGSTON/embeddings_v7_purple/db/extended_data.db",
    #     table_name="predictions",
    #     out_dir="/Volumes/KINGSTON/embeddings_v7_purple/results/cluster_grids",
    #     rows=10,
    #     cols=10,
    #     s3_path_builder=s3_path_builder,
    # )
    # for cid in [96, 74, 23, 65, 79, 78, 72, 31, 69, 1]:
    #     visualize_cluster_samples_duckdb(
    #         embed_db_path="/Volumes/KINGSTON/embeddings_v7_purple/db/embeddings.db",
    #         pred_db_path="/Volumes/KINGSTON/embeddings_v7_purple/db/predictions.db",
    #         extended_db_path="/Volumes/KINGSTON/embeddings_v7_purple/db/extended_data.db",
    #         table_name="predictions",
    #         out_dir="/Volumes/KINGSTON/embeddings_v7_purple/results/cluster_grids_above_1000",
    #         cluster_id=cid,
    #         rows=10,
    #         cols=10,
    #         s3_path_builder=s3_path_builder,
    #     )
#     # exit()
#     for cid in range(-2, 112):
#         visualize_cluster_samples_duckdb(
#             embed_db_path="/Volumes/KINGSTON/embeddings_v7/db/embeddings.db",
#             pred_db_path="/Volumes/KINGSTON/embeddings_v7/db/predictions_above_20_extended.db",
#             extended_db_path="/Volumes/KINGSTON/embeddings_v7/db/extended_data.db",
#             table_name="predictions",
#             out_dir="/Volumes/KINGSTON/embeddings_v7/results/cluster_grids_new_above_20_detailed",
#             cluster_id=cid,
#             rows=10,
#             cols=10,
#             s3_path_builder=s3_path_builder,
#             num_grids=5,
#             min_tissue_pct=0
#         )
#     # visualize_cluster_samples_duckdb(
#     #     embed_db_path="/Volumes/KINGSTON/embeddings_v7/db/embeddings.db",
#     #     pred_db_path="/Volumes/KINGSTON/embeddings_v7/db/predictions_extended.db",
#     #     extended_db_path="/Volumes/KINGSTON/embeddings_v7/db/extended_data.db",
#     #     table_name="predictions",
#     #     out_dir="/Volumes/KINGSTON/embeddings_v7/results/cluster_grids_extended_multi",
#     #     rows=10,
#     #     cols=10,
#     #     num_grids=5,
#     #     s3_path_builder=s3_path_builder,
#     #     min_tissue_pct=0
#     # )
#
#     #
#     # visualize_cluster_samples_s3(
#     #     clustering_result_path="/Volumes/KINGSTON/embeddings_v7/results/clustering_results_all_k112.parquet",
#     #     out_dir="/Volumes/KINGSTON/embeddings_v7/cluster_grids",
#     #     tile_key="tile_key",
#     #     rows=10,
#     #     cols=10,
#     #     s3_path_builder=s3_path_builder,
#     # )
#
#     #
#     # visualize_cluster_samples_s3_parallel(
#     #     clustering_result_path="/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_all_resize/data/clustering_results_all_k112.parquet",
#     #     out_dir="cluster_grids",
#     #     rows=5,
#     #     cols=5,
#     #     s3_path_builder=s3_path_builder,
#     #     processes=16,  # use 16 parallel workers
#     #     downloader_workers=8  # each worker downloads in parallel too
#     # )