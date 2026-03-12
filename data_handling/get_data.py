import boto3
import argparse
import os
import configparser
import threading
from queue import Queue, Empty
from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm


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


def read_s3_paths_from_file(file_path):
    """Read S3 paths from a text file"""
    if file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            lines = f.readlines()
            return sorted([f"s3://gi-registration/{x.strip().split(',')[-1]}" for x in lines])
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
        paths = df.values.flatten().tolist()
        result = []
        for x in paths:
            if pd.notna(x):
                path = x.strip().split(',')[-1]
                if not path.startswith('s3://gi-registration/'):
                    path = f"s3://gi-registration/{path}"
                result.append(path)
        return sorted(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download files from S3 using multiple threads')
    parser.add_argument('--s3-paths', type=str, nargs='+',
                        help='S3 paths to process images from (space separated)',
                        # default='/root/minimodels/embeddings/datasets/smallbowel.txt')
                        default='/Volumes/KINGSTON/test_extraction_script/test_true.parquet')
    parser.add_argument('--dl-dir', type=str,
                        help='Directory to download files to',
                        default='/Volumes/KINGSTON/test_extraction_script/tiffs')
    parser.add_argument('--threads', type=int,
                        help='Number of download threads to use',
                        default=32)

    args = parser.parse_args()

    # Process input paths
    if os.path.isfile(args.s3_paths):
        s3_paths = read_s3_paths_from_file(args.s3_paths)
    else:
        s3_paths = args.s3_paths

    # Create downloader
    downloader = S3Downloader(
        s3_paths=s3_paths,
        dl_dir=args.dl_dir,
        num_workers=args.threads
    )

    # Build file list
    tiffs = downloader.build_s3_paths()
    print(f"Found {len(tiffs)} files to download")

    # Run the downloader
    print(f"Starting download with {args.threads} threads to {args.dl_dir}")
    downloader.run(tiffs)

    print("Download complete!")