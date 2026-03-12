import os
import cv2
import boto3
import threading
from queue import Queue, Empty
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Optional
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import csv
import configparser
import torch
import traceback
from async_funcs import LogItem
from multiprocessing.pool import ThreadPool
import random


class S3ImageDataset(Dataset):
    def __init__(
        self, 
        s3_paths: List[str],
        local_temp_dir: str = "temp_images",
        batch_size: int = 6,
        buffer_batches: int = 10,
        image_size: tuple = (512, 512),
        log_queue: Optional[Queue] = None,
        num_download_workers: int = 6,
        scan: bool = False,
    ):
        # First initialize all instance variables
        self.log_queue = log_queue
        self._shutdown = False
        self.s3_paths = s3_paths
        self.batch_size = batch_size
        self.buffer_batches = buffer_batches
        self.image_size = image_size
        self.local_temp_dir = Path(local_temp_dir)
        self.local_temp_dir.mkdir(exist_ok=True)

        print(f"\n[S3ImageDataset] Initializing dataset")
        # print(f"[S3ImageDataset] S3 paths: {s3_paths}")
        print(f"[S3ImageDataset] Local temp dir: {local_temp_dir}")
        print(f"[S3ImageDataset] Batch size: {batch_size}")
        print(f"[S3ImageDataset] Buffer batches: {buffer_batches}")
        print(f"[S3ImageDataset] Image size: {image_size}")
        print(f"[S3ImageDataset] Download workers: {num_download_workers}")

        # Initialize total buffer files
        self.total_buffer_files = self.batch_size * self.buffer_batches
        
        # Initialize queues and buffers before logging
        self.download_queue = Queue()
        self.processed_files: Set[str] = set()
        self.failed_files = Queue()
        self.cleanup_queue = Queue()
        
        # Now we can safely log
        self._log_event("init", "startup", {
            "s3_paths": s3_paths,
            "local_temp_dir": local_temp_dir,
            "batch_size": batch_size,
            "buffer_batches": buffer_batches,
            "image_size": image_size,
            "process_id": os.getpid()
        })

        # Add paths for logging files
        self.total_images_file = "logs/total_images.txt"

        self.r2_config = None
        
        # Initialize S3 client
        print(f"[S3ImageDataset] Setting up S3 client...")
        self._setup_s3_client()
        
        self._log_event("init", "s3_client_setup", {
            "endpoint_url": self.r2_config['endpoint_url'],
            "bucket_name": self.r2_config['bucket_name'],
            "artefact_folder": self.r2_config['artefact_folder_name']
        })

        if scan:
            # Get initial file list
            print(f"[S3ImageDataset] Scanning S3 for .tiff files...")
            self._log_event("scan", "start", {
                "message": "Scanning S3 for .tiff files..."
            })
            self.file_list = self._get_all_tiff_files()
        else:
            # Get initial file list
            print(f"[S3ImageDataset] Not Scanning S3 for .tiff files...")
            self._log_event("scan", "start", {
                "message": "Not Scanning S3 for .tiff files..."
            })
            self.file_list = self._get_all_tiff_files2()
        self.file_list_for_download = self.file_list.copy()
        print(f"[S3ImageDataset] Found {len(self.file_list)} files")
        self._log_event("scan", "complete", {
            "files_found": len(self.file_list)
        })

        # Log total images to file
        with open(self.total_images_file, 'w') as f:
            for file_info in self.file_list:
                f.write(f"{file_info['key']}\n")
        
        # Initialize multiple download workers
        print(f"[S3ImageDataset] Starting {num_download_workers} download workers...")
        self.download_threads = []
        for i in range(num_download_workers):
            thread = threading.Thread(target=self._download_worker, daemon=True)
            thread.name = f"DownloadWorker-{i}"
            self.download_threads.append(thread)
            thread.start()

        # Start cleanup and error logging workers
        print(f"[S3ImageDataset] Starting background workers...")
        self._start_background_workers()

        self._log_event("init", "workers_started", {
            "num_download_workers": num_download_workers,
            "background_workers": ["error_logger", "cleanup"]
        })
        
        # Initialize file ready tracking
        self.file_ready_events = {}
        self.file_ready_lock = threading.Lock()
        self.files_in_use = set()
        self.files_in_use_lock = threading.Lock()

        # Initialize workers and buffer
        self._initialize_buffer()
        
        print(f"[S3ImageDataset] Initialization complete. Ready to process {len(self.file_list)} images.")

    def _setup_s3_client(self):
        """Initialize S3 client from r2.conf"""
        # Load R2 configuration
        config = configparser.ConfigParser()
        config.read('/root/minimodels/embeddings/r2.conf')
        # config.read('/Users/miloszivkovic/GIT/minimodels/r2.conf')
        self.r2_config = {
            'endpoint_url': config['r2']['endpoint_url'],
            'access_key': config['r2']['access_key'],
            'secret_key': config['r2']['secret_key'],
            'bucket_name': config['r2']['bucket_name'],
            'artefact_folder_name': config['r2']['artefact_folder_name']
        }

        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.r2_config['endpoint_url'],
            aws_access_key_id=self.r2_config['access_key'],
            aws_secret_access_key=self.r2_config['secret_key']
        )

    def _get_all_tiff_files2(self) -> List[str]:
        """
        Recursively scan S3 bucket for *-stained.tiff files
        Returns a list of S3 keys for matching files
        """
        try:
            tiff_files = []
            suffixes = ["-stained.tiff"]

            # Extract bucket and prefix from s3_path
            for s3_path in tqdm(self.s3_paths, total=len(self.s3_paths)):
                if not s3_path.startswith('s3://'):
                    raise ValueError(f"Invalid S3 path format: {s3_path}")
                
                bucket = s3_path.split('/')[2]
                prefix = '/'.join(s3_path.split('/')[3:])
                
                self._log_event("scan", "info", {
                    "message": f"Scanning path: {s3_path}",
                    "bucket": bucket,
                    "prefix": prefix
                })
                key = s3_path.replace("s3://", "").replace(f"{bucket}/", "")
                for suffix in suffixes:
                    key += suffix
                    tiff_files.append({
                        'bucket': bucket,
                        'key': key,
                        'size': 1
                    })

            
            if not tiff_files:
                self._log_event("scan", "warning", {
                    "message": f"No *-stained.tiff files found in any of the provided paths",
                    "paths": self.s3_paths
                })
            else:
                self._log_event("scan", "success", {
                    "total_files_found": len(tiff_files),
                    "paths_scanned": len(self.s3_paths)
                })
            
            # Shuffle the files to mix content from different paths
            random.shuffle(tiff_files)
            
            return tiff_files
            
        except Exception as e:
            self._log_event("scan", "error", {
                "message": str(e),
                "traceback": traceback.format_exc()
            })
            raise

    def _get_all_tiff_files(self) -> List[str]:
        """
        Recursively scan S3 bucket for *-stained.tiff files
        Returns a list of S3 keys for matching files
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            # Use paginator for handling large directories
            tiff_files = []

            # Extract bucket and prefix from s3_path
            for s3_path in tqdm(self.s3_paths, total=len(self.s3_paths)):
                if not s3_path.startswith('s3://'):
                    raise ValueError(f"Invalid S3 path format: {s3_path}")

                bucket = s3_path.split('/')[2]
                prefix = '/'.join(s3_path.split('/')[3:])

                self._log_event("scan", "info", {
                    "message": f"Scanning path: {s3_path}",
                    "bucket": bucket,
                    "prefix": prefix
                })

                path_files = []
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    if 'Contents' not in page:
                        continue

                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('-stained.tiff'):
                            path_files.append({
                                'bucket': bucket,
                                'key': key,
                                'size': obj['Size']
                            })

                self._log_event("scan", "info", {
                    "path": s3_path,
                    "files_found": len(path_files)
                })

                tiff_files.extend(path_files)

            if not tiff_files:
                self._log_event("scan", "warning", {
                    "message": f"No *-stained.tiff files found in any of the provided paths",
                    "paths": self.s3_paths
                })
            else:
                self._log_event("scan", "success", {
                    "total_files_found": len(tiff_files),
                    "paths_scanned": len(self.s3_paths)
                })

            # Shuffle the files to mix content from different paths
            random.shuffle(tiff_files)

            return tiff_files

        except Exception as e:
            self._log_event("scan", "error", {
                "message": str(e),
                "traceback": traceback.format_exc()
            })
            raise

    def _start_background_workers(self):
        """Initialize and start background threads for downloading and error logging"""
        # Download workers are now started in __init__
        
        # Start error logger thread
        self.error_thread = threading.Thread(target=self._error_logger_worker, daemon=True)
        self.error_thread.start()

        # Start cleanup worker thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        self._log_event("scan", "info", {
            "message": "Background workers started"
        })

    def _initialize_buffer(self):
        """Initialize buffer with first set of images"""
        # Queue initial batches (buffer_batches * batch_size files)
        initial_files = self.file_list[:self.total_buffer_files]
        print(f"\n[S3ImageDataset] Initializing buffer with {len(initial_files)} files...")
        
        self._log_event("buffer", "info", {
            "message": f"Queueing initial {len(initial_files)} files for download",
            "total_files": len(self.file_list),
            "paths": self.s3_paths
        })
        
        for file_info in tqdm(initial_files, desc="Queueing initial files", unit="file"):
            if file_info['key'] not in self.processed_files:
                self.download_queue.put(file_info)
                self.file_list_for_download.remove(file_info)  # Remove after queuing
        
        print(f"[S3ImageDataset] Waiting for initial buffer to fill...")
        self.download_queue.join()
        print(f"[S3ImageDataset] Initial buffer filled with {len(os.listdir(self.local_temp_dir))} files")
        
        self._log_event("buffer", "info", {
            "message": "Initial buffer filled",
            "remaining_files": len(self.file_list_for_download)
        })

    def _ensure_buffer_filled(self):
        """Ensure buffer has enough files queued for download"""
        with self.file_ready_lock:  # Add lock here
            files_in_buffer = len([f for f in os.listdir(self.local_temp_dir) 
                                 if not f.startswith('.')])  # Ignore hidden files
                                 
            files_in_download = self.download_queue.qsize()
            total_buffered = files_in_buffer + files_in_download
            
            # Log current buffer state
            self._log_event("buffer", "status", {
                "files_in_buffer": files_in_buffer,
                "files_in_download": files_in_download,
                "total_buffered": total_buffered,
                "target_buffer": self.total_buffer_files
            })
            
            # Don't queue more files if we're at or above the buffer limit
            if total_buffered >= self.total_buffer_files:
                return
            
            # Calculate how many more files we can add while respecting the limit
            available_slots = self.total_buffer_files - total_buffered
            files_to_queue = min(available_slots, len(self.file_list_for_download))
            
            if files_to_queue > 0:
                print(f"[S3ImageDataset] Refilling buffer: {files_in_buffer} in buffer, {files_in_download} downloading, adding {files_to_queue} more")
            
            queued_count = 0
            for _ in range(files_to_queue):
                if not self.file_list_for_download:
                    break
                
                file_info = self.file_list_for_download.pop(0)
                if file_info['key'] not in self.processed_files:
                    self.download_queue.put(file_info)
                    queued_count += 1
                
            if queued_count > 0:
                print(f"[S3ImageDataset] Added {queued_count} files to download queue. {len(self.file_list_for_download)} files remaining")
            
            self._log_event("buffer", "status", {
                "files_ready": files_in_buffer,
                "files_downloading": files_in_download,
                "new_files_queued": queued_count,
                "total_processed": len(self.processed_files),
                "remaining_files": len(self.file_list_for_download),
                "buffer_capacity": self.total_buffer_files,
            })

    def _download_worker(self):
        """Background worker for downloading images from S3"""
        worker_id = threading.get_ident()
        worker_name = threading.current_thread().name
        total_download_time = 0
        files_downloaded = 0
        
        print(f"[{worker_name}] Started download worker (thread ID: {worker_id})")
        
        self._log_event("worker", "download_worker_start", {
            "thread_id": worker_id
        })

        while not self._shutdown:
            try:
                file_info = self.download_queue.get()
                if file_info is None:  # Poison pill
                    print(f"[{worker_name}] Received shutdown signal, stopping...")
                    break
                
                local_path = self.local_temp_dir / Path(file_info['key']).name
                
                try:
                    start_time = datetime.now()
                    self.s3_client.download_file(
                        file_info['bucket'],
                        file_info['key'],
                        str(local_path)
                    )
                    download_time = (datetime.now() - start_time).total_seconds()
                    total_download_time += download_time
                    files_downloaded += 1
                    avg_download_time = total_download_time / files_downloaded
                    
                    # Only mark as processed after successful download
                    if file_info['key'] in self.processed_files:
                        if local_path.exists():
                            local_path.unlink()  # Remove duplicate download
                    else:
                        # Print status every 10 files
                        if files_downloaded % 10 == 0:
                            print(f"[{worker_name}] Downloaded {files_downloaded} files, avg time: {avg_download_time:.2f}s")
                        
                        self._log_event("download", "success", {
                            "worker_id": worker_id,
                            "file": file_info['key'],
                            "download_time_seconds": download_time,
                            "avg_download_time": avg_download_time,
                            "speed_mbps": (file_info['size'] / 1024 / 1024) / download_time if download_time > 0 else 0
                        })
                        
                        # Signal that file is ready
                        with self.file_ready_lock:
                            if file_info['key'] in self.file_ready_events:
                                self.file_ready_events[file_info['key']].set()

                except Exception as e:
                    print(f"[{worker_name}] Error downloading {file_info['key']}: {str(e)}")
                    self._log_event("download", "error", {
                        "worker_id": worker_id,
                        "file": file_info['key'],
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
                    self.failed_files.put({
                        'time': datetime.now().isoformat(),
                        'file': file_info['key'],
                        'error': f"Download error: {str(e)}"
                    })
                
                self.download_queue.task_done()
                
            except Exception as e:
                print(f"[{worker_name}] Worker error: {str(e)}")
                self._log_event("worker", "error", {
                    "worker_id": worker_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                continue

    def _error_logger_worker(self):
        """Background worker for logging errors to CSV"""
        csv_path = 'logs/failed.csv'
        csv_exists = Path(csv_path).exists()
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['time', 'file', 'error'])
            if not csv_exists:
                writer.writeheader()
            
            while not self._shutdown:
                try:
                    error_info = self.failed_files.get()
                    if error_info is None:  # Poison pill
                        break
                    
                    writer.writerow(error_info)
                    f.flush()  # Ensure immediate write
                    self.failed_files.task_done()
                    
                except Exception as e:
                    self._log_event("error_logger", "error", {
                        "message": f"Error logger worker error: {str(e)}"
                    })
                    continue

    def mark_file_processed(self, key):
        """Mark a file as done processing and ready for cleanup"""
        with self.files_in_use_lock:
            if key in self.files_in_use:
                self.files_in_use.remove(key)
                self.cleanup_queue.put([key])
        self._ensure_buffer_filled()

    def _cleanup_worker(self):
        """Background worker for cleaning up processed images"""
        while not self._shutdown:
            try:
                # Get batch of keys to cleanup
                try:
                    keys = self.cleanup_queue.get(timeout=1.0)
                except Empty:
                    continue

                if keys is None:  # Poison pill
                    break

                # Only delete files that are not in use
                keys_to_delete = []
                with self.files_in_use_lock:
                    for key in keys:
                        if key not in self.files_in_use:
                            keys_to_delete.append(key)

                # Delete local files and mark as processed
                for key in keys_to_delete:
                    local_path = self.local_temp_dir / Path(key).name
                    try:
                        if local_path.exists():
                            local_path.unlink()
                            self._log_event("cleanup", "success", {
                                "file": str(local_path)
                            })
                    except Exception as e:
                        self._log_event("cleanup", "error", {
                            "file": str(local_path),
                            "error": str(e)
                        })

                    # Update tracking sets/lists
                    self.processed_files.add(key)

                # Trigger buffer refill after cleanup
                self._ensure_buffer_filled()

                self.cleanup_queue.task_done()

            except Exception as e:
                self._log_event("cleanup", "error", {
                    "message": f"Cleanup worker error: {str(e)}"
                })
                continue

    def __len__(self):
        """Return total number of files in dataset"""
        return len(self.file_list)

    def _prepare_image(self, img):
        """Default image preparation"""
        if img.shape[2] == 3:  # If BGR/RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != self.image_size:
            img = cv2.resize(img, self.image_size)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, idx):
        start_time = datetime.now()
        file_info = self.file_list[idx]
        key = file_info['key']

        # Print status every 10 items
        if idx % 10 == 0:
            print(f"[S3ImageDataset] Processing item {idx}/{len(self.file_list)}: {Path(key).name}")

        self._log_event("getitem", "request", {
            "idx": idx,
            "file": key,
            "buffer_size": len(os.listdir(self.local_temp_dir)),
            "queue_size": self.download_queue.qsize()
        })

        # Add timeout to prevent infinite waiting
        max_wait_time = 60  # seconds
        wait_start = datetime.now()
        local_path = self.local_temp_dir / Path(key).name

        # Create event for this file if needed
        with self.file_ready_lock:
            if key not in self.file_ready_events:
                self.file_ready_events[key] = threading.Event()

        # Wait for file with timeout
        if not local_path.exists():
            print(f"[S3ImageDataset] Waiting for file to download: {Path(key).name}")
            if not self.file_ready_events[key].wait(timeout=max_wait_time):
                wait_time = (datetime.now() - wait_start).total_seconds()
                print(f"[S3ImageDataset] WARNING: Timeout waiting for download: {Path(key).name}")

                # Check if the key actually exists in S3
                try:
                    # Use head_object to check if file exists without downloading it
                    bucket = file_info['bucket']
                    self.s3_client.head_object(Bucket=bucket, Key=key)

                    # If we get here, file exists but download timed out
                    self._log_event("getitem", "error", {
                        "file": key,
                        "error": "Timeout waiting for download, but file exists in S3",
                        "wait_time": wait_time
                    })
                    raise TimeoutError(f"Timeout waiting for file download: {key}")

                except self.s3_client.exceptions.ClientError as e:
                    # If 404 error, file doesn't exist
                    if e.response['Error']['Code'] == '404':
                        print(f"[S3ImageDataset] INFO: File {Path(key).name} not found in S3, skipping")
                        self._log_event("getitem", "skip", {
                            "file": key,
                            "reason": "File not found in S3 after timeout",
                            "wait_time": wait_time
                        })
                        # Add to processed files to avoid trying again
                        self.processed_files.add(key)

                        # Try to get next available file instead
                        new_idx = (idx + 1) % len(self.file_list)
                        print(f"[S3ImageDataset] Trying next file at index {new_idx}")
                        return self.__getitem__(new_idx)
                    else:
                        # Other S3 error
                        self._log_event("getitem", "error", {
                            "file": key,
                            "error": f"S3 error: {str(e)}",
                            "wait_time": wait_time
                        })
                        raise

        # Clean up event
        with self.file_ready_lock:
            self.file_ready_events.pop(key, None)

        wait_time = (datetime.now() - wait_start).total_seconds()

        try:
            process_start = datetime.now()
            img = cv2.imread(str(local_path))
            if img is None:
                print(f"[S3ImageDataset] ERROR: Failed to read image: {local_path}")
                self._log_event("getitem", "error", {
                    "file": key,
                    "error": "Failed to read image"
                })
                raise ValueError(f"Failed to read image: {local_path}")

            img = self._prepare_image(img)
            processing_time = (datetime.now() - process_start).total_seconds()
            total_time = (datetime.now() - start_time).total_seconds()

            with self.files_in_use_lock:
                self.files_in_use.add(key)
            self._ensure_buffer_filled()

            self._log_event("getitem", "complete", {
                "file": key,
                "wait_time_seconds": wait_time,
                "processing_time_seconds": processing_time,
                "total_time_seconds": total_time,
                "remaining_buffer": len(os.listdir(self.local_temp_dir)),
                "remaining_queue": self.download_queue.qsize()
            })

            local_path_str = str(self.local_temp_dir / Path(key).name)
            return img, key, local_path_str

        except Exception as e:
            print(f"[S3ImageDataset] ERROR processing {Path(key).name}: {str(e)}")
            self._log_event("getitem", "error", {
                "file": key,
                "error": str(e)
            })
            self.failed_files.put({
                'time': datetime.now().isoformat(),
                'file': key,
                'error': f"Processing error: {str(e)}"
            })

            raise

    def cleanup(self):
        """Cleanup resources and stop workers"""
        print("\n[S3ImageDataset] Starting cleanup process...")
        self._log_event("cleanup", "start", {
            "message": "Starting cleanup process"
        })
        
        # Signal shutdown to all workers
        self._shutdown = True
        
        # Clear queues first to prevent blocking
        while not self.download_queue.empty():
            try:
                self.download_queue.get_nowait()
                self.download_queue.task_done()
            except Empty:
                break

        # Stop all workers with poison pills
        print(f"[S3ImageDataset] Stopping {len(self.download_threads)} download workers...")
        for _ in self.download_threads:
            self.download_queue.put(None)
        self.failed_files.put(None)
        self.cleanup_queue.put(None)

        # Use a thread pool to delete files in parallel
        if self.local_temp_dir.exists():
            files = list(self.local_temp_dir.iterdir())
            print(f"[S3ImageDataset] Cleaning up {len(files)} temporary files...")
            with ThreadPool() as pool:
                pool.map(self._safe_delete, files)
        
        # Join threads with timeout
        print("[S3ImageDataset] Waiting for worker threads to finish...")
        for thread in self.download_threads:
            thread.join(timeout=1.0)
        self.error_thread.join(timeout=1.0)
        self.cleanup_thread.join(timeout=1.0)

        print("[S3ImageDataset] Cleanup complete")
        self._log_event("cleanup", "complete", {
            "message": "Cleanup process finished"
        })

    def _safe_delete(self, file_path):
        """Safely delete a single file"""
        try:
            file_path.unlink(missing_ok=True)
        except Exception as e:
            self._log_event("cleanup", "error", {
                "message": f"Error deleting {file_path}: {e}"
            })

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()

    def _log_event(self, source, event_type, details):
        """Add local logging method"""
        if self.log_queue:
            # Add basic context to all log events
            details.update({
                "process_id": os.getpid(),
            })
            
            self.log_queue.put(LogItem(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                source=f"S3ImageDataset.{source}",
                event_type=event_type,
                details=details
            ))
