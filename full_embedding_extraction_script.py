import argparse
import gc
import glob
import logging
import math
import os
import pickle
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from queue import Queue

import huggingface_hub
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ModelOptimizer:

    @staticmethod
    def create_optimized_models(batch_size=512):
        print("Loading and optimizing model...")

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tfgpu = gpus[len(gpus) - 1]
                tf.config.experimental.set_memory_growth(tfgpu, True)
                if len(gpus) > 1:
                    tf.config.set_visible_devices(tfgpu, 'GPU')
                print(f"GPU configured: {tfgpu}")
            except RuntimeError as e:
                print(f"GPU configuration failed: {e}")

        tf.config.optimizer.set_jit(True)
        huggingface_hub.login(token="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")

        print("Loading model from Hugging Face snapshot: google/path-foundation@main")
        snapshot_dir = huggingface_hub.snapshot_download(
            "google/path-foundation",
            revision="main",
            use_auth_token=True
        )

        embedding_model = keras.layers.TFSMLayer(snapshot_dir, call_endpoint="serving_default")

        @tf.function(jit_compile=True)
        def optimized_inference_main(inputs):
            batch_normalized = tf.cast(inputs, tf.float32) / 255.0
            return embedding_model(batch_normalized)

        @tf.function(jit_compile=True)
        def optimized_inference_flexible(inputs):
            batch_normalized = tf.cast(inputs, tf.float32) / 255.0
            return embedding_model(batch_normalized)

        print(f"Warming up main model with batch size {batch_size}...")
        dummy_batch = tf.zeros([batch_size, 224, 224, 3], dtype=tf.float32)
        _ = optimized_inference_main(dummy_batch)

        print("Warming up flexible model...")
        for size in [64, 128, 256]:
            if size < batch_size:
                dummy_small = tf.zeros([size, 224, 224, 3], dtype=tf.float32)
                _ = optimized_inference_flexible(dummy_small)

        print("Models warmed up successfully")
        return optimized_inference_main, optimized_inference_flexible


class BatchImageLoader:
    def __init__(self, num_workers=32, preprocessing_fcn="resize"):
        self.num_workers = num_workers
        self.embedding_types = {
            "none": load_image,
            "center_crop": center_crop,
            "top_left_crop": top_left_crop,
            "resize": resize_image,
        }
        self.preprocess_fcn = self.embedding_types.get(preprocessing_fcn, resize_image)

    def load_single_image(self, path):
        try:
            img_resized = self.preprocess_fcn(path)
            return np.array(img_resized, dtype=np.uint8)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros((224, 224, 3), dtype=np.uint8)

    def load_batch(self, image_paths):
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            images = list(executor.map(self.load_single_image, image_paths))
        return np.stack(images)


class HighPerformanceEmbeddingGenerator:

    def __init__(
            self,
            output_file,
            img_path,
            batch_size=512,
            save_every_n_batches=50,
            num_loader_workers=32,
            prefetch_size=2,
            preprocessing_fcn="none",
            s3_path="s3://gi-registration/slide-registration-production/tiles_nanozoomers360md/512/"
    ):
        self.batch_size = batch_size
        self.output_file = output_file
        self.save_every_n_batches = save_every_n_batches
        self.prefetch_size = prefetch_size
        self.img_path = img_path
        self.s3_path = s3_path

        self.model_main, self.model_flexible = ModelOptimizer.create_optimized_models(batch_size)
        self.image_loader = BatchImageLoader(num_workers=num_loader_workers, preprocessing_fcn=preprocessing_fcn)

        self.batch_queue = Queue(maxsize=prefetch_size)
        self.result_buffer = []
        self.stop_prefetch = threading.Event()

        self.seen_paths = self._load_existing_paths()

        self.batch_times = []
        self.inference_times = []

        print("Scanning for images...")
        if os.path.isdir(self.img_path):
            self.image_paths = get_all_tiff_files(self.img_path)
        else:
            print(f"Invalid directory: {self.img_path}")
            sys.exit(1)

        if not self.image_paths:
            print("No TIFF files found!")
            sys.exit(1)

        self.image_paths = self.image_paths[:1000]

    def _load_existing_paths(self):
        """Load already processed image paths"""
        if os.path.exists(self.output_file):
            try:
                df = pd.read_parquet(self.output_file)
                paths = set(df['full_file_path'].tolist())
                print(f"Found {len(paths)} existing embeddings")
                return paths
            except:
                print("Could not load existing file, starting fresh")
                return set()
        return set()

    def _get_s3_path(self, local_path):
        """Convert local path to S3 path"""
        if self.s3_path is None:
            return local_path
        else:
            return local_path.replace(self.img_path, self.s3_path)

    def _prefetch_worker(self, image_path_batches, remainder_batch=None):
        for batch_paths in image_path_batches:
            if self.stop_prefetch.is_set():
                break

            try:
                start_time = time.time()
                batch_images = self.image_loader.load_batch(batch_paths)
                load_time = time.time() - start_time

                batch_tensor = tf.constant(batch_images, dtype=tf.uint8)

                self.batch_queue.put((batch_tensor, batch_paths, load_time, 'main'))

            except Exception as e:
                print(f"Error in prefetch worker: {e}")
                continue

        if remainder_batch and not self.stop_prefetch.is_set():
            try:
                start_time = time.time()
                batch_images = self.image_loader.load_batch(remainder_batch)
                load_time = time.time() - start_time

                batch_tensor = tf.constant(batch_images, dtype=tf.uint8)
                self.batch_queue.put((batch_tensor, remainder_batch, load_time, 'remainder'))

            except Exception as e:
                print(f"Error processing remainder batch: {e}")

        self.batch_queue.put(None)

    def _save_results(self):
        """Save accumulated results to file"""
        if not self.result_buffer:
            return

        try:
            new_df = pd.DataFrame(self.result_buffer)

            if os.path.exists(self.output_file):
                existing_df = pd.read_parquet(self.output_file)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df

            combined_df.to_parquet(self.output_file, index=False)
            print(f"Saved {len(self.result_buffer)} new embeddings (total: {len(combined_df)})")

            self.result_buffer = []

        except Exception as e:
            print(f"Error saving results: {e}")

    def generate_embeddings(self):
        new_paths = [p for p in self.image_paths if self._get_s3_path(p) not in self.seen_paths]
        print(f"Processing {len(new_paths)} new images (skipping {len(self.seen_paths)} existing)")

        if not new_paths:
            print("No new images to process!")
            return pd.read_parquet(self.output_file) if os.path.exists(self.output_file) else pd.DataFrame()

        main_batches = []
        remainder_batch = None

        for i in range(0, len(new_paths), self.batch_size):
            batch = new_paths[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                main_batches.append(batch)
            else:
                remainder_batch = batch

        total_images = len(main_batches) * self.batch_size
        if remainder_batch:
            total_images += len(remainder_batch)
            print(
                f"Processing {len(main_batches)} main batches of {self.batch_size} + 1 remainder batch of {len(remainder_batch)} images")
        else:
            print(f"Processing {len(main_batches)} complete batches of {self.batch_size} images each")

        if not main_batches and not remainder_batch:
            print("No batches to process!")
            return pd.read_parquet(self.output_file) if os.path.exists(self.output_file) else pd.DataFrame()

        prefetch_thread = threading.Thread(target=self._prefetch_worker, args=(main_batches, remainder_batch))
        prefetch_thread.start()

        pbar = tqdm(total=total_images, desc="Processing", unit="img")
        batch_count = 0

        try:
            while True:
                batch_data = self.batch_queue.get(timeout=300)

                if batch_data is None:
                    break

                batch_tensor, batch_paths, load_time, batch_type = batch_data

                inference_start = time.time()

                try:
                    with tf.device('/GPU:0'):
                        if batch_type == 'main':
                            outputs = self.model_main(batch_tensor)
                        else:
                            outputs = self.model_flexible(batch_tensor)

                        embeddings = outputs['output_0']
                        embeddings_np = embeddings.numpy()

                except Exception as e:
                    print(f"Inference error for {batch_type} batch: {e}")
                    continue

                inference_time = time.time() - inference_start
                total_batch_time = load_time + inference_time

                for i, path in enumerate(batch_paths):
                    s3_path = self._get_s3_path(path)
                    embedding = embeddings_np[i].flatten()

                    self.result_buffer.append({
                        'full_file_path': s3_path,
                        'vector': embedding
                    })
                    self.seen_paths.add(s3_path)

                batch_count += 1
                self.batch_times.append(total_batch_time)
                self.inference_times.append(inference_time)

                pbar.update(len(batch_paths))

                if len(self.batch_times) >= 5:
                    recent_times = self.batch_times[-5:]
                    recent_inference = self.inference_times[-5:]

                    avg_total_time = np.mean(recent_times)
                    avg_inference_time = np.mean(recent_inference)
                    throughput = len(batch_paths) / avg_total_time
                    gpu_efficiency = (avg_inference_time / avg_total_time) * 100

                    pbar.set_postfix({
                        'imgs/s': f'{throughput:.1f}',
                        'inference_ms': f'{avg_inference_time * 1000:.1f}',
                        'gpu_eff': f'{gpu_efficiency:.1f}%',
                        'batch': batch_count,
                        'type': batch_type
                    })

                if batch_count % self.save_every_n_batches == 0:
                    self._save_results()

                del batch_tensor, embeddings, embeddings_np
                if batch_count % 10 == 0:
                    gc.collect()

        except Exception as e:
            print(f"Processing error: {e}")

        finally:
            self.stop_prefetch.set()
            prefetch_thread.join(timeout=60)
            self._save_results()
            pbar.close()

            if self.batch_times:
                avg_time = np.mean(self.batch_times)
                avg_inference = np.mean(self.inference_times)
                final_throughput = self.batch_size / avg_time
                final_efficiency = (avg_inference / avg_time) * 100

                print(f"\n=== Final Performance Stats ===")
                print(f"Average throughput: {final_throughput:.1f} imgs/sec")
                print(f"Average inference time: {avg_inference * 1000:.1f} ms")
                print(f"GPU efficiency: {final_efficiency:.1f}%")
                print(f"Total batches processed: {batch_count}")

        return pd.read_parquet(self.output_file) if os.path.exists(self.output_file) else pd.DataFrame()


class EmbeddingsConverter:
    def __init__(self, embedding_path):
        """
        Initialize the EmbeddingInsights class.

        Parameters:
            embedding_path (str): Path to the embedding file (.pkl, .parquet, or .csv)
        """
        self.embedding_path = embedding_path
        self.embeddings = {}  # Nested structure: {"unstained": [...], "stained-sr": [...]}
        self.lookup_table = {}
        self.embedding_pairs = {}
        self.flat_df = None  # will hold the flat DataFrame if loaded from parquet

    def get_flat_dataframe(self, path):
        if self.flat_df is None:
            self.load_embeddings_flat(path)
        return self.flat_df

    def load_embeddings(self, path):
        full_path = path if self.embedding_path is None else self.embedding_path
        with open(full_path, "rb") as f:
            self.embeddings = pickle.load(f)

    def load_embeddings_flat(self, path):
        full_path = path if self.embedding_path is None else self.embedding_path
        ext = os.path.splitext(full_path)[-1].lower()
        if ext == ".parquet":
            self.flat_df = self.load_flat_embeddings()
        elif ext == ".csv":
            self.flat_df = pd.read_csv(full_path)
        elif ext == ".pkl":
            with open(full_path, "rb") as f:
                self.flat_df = pickle.load(f)
        else:
            raise ValueError("Unsupported file extension. Use .pkl, .parquet, or .csv")

    def load_flat_embeddings(self, path=None):
        full_path = path if self.embedding_path is None else self.embedding_path
        ext = os.path.splitext(full_path)[-1].lower()
        if ext == ".parquet":
            return pd.read_parquet(full_path)
        elif ext == ".csv":
            return pd.read_csv(full_path)
        else:
            raise ValueError("Unsupported file extension. Use .parquet or .csv")

    def _create_slide_key(self, metadata):
        return "_".join([
            metadata["tissue_type"],
            metadata["block_id"],
            metadata["slice_id"],
            metadata["scan_date"],
            metadata["box_id"],
            metadata["filename"]
        ])

    def flatten_embeddings_to_dataframe(self, modalities=["unstained", "stained-sr", "stained", "inferred"]):
        """
        Flatten all embeddings into a DataFrame with metadata and vector columns.
        """
        records = []
        for modality in modalities:
            print(f"Flattening {modality} embeddings...")
            for sample in tqdm(self.embeddings.get(modality, [])):
                slide_key = self._create_slide_key(sample["metadata"])
                for emb in sample["embeddings"]:
                    base = {
                        "slide_key": slide_key,
                        "modality": modality,
                        "preprocessing": emb["metadata"]["preprocessing"],
                        "dimension": emb["metadata"].get("dimension"),
                        "model": emb["metadata"].get("model"),
                        "vector": emb.get("vector"),
                        "id": sample["id"],
                        **sample["metadata"]
                    }
                    records.append(base)
        return pd.DataFrame(records)

    def save_flat_embeddings(self, path, format="parquet"):
        """
        Save flattened embeddings to disk.
        Supported formats: parquet, csv
        """
        print("Flattening embeddings...")
        df = self.flatten_embeddings_to_dataframe()
        print("Saving flattened embeddings...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if format == "parquet":
            df.to_parquet(path, index=False)
        elif format == "csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError("Unsupported format. Use 'parquet' or 'csv'.")
        print(f"[INFO] Flattened embeddings saved to {path} ({format})")

    def create_chunks(self, tgt_dir, max_rows=200_000):
        input_files = glob.glob(f"{tgt_dir}/*.parquet")
        output_dir = f"{tgt_dir}/splits"
        os.makedirs(output_dir, exist_ok=True)

        # Mapping of label → condition
        conditions = {
            "stained_sr": "modality == 'stained-sr'",
            "stained": "modality == 'stained'",
            "unstained": "modality == 'unstained'",
            "inferred": "modality == 'inferred'",
        }
        MAX_ROWS_PER_FILE = max_rows

        for i, file in enumerate(input_files):
            print(f"Processing: {file}")
            df = pd.read_parquet(file)
            # df["unique_id"] = df.apply(generate_unique_id, axis=1)
            # ids = pd.read_parquet(f"{file.split('/')[-1].replace('_resize', '_unique_ids').replace('_pt2', '')}").squeeze().tolist()
            # df = df[~df["unique_id"].isin(ids)]
            for key, cond in conditions.items():
                subset = df.query(cond)
                if not subset.empty:
                    num_chunks = (len(subset) + MAX_ROWS_PER_FILE - 1) // MAX_ROWS_PER_FILE
                    for j in range(num_chunks):
                        start = j * MAX_ROWS_PER_FILE
                        end = start + MAX_ROWS_PER_FILE
                        chunk = subset.iloc[start:end].copy()
                        out_path = os.path.join(output_dir, f"{key}_part_{i}_chunk_{j}.parquet")
                        chunk.to_parquet(out_path, index=False)
                        print(f"  ➤ Saved {len(chunk)} rows to {out_path}")

            del df  # free memory

        print("Done splitting embeddings into subsets with parts")


def center_crop(image_path, crop_size=224):
    """Center crop the image"""
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    img = Image.open(image_path)
    width, height = img.size

    left = (width - crop_size[0]) // 2
    top = (height - crop_size[1]) // 2
    right = left + crop_size[0]
    bottom = top + crop_size[1]

    cropped_img = img.crop((left, top, right, bottom)).convert('RGB')
    return cropped_img

def resize_image(image_path, size=224):
    """Resize the image"""
    if isinstance(size, int):
        size = (size, size)
    img = Image.open(image_path)
    resized_img = img.resize(size, Image.LANCZOS).convert('RGB')
    return resized_img

def top_left_crop(image_path, size=224):
    """Crop from top left corner"""
    img = Image.open(image_path)
    cropped_img = img.crop((0, 0, size, size)).convert('RGB')
    return cropped_img

def load_image(image_path):
    img = Image.open(image_path)
    return img

def get_all_tiff_files(directory):
    """Get all TIFF files from directory"""
    pattern = os.path.join(directory, "**/*.tiff")
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} TIFF files")
    return files

def create_embedding_data(embedding_type, embedding_vector, is_torch_tensor=True):
    return {
        "metadata": {
            "preprocessing": embedding_type,
            "dimension": len(embedding_vector),
            "model": 'path_foundation',
        },
        "created_at": datetime.now().isoformat(),
        "vector": embedding_vector.float().cpu().numpy() if is_torch_tensor else embedding_vector,
    }

def format_embedding(img_path, tile_type, embedding_type, img_embedding_vector, is_torch_tensor=True):
    embeddings_list = []
    embeddings_list.append(create_embedding_data(embedding_type, img_embedding_vector, is_torch_tensor=is_torch_tensor))

    tissue_type, block_id, slice_id, scan_date, box_id, filename = img_path.split("/")[-6:]
    filename = filename.replace(".tiff", "").replace(".png", "").replace(".jpg", "").replace("-inferred", "")
    consolidated_record = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS,
                             f"{tissue_type}/{block_id}/{slice_id}/{scan_date}/{box_id}/{filename}/{tile_type}")),
        "created_at": datetime.now().isoformat(),
        "metadata": {
            "tissue_type": tissue_type,
            "block_id": block_id,
            "slice_id": slice_id,
            "scan_date": scan_date,
            "box_id": box_id,
            "filename": filename.replace("-stained", "").replace("-unstained", "").replace("-inferred", ""),
            "tile_type": tile_type,
        },
        "embeddings": embeddings_list
    }

    return consolidated_record

def load_or_initialize_data(pickle_path):
    if os.path.exists(pickle_path) and os.path.getsize(pickle_path) > 0:
        try:
            with open(pickle_path, "rb") as f:
                existing_data = pickle.load(f)
                print(f"Loaded existing data from {pickle_path}")
                return existing_data
        except Exception as e:
            print(f"Error loading existing pickle: {e}")
            print("Initializing new data structure")

    # Initialize new data structure if file doesn't exist or loading failed
    return {
        "unstained": [],
        "stained": [],
        "inferred": []
    }

def cleanup_old_backups(pickle_path, max_backups=3):
    try:
        # Find all backup files for this pickle file
        backup_pattern = f"{pickle_path}.bak.*"
        backup_files = glob.glob(backup_pattern)

        if len(backup_files) <= max_backups:
            return True  # No need to clean up if we have fewer backups than the limit

        # Sort backup files by creation time (newest first)
        backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Remove old backups, keeping only the max_backups most recent ones
        files_to_remove = backup_files[max_backups:]
        for old_backup in files_to_remove:
            try:
                os.remove(old_backup)
                logging.info(f"Removed old backup: {old_backup}")
            except Exception as e:
                logging.warning(f"Failed to remove old backup {old_backup}: {e}")

        return True
    except Exception as e:
        logging.error(f"Error during backup cleanup: {e}")
        return False

def save_manifest_files(data, pickle_path):
    try:
        # Get directory where pickle file is stored
        pickle_dir = os.path.dirname(pickle_path)

        # Create paths for manifest files
        processed_manifest_path = os.path.join(pickle_dir, "processed_manifest.txt")

        # Load existing processed samples if the file exists
        existing_processed_samples = set()
        if os.path.exists(processed_manifest_path):
            try:
                with open(processed_manifest_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            existing_processed_samples.add(line)
                print(f"Loaded {len(existing_processed_samples)} existing processed samples")
            except Exception as e:
                print(f"Warning: Could not read existing processed manifest: {e}")

        # Extract base paths of newly processed samples from data
        processed_samples = set()
        for item in data.get("unstained", []):
            # Extract base path from metadata
            metadata = item.get("metadata", {})
            tissue_type = metadata.get("tissue_type", "")
            block_id = metadata.get("block_id", "")
            slice_id = metadata.get("slice_id", "")
            scan_date = metadata.get("scan_date", "")
            box_id = metadata.get("box_id", "")
            filename = metadata.get("filename", "")
            base_path = f"{tissue_type}/{block_id}/{slice_id}/{scan_date}/{box_id}/{filename}"
            processed_samples.add(base_path)

        # Combine existing and new processed samples
        all_processed_samples = existing_processed_samples.union(processed_samples)

        # Write processed samples to file (append mode)
        with open(processed_manifest_path, "w") as f:
            for sample in sorted(all_processed_samples):
                f.write(f"{sample}\n")

        print(f"Saved {len(all_processed_samples)} processed samples to {processed_manifest_path}")
        return True

    except Exception as e:
        print(f"Error saving manifest files: {e}")
        logging.error(f"Error saving manifest files: {e}")
        return False

def save_data_incrementally(data, pickle_path, create_backup=True, max_backups=3, samples=None):

    backup_created = False
    backup_path = None

    # Optionally create a backup of the existing file
    if create_backup and os.path.exists(pickle_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{pickle_path}.bak.{timestamp}"
        try:
            import shutil
            shutil.copy2(pickle_path, backup_path)
            print(f"Created backup at {backup_path}")
            backup_created = True
        except Exception as e:
            print(f"Failed to create backup: {e}")

    # Save the updated data
    temp_path = f"{pickle_path}.temp"
    try:
        with open(temp_path, "wb") as f:
            pickle.dump(data, f)

        # Atomic replace to prevent data corruption if process is interrupted
        if os.path.exists(temp_path):
            os.replace(temp_path, pickle_path)
            print(f"Successfully saved data to {pickle_path}")

            # Save manifest files if samples list is provided and not empty
            # Save manifest files if samples list is provided and not empty
            if samples is not None and len(samples) > 0:
                save_manifest_files(data, pickle_path)
            # Clean up old backups after successful save
            if backup_created:
                if cleanup_old_backups(pickle_path, max_backups):
                    print(f"Cleaned up old backups, keeping {max_backups} most recent")
                else:
                    print("Warning: Failed to clean up some old backups")

            return True
    except Exception as e:
        print(f"Error saving data: {e}")
        # If backup was created but save failed, remove the backup to avoid confusion
        if backup_created and backup_path and os.path.exists(backup_path):
            try:
                os.remove(backup_path)
                print(f"Removed backup {backup_path} due to failed save")
            except Exception as be:
                print(f"Failed to remove backup after save error: {be}")

        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return False

def convert_embeddings_to_pickle(in_parquet, out_pkl, s3_path):
    SAVE_INTERVAL = 2048
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)

    all_data = load_or_initialize_data(out_pkl)
    df = pd.read_parquet(in_parquet)

    batch_size = 2048
    num_samples = len(df)
    num_batches = math.ceil(num_samples / batch_size)

    extracted_embeddings = []

    print(f"Total samples: {num_samples}, Total batches: {num_batches}")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_df = df.iloc[start_idx:end_idx]

        # Prepare lists for each batch
        embs_out_list = []
        bases_list = []

        for _, row in batch_df.iterrows():
            emb_out = row['vector']
            embs_out_list.append(emb_out)
            bases_list.append(row['full_file_path'])

        # Process the batch
        for emb_out, base in zip(embs_out_list, bases_list):
            if s3_path in base:
                base = base.replace(f"{s3_path}/", "")

            tile_type = base.split("-")[-1].split(".")[0]
            emb = format_embedding(base, tile_type, "resize", emb_out,
                                   is_torch_tensor=False)  # <- both same for now

            all_data[tile_type].append(emb)
            extracted_embeddings.append(base)

        # Progress reporting
        unique_count = len(set(extracted_embeddings))
        total_count = len(extracted_embeddings)
        print(f"Iteration {i + 1}/{num_batches}: Unique embeddings: {unique_count}, Total: {total_count}")

        if (i + 1) % SAVE_INTERVAL == 0:
            print(f"Saving data at iteration {i + 1}...")
            save_data_incrementally(all_data, out_pkl, create_backup=False, samples=extracted_embeddings)

    # Final save
    print("Performing final save...")
    save_data_incrementally(all_data, out_pkl, create_backup=False, samples=extracted_embeddings)

    print(f"\nDataset iteration completed. Total unique embeddings: {len(set(extracted_embeddings))}")
    print(f"Total records: "
          f"\nUnstained: {len(all_data.get('unstained', []))},"
          f"\nStained: {len(all_data.get('stained', []))}"
          f"\nInferred: {len(all_data.get('inferred', []))}"
          f"")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default="/Volumes/KINGSTON/data",
                        help="This should be the main directory that contains data downloaded  from s3 that "
                             "has the following structure, this is important to extract the metadata => "
                             "/Volumes/KINGSTON/data/{tissue_type}/{block_id}/{slice_id}/{scan_date}/{box_id}/*.tiff")
    parser.add_argument('--preprocessing_fcn', type=str, default="resize", help="Options: none, center_crop, top_left_crop, resize")
    parser.add_argument('--output', type=str, default="/Volumes/KINGSTON/embedding_tester/embeddings_extracted.parquet")
    parser.add_argument('--s3_path', type=str, default="s3://gi-registration/slide-registration-production/tiles_nanozoomers360md/512")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--save_every_n_batches', type=int, default=1)
    parser.add_argument('--num_loader_workers', type=int, default=32)

    args = parser.parse_args()

    tf.config.threading.set_intra_op_parallelism_threads(16)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    print("Initializing embedding generator...")
    generator = HighPerformanceEmbeddingGenerator(
        output_file=args.output,
        img_path=args.img_path,
        batch_size=args.batch_size,
        save_every_n_batches=args.save_every_n_batches,
        num_loader_workers=args.num_loader_workers,
        preprocessing_fcn=args.preprocessing_fcn,
        s3_path=args.s3_path
    )

    print("Starting embedding generation...")
    result = generator.generate_embeddings()
    print(f"\nProcessing complete! Total embeddings: {len(result)}")

    convert_embeddings_to_pickle(args.output, args.output.replace(".parquet", ".pickle"), args.s3_path)

    ec = EmbeddingsConverter(args.output.replace(".parquet", ".pickle"))
    ec.load_embeddings(None)
    save_path = args.output.replace(".parquet", "_flat.parquet")
    ec.save_flat_embeddings(save_path, format="parquet")
