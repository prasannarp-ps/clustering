import glob
import logging
import math
import os
import pickle
import uuid
from datetime import datetime
import pandas as pd


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
        f"{STAINED_TYPE}": []
    }

def cleanup_old_backups(pickle_path, max_backups=3):
    """
    Clean up old backup files, keeping only the specified number of most recent backups.

    Args:
        pickle_path: The base path of the pickle file
        max_backups: Maximum number of backup files to keep

    Returns:
        bool: True if cleanup was successful, False otherwise
    """
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
    """
    Save two manifest files containing processed and unprocessed samples

    Args:
        data: The current data containing processed samples
        pickle_path: The path to the pickle file (manifest files will be saved in the same directory)
        samples: Full list of samples being processed in this run

    Returns:
        bool: True if saving was successful, False otherwise
    """
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


if __name__ == "__main__":
    PICKLE_PATH = "/Volumes/KINGSTON/embedding_tester/embeddings_extracted.pkl"
    STAINED_TYPE = "inferred"
    SAVE_INTERVAL = 2048 * 10
    s3_path = "s3://gi-registration/slide-registration-production/tiles_nanozoomers360md/512/"
    os.makedirs(os.path.dirname(PICKLE_PATH), exist_ok=True)

    all_data = load_or_initialize_data(PICKLE_PATH)
    parquet_path = "/Volumes/KINGSTON/embedding_tester/embeddings_extracted.parquet"
    df = pd.read_parquet(parquet_path)

    df = df[df["full_file_path"].str.contains(STAINED_TYPE)]

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
            # Assume only 'vector' is present, adapt if you also have emb_in
            emb_out = row['vector']
            embs_out_list.append(emb_out)
            bases_list.append(row['full_file_path'])

        # Process the batch
        for emb_out, base in zip(embs_out_list, bases_list):
            if s3_path in base:
                base = base.replace(f"{s3_path}/", "")
            tile_type = base.split("-")[-1].split(".")[0]
            # unst_emb = format_embedding(base, "unstained", "resize", emb_out)
            stain_emb = format_embedding(base, tile_type, "resize", emb_out, is_torch_tensor=False)  # <- both same for now

            # all_data["unstained"].append(unst_emb)
            all_data[tile_type].append(stain_emb)
            extracted_embeddings.append(base)

        # Progress reporting
        unique_count = len(set(extracted_embeddings))
        total_count = len(extracted_embeddings)
        print(f"Iteration {i + 1}/{num_batches}: Unique embeddings: {unique_count}, Total: {total_count}")

        if (i + 1) % SAVE_INTERVAL == 0:
            print(f"Saving data at iteration {i + 1}...")
            save_data_incrementally(all_data, PICKLE_PATH, create_backup=False, samples=extracted_embeddings)

    # Final save
    print("Performing final save...")
    save_data_incrementally(all_data, PICKLE_PATH, create_backup=False, samples=extracted_embeddings)

    print(f"\nDataset iteration completed. Total unique embeddings: {len(set(extracted_embeddings))}")
    print(f"Total records: "
          f"\nUnstained: {len(all_data.get('unstained', []))},"
          f"\nStained: {len(all_data.get('stained', []))}"
          f"\nInferred: {len(all_data.get('inferred', []))}"
          f"")
