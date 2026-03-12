import os
import pandas as pd
import numpy as np
import joblib
import shutil
import glob
import uuid

def predict_from_local_clustering_model(
        model_path,
        parquet_paths,
        output_dir,
        result_file_name="clustering_results.parquet",
        batch_size=10000,
        filter_modality="stained",
        generate_unique_id_fn=None
    ):

    result_path = os.path.join(output_dir, result_file_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = joblib.load(model_path)

    all_predictions = []

    for parquet_path in parquet_paths:
        print(f"Running on: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        df = df[df["modality"] == filter_modality]

        if generate_unique_id_fn:
            df["unique_id"] = df.apply(generate_unique_id_fn, axis=1)
        else:
            raise ValueError("❗ generate_unique_id_fn must be provided")

        all_ids = df["unique_id"].tolist()

        # Resume from existing results if available
        if os.path.exists(result_path):
            print(f"🔄 Found existing results. Resuming from disk...")
            existing_df = pd.read_parquet(result_path)
            predicted_ids = set(existing_df["unique_id"].tolist())
        else:
            print(f"🆕 No previous results found. Starting fresh...")
            existing_df = pd.DataFrame(columns=["unique_id", "slide_key", "cluster"])
            predicted_ids = set()

        # Filter already predicted
        remaining_df = df[~df["unique_id"].isin(predicted_ids)]
        print(f"Remaining samples: {len(remaining_df)} / {len(all_ids)}")

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
            print(f"Batch {batch_counter}: saved {len(existing_df)} total preds")

        all_predictions.append(existing_df)
        del df

        print(f"\nDone with file. Saved to: {result_path}")

    final_df = pd.concat(all_predictions, ignore_index=True)
    return final_df

def unique_id_fn(row):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, row["slide_key"]))

if __name__ == "__main__":
    out = predict_from_local_clustering_model(
        model_path="/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_resize/results/models_zoo/kmeans_model_k112_50000.joblib",
        parquet_paths=glob.glob("/Volumes/KINGSTON/extracted_embeddings_flat/*.parquet")[:2],
        output_dir="outputs/clustering_predictions",
        generate_unique_id_fn=unique_id_fn
    )