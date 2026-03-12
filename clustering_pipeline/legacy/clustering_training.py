import os
import glob
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uuid
from collections import defaultdict, Counter
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------
# 1. STRATIFIED SAMPLING FROM LOCAL PARQUET FILES
# ---------------------------------------------------------
def create_stratified_dataset(
    parquet_paths,
    stratify_field="tissue_type",
    per_class=25000,
    max_samples=100000,
    batch_size=1_000_000,
    stats=None,
    id_col="unique_id",
    random_state=42
):
    rng = np.random.RandomState(random_state)
    print("PASS 1 — Counting values per class per file...")
    per_file_counts = []
    total_per_class = Counter()

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

    classes = [c for c in total_per_class.keys() if total_per_class[c] > 0 and not str(c).isdigit()]
    print("Classes:", classes)

    # Compute per-class targets
    raw_targets = {c: min(per_class, total_per_class[c]) for c in classes}
    total_target = sum(raw_targets.values())

    if total_target > max_samples:
        scale = max_samples / float(total_target)
        floored = {c: int(np.floor(raw_targets[c] * scale)) for c in classes}
        remaining = max_samples - sum(floored.values())

        # allocate the remainders
        remainders = {c: raw_targets[c]*scale - floored[c] for c in classes}
        for c in sorted(classes, key=lambda x: remainders[x], reverse=True)[:remaining]:
            floored[c] += 1

        targets = floored
    else:
        targets = raw_targets

    print("Computed per-class target quotas:", targets)

    # ------------- PER-FILE QUOTAS DISTRIBUTION ----------
    quotas = [defaultdict(int) for _ in parquet_paths]

    for c in classes:
        total_count = total_per_class[c]
        if total_count == 0 or targets[c] == 0:
            continue

        proportional = [(i, per_file_counts[i][c] / total_count * targets[c])
                        for i in range(len(parquet_paths))]

        floors = {i: int(np.floor(val)) for i, val in proportional}
        assigned = sum(floors.values())
        need = targets[c] - assigned

        # assign the remaining based on largest fractional remainder
        rema = sorted([(i, proportional[i][1] - floors[i]) for i in range(len(parquet_paths))],
                      key=lambda x: x[1], reverse=True)

        for i, _ in rema[:need]:
            floors[i] += 1

        for i, q in floors.items():
            if q > 0:
                quotas[i][c] = q

    # ---------------- PASS 2: SAMPLING --------------------
    print("\nPASS 2 — Sampling according to quotas...")

    collected = []
    seen_ids = set()

    for file_idx, path in enumerate(parquet_paths):
        file_quota = quotas[file_idx]

        if not file_quota:
            continue

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            df = df[df["modality"] == "stained"]

            if stratify_field not in df.columns:
                continue

            if id_col in df.columns:
                df = df.drop_duplicates(subset=[id_col])
                df = df[~df[id_col].isin(seen_ids)]

            pending = {c: q for c, q in file_quota.items() if q > 0}
            if not pending:
                break

            for c, q in list(pending.items()):
                grp = df[df[stratify_field] == c]
                if grp.empty:
                    continue

                take_n = min(q, len(grp))
                sample = grp.sample(n=take_n, random_state=random_state)

                collected.append(sample)
                if id_col in sample.columns:
                    seen_ids.update(sample[id_col].tolist())

                file_quota[c] -= take_n

        print("Completed file:", path)

    sampled_df = pd.concat(collected, ignore_index=True)
    print(f"\nFinal sampled dataset size: {len(sampled_df)}")

    if stats:
        for col in stats:
            if col in sampled_df.columns:
                print(f"\nStats: {col}")
                print(sampled_df[col].value_counts())

    return sampled_df


def fit_kmeans_model(df, k, batch_size=100_000, suffix=None, out_dir="./models"):
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"kmeans_model_k{k}"
    if suffix:
        out_name += f"_{suffix}"

    mbk = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    total = len(df)

    done = 0
    for start in range(0, total, batch_size):
        batch = df.iloc[start:start + batch_size]
        vectors = np.vstack(batch["vector"].tolist())
        mbk.partial_fit(vectors)
        done += len(batch)
        print(f"Progress: {done}/{total}")

    model_path = os.path.join(out_dir, out_name + ".joblib")
    joblib.dump(mbk, model_path)
    print("Saved model to:", model_path)
    return model_path


def predict_kmeans_model(
    model_path,
    parquet_paths,
    out_path,
    batch_size=10000
):
    model = joblib.load(model_path)

    if os.path.exists(out_path):
        print("Resuming from:", out_path)
        existing = pd.read_parquet(out_path)
        predicted_ids = set(existing["unique_id"].tolist())
    else:
        existing = pd.DataFrame(columns=["unique_id", "slide_key", "cluster"])
        predicted_ids = set()

    for parquet_path in parquet_paths:
        print("Predicting:", parquet_path)

        df = pd.read_parquet(parquet_path)
        df = df[df["modality"] == "stained"]

        df["unique_id"] = df.apply(generate_unique_id, axis=1)

        remaining = df[~df["unique_id"].isin(predicted_ids)]

        for start in range(0, len(remaining), batch_size):
            batch = remaining.iloc[start:start + batch_size]
            vectors = np.vstack(batch["vector"].to_numpy())
            labels = model.predict(vectors)

            batch_df = pd.DataFrame({
                "unique_id": batch["unique_id"].tolist(),
                "slide_key": batch["slide_key"].tolist(),
                "cluster": labels
            })

            existing = pd.concat([existing, batch_df], ignore_index=True)
            existing.to_parquet(out_path, index=False)

    print("Saved final predictions to:", out_path)
    return existing


def generate_unique_id(row):
    key = f"{row['tissue_type']}/{row['block_id']}/{row['slice_id']}/{row['scan_date']}/{row['box_id']}/{row['filename']}/{row['tile_type']}/{row['preprocessing']}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


if __name__ == "__main__":
    PARQUET_FILES_PATH = glob.glob("/Volumes/KINGSTON/extracted_embeddings_flat/*.parquet")
    OUT_DIR = "/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/outputs"
    OUT_DIR_MODEL = os.path.join(OUT_DIR, "models")
    OUT_DIR_DATA = os.path.join(OUT_DIR, "data")
    OUT_DIR_RESULTS = os.path.join(OUT_DIR, "results")
    SAMPLE_NB = 250_000

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR_MODEL, exist_ok=True)
    os.makedirs(OUT_DIR_DATA, exist_ok=True)
    os.makedirs(OUT_DIR_RESULTS, exist_ok=True)

    df = create_stratified_dataset(
        PARQUET_FILES_PATH,
        stratify_field="tissue_type",
        per_class=10_000,
        max_samples=SAMPLE_NB
    )
    df.to_parquet(os.path.join(OUT_DIR_DATA, "sample_stratified.parquet"), index=False)

    df = pd.read_parquet(os.path.join(OUT_DIR_DATA, "sample_stratified.parquet"))
    fit_kmeans_model(df, k=112, suffix=str(SAMPLE_NB), out_dir=OUT_DIR_MODEL)

    # predict_kmeans_model(
    #     model_path=f"{OUT_DIR_MODEL}/kmeans_model_k112_100000.joblib",
    #     parquet_paths=PARQUET_FILES_PATH,
    #     out_path=f"{OUT_DIR_RESULTS}/clustering_results_all_k112.parquet"
    # )

