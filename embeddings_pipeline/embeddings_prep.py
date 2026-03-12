import glob
import os
import pickle

import pandas as pd
from tqdm import tqdm
from embeddings.embeddings_pipeline.utils import generate_unique_id


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