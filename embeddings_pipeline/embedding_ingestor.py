import argparse
import glob
import json
import logging
import os
import re
from datetime import datetime
from pprint import pprint
from uuid import UUID
import random

import numpy as np
import pandas as pd
import urllib3
from elasticsearch import helpers
from tqdm import tqdm

from embeddings.embeddings_pipeline.embeddings_prep import EmbeddingsConverter
from embeddings.embeddings_pipeline.utils import ElasticEmbeddingFetcher, ElasticsearchFilterBuilder, generate_unique_id

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def create_logger(log_file):
    logger = logging.getLogger("es_ingest")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    if os.path.exists(log_file):
        os.remove(log_file)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def is_valid_doc(doc):
    try:
        # Validate _id and unique_id (UUID format)
        UUID(doc["unique_id"], version=5)

        # block_id: lowercase letters and numbers only
        if not re.fullmatch(r"[a-z0-9]+", doc["block_id"]):
            print(f"Invalid block_id: {doc['block_id']}")
            return False

        # box_id: two numbers separated by "-"
        if not re.fullmatch(r"\d+-\d+", doc["box_id"]):
            print(f"Invalid box_id: {doc['box_id']}")
            return False

        # created_at: ISO format datetime
        datetime.fromisoformat(doc["created_at"])

        # filename: must start with "tile_" followed by digits-digits-digits-digits
        if not re.fullmatch(r"tile_\d+-\d+-\d+-\d+", doc["filename"]):
            print(f"Invalid filename: {doc['filename']}")
            return False

        # modality, tile_type: allowed values
        if doc["modality"] not in {"stained", "unstained", "stained-sr"}:
            print(f"Invalid modality: {doc['modality']}")
            return False
        if doc["tile_type"] not in {"stained", "unstained", "stained-sr"}:
            print(f"Invalid tile_type: {doc['tile_type']}")
            return False

        # preprocessing: allowed values
        if doc["preprocessing"] not in {"resize", "center_crop"}:
            print(f"Invalid preprocessing: {doc['preprocessing']}")
            return False

        # scan_date: YYYY-MM-DD
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", doc["scan_date"]):
            print(f"Invalid scan_date: {doc['scan_date']}")
            return False

        # slice_id: positive integer (as string)
        if not str(doc["slice_id"]).isdigit():
            print(f"Invalid slice_id: {doc['slice_id']}")
            return False

        # tissue_type: only letters
        if not re.fullmatch(r"[a-zA-Z]+", doc["tissue_type"]):
            print(f"Invalid tissue_type: {doc['tissue_type']}")
            return False

        # slide_key: must follow the format {tissue_type}_{block_id}_{slice_id}_{scan_date}_{box_id}_{filename}
        expected_key = f"{doc['tissue_type']}_{doc['block_id']}_{doc['slice_id']}_{doc['scan_date']}_{doc['box_id']}_{doc['filename']}"
        if doc["slide_key"] != expected_key:
            print(f"Expected slide_key: {expected_key}, but got: {doc['slide_key']}")
            return False

        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False


class EmbeddingIngestor:
    def __init__(self, es_fetcher: ElasticEmbeddingFetcher, log_file="ingest.log", vector_dims=384,
                 conflict_mode="skip_if_exists", recreate_index=False, dry_run=False, filter_query=None):

        self.logger = create_logger(log_file)
        self.fetcher = es_fetcher
        self.index = self.fetcher.index
        self.batch_size = self.fetcher.batch_size
        self.vector_dims = vector_dims
        self.conflict_mode = conflict_mode
        self.recreate_index = recreate_index
        self.default_dataset = "train"
        self.default_dataset_version = "v4"
        self.dry_run = dry_run
        if filter_query:
            tissue_type = None
            for i in range(0, len(filter_query['bool']['must'])):
                if filter_query['bool']['must'][i]['term'].get('tissue_type'):
                    tissue_type = filter_query['bool']['must'][i]['term']['tissue_type']
                    break
            name = f"{tissue_type}_unique_ids.parquet" if tissue_type else "unique_ids.parquet"
            if not os.path.exists(name):
                ids = self.fetcher.fetch_and_save_ids(filter_query)
                ids.to_parquet(name, index=False)

        ids = glob.glob(f"*unique_ids.parquet")
        self.ids = []
        for x in ids:
            self.ids.extend(pd.read_parquet(x).squeeze().tolist())
        print(f"Loaded {len(self.ids)} unique_ids")

    def _recreate_index(self):
        if self.fetcher.es.indices.exists(index=self.index):
            self.fetcher.es.indices.delete(index=self.index)
            self.logger.info(f"Deleted existing index: {self.index}")
        self._create_index()

    def _create_index(self):
        if not self.fetcher.es.indices.exists(index=self.index):
            self.fetcher.es.indices.create(
                index=self.index,
                mappings={
                    "properties": {
                        "block_id": {"type": "keyword"},
                        "box_id": {"type": "keyword"},
                        "created_at": {"type": "date"},
                        "vector": {
                            "type": "dense_vector",
                            "dims": self.vector_dims,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {"type": "int8_hnsw", "m": 16, "ef_construction": 100}
                        },
                        "filename": {"type": "keyword"},
                        "model": {"type": "keyword"},
                        "preprocessing": {"type": "keyword"},
                        "scan_date": {"type": "keyword"},
                        "slice_id": {"type": "keyword"},
                        "slide_key": {"type": "keyword"},
                        "tile_type": {"type": "keyword"},
                        "tissue_type": {"type": "keyword"},
                        "unique_id": {"type": "keyword"},
                        "cluster": {"type": "keyword"},
                        "dataset": {"type": "keyword"},
                        "dataset_version": {"type": "keyword"},
                    }
                }
            )
            self.logger.info(f"Created index: {self.index} with mapping")
        else:
            self.logger.info(f"Index {self.index} already exists")

    def prepare_docs(self, df):
        actions = []
        skip_check = self.conflict_mode in {"skip_if_exists", "compare_and_update"}
        existing_docs = {}

        if skip_check:
            try:
                mget_body = [{"_id": _id} for _id in df["unique_id"].tolist()]
                results = self.fetcher.es.mget(index=self.index, body={"docs": mget_body})
                existing_docs = {
                    doc["_id"]: doc["_source"]
                    for doc in results["docs"]
                    if doc.get("found")
                }
            except Exception as e:
                self.logger.warning(f"Could not pre-fetch existing document IDs. Falling back to per-doc checks. {e}")
                if self.conflict_mode == "skip_if_exists":
                    existing_docs = self.ids  # Trigger fallback path
                else:
                    existing_docs = None

        if self.conflict_mode == "skip_if_exists" and existing_docs is not None:
            df = df[~df["unique_id"].isin(existing_docs)]
        elif self.conflict_mode == "compare_and_update" and existing_docs is not None:
            def is_changed(row):
                doc = row.to_dict()
                doc["created_at"] = datetime.utcnow().isoformat()
                doc["vector"] = doc.pop("vector", [])
                doc.setdefault("dataset", self.default_dataset)
                doc.setdefault("dataset_version", self.default_dataset_version)
                return existing_docs.get(row["unique_id"]) != doc

            df = df[df.apply(is_changed, axis=1)]

        print(f"Processing {len(df)} documents")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing documents"):
            unique_id = row["unique_id"]
            doc = row.to_dict()
            doc["created_at"] = datetime.utcnow().isoformat()
            doc["vector"] = doc.pop("vector", [])

            doc.setdefault("dataset", self.default_dataset)
            doc.setdefault("dataset_version", self.default_dataset_version)

            if not is_valid_doc(doc):
                self.logger.warning(f"Document failed validation and was skipped: {unique_id}")
                continue

            actions.append({
                "_index": self.index,
                "_id": unique_id,
                "_source": doc
            })

        return actions

    def batch_upload(self, actions):
        flag_file = True
        for i in tqdm(range(0, len(actions), self.batch_size), desc="Uploading batches"):
            batch = actions[i:i + self.batch_size]
            try:
                success, failed = helpers.bulk(self.fetcher.es, batch, raise_on_error=False, stats_only=True)
                self.logger.info(f"Uploaded batch {i // self.batch_size + 1}: Success: {success}, Failed: {failed}")
                if failed > 0:
                    flag_file = False
            except Exception as e:
                self.logger.error(f"Batch {i // self.batch_size + 1} failed: {e}")
        return flag_file

    def ingest_single(self, file_path):
        self.logger.info(f"Ingesting single sample from: {file_path}")
        df = pd.read_parquet(file_path)
        row = df.iloc[0].copy()
        row["unique_id"] = generate_unique_id(row)
        row["vector"] = list(row["vector"]) if isinstance(row["vector"], (np.ndarray, list)) else []
        doc = row.to_dict()
        doc["created_at"] = datetime.utcnow().isoformat()
        doc["vector"] = doc.pop("vector", [])

        if self.dry_run:
            print(json.dumps(doc, indent=2))
            return

        try:
            self.fetcher.es.index(index=self.index, id=row["unique_id"], document=doc)
            self.logger.info(f"Successfully inserted doc with ID {row['unique_id']}")
        except Exception as e:
            self.logger.error(f"Failed to insert doc: {e}")

    def ingest_batch(self, file_path):
        self.logger.info(f"Ingesting from file: {file_path}")
        df = pd.read_parquet(file_path)
        df["unique_id"] = df.apply(generate_unique_id, axis=1)
        df["vector"] = df["vector"].apply(lambda x: list(x) if isinstance(x, (np.ndarray, list)) else [])
        docs = self.prepare_docs(df)

        if self.dry_run:
            self.logger.info("Dry run: showing first 5 documents")
            for doc in docs[:5]:
                doc["_source"].pop("vector", None)
                pprint(doc)
            return

        self.logger.info(f"Prepared {len(docs)} docs for ingestion")
        if len(docs) != 0:
            success = self.batch_upload(docs)
            if success:
                self.logger.info("Ingestion complete")
                os.remove(file_path)
                self.logger.info(f"{file_path} removed because it was ingested")
                pd.DataFrame({"unique_id": [x.get("_id") for x in docs]}).to_parquet(
                    f"{random.randint(0, 5000000)}_unique_ids.parquet", index=False)
            else:
                self.logger.info(f"{file_path} had problems and not all docs are ingested")
        else:
            self.logger.info("No documents to ingest")

    def ingest_directory(self, directory_path):
        parquet_files = glob.glob(f"{directory_path}/*.parquet")
        self.logger.info(f"Found {len(parquet_files)} files in {directory_path}")
        for path in list(sorted(parquet_files)):
            self.ingest_batch(path)

    def run(self, file_path=None, directory_path=None, single=False):
        self._create_index()
        if single and file_path:
            self.ingest_single(file_path)
        elif directory_path:
            self.ingest_directory(directory_path)
        elif file_path:
            self.ingest_batch(file_path)
        else:
            self.logger.error("No input path provided. Exiting.")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Elasticsearch Bulk Ingest Tool")
    parser.add_argument("--es_host", type=str, default="https://10.88.0.3:9200", help="Elasticsearch host URL")
    parser.add_argument("--username", type=str, default="milos_z", help="Elasticsearch username")
    parser.add_argument("--password", type=str, default="wcV4!w^eYQSuqzRK", help="Elasticsearch password")
    parser.add_argument("--index", type=str, default="app_proto_data_v4", help="Index name to write to")
    parser.add_argument("--pipeline", type=str, choices=["convert", "ingest", "clean"], default="convert")
    args = parser.parse_args()

    # ######################################################################################
    # # Init
    # ######################################################################################

    # Init fetcher
    fetcher = ElasticEmbeddingFetcher(
        es_host=args.es_host,
        index=args.index,
        username=args.username,
        password=args.password,
    )
    # ######################################################################################
    # # Init
    # ######################################################################################
    BASE_DIR = "/Volumes/KINGSTON"
    EXTRACTED_DIR = "testing"
    FLAT_DIR = "testing"
    EXTRACTED_EMBEDDINGS_PATH = f"{BASE_DIR}/{EXTRACTED_DIR}"
    FLATTENED_EMBEDDINGS_PATH = f"{BASE_DIR}/{FLAT_DIR}"

    if args.pipeline == "convert":
        # ######################################################################################
        # # Transform embeddings extracted with dataserver_dataset to flat embeddings ready for ingestion
        # ######################################################################################


        ec = EmbeddingsConverter(None)
        emb_list = glob.glob(f"{EXTRACTED_EMBEDDINGS_PATH}/*.pkl")
        for emb in emb_list:
            ec.load_embeddings(emb)
            save_path = emb.replace(EXTRACTED_DIR,
                                    FLAT_DIR).replace(".pkl",".parquet")
            if os.path.exists(save_path):
                continue
            ec.save_flat_embeddings(save_path, format="parquet")

        # Create chunks for ingestion
        # ec.create_chunks(tgt_dir=FLATTENED_EMBEDDINGS_PATH)

    elif args.pipeline == "clean":
        ######################################################################################
        # Cleaning
        ######################################################################################
        approved = ["adrenalgland","ovary","brain","breast","skin","colon","endometrium","uterus",
                    "falltube","prostate","kidney","lung","lymphnodes","smallbowel","placenta",
                    "thyroid","salivarygland","stomach",]
        fetcher.review_and_clean_field(approved_types=approved, field="tissue_type")

    elif args.pipeline == "ingest":
        # ######################################################################################
        # # Ingest embeddings
        # ######################################################################################
        filter_builder = ElasticsearchFilterBuilder(keyword_suffix=False)
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
            dataset_version="v4",
            # modality="stained",
            # preprocessing="resize",
            # tissue_type="thyroid",
        )
        filter_query = filter_builder.build()

        filter_query = None
        ingestor = EmbeddingIngestor(fetcher, dry_run=False, filter_query=filter_query)
        ingestor.run(directory_path=f"{FLATTENED_EMBEDDINGS_PATH}/splits")

