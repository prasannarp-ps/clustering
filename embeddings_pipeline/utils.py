import glob
import os
import time
import uuid
from pprint import pprint
from typing import Literal, Optional

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from elasticsearch import Elasticsearch


def load_clustering_results(results_dir: str):
    rows = []
    for file in glob.glob(os.path.join(results_dir, "*.parquet")):
        if "model" not in file:
            continue
        df = pd.read_parquet(file)

        # Extract metadata from filename
        filename = os.path.basename(file).replace(".parquet", "")
        parts = filename.split("_")
        try:
            k = int(parts[2][1:])  # from e.g. kmeans_model_k28_min10000
            min_samples = int(parts[3])
        except Exception:
            print(f"Skipping {filename}, could not parse k or min_samples")
            continue

        silhouette = df["silhouette_score"].iloc[0]
        db = df["davies_bouldin_score"].iloc[0]
        ch = df["calinski_harabasz_score"].iloc[0]

        rows.append({
            "k": k,
            "min_samples": min_samples,
            "silhouette_score": silhouette,
            "davies_bouldin_score": db,
            "calinski_harabasz_score": ch,
            "file": file
        })

    return pd.DataFrame(rows)

def plot_clustering_summary(df):
    metrics = ["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="k", y=metric, hue="min_samples", marker="o")
        plt.title(f"{metric.replace('_', ' ').title()} vs K")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def retry_with_backoff(func, max_retries=5, initial_wait=5, exceptions=(Exception,), **kwargs):
    wait_time = initial_wait
    for attempt in range(max_retries):
        try:
            return func(**kwargs)
        except exceptions as e:
            print(f"[Retry {attempt+1}/{max_retries}] Failed with: {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
                wait_time *= 2
            else:
                raise

def generate_unique_id(row):
    key = f"{row['tissue_type']}/{row['block_id']}/{row['slice_id']}/{row['scan_date']}/{row['box_id']}/{row['filename']}/{row['tile_type']}/{row['preprocessing']}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))



class ElasticsearchFilterBuilder:
    def __init__(self, keyword_suffix=True):
        self.fields = {
            "block_id": None,
            "box_id": None,
            "cluster": None,
            "created_at": None,
            "dimension": None,
            "dataset": None,
            "dataset_version": None,
            "filename": None,
            "id": None,
            "modality": None,
            "model": None,
            "preprocessing": None,
            "scan_date": None,
            "slice_id": None,
            "slide_key": None,
            "tile_type": None,
            "tissue_type": None,
            "unique_id": None
        }
        self.keyword_suffix = keyword_suffix

    def set_filter(
            self,
            block_id: Optional[str] = None,
            box_id: Optional[str] = None,
            cluster: Optional[int] = None,
            created_at: Optional[str] = None,
            dimension: Optional[int] = None,
            dataset: Optional[str] = None,
            dataset_version: Optional[str] = None,
            filename: Optional[str] = None,
            id: Optional[str] = None,
            modality: Optional[Literal["unstained", "stained-sr", "stained"]] = None,
            model: Optional[Literal["path_foundation"]] = None,
            preprocessing: Optional[Literal["resize", "center_crop"]] = None,
            scan_date: Optional[str] = None,
            slice_id: Optional[str] = None,
            slide_key: Optional[str] = None,
            tile_type: Optional[Literal["unstained", "stained-sr"]] = None,
            tissue_type: Optional[Literal["adrenalgland","ovary","brain","breast","skin","colon","endometrium","uterus",
                    "falltube","prostate","kidney","lung","lymphnodes","smallbowel","placenta", "thyroid","salivarygland","stomach"]] = None,
            unique_id: Optional[str] = None
    ):
        for key, value in locals().items():
            if key != "self" and value is not None:
                self.fields[key] = value

    def build(self):
        must_clauses = []
        for field, value in self.fields.items():
            if value is not None:
                field_name = f"{field}.keyword" if self.keyword_suffix else field
                must_clauses.append({"term": {field_name: value}})

        query = {"bool": {"must": must_clauses}}
        pprint(query)
        return query

    def add_term_filter(self, field: str, value):
        if field not in self.fields:
            raise ValueError(f"Field '{field}' is not supported in the filter builder.")
        self.fields[field] = value



class EmbeddingIO:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.data_dir = os.path.join(out_dir, "data")
        self.results_dir = os.path.join(out_dir, "results")
        self.grid_search_dir = os.path.join(self.results_dir, "grid_search")
        self.models_dir = os.path.join(self.results_dir, "models_zoo")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.s3_paths_dir = os.path.join(self.results_dir, "s3_paths")
        self.grid_images_dir = os.path.join(self.results_dir, "cluster_grids")
        self.cluster_stats_dir = os.path.join(self.results_dir, "cluster_stats")
        self.ensure_dirs()

    def ensure_dirs(self):
        for d in [self.data_dir, self.results_dir, self.grid_search_dir, self.models_dir, self.plots_dir,
                  self.s3_paths_dir, self.grid_images_dir, self.cluster_stats_dir]:
            os.makedirs(d, exist_ok=True)

    def save_parquet(self, df, name, subfolder="data", index=False):
        path = os.path.join(self.out_dir, subfolder, f"{name}.parquet")
        df.to_parquet(path, index=index)
        print(f"Saved: {path}")

    def load_parquet(self, name, subfolder="data"):
        path = os.path.join(self.out_dir, subfolder, f"{name}.parquet")
        return pd.read_parquet(path)

    def save_csv(self, df, name, subfolder="results", index=False, out_dir=None):
        if out_dir is None:
            path = os.path.join(self.out_dir, subfolder, f"{name}.csv")
            df.to_csv(path, index=index)
            print(f"Saved: {path}")
        else:
            path = os.path.join(out_dir, f"{name}.csv")
            df.to_csv(path, index=index)
            print(f"Saved: {path}")

    def load_csv(self, name, subfolder="results"):
        path = os.path.join(self.out_dir, subfolder, f"{name}.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        else:
            return None

    def save_model(self, model, name):
        path = os.path.join(self.models_dir, f"{name}.joblib")
        joblib.dump(model, path)
        print(f"Model saved: {path}")

    def load_model(self, name):
        path = os.path.join(self.models_dir, f"{name}.joblib")
        return joblib.load(path)

    def save_plot(self, fig, name, override_default=False):
        path = os.path.join(self.plots_dir, f"{name}.png")
        if override_default:
            path = os.path.join(self.results_dir, f"{name}.png")
        fig.savefig(path)
        print(f"Plot saved: {path}")

    def save_html(self, fig, name, cluster_id=None, out_dir=None):
        if out_dir is None:
            path = os.path.join(self.plots_dir, f"{name}.html")
            if cluster_id:
                path = os.path.join(self.cluster_stats_dir, f"{cluster_id}_{name}.html")
            fig.write_html(path, include_plotlyjs='cdn')
            print(f"HTML saved: {path}")
        else:
            path = os.path.join(out_dir, f"{name}.html")
            if cluster_id:
                path = os.path.join(out_dir, f"{cluster_id}_{name}.html")
            fig.write_html(path, include_plotlyjs='cdn')
            print(f"HTML saved: {path}")


    def save_s3_paths(self, paths, cluster_id):
        path = os.path.join(self.s3_paths_dir, f"cluster_{cluster_id}_s3_paths.txt")
        with open(path, "w") as f:
            f.write("\n".join(paths))
        print(f"S3 paths saved: {path}")

    def save_grid_image(self, image, cluster_id, grid_index=None, k=None):
        suffix = f"_grid_{grid_index}" if grid_index is not None else "_grid"
        path = os.path.join(self.grid_images_dir, f"cluster_{cluster_id}{suffix}.png") if k is None else os.path.join(self.grid_images_dir, f"{k}", f"cluster_{cluster_id}{suffix}.png")
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Grid image saved: {path}")


class ElasticEmbeddingFetcher:
    def __init__(self, es_host, index, username, password, batch_size=10_000, scroll_time='5m', keyword_field=False):
        self.index = index
        self.batch_size = batch_size
        self.scroll_time= scroll_time
        self.es = Elasticsearch(es_host, basic_auth=(username, password), verify_certs=False)
        self.keyword_field = keyword_field

    def get_doc_by_id(self, unique_id):
        try:
            return self.es.get(index=self.index, id=unique_id)['_source']
        except Exception as e:
            print(f"Error fetching document for {unique_id}: {e}")
            return None

    def get_vector_by_id(self, unique_id):
        try:
            doc = self.get_doc_by_id(unique_id)
            return np.array(doc['vector'], dtype=np.float32) if doc else None
        except Exception as e:
            print(f"Error fetching vector for {unique_id}: {e}")
            return None

    def mget(self, batch):
        return self.es.mget(index=self.index, body={"ids": batch})

    def get_docs_by_tile_group(self,
                               tissue_type=None,
                               block_id=None,
                               slice_id=None,
                               scan_date=None,
                               box_id=None,
                               filename=None,
                               fields=None,
                               size=10000):
        must = []

        if tissue_type:
            must.append({"term": {"tissue_type.keyword": tissue_type}})
        if block_id:
            must.append({"term": {"block_id.keyword": block_id}})
        if slice_id:
            must.append({"term": {"slice_id.keyword": slice_id}})
        if scan_date:
            must.append({"term": {"scan_date": scan_date}})
        if box_id:
            must.append({"term": {"box_id.keyword": box_id}})
        if filename:
            must.append({"term": {"filename.keyword": filename}})

        query = {
            "size": size,
            "_source": fields if fields else True,
            "query": {
                "bool": {
                    "must": must
                }
            }
        }

        try:
            res = self.es.search(index=self.index, body=query)
            return [hit["_source"] for hit in res["hits"]["hits"]]
        except Exception as e:
            print(f"Failed to fetch documents for tile group: {e}")
            return []

    def knn_vector_search(self, vector, top_k=10, num_candidates=10000,
                          filter_query=None, exclude_id=None, exclude_block_id=None):

        base_query = filter_query["bool"] if filter_query and "bool" in filter_query else {"must": []}
        must_not = []

        if exclude_id:
            must_not.append({"ids": {"values": [exclude_id]}})

        if must_not:
            base_query["must_not"] = base_query.get("must_not", []) + must_not

        full_query = {
            "size": top_k,
            "knn": {
                "field": "vector",
                "query_vector": vector,
                "k": top_k,
                "num_candidates": num_candidates
            },
            "query": {"bool": base_query}
        }
        print("Full query:")
        pprint(full_query)
        try:
            res = self.es.search(index=self.index, body=full_query)
            if exclude_block_id:
                return [hit for hit in res["hits"]["hits"] if hit["_source"].get("block_id") != exclude_block_id]
            else:
                return res["hits"]["hits"]
        except Exception as e:
            print(f"kNN search failed: {e}")
            return []

    def fetch_filtered_vectors_generator(self, filter_query, max_docs=10_000_000):
        query = {
            "query": filter_query,
            "_source": [
                "unique_id", "slide_key", "vector",
                "modality", "preprocessing", "tissue_type",
                "block_id", "slice_id", "scan_date", "box_id", "filename"
            ],
            "size": self.batch_size
        }

        response = self.es.search(index=self.index, body=query, scroll=self.scroll_time)
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        total_fetched = 0

        while hits and total_fetched < max_docs:
            records = []
            for hit in hits:
                src = hit['_source']
                records.append({
                    "unique_id": src.get("unique_id"),
                    "vector": np.array(src["vector"], dtype=np.float32),
                    "slide_key": src.get("slide_key", ""),
                    "modality": src.get("modality"),
                    "preprocessing": src.get("preprocessing"),
                    "tissue_type": src.get("tissue_type"),
                    "block_id": src.get("block_id"),
                    "slice_id": src.get("slice_id"),
                    "scan_date": src.get("scan_date"),
                    "box_id": src.get("box_id"),
                    "filename": src.get("filename"),
                })

            total_fetched += len(hits)
            yield pd.DataFrame.from_records(records)

            response = self.es.scroll(scroll_id=scroll_id, scroll=self.scroll_time)
            hits = response['hits']['hits']
            scroll_id = response['_scroll_id']

        self.es.clear_scroll(scroll_id=scroll_id)

    def get_unique_field_values(self, field="tissue_type", size=100):
        aggs_query = {
            "size": 0,
            "aggs": {
                "unique_terms": {
                    "terms": {
                        "field": f"{field}.keyword" if self.keyword_field else f"{field}",
                        "size": size
                    }
                }
            }
        }
        try:
            response = self.es.search(index=self.index, body=aggs_query)
            return [bucket["key"] for bucket in response["aggregations"]["unique_terms"]["buckets"]]
        except Exception as e:
            print(f"Aggregation query failed: {e}")
            return []

    def review_and_clean_field(self, approved_types, field="tissue_type", sample_size=1000):
        all_types = self.get_unique_field_values(field=field)
        unexpected = [t for t in all_types if t not in approved_types]

        print(f"\n🔍 Found {len(unexpected)} unexpected tissue_type values: {unexpected}")

        for tissue_type in unexpected:
            print(f"\nPreviewing documents for unexpected tissue_type: {tissue_type}")
            query = {
                "size": sample_size,
                "query": {
                    "term": {f"{field}.keyword": tissue_type}
                }
            }

            res = self.es.search(index=self.index, body=query)
            for doc in res["hits"]["hits"]:
                pprint(doc["_source"])

            print(f'Found {len(res["hits"]["hits"])} documents with tissue_type = {tissue_type}')
            decision = input(f"\n🗑️ Delete ALL documents with tissue_type = '{tissue_type}'? [y/N]: ").strip().lower()
            if decision == "y":
                delete_query = {
                    "query": {
                        "term": {f"{field}.keyword": tissue_type}
                    }
                }
                deleted = self.es.delete_by_query(index=self.index, body=delete_query, refresh=True)
                print(f"Deleted {deleted['deleted']} documents with tissue_type = '{tissue_type}'")
            else:
                print("Skipping deletion.")

    def fetch_and_save_ids(self, filter_query):
        res = self.es.count(index=self.index)
        print(f"Total documents in index: {res['count']}")
        res = self.es.count(index=self.index, body={"query": filter_query})
        print("Matching documents:", res["count"])

        print(f"Fetching unique_ids with filter_query...")
        ids = []

        query = {
            "query": filter_query,
            "_source": ["unique_id"],
            "size": self.batch_size
        }

        response = self.es.search(index=self.index, body=query, scroll=self.scroll_time)
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']

        while hits:
            ids.extend(hit['_source']['unique_id'] for hit in hits)
            response = self.es.scroll(scroll_id=scroll_id, scroll=self.scroll_time)
            hits = response['hits']['hits']
            scroll_id = response['_scroll_id']
            print(f"Fetched {len(ids)}/{res['count']} unique_ids...")

        self.es.clear_scroll(scroll_id=scroll_id)
        return pd.DataFrame({"unique_id": ids})

    def fetch_vectors_by_ids(self, batch_ids):
        vectors = []
        ids = []

        for i in range(0, len(batch_ids), 1000):
            sub_batch = batch_ids[i:i + 1000]
            body = {"ids": sub_batch}
            try:
                res = self.es.mget(index=self.index, body=body)
                for doc in res['docs']:
                    if doc.get('found'):
                        vectors.append(np.array(doc['_source']['vector'], dtype=np.float32))
                        ids.append(doc['_id'])
            except Exception as e:
                print(f"Error fetching batch ids {sub_batch}: {e}")
        return ids, vectors


def generate_per_plot_type_dashboards(html_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Collect all HTML files and organize them by plot type
    plot_groups = {}
    for fname in os.listdir(html_dir):
        if fname.endswith(".html"):
            try:
                cluster_id, *plot_parts = fname.replace(".html", "").split("_")
                plot_type = "_".join(plot_parts)
                if plot_type not in plot_groups:
                    plot_groups[plot_type] = []
                plot_groups[plot_type].append((int(cluster_id), fname))
            except Exception:
                continue

    # Create a dashboard for each plot type
    for plot_type, files in plot_groups.items():
        files.sort()  # sort by cluster number

        html_body = ""
        for cluster_id, fname in files:
            file_path = os.path.join(html_dir, fname)
            with open(file_path, "r") as f:
                embedded_html = f.read()

            title = f"<h3>Model: {cluster_id} Clusters</h3>\n"
            html_body += title + embedded_html + "<hr/>\n"

        final_html = f"""
        <html>
        <head>
            <title>{plot_type} Comparison</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>{plot_type.replace('_', ' ').title()} - Model Comparison</h1>
            {html_body}
        </body>
        </html>
        """

        output_path = os.path.join(output_dir, f"{plot_type}_dashboard.html")
        with open(output_path, "w") as f:
            f.write(final_html)

    return output_dir
