import argparse
import time
from pprint import pprint

import glob
import os
import pandas as pd

from embeddings.embeddings_pipeline.utils import ElasticsearchFilterBuilder, ElasticEmbeddingFetcher, load_clustering_results, plot_clustering_summary
from embeddings.embeddings_pipeline.embeddings_analyzer import ElasticEmbeddingAnalyzer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elasticsearch Analyzer Tool")
    parser.add_argument("--es_host", type=str, default="https://10.88.0.3:9200", help="Elasticsearch host URL")
    parser.add_argument("--username", type=str, default="milos_z", help="Elasticsearch username")
    parser.add_argument("--password", type=str, default="wcV4!w^eYQSuqzRK", help="Elasticsearch password")
    parser.add_argument("--index", type=str, default="app_proto_data_v4", help="Index name to write to")
    parser.add_argument("--pipeline", type=str, choices=["sim_search", "grid_search", "fit_all", "fit_local",
                                                         "predict_all", "predict_local", "cluster_info", "assign_clusters",
                                                         "generate_local_stratified_dataset"],
                        default="predict_local")
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
        keyword_field=False
    )
    # Init analyzer
    analyzer = ElasticEmbeddingAnalyzer(
        fetcher=fetcher,
        out_dir="data/grid_search_fine_all_resize",
        batch_size=100000
    )


    # ######################################################################################
    # # Filter builder
    # ######################################################################################



    if args.pipeline == "sim_search":
        # ######################################################################################
        # # Similarity search
        # ######################################################################################

        # Init filter builder
        filter_builder = ElasticsearchFilterBuilder()

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
            modality="stained",
            preprocessing="resize",
            # tissue_type="prostate",
        )
        filter_query = filter_builder.build()
        target_id = ""
        target_doc = fetcher.get_doc_by_id(target_id)
        vector = target_doc["vector"]

        print("Target sample:")
        target_doc.pop("vector", None)
        pprint(target_doc)

        start_time = time.time()


        matches = fetcher.knn_vector_search(vector, top_k=50, num_candidates=1000, filter_query={"bool": filter_query["bool"]},
                                             # exclude_id=target_id, exclude_block_id=target_doc["block_id"])
                                             exclude_id=target_id)

        analyzer.visualize_similar_images_grid(target_doc, matches, N=10)

    elif args.pipeline == "cleaning":
        ######################################################################################
        # Cleaning
        ######################################################################################
        approved = ["colon", "endometrium", "falltube", "ovary", "placenta", "prostate",
                    "skin", "smallbowel", "thyroid", "uterus"]
        fetcher.review_and_clean_field(approved_types=approved)

    elif args.pipeline == "grid_search":
        ######################################################################################
        # Grid Search
        ######################################################################################
        # preprocess_options = ["resize", "center_crop"]
        preprocess_options = ["resize"]
        # preprocess_options = ["center_crop"]
        apply_pca_options = [True, False]
        # apply_pca_options = [False]
        min_samples_options = [1_000, 10_000, 50_000, 100_000]
        # min_samples_options = [200_000]
        n_clusters_list = list(range(5, 100, 1))

        # analyzer.grid_search(preprocess_options=preprocess_options,
        #                      apply_pca_options=apply_pca_options,
        #                      min_samples_options=min_samples_options,
        #                      n_clusters_list=n_clusters_list,
        #                      modality="stained")

        analyzer.visualize_grid_search()
        analyzer.results_summary()

    elif args.pipeline == "fit_all":
        # ######################################################################################
        # Fit model on data
        # ######################################################################################
        # Init filter builder
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
            # modality="stained",
            # preprocessing="resize",
            # tissue_type="prostate",
            dataset_version="v4"
        )
        filter_query = filter_builder.build()
        ids = fetcher.fetch_and_save_ids(filter_query)
        analyzer.io.save_parquet(ids, name="unique_ids")


        analyzer.partial_fit_clustering(k=28)
        analyzer.partial_fit_clustering(k=31)
        analyzer.partial_fit_clustering(k=34)
        analyzer.partial_fit_clustering(k=41)

    elif args.pipeline == "generate_local_stratified_dataset":

        parquet_paths = glob.glob("/Volumes/KINGSTON/v6_2_missing_flat/*.parquet")
        parquet_paths.extend(glob.glob("/Volumes/KINGSTON/extracted_embeddings_flat/*.parquet"))
        df = analyzer.get_stratified_dataset_quota_streaming(
            parquet_paths=parquet_paths,
            stratify_field="tissue_type",
            per_class=25_000,
            max_samples=100_000,
            stats=["tissue_type", "block_id", "box_id", "modality", "preprocessing"],
            batch_size=1_000_000
        )
        df.to_parquet(f"{analyzer.io.data_dir}/sample_stratified.parquet", index=False)

    elif args.pipeline == "fit_local":

        paths=[
            # "/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_resize/results/grid_search/stained_resize_pca0_min1000/data/sample_stratified.parquet",
            # "/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_resize/results/grid_search/stained_resize_pca0_min10000/data/sample_stratified.parquet",
            # "/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_resize/results/grid_search/stained_resize_pca0_min50000/data/sample_stratified.parquet",
            # "/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_resize/results/grid_search/stained_resize_pca0_min100000/data/sample_stratified.parquet",
            # "/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_resize/results/grid_search/stained_resize_pca0_min200000/data/sample_stratified.parquet",
            "/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_all_resize/data/sample_stratified.parquet"
        ]
        import glob
        for path in paths:
            # num_samples = path.split("_")[-2].split("/")[0].replace("min", "")
            num_samples = "100000"
            df = pd.read_parquet(path)
            # analyzer.partial_fit_clustering_from_df(k=28, df=df, suffix=num_samples)
            # analyzer.partial_fit_clustering_from_df(k=31, df=df, suffix=num_samples)
            # analyzer.partial_fit_clustering_from_df(k=41, df=df, suffix=num_samples)
            # analyzer.partial_fit_clustering_from_df(k=56, df=df, suffix=num_samples)
            analyzer.partial_fit_clustering_from_df(k=112, df=df, suffix=num_samples)
            # analyzer.partial_fit_clustering_from_df(k=140, df=df, suffix=num_samples)
        # analyzer.partial_fit_clustering_from_df(k=34, df=df, suffix=num_samples)
        # analyzer.partial_fit_clustering_from_df(k=41, df=df, suffix=num_samples)

    elif args.pipeline == "predict_all":
        # ######################################################################################
        # Predict clusters for all data
        # ######################################################################################

        analyzer.predict_from_model(model_path="/Users/miloszivkovic/GIT/minimodels/embeddings/data/grid_search_fine_resize/results/models_zoo/kmeans_model_k28.joblib")
        analyzer.predict_from_model(model_path="/Users/miloszivkovic/GIT/minimodels/embeddings/data/grid_search_fine_center_crop/results/models_zoo/kmeans_model_k28.joblib")
        analyzer.visualize_cluster_samples(rows=10, cols=10)
        for cl_id in range(0, 28):
            analyzer.describe_cluster(cluster_id=cl_id, batch_size=10000)
        analyzer.plot_cluster_tissue_info(preprocessing="resize")

    elif args.pipeline == "predict_local":
        parquet_paths = glob.glob("/Volumes/KINGSTON/extracted_embeddings_flat/*.parquet")

        data_type = "all"
        # data_type = "inferred"
        for x in [112]:
            print("Running clustering for k=", x)
            # if not os.path.exists(f"/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_resize/data/clustering_results_{data_type}_k{x}.parquet"):
            #     analyzer.predict_from_model_from_local_file(model_path=f"/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_resize/results/models_zoo/kmeans_model_k{x}_50000.joblib",
            #     analyzer.predict_from_model_from_local_file(model_path=f"/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_all_resize/results/models_zoo/kmeans_model_k{x}_100000.joblib",
                #                                             parquet_paths=parquet_paths,
                #                                             result_file_name=f"clustering_results_{data_type}_k{x}.parquet", data_type=data_type)
            analyzer.visualize_cluster_samples(rows=3, cols=3, result_file_name=f"clustering_results_{data_type}_k{x}.parquet", k=x, local=True)

            # analyzer.plot_cluster_tissue_info_local(parquet_paths=parquet_paths,
            #                                         result_file_name=f"clustering_results_{data_type}_k{x}.parquet",
            #                                         preprocessing=f"{data_type}_resize",
            #                                         k=x)

    elif args.pipeline == "cluster_info":
        analyzer.export_s3_paths_for_cluster(cluster_id=14)
        analyzer.visualize_cluster_samples(cluster_id=14, rows=3, cols=3, num_grids=5)
        analyzer.describe_cluster(cluster_id=14, batch_size=10000)
        analyzer.plot_cluster_tissue_info(preprocessing="center_crop")
    elif args.pipeline == "assign_clusters":
        analyzer.assign_clusters_to_documents_batch(batch_size=10000)
    elif args.pipeline == "summary":
        summary_df = load_clustering_results("/Users/miloszivkovic/GIT/minimodels/embeddings/embeddings_pipeline/data/grid_search_fine_all_resize/data")
        summary_df.to_csv("info.csv")
        plot_clustering_summary(summary_df)
    else:
        print("??? Invalid pipeline option selected. Please choose a valid pipeline.")
