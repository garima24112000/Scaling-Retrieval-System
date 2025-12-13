import argparse
import os
import json
import numpy as np
import pandas as pd
import faiss

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index and output config JSON.")
    parser.add_argument("--subset_dir", required=True,
                        help="Folder containing <domain>_meta_<n>.parquet and <domain>_embeddings_<n>.npy")
    parser.add_argument("--index_name", required=True,
                        help="Name of this index entry (e.g. faiss_single_30k)")
    parser.add_argument("--out_config", default="config_subset.json",
                        help="Where to write JSON config")
    parser.add_argument("--index_group", default="default",
                        help="Index group directory inside indices/")
    parser.add_argument("--domain", required=True,
                        help="Domain prefix (e.g., wiki, pubmed, so, arxiv)")
    parser.add_argument("--model_name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--metric", default="ip")
    parser.add_argument("--index_metric", default="ip")
    parser.add_argument("--normalize_query", type=bool, default=True)
    parser.add_argument("--vector_normed", type=bool, default=True)
    parser.add_argument("--skip_path_checks", type=bool, default=False)
    args = parser.parse_args()

    subset_dir = args.subset_dir
    domain = args.domain
    index_name = args.index_name

    # Detect input files using dynamic domain match
    meta_file = None
    emb_file = None
    for f in os.listdir(subset_dir):
        if f.startswith(f"{domain}_meta_") and f.endswith(".parquet"):
            meta_file = os.path.join(subset_dir, f)
        if f.startswith(f"{domain}_embeddings_") and f.endswith(".npy"):
            emb_file = os.path.join(subset_dir, f)

    if meta_file is None or emb_file is None:
        raise FileNotFoundError(
            f"subset_dir must contain {domain}_meta_<n>.parquet and {domain}_embeddings_<n>.npy"
        )

    print("Metadata file:", meta_file)
    print("Embeddings file:", emb_file)

    # Load data
    meta = pd.read_parquet(meta_file)
    vectors = np.load(emb_file).astype("float32")

    print("Metadata rows:", len(meta))
    print("Embedding matrix:", vectors.shape)

    dim = vectors.shape[1]

    # Build FAISS index
    print("Building FAISS HNSW index...")
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efSearch = 128
    index.add(vectors)

    print("Indexed vectors:", index.ntotal)

    # Save FAISS index
    index_dir = os.path.join("indices", args.index_group)
    os.makedirs(index_dir, exist_ok=True)

    index_path = os.path.join(index_dir, f"{index_name}.faiss")
    faiss.write_index(index, index_path)
    print("Saved FAISS index:", index_path)

    # Write config JSON
    config = {
        "indices": {
            index_name: {
                "backend": "faiss",
                "faiss_path": index_path,
                "meta_path": meta_file,
                "model_name": args.model_name,
                "metric": args.metric,
                "normalize_query": args.normalize_query,
                "index_metric": args.index_metric,
                "vector_normed": args.vector_normed,
                "skip_path_checks": args.skip_path_checks
            }
        }
    }

    with open(args.out_config, "w") as f:
        json.dump(config, f, indent=2)

    print("Config written to:", args.out_config)
    print("Done.")

if __name__ == "__main__":
    main()
