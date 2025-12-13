#!/usr/bin/env python3
import json
import argparse
import os


def make_spark_config(index_key, manifest_path, model_name,
                      normalize_query, efSearch, out_path):

    cfg = {
        "active_index": index_key,
        "indices": {
            index_key: {
                "backend": "spark",
                "model_name": model_name,
                "normalize_query": bool(normalize_query),
                "efSearch": efSearch,
                "extras": {
                    "manifest_path": manifest_path
                }
            }
        }
    }

    # ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print("✓ Spark config.json created")
    print("  index_key     :", index_key)
    print("  manifest_path :", manifest_path)
    print("  out_path      :", out_path)


def main():
    ap = argparse.ArgumentParser(description="Generate Spark backend config.json")

    ap.add_argument("--index_key", required=True,
                    help="Name of the index (e.g., wiki_30000)")
    ap.add_argument("--manifest_path", required=True,
                    help="Path to manifest.json (sharded FAISS)")
    ap.add_argument("--out_path", required=True,
                    help="Where to save the generated config.json")
    ap.add_argument("--model_name", default="BAAI/bge-small-en-v1.5",
                    help="Embedding model to use")
    ap.add_argument("--normalize_query", type=int, default=1,
                    help="1 = normalize query embeddings, 0 = no")
    ap.add_argument("--efSearch", type=int, default=128,
                    help="FAISS efSearch parameter")

    args = ap.parse_args()

    make_spark_config(
        index_key=args.index_key,
        manifest_path=args.manifest_path,
        model_name=args.model_name,
        normalize_query=args.normalize_query,
        efSearch=args.efSearch,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    main()
