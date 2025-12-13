import os, json, glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Inject FAISS-sharded index into existing config.json")
    parser.add_argument("--config_path", required=True,
                        help="Path to existing config.json to update")
    parser.add_argument("--index_name", required=True,
                        help="Name of new index entry (e.g., wiki_30000_sharded)")
    parser.add_argument("--faiss_dir", required=True,
                        help="Directory containing shard_*.faiss and manifest.json")
    parser.add_argument("--meta_dir", required=True,
                        help="Directory containing meta_shard_*.parquet")
    parser.add_argument("--model_name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--metric", default="ip")
    parser.add_argument("--index_metric", default="ip")
    parser.add_argument("--normalize_query", type=bool, default=True)
    parser.add_argument("--vector_normed", type=bool, default=True)
    parser.add_argument("--skip_path_checks", type=bool, default=False)
    parser.add_argument("--set_active", action="store_true",
                        help="Set this index as active_index")
    args = parser.parse_args()

    # -----------------------------------------------
    # Detect shards
    # -----------------------------------------------
    faiss_shards = sorted(glob.glob(os.path.join(args.faiss_dir, "*shard_*.faiss")))
    if not faiss_shards:
        raise FileNotFoundError("No shard_*.faiss files in " + args.faiss_dir)

    meta_shards = sorted(glob.glob(os.path.join(args.meta_dir, "*meta_shard_*.parquet")))
    if not meta_shards:
        raise FileNotFoundError("No meta_shard_*.parquet in " + args.meta_dir)

    # Manifest
    manifest_list = glob.glob(os.path.join(args.faiss_dir, "*manifest*.json"))
    if not manifest_list:
        raise FileNotFoundError("No manifest JSON in " + args.faiss_dir)
    manifest_path = manifest_list[0]

    # -----------------------------------------------
    # Load existing config.json (do NOT overwrite)
    # -----------------------------------------------
    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                config = {"indices": {}}
    else:
        # Create new empty config
        os.makedirs(os.path.dirname(args.config_path), exist_ok=True)
        config = {"indices": {}}

    config.setdefault("indices", {})

    # -----------------------------------------------
    # Build FAISS-sharded config block
    # -----------------------------------------------
    config["indices"][args.index_name] = {
        "backend": "faiss",
        "index_paths": faiss_shards,
        "meta_paths": meta_shards,
        "model_name": args.model_name,
        "metric": args.metric,
        "normalize_query": args.normalize_query,
        "index_metric": args.index_metric,
        "vector_normed": args.vector_normed,
        "skip_path_checks": args.skip_path_checks,
        "extras": {
            "manifest_path": manifest_path
        }
    }

    # -----------------------------------------------
    # Optionally set as active index
    # -----------------------------------------------
    if args.set_active:
        config["active_index"] = args.index_name

    # -----------------------------------------------
    # Save updated config.json
    # -----------------------------------------------
    with open(args.config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\nSuccessfully added FAISS-sharded entry to config json file")
    print("New index name:", args.index_name)
    print("FAISS shards:", len(faiss_shards))
    print("META shards:", len(meta_shards))
    print("Manifest:", manifest_path)
    print("Updated config:", args.config_path)


if __name__ == "__main__":
    main()
