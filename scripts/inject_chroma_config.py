import json
import os
import argparse


def create_chroma_config(
    output_config_path: str,
    index_name: str,
    persist_dir: str,
    collection_name: str,
    set_active: bool = True
):
    """Create a separate Chroma config file for any dataset/domain."""

    chroma_entry = {
        "backend": "chroma",
        "persist_dir": persist_dir,
        "collection_name": collection_name,
        "model_name": "BAAI/bge-small-en-v1.5",
        "metric": "cosine",
        "normalize_query": True,
        "vector_normed": True,
        "skip_path_checks": False
    }

    cfg = {
        "indices": {
            index_name: chroma_entry
        }
    }

    if set_active:
        cfg["active_index"] = index_name

    os.makedirs(os.path.dirname(output_config_path), exist_ok=True)
    with open(output_config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"✓ Created new Chroma config: {output_config_path}")
    print(f"✓ Added index: {index_name}")
    if set_active:
        print(f"✓ active_index = {index_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_config", required=True)
    parser.add_argument("--index_name", required=True)
    parser.add_argument("--persist_dir", required=True)
    parser.add_argument("--collection_name", required=True)
    parser.add_argument("--no_set_active", action="store_true")

    args = parser.parse_args()

    create_chroma_config(
        output_config_path=args.output_config,
        index_name=args.index_name,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        set_active=not args.no_set_active
    )
