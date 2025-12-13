import argparse
import os
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Slice N embeddings + metadata for any domain.")
    parser.add_argument("--emb_path", required=True, help="Path to vectors .npy file")
    parser.add_argument("--meta_path", required=True, help="Path to metadata parquet file")
    parser.add_argument("--out_dir", required=True, help="Output directory for the sliced subset")
    parser.add_argument("--n", type=int, required=True, help="Number of rows to slice (e.g., 30000)")
    parser.add_argument("--domain", required=True, help="Domain name prefix (e.g., wiki, pubmed, so)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    domain = args.domain
    n = args.n

    print("Loading metadata:", args.meta_path)
    meta = pd.read_parquet(args.meta_path)
    total_rows = len(meta)
    print(f"Metadata rows available: {total_rows}")

    if total_rows < n:
        raise ValueError(f"Requested {n} rows but only {total_rows} exist.")

    # Slice metadata
    print(f"Slicing first {n} metadata rows...")
    meta_n = meta.iloc[:n]
    meta_out = os.path.join(args.out_dir, f"{domain}_meta_{n}.parquet")
    meta_n.to_parquet(meta_out)
    print("Saved:", meta_out)

    # Load embeddings
    print("Loading embeddings:", args.emb_path)
    vectors = np.load(args.emb_path)
    print("Embedding matrix shape:", vectors.shape)

    if vectors.shape[0] < n:
        raise ValueError("Embedding count is smaller than requested slice size — mismatch!")

    # Slice vectors
    print(f"Slicing first {n} embedding rows...")
    vectors_n = vectors[:n]
    emb_out = os.path.join(args.out_dir, f"{domain}_embeddings_{n}.npy")
    np.save(emb_out, vectors_n)
    print("Saved:", emb_out)

    print("Files created:")
    print("  -", meta_out)
    print("  -", emb_out)

if __name__ == "__main__":
    main()
