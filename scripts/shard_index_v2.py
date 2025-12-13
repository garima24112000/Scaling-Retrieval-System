#!/usr/bin/env python3
import os, json, argparse, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss


# --------------------------
# Utility functions
# --------------------------

def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _hnsw_index(dim: int, M: int, metric: str):
    metric = metric.lower()
    if metric not in {"ip", "l2"}:
        raise ValueError("metric must be 'ip' or 'l2'")
    METRIC = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2

    try:
        return faiss.IndexHNSWFlat(dim, M, METRIC)
    except TypeError:
        idx = faiss.IndexHNSWFlat(dim, M)
        try:
            idx.metric_type = METRIC
        except Exception:
            pass
        return idx


def _hash_bucket(val: str, n: int) -> int:
    h = hashlib.sha1(val.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16) % n


def _value_counts_int(series: pd.Series) -> dict:
    vc = series.value_counts(dropna=False)
    keys = [("(null)" if (isinstance(k, float) and np.isnan(k)) else str(k)) for k in vc.index.tolist()]
    vals = [int(v) for v in vc.values.tolist()]
    return dict(zip(keys, vals))


# --------------------------
# Main sharding function
# --------------------------

def shard_index(
    meta_path: str,
    emb_path: str,
    domain: str,
    out_meta_dir: str,
    out_index_dir: str,
    out_vectors_dir: str,
    out_manifest_dir: str,
    n_shards: int,
    metric: str,
    normalize: int,
    M: int,
    efC: int,
    split: str,
    seed: int,
    save_vectors: int,
):

    dom = domain.lower()

    # Ensure directories exist
    os.makedirs(out_meta_dir, exist_ok=True)
    os.makedirs(out_index_dir, exist_ok=True)
    os.makedirs(out_manifest_dir, exist_ok=True)
    if save_vectors:
        os.makedirs(out_vectors_dir, exist_ok=True)

    # Load inputs
    meta = pd.read_parquet(meta_path)
    vecs = np.load(emb_path).astype("float32")
    if len(meta) != len(vecs):
        raise ValueError(f"metadata rows {len(meta)} != vectors {len(vecs)}")

    N, dim = vecs.shape
    print(f"Loaded {N:,} vectors (dim={dim})")

    # Optional normalization
    if bool(int(normalize)):
        print("Normalizing vectors (L2)...")
        vecs = _normalize_rows(vecs)

    # Detect domain column if any
    domain_col = None
    for c in ["domain", "source"]:
        if c in meta.columns:
            domain_col = c
            break

    # Compute shard assignment
    if split == "hash":
        if "id" in meta.columns:
            assign = meta["id"].astype(str).map(lambda s: _hash_bucket(s, n_shards)).to_numpy()
        else:
            print("[WARN] No 'id' column → using contiguous splits.")
            split = "contiguous"

    if split == "contiguous":
        shard_size = int(np.ceil(N / n_shards))
        assign = np.repeat(np.arange(n_shards), shard_size)[:N]

    # Manifest
    manifest = {
        "domain": dom,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "n_total": int(N),
        "n_shards": int(n_shards),
        "dim": int(dim),
        "normalized": bool(int(normalize)),
        "index_kind": "hnsw_flat",
        "index_params": {"M": M, "efConstruction": efC},
        "index_metric": metric,
        "split_strategy": split,
        "seed": int(seed),
        "save_vectors": bool(save_vectors),
        "shards": [],
    }

    # Build shards
    pbar = tqdm(range(n_shards), desc=f"Sharding domain '{dom}'")
    for shard_i in pbar:
        shard_id = f"{shard_i:03d}"
        mask = (assign == shard_i)
        n_i = int(mask.sum())

        sub_meta = meta.loc[mask].reset_index(drop=True)
        sub_vecs = vecs[mask].copy()

        # Output file names (domain-prefixed)
        meta_out = os.path.join(out_meta_dir, f"{dom}_{N}_meta_shard_{shard_id}.parquet")
        index_out = os.path.join(out_index_dir, f"{dom}_{N}_shard_{shard_id}.faiss")
        vectors_out = (
            os.path.join(out_vectors_dir, f"{dom}_{N}_vecs_shard_{shard_id}.npy")
            if save_vectors else None
        )

        # Write metadata shard
        sub_meta.to_parquet(meta_out)

        # Build FAISS index
        idx = _hnsw_index(dim, M, metric)
        try:
            idx.hnsw.efConstruction = efC
        except Exception:
            pass
        if n_i > 0:
            idx.add(sub_vecs)
        faiss.write_index(idx, index_out)

        # Optional vec save
        if save_vectors and n_i > 0:
            np.save(vectors_out, sub_vecs)

        # Manifest entry
        manifest["shards"].append({
            "id": shard_id,
            "n": n_i,
            "meta_path": meta_out,
            "index_path": index_out,
            **({"emb_path": vectors_out} if save_vectors else {})
        })

    # Write manifest
    manifest_path = os.path.join(out_manifest_dir, f"{dom}_{N}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Manifest written:", manifest_path)
    print("Total rows =", sum(s["n"] for s in manifest["shards"]))


# --------------------------
# CLI
# --------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Shard FAISS index into multiple domain-named shards.")

    # Inputs
    ap.add_argument("--meta_path", required=True, help="Path to metadata parquet file.")
    ap.add_argument("--emb_path", required=True, help="Path to embeddings .npy file.")

    # Domain
    ap.add_argument("--domain", required=True, help="Domain prefix for all output files (e.g., wiki, pubmed).")

    # Four explicit output directories
    ap.add_argument("--out_meta_dir", required=True, help="Directory for domain_meta_shard_*.parquet files.")
    ap.add_argument("--out_index_dir", required=True, help="Directory for domain_shard_*.faiss files.")
    ap.add_argument("--out_vectors_dir", required=True, help="Directory for domain_vecs_shard_*.npy files.")
    ap.add_argument("--out_manifest_dir", required=True, help="Directory for domain_manifest.json.")

    # Settings
    ap.add_argument("--n_shards", type=int, default=8, help="Number of shards to create.")
    ap.add_argument("--metric", type=str, default="ip", choices=["ip", "l2"], help="FAISS metric.")
    ap.add_argument("--normalize", type=int, default=1, help="L2-normalize vectors (1=yes).")
    ap.add_argument("--M", type=int, default=32, help="HNSW M parameter.")
    ap.add_argument("--efC", type=int, default=200, help="HNSW efConstruction parameter.")
    ap.add_argument("--split", type=str, default="hash", choices=["hash", "contiguous"], help="Sharding strategy.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--save_vectors", type=int, default=0, help="Save vector shards as .npy files (1=yes).")

    args = ap.parse_args()

    shard_index(
        meta_path=args.meta_path,
        emb_path=args.emb_path,
        domain=args.domain,
        out_meta_dir=args.out_meta_dir,
        out_index_dir=args.out_index_dir,
        out_vectors_dir=args.out_vectors_dir,
        out_manifest_dir=args.out_manifest_dir,
        n_shards=args.n_shards,
        metric=args.metric,
        normalize=args.normalize,
        M=args.M,
        efC=args.efC,
        split=args.split,
        seed=args.seed,
        save_vectors=args.save_vectors,
    )
