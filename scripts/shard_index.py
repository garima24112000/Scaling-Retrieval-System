"""
Shard the merged FAISS dataset into N shards with metric parity + manifest.

Usage (typical):
    python scripts/shard_index.py --n_shards 8 \
        --metric ip --normalize 1 --M 32 --efC 200 --split hash --seed 42 \
        --save_vectors 0

Inputs (fixed paths under project_root):
    data/embeddings/multi_default/multi_meta.parquet
    data/embeddings/multi_default/multi_vectors.npy

Outputs:
    data/embeddings/multi_default_sharded/meta_shard_000.parquet ...
    indices/multi_default_sharded/shard_000.faiss ...
    indices/multi_default_sharded/manifest.json
    # (optional, only if --save_vectors 1)
    data/embeddings/multi_default_sharded/vecs_shard_000.npy ...
"""

import os, json, argparse, time, hashlib, sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss


project_root = "/content/drive/MyDrive/AMS 560 PROJECT"
EMB_DIR  = f"{project_root}/data/embeddings/multi_default"
OUT_EMB  = f"{project_root}/data/embeddings/multi_default_sharded"
OUT_IDX  = f"{project_root}/indices/multi_default_sharded"
os.makedirs(OUT_EMB, exist_ok=True)
os.makedirs(OUT_IDX, exist_ok=True)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _hnsw_index(dim: int, M: int, metric: str):
    """
    Build IndexHNSWFlat with the requested metric.
    For FAISS versions without metric in constructor, fall back & try to set it.
    """
    metric = metric.lower()
    if metric not in {"ip", "l2"}:
        raise ValueError("metric must be 'ip' or 'l2'")
    METRIC = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2

    # Try modern signature (d, M, metric)
    try:
        return faiss.IndexHNSWFlat(dim, M, METRIC)
    except TypeError:
        idx = faiss.IndexHNSWFlat(dim, M)
        # Some builds expose metric_type; if not, accept default (L2)
        try:
            idx.metric_type = METRIC  # may no-op on older builds
        except Exception:
            if metric == "ip":
                print("[WARN] Index metric could not be set to IP; "
                      "ensure vectors are L2-normalized and use IP-compatible search.")
        return idx


def _hash_bucket(val: str, n: int) -> int:
    """Stable shard id via SHA1(id) % n for better topic balance."""
    h = hashlib.sha1(val.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16) % n


def _value_counts_int(series: pd.Series) -> dict:
    """Return a plain dict of str->int counts (safe for JSON)."""
    vc = series.value_counts(dropna=False)
    # Convert NaN to "(null)" for readability
    keys = [("(null)" if (isinstance(k, float) and np.isnan(k)) else str(k)) for k in vc.index.tolist()]
    vals = [int(v) for v in vc.values.tolist()]
    return dict(zip(keys, vals))


def shard_index(
    n_shards: int = 8,
    metric: str = "ip",
    normalize: int = 1,
    M: int = 32,
    efC: int = 200,
    split: str = "hash",  # 'hash' or 'contiguous'
    seed: int = 42,
    save_vectors: int = 0,  # 1 = save vecs_shard_XXX.npy (optional)
):
    rng = np.random.default_rng(seed)

    meta_pq = f"{EMB_DIR}/multi_meta.parquet"
    vec_np  = f"{EMB_DIR}/multi_vectors.npy"

    if not os.path.exists(meta_pq) or not os.path.exists(vec_np):
        raise FileNotFoundError(
            f"Missing inputs:\n  {meta_pq if os.path.exists(meta_pq) else '(not found)'}\n"
            f"  {vec_np if os.path.exists(vec_np) else '(not found)'}"
        )

    print("Loading meta & vectors ...")
    meta = pd.read_parquet(meta_pq)
    vecs = np.load(vec_np).astype("float32")
    if len(meta) != len(vecs):
        raise ValueError(f"meta rows {len(meta)} != vectors {len(vecs)}")

    N, dim = vecs.shape
    print(f"Loaded {N:,} vectors (dim={dim})")

    # Decide domain column (for diagnostics)
    domain_col = "domain" if "domain" in meta.columns else ("source" if "source" in meta.columns else None)
    if domain_col is None:
        print("[INFO] No 'domain' or 'source' column found; domain counts will be omitted.")

    # Optional normalization (recommended for IP/cosine semantics)
    do_norm = bool(int(normalize))
    if do_norm:
        print("Normalizing vectors (L2) ...")
        vecs = _normalize_rows(vecs)

    # Compute shard assignment
    print(f"Sharding strategy: {split}  |  n_shards={n_shards}")
    if split == "hash":
        if "id" not in meta.columns:
            print("[WARN] meta has no 'id' column; using contiguous splits.")
            split = "contiguous"
        else:
            assign = meta["id"].astype(str).map(lambda s: _hash_bucket(s, n_shards)).to_numpy()
    if split == "contiguous":
        shard_size = int(np.ceil(N / n_shards))
        assign = np.repeat(np.arange(n_shards), shard_size)[:N]

    # Global domain counts (pre-sharding)
    global_domain_counts = None
    if domain_col is not None:
        global_domain_counts = _value_counts_int(meta[domain_col])

    # Manifest skeleton
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "n_total": int(N),
        "n_shards": int(n_shards),
        "dim": int(dim),
        "normalized": bool(do_norm),
        "index_kind": "hnsw_flat",
        "index_params": {"M": int(M), "efConstruction": int(efC)},
        "index_metric": metric.lower(),  # logical metric used when building
        "split_strategy": split,
        "seed": int(seed),
        "save_vectors": bool(int(save_vectors)),
        "domain_counts_total": global_domain_counts if global_domain_counts is not None else {},
        "shards": [],
    }

    # Build each shard
    pbar = tqdm(range(n_shards), desc="Building shards")
    for i in pbar:
        mask = (assign == i)
        n_i = int(mask.sum())
        if n_i == 0:
            sub_meta = meta.iloc[0:0].copy()
            sub_vecs = np.zeros((0, dim), dtype="float32")
        else:
            sub_meta = meta.loc[mask].reset_index(drop=True)
            sub_vecs = vecs[mask].copy()

        shard_id = f"{i:03d}"
        meta_p = f"{OUT_EMB}/meta_shard_{shard_id}.parquet"
        idx_p  = f"{OUT_IDX}/shard_{shard_id}.faiss"
        vec_p  = f"{OUT_EMB}/vecs_shard_{shard_id}.npy"

        # Save meta
        sub_meta.to_parquet(meta_p)

        # Build FAISS HNSW
        index = _hnsw_index(dim, M, metric)
        try:
            index.hnsw.efConstruction = int(efC)
        except Exception:
            pass
        if n_i > 0:
            index.add(sub_vecs)

        # Persist index
        faiss.write_index(index, idx_p)

        # Optional: save vectors for this shard
        vectors_path = None
        if int(save_vectors) and n_i > 0:
            np.save(vec_p, sub_vecs)
            vectors_path = vec_p

        # Domain counts for this shard
        shard_domain_counts = {}
        if (domain_col is not None) and (n_i > 0):
            shard_domain_counts = _value_counts_int(sub_meta[domain_col])

        manifest["shards"].append({
            "id": shard_id,
            "n": n_i,
            "meta_path": meta_p,
            "index_path": idx_p,
            **({"vectors_path": vectors_path} if vectors_path else {}),
            "domain_counts": shard_domain_counts
        })
        pbar.set_postfix({"shard": shard_id, "rows": n_i})

    # Write manifest
    man_path = f"{OUT_IDX}/manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Manifest written:", man_path)
    print("Sum(ntotal) =", sum(s["n"] for s in manifest["shards"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_shards", type=int, default=8)
    ap.add_argument("--metric", type=str, default="ip", choices=["ip", "l2"])
    ap.add_argument("--normalize", type=int, default=1)   # 1=True (normalize rows)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efC", type=int, default=200)
    ap.add_argument("--split", type=str, default="hash", choices=["hash", "contiguous"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_vectors", type=int, default=0)  # 1 to save vecs_shard_XXX.npy
    args = ap.parse_args()

    shard_index(
        n_shards=args.n_shards,
        metric=args.metric,
        normalize=args.normalize,
        M=args.M,
        efC=args.efC,
        split=args.split,
        seed=args.seed,
        save_vectors=args.save_vectors,
    )
