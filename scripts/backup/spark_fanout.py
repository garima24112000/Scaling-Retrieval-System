import os, json, time, argparse, math
from typing import List, Tuple, Dict, Any, Iterator

# Quieter HuggingFace tokenizers in forked workers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer

try:
    import faiss  # faiss-cpu
except Exception:
    import faiss_cpu as faiss  # fallback name if env aliases it


def _set_efsearch(index, ef: int) -> None:
    try:
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "efSearch", int(ef))
    except Exception:
        pass


def _encode_queries(qfile: str, model_name: str, normalize: bool) -> Tuple[List[str], np.ndarray]:
    queries: List[str] = []
    with open(qfile, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                queries.append(ln)
    if not queries:
        raise SystemExit(f"No queries found in {qfile}")

    model = SentenceTransformer(model_name)
    qvecs = model.encode(queries, normalize_embeddings=normalize)
    qvecs = np.asarray(qvecs, dtype="float32")
    return queries, qvecs


def _row_to_rd(row: pd.Series) -> Dict[str, Any]:
    src_raw = row.get("domain", "") or row.get("source", "")
    src = str(src_raw).strip().lower()
    cid = row.get("chunk_id", None)
    try:
        cid = int(cid) if cid is not None and cid == cid else None
    except Exception:
        cid = None
    uid = f"{row.get('id','')}::c{cid if cid is not None else -1}"
    text = (row.get("chunk_text", "") or row.get("text", "") or row.get("preview", "") or "")
    return {
        "id": str(row.get("id", "")),
        "uid": uid,
        "source": src,
        "title": str(row.get("title", "")),
        "url": str(row.get("url", "")),
        "text": str(text),
        "chunk_id": cid,
    }


def _search_shard_for_batch(
    ipath: str,
    mpath: str,
    qvecs_b: np.ndarray,
    top_k_prime: int,
    efsearch: int,
    filter_source: str | None,
) -> List[Tuple[int, float, Dict[str, Any]]]:
    """
    Search one shard for all query vectors in the broadcast batch.
    Returns flat (qidx, score, rowdict) tuples.
    """
    index = faiss.read_index(ipath)
    _set_efsearch(index, efsearch)
    meta = pd.read_parquet(mpath).reset_index(drop=True)

    out: List[Tuple[int, float, Dict[str, Any]]] = []

    # Vectorized search per shard (one FAISS call per qvec)
    for qi in range(qvecs_b.shape[0]):
        qv = qvecs_b[qi:qi+1]
        D, I = index.search(qv, top_k_prime)
        for score, li in zip(D[0], I[0]):
            if li < 0:
                continue
            row = meta.iloc[int(li)]
            src_raw  = row.get("domain", "") or row.get("source", "") or ""
            src_norm = str(src_raw).strip().lower()
            if filter_source and src_norm != filter_source.strip().lower():
                continue
            out.append((qi, float(score), _row_to_rd(row)))

    return out


def _map_partitions_search(
    it: Iterator[Tuple[str, str]],
    qvecs_b: np.ndarray,
    top_k_prime: int,
    efsearch: int,
    filter_source: str | None,
) -> Iterator[List[Tuple[int, float, Dict[str, Any]]]]:
    """
    mapPartitions: each partition receives one or more (index_path, meta_path)
    tuples; for each shard in the partition, load index ONCE and search all queries.
    """
    results_all: List[Tuple[int, float, Dict[str, Any]]] = []
    for ipath, mpath in it:
        part_out = _search_shard_for_batch(
            ipath=ipath,
            mpath=mpath,
            qvecs_b=qvecs_b,
            top_k_prime=top_k_prime,
            efsearch=efsearch,
            filter_source=filter_source,
        )
        results_all.extend(part_out)
    yield results_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="Path to indices/multi_default_sharded/manifest.json")
    ap.add_argument("--qfile", required=True, help="Newline-delimited queries file")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--normalize_query", type=int, default=1)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--alpha", type=int, default=3, help="overfetch factor per shard")
    ap.add_argument("--efsearch", type=int, default=128)
    ap.add_argument("--filter_source", default=None, help="wikipedia|pubmed|stackexchange")
    ap.add_argument("--results_csv", default="results/spark_fanout_hits.csv")
    ap.add_argument("--log", default="logs/spark_fanout_test.log")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    # ---- Load manifest
    man = json.load(open(args.manifest))
    # Support both "index_paths/meta_paths" and richer shard list
    if "shards" in man and isinstance(man["shards"], list):
        index_paths = [s["index_path"] for s in man["shards"]]
        meta_paths  = [s["meta_path"]  for s in man["shards"]]
        metric      = (man.get("metric") or man.get("index_metric") or "ip").lower()
        normalized  = bool(man.get("normalized", True))
    else:
        index_paths = man.get("index_paths") or []
        meta_paths  = man.get("meta_paths") or []
        metric      = (man.get("metric") or man.get("index_metric") or "ip").lower()
        normalized  = bool(man.get("normalized", True))

    if not index_paths or not meta_paths or len(index_paths) != len(meta_paths):
        raise SystemExit("Manifest missing or paths length mismatch.")

    # ---- Metric parity & normalization checks (fail-fast)
    if metric not in {"ip", "cosine"}:
        # If shards were built as L2, we can still work but ranking semantics change (lower=better).
        # For fan-out we assume IP/cosine (higher=better). Enforce now to keep heap logic consistent.
        raise SystemExit(
            f"Manifest reports metric='{metric}'. Expected 'ip' or 'cosine'. "
            f"Rebuild shards with IP on normalized vectors (see Step 3.3)."
        )
    if not normalized:
        # IP≈cosine requires normalized corpus + (usually) normalized queries
        raise SystemExit(
            "Manifest indicates 'normalized=false'. Rebuild corpus vectors as unit-norm for IP/cosine parity."
        )

    shards = list(zip(index_paths, meta_paths))

    # ---- Encode on driver (broadcast vectors, not the model)
    t0 = time.perf_counter()
    queries, qvecs = _encode_queries(args.qfile, args.model, bool(args.normalize_query))
    t1 = time.perf_counter()

    # ---- Spark session (local; disable Arrow—FAISS is native)
    spark = (
        SparkSession.builder
        .appName("spark_fanout_local")
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext

    # ---- Broadcast query vectors and log broadcast size
    qvecs_b = sc.broadcast(qvecs)
    est_bcast_bytes = qvecs.nbytes
    # Simple size guard: if > 50MB suggest batch-splitting
    bcast_warn = (est_bcast_bytes > 50 * 1024 * 1024)

    per_shard = max(args.top_k * args.alpha, args.top_k)

    # ---- Map each shard → top-k' per query using mapPartitions (load index once per shard)
    t2 = time.perf_counter()
    # One shard per partition for clean locality
    rdd = sc.parallelize(shards, len(shards))
    shard_results_nested = rdd.mapPartitions(
        lambda it: _map_partitions_search(
            it=it,
            qvecs_b=qvecs_b.value,
            top_k_prime=per_shard,
            efsearch=args.efsearch,
            filter_source=args.filter_source,
        )
    ).collect()
    t3 = time.perf_counter()

    # ---- Merge to global top-k per query, deterministic tie-break (score desc, uid asc)
    flat: List[Tuple[int, float, Dict[str, Any]]] = []
    for part in shard_results_nested:
        flat.extend(part)

    # Dedup by uid with best score kept
    by_q: Dict[int, Dict[str, Tuple[float, Dict[str, Any]]]] = {}
    for qi, score, rd in flat:
        bucket = by_q.setdefault(qi, {})
        uid = rd["uid"]
        if (uid not in bucket) or (score > bucket[uid][0]):
            bucket[uid] = (score, rd)

    rows = []
    for qi, uid_map in by_q.items():
        # stable key: (score desc, uid asc)
        candidates = [(score, rd["uid"], rd) for uid, (score, rd) in uid_map.items()]
        candidates.sort(key=lambda x: (-x[0], x[1]))
        top = candidates[: args.top_k]
        for rank, (score, _uid, rd) in enumerate(top, start=1):
            rows.append({
                "query": queries[qi],
                "rank": rank,
                "score": float(score),
                "id": rd["id"],
                "uid": rd["uid"],
                "source": rd["source"],
                "title": rd["title"],
                "url": rd["url"],
                "chunk_id": rd["chunk_id"],
                "text": rd["text"],
            })

    rows.sort(key=lambda r: (r["query"], r["rank"]))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    df.to_csv(args.results_csv, index=False)

    # ---- timings & log
    t4 = time.perf_counter()
    timings = {
        "n_shards": len(shards),
        "n_queries": len(queries),
        "top_k": args.top_k,
        "alpha": args.alpha,
        "efsearch": args.efsearch,
        "encode_ms": round((t1 - t0) * 1000, 1),
        "spark_map_ms": round((t3 - t2) * 1000, 1),
        "merge_ms": round((t4 - t3) * 1000, 1),
        "total_ms": round((t4 - t0) * 1000, 1),
        "filter_source": args.filter_source or None,
        "results_csv": args.results_csv,
        "metric": metric,
        "normalized": bool(args.normalize_query),
        "broadcast_bytes": est_bcast_bytes,
        "broadcast_large_hint": bcast_warn,
    }
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(timings, indent=2) + "\n")

    print("=== Spark fan-out summary ===")
    print(json.dumps(timings, indent=2))

    spark.stop()


if __name__ == "__main__":
    main()
