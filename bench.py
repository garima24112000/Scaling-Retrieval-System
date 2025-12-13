
import argparse, os, sys, json, time, csv, math, datetime, hashlib, platform
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np

# Optional libs for env capture
try:
    import faiss
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

try:
    import chromadb
    _CHROMA_VER = getattr(chromadb, "__version__", "unknown")
except Exception:
    _CHROMA_VER = "unknown"

try:
    from sentence_transformers import __version__ as _ST_VER
except Exception:
    _ST_VER = "unknown"

from pipeline.runner import load_pipeline
from pipeline.config import load_config, ConfigError

# ---------------- util ---------------- #

def read_queries(qfile: str) -> List[str]:
    qs = []
    with open(qfile, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                qs.append(ln)
    if not qs:
        raise SystemExit(f"No queries found in {qfile}")
    return qs

def sha1_of_json(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted)-1) * (p/100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c: return values_sorted[int(k)]
    d0 = values_sorted[f] * (c-k); d1 = values_sorted[c] * (k-f)
    return d0 + d1

def flat_hits_rows(query, hits, iter_id):
    rows = []
    for h in hits:
        rows.append({
            "query": query,
            "iter": iter_id,
            "rank": h.get("rank"),
            "score": h.get("score"),
            "id": h.get("id"),
            "uid": h.get("uid"),
            "source": h.get("source"),
            "title": h.get("title"),
            "url": h.get("url"),
            "chunk_id": h.get("chunk_id"),
            "text": h.get("text"),
        })
    return rows

def load_truth_map(path: Optional[str]) -> Dict[str, List[str]]:
    """
    Optional gold file (JSON) mapping query -> list of relevant uids/ids.
    Example:
      {"Explain PageRank algorithm": ["doc123::c0","doc456::c2", ...], ...}
    """
    if not path: return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_recall_mrr(hits_by_query: Dict[str, List[Dict[str, Any]]], gold: Dict[str, List[str]]) -> Tuple[Optional[float], Optional[float]]:
    """
    Computes recall@k and MRR using uid if present, else falls back to id.
    """
    recall_list = []
    rr_list = []
    for q, hits in hits_by_query.items():
        if q not in gold:
            continue
        gold_set = set(gold[q])
        if not gold_set:
            continue
        seq = [h.get("uid") or h.get("id") for h in hits]
        seq = [x for x in seq if x]
        hit_any = any(s in gold_set for s in seq)
        if hit_any:
            rank = None
            for i, s in enumerate(seq, start=1):
                if s in gold_set:
                    rank = i
                    break
            rr_list.append(1.0 / rank if rank else 0.0)
        else:
            rr_list.append(0.0)
        retrieved = set(seq)
        inter = len(retrieved.intersection(gold_set))
        recall_list.append(inter / max(1, len(gold_set)))
    rec = round(float(np.mean(recall_list)), 4) if recall_list else None
    mrr = round(float(np.mean(rr_list)), 4) if rr_list else None
    return rec, mrr

def _dedup_and_rerank(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep the best-scoring occurrence per UID (or per (id,chunk_id) fallback), then re-rank.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        uid = h.get("uid")
        if not uid:
            cid = h.get("chunk_id")
            uid = f"{h.get('id','')}::c{int(cid) if cid is not None else -1}"
        prev = best.get(uid)
        if (prev is None) or (float(h.get("score", -1e30)) > float(prev.get("score", -1e30))):
            best[uid] = {**h, "uid": uid}
    # Re-rank by (score desc, uid asc)
    merged = list(best.values())
    merged.sort(key=lambda r: (-float(r.get("score", 0.0)), str(r.get("uid",""))))
    for i, h in enumerate(merged, start=1):
        h["rank"] = i
    return merged

# ---------------- main bench logic ---------------- #

def run_bench(
    config_path: str,
    index_key: str | None,
    backend: str | None,
    qfile: str,
    top_k: int,
    iters: int,
    warmup: int,
    filter_source: str | None,
    out_format: str,
    results_dir: str,
    truth_json: Optional[str] = None,
):
    # Load raw config to determine effective index_key and some knobs for env capture
    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    effective_index = index_key or raw_cfg.get("active_index")
    if not effective_index:
        raise SystemExit("No index specified and no active_index in config.")
    icfg = raw_cfg["indices"][effective_index]
    cfg_hash = sha1_of_json(icfg)

    # Figure out score semantics (for summary)
    metric = (icfg.get("metric") or "ip").lower()
    # FAISS returns similarity for ip/cosine; Chroma backend converts distance->similarity
    score_type = "similarity" if metric in ("ip", "cosine") else "distance"

    # Identify shard count (if sharded FAISS)
    shard_count = None
    if isinstance(icfg.get("index_paths"), list):
        shard_count = len(icfg.get("index_paths"))

    # Pull common knobs
    ef_search = icfg.get("efSearch")
    extras = icfg.get("extras", {})
    alpha = extras.get("alpha")
    model_name = icfg.get("model_name")

    # Prepare timestamped run dir: results/<index_key>/<YYYYMMDD-HHMMSS>-<cfg_hash>/
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(results_dir) / effective_index / f"{ts}-{cfg_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "run.log"

    def log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    # Save a config snapshot and args for reproducibility
    with open(run_dir / "config_effective.json", "w", encoding="utf-8") as f:
        json.dump(icfg, f, indent=2)
    args_snapshot = {
        "config": config_path,
        "index": index_key,
        "backend": backend,
        "qfile": qfile,
        "top_k": top_k,
        "iters": iters,
        "warmup": warmup,
        "filter_source": filter_source,
        "results_dir": results_dir,
        "truth_json": truth_json,
    }
    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(args_snapshot, f, indent=2)

    # Echo basic setup
    queries = read_queries(qfile)
    n_q = len(queries)
    log(f"[bench] config={config_path} index={effective_index} backend={backend or '(config)'}")
    log(f"[bench] queries={n_q} top_k={top_k} iters={iters} warmup={warmup} filter_source={filter_source or 'None'}")
    log(f"[bench] run_dir={run_dir}")

    # Load pipeline once (measure)
    t_load0 = time.perf_counter()
    P = load_pipeline(config_path, index_key=effective_index, backend_override=backend)
    t_load1 = time.perf_counter()
    load_ms = (t_load1 - t_load0) * 1000.0
    log(f"[bench] pipeline loaded in {load_ms:.1f} ms")

    # Warmup
    if warmup > 0:
        log(f"[bench] warmup passes={warmup}")
        for _ in range(warmup):
            for q in queries:
                _ = P.search_one(q, top_k=top_k, filter_source=filter_source)

    # Measured runs
    per_query_ms: List[float] = []
    all_hits_for_mix: List[Dict[str, Any]] = []
    hits_rows_accum: List[Dict[str, Any]] = []
    hits_by_query_last_iter: Dict[str, List[Dict[str, Any]]] = {}

    t0 = time.perf_counter()
    for it in range(iters):
        log(f"[bench] iter={it+1}/{iters}")
        for q in queries:
            t_q0 = time.perf_counter()
            raw_hits = P.search_one(q, top_k=top_k, filter_source=filter_source)
            # de-dup + re-rank to ensure apples-to-apples
            hits = _dedup_and_rerank(raw_hits)[:top_k]
            t_q1 = time.perf_counter()

            per_query_ms.append((t_q1 - t_q0) * 1000.0)
            all_hits_for_mix.extend(hits)
            hits_by_query_last_iter[q] = hits  # keep last iter (already deduped)
            if out_format in ("csv", "jsonl"):
                hits_rows_accum.extend(flat_hits_rows(q, hits, iter_id=it+1))
    t1 = time.perf_counter()
    P.close()

    total_ms = (t1 - t0) * 1000.0
    total_queries = n_q * iters
    avg_ms = (sum(per_query_ms) / len(per_query_ms)) if per_query_ms else 0.0
    p50_ms = percentile(per_query_ms, 50.0)
    p90_ms = percentile(per_query_ms, 90.0)
    qps = (total_queries / (total_ms / 1000.0)) if total_ms > 0 else 0.0

    # Source mix (normalize)
    mix_counts: Dict[str, int] = {}
    for h in all_hits_for_mix:
        src = (h.get("source") or "").strip().lower() or "unknown"
        mix_counts[src] = mix_counts.get(src, 0) + 1
    total_hits = sum(mix_counts.values()) or 1
    source_mix = {k: round(v / total_hits, 4) for k, v in sorted(mix_counts.items())}

    # Quality probes (optional)
    truth = load_truth_map(truth_json)
    recall_at_k, mrr = compute_recall_mrr(hits_by_query_last_iter, truth) if truth else (None, None)

    # Environment capture
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "faiss": getattr(faiss, "__version__", "unknown") if _FAISS_OK else "not_installed",
        "faiss_threads": int(getattr(faiss, "omp_get_max_threads", lambda: 0)()) if _FAISS_OK else None,
        "chromadb": _CHROMA_VER,
        "sentence_transformers": _ST_VER,
        "cpu_count": os.cpu_count(),
    }

    summary = {
        "backend": (backend or "config.default"),
        "index_key": effective_index,
        "config_hash": cfg_hash,
        "model_name": model_name,
        "metric": metric,
        "score_type": score_type,                 # similarity|distance
        "efSearch": ef_search,
        "alpha": alpha,
        "shard_count": shard_count,
        "n_queries": n_q,
        "iters": iters,
        "top_k": top_k,
        "avg_ms": round(avg_ms, 2),
        "p50_ms": round(p50_ms, 2),
        "p90_ms": round(p90_ms, 2),
        "qps": round(qps, 2),
        "source_mix": source_mix,
        "filter_source": (filter_source.strip().lower() if filter_source else None),
        "total_elapsed_ms": round(total_ms, 1),
        "quality": {
            "recall_at_k": recall_at_k,
            "mrr": mrr
        },
        "environment": env,
        "run_dir": str(run_dir),
        "hits_written": bool(out_format in ("csv","jsonl")),
    }

    # Write summary to timestamped run_dir
    out_summary_path = run_dir / "bench_summary.json"
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"[bench] wrote summary → {out_summary_path}")

    # Optional detailed hits (also under run_dir)
    if out_format == "jsonl":
        out_hits = run_dir / "hits.jsonl"
        with open(out_hits, "w", encoding="utf-8") as f:
            for row in hits_rows_accum:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(f"[bench] wrote hits jsonl → {out_hits}")
    elif out_format == "csv":
        out_hits = run_dir / "hits.csv"
        if hits_rows_accum:
            fieldnames = list(hits_rows_accum[0].keys())
            with open(out_hits, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(hits_rows_accum)
        log(f"[bench] wrote hits csv   → {out_hits}")

    # Echo summary
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/index/config.json")
    ap.add_argument("--index", default=None, help="Index key in config (defaults to active_index)")
    ap.add_argument("--backend", default=None, choices=["faiss","chroma"], help="Override backend")
    ap.add_argument("--qfile", required=True, help="Newline-delimited queries file")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--filter_source", default=None, help="wikipedia | pubmed | stackexchange")
    ap.add_argument("--format", default="jsonl", choices=["jsonl","csv","none"])
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--truth_json", default=None, help="Optional JSON with gold uids/ids per query for recall/MRR")
    args = ap.parse_args()

    run_bench(
        config_path=args.config,
        index_key=args.index,
        backend=args.backend,
        qfile=args.qfile,
        top_k=args.top_k,
        iters=args.iters,
        warmup=args.warmup,
        filter_source=args.filter_source,
        out_format=args.format if args.format != "none" else "",
        results_dir=args.results_dir,
        truth_json=args.truth_json,
    )

if __name__ == "__main__":
    main()
