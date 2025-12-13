import os, json, argparse, time
import numpy as np, pandas as pd
try:
    import faiss
except Exception:
    import faiss_cpu as faiss
from sentence_transformers import SentenceTransformer

class RetrievalRunner:
    """Loads FAISS + metadata and returns search results."""
    def __init__(self, config_path: str, index_key: str | None = None):
        # Load config
        with open(config_path, "r") as f:
            cfg = json.load(f)
        index_key = index_key or cfg.get("active_index")
        if not index_key or index_key not in cfg["indices"]:
            raise ValueError(f"Index key '{index_key}' missing or not found in config.")
        self.icfg = cfg["indices"][index_key]

        # 1) Load FAISS index
        if not os.path.exists(self.icfg["faiss_path"]):
            raise FileNotFoundError(f"FAISS index not found: {self.icfg['faiss_path']}")
        self.index = faiss.read_index(self.icfg["faiss_path"])
        print(f"Loaded FAISS index: {self.icfg['faiss_path']}")

        # 2) Load metadata
        if not os.path.exists(self.icfg["meta_path"]):
            raise FileNotFoundError(f"Meta parquet not found: {self.icfg['meta_path']}")
        self.meta  = pd.read_parquet(self.icfg["meta_path"])
        print(f"Loaded metadata rows: {len(self.meta)}")

        # 3) Load query encoder
        self.model = SentenceTransformer(self.icfg.get("model_name", "BAAI/bge-small-en-v1.5"))
        print(f"Loaded model: {self.icfg.get('model_name')}")

        # Search settings
        self.normalize_query = bool(self.icfg.get("normalize_query", True))
        try:
            faiss.ParameterSpace().set_index_parameter(self.index, "efSearch", 64)
        except Exception:
            pass

    def _encode(self, q: str) -> np.ndarray:
        """Encode text query to a vector."""
        if not q or not q.strip():
            raise ValueError("Empty query.")
        vec = self.model.encode([q], normalize_embeddings=self.normalize_query).astype("float32")
        return vec

    def search(self, query: str, top_k: int = 5, filter_source: str | None = None) -> list[dict]:
        """
        Returns list of dicts:
        rank, score, id, source, title, url, text, chunk_id
        """
        qv = self._encode(query)
        overfetch = top_k * 4 if filter_source else top_k
        D, I = self.index.search(qv, overfetch)
        results, taken = [], 0

        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue
            row = self.meta.iloc[idx]
            source = row.get("domain", "") or row.get("source", "")
            if filter_source and source != filter_source:
                continue
            results.append({
                "rank": taken + 1,
                "score": float(score),
                "id": row.get("id", ""),
                "source": source,
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "text": row.get("text", row.get("chunk_text", "")),
                "chunk_id": row.get("chunk_id", None)
            })
            taken += 1
            if taken >= top_k:
                break
        return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/index/config.json")
    ap.add_argument("--index", default=None, help="Index key to use (overrides active_index)")
    ap.add_argument("--query", default=None, help="Single query string")
    ap.add_argument("--qfile", default=None, help="Optional newline-delimited queries file")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--filter_source", default=None, help="e.g., wikipedia | pubmed | stackexchange")
    ap.add_argument("--out_csv", default=None, help="If set, write results to CSV at this path")
    args = ap.parse_args()

    rr = RetrievalRunner(args.config, args.index)

    # collect queries
    queries = []
    if args.query:
        queries.append(args.query)
    if args.qfile and os.path.exists(args.qfile):
        with open(args.qfile) as f:
            queries += [ln.strip() for ln in f if ln.strip()]
    if not queries:
        raise SystemExit("Provide --query or --qfile")

    # run
    t0 = time.time()
    rows = []
    for q in queries:
        hits = rr.search(q, top_k=args.top_k, filter_source=args.filter_source)
        for h in hits:
            rows.append({"query": q, **h})
    dt = time.time() - t0
    print(f"🕒 Ran {len(queries)} queries in {dt:.3f}s; avg {dt/len(queries):.3f}s/query")

    # output
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print("💾 Saved:", args.out_csv)
    else:
        for r in rows:
            print(f"\nQ: {r['query']}")
            print(f"{r['rank']}. [{r['source']}] {r['title']} (score={r['score']:.4f})")
            if r.get("url"):
                print(r["url"])

if __name__ == "__main__":
    main()
