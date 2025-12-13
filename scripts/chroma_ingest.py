import os, argparse, json
import numpy as np, pandas as pd
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTORS_ARE_NORMALIZED = True  # ✅ you already normalized corpus vectors at build time

def _build_unique_ids(meta: pd.DataFrame) -> pd.Series:
    if "chunk_id" in meta.columns:
        ch = meta["chunk_id"].fillna(-1).astype("int64").astype(str)
        uids = meta["id"].astype(str) + "::c" + ch
    else:
        uids = meta["id"].astype(str) + "::row" + pd.Series(range(len(meta)), index=meta.index).astype(str)
    if not uids.is_unique:
        uids = uids + "::" + pd.Series(range(len(uids)), index=uids.index).astype(str)
    return uids

def _sanitize_metadata(meta: pd.DataFrame) -> list[dict]:
    m = meta.copy()

    # Normalize domain/source consistently (lowercase, strip)
    if "domain" in m.columns:
        m["domain"] = m["domain"].astype(str).str.strip().str.lower()
    if "source" in m.columns:
        m["source"] = m["source"].astype(str).str.strip().str.lower()

    # Ensure common string fields are strings
    for col in ["title", "url", "source_id", "id"]:
        if col in m.columns:
            m[col] = m[col].astype(str)

    # chunk_id as int or None
    if "chunk_id" in m.columns:
        m["chunk_id"] = m["chunk_id"].apply(lambda x: int(x) if pd.notna(x) else None)

    return m.to_dict(orient="records")

def ingest_chroma(project_root: str, collection_name: str, persist_dir: str):
    emb_dir = f"{project_root}/data/embeddings/multi_default"
    meta_pq = f"{emb_dir}/multi_meta.parquet"
    vec_np  = f"{emb_dir}/multi_vectors.npy"
    if not (os.path.exists(meta_pq) and os.path.exists(vec_np)):
        raise FileNotFoundError(f"Missing merged files:\n- {meta_pq}\n- {vec_np}")

    meta = pd.read_parquet(meta_pq)
    vecs = np.load(vec_np).astype("float32")
    if len(meta) != len(vecs):
        raise ValueError(f"meta/vector length mismatch: {len(meta)} vs {len(vecs)}")

    # Normalize domain/source columns BEFORE batching (helps server-side filtering)
    if "domain" in meta.columns:
        meta["domain"] = meta["domain"].astype(str).str.strip().str.lower()
    elif "source" in meta.columns:
        meta["source"] = meta["source"].astype(str).str.strip().str.lower()

    os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))

    # Clean create the collection
    try:
        if collection_name in [c.name for c in client.list_collections()]:
            client.delete_collection(collection_name)
    except Exception:
        pass
    coll = client.create_collection(name=collection_name)

    ids = _build_unique_ids(meta)
    print(f"Ingesting {len(meta):,} into '{collection_name}' @ {persist_dir}")
    bs = 1000
    for i in tqdm(range(0, len(meta), bs)):
        batch = meta.iloc[i:i+bs]
        coll.add(
            ids=ids.iloc[i:i+bs].tolist(),
            embeddings=vecs[i:i+bs].tolist(),
            metadatas=_sanitize_metadata(batch)
        )

    info = {
        "collection": collection_name,
        "persist_dir": persist_dir,
        "n_records": len(meta),
        "fields": list(meta.columns),
        "id_format": "uid = id::c<chunk_id> (or ::row<i>)",
        "model_name": MODEL_NAME,
        "vector_normed": VECTORS_ARE_NORMALIZED,
        "metric": "cosine",  # Chroma uses cosine distance
        "score_semantics": "distance=1-cosine_similarity; report similarity=1-distance"
    }
    man_path = f"{project_root}/indices/{collection_name}_chroma_manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("✓ Manifest:", man_path)
    print("✓ Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="multi_default")
    ap.add_argument("--persist_dir", required=True)
    args = ap.parse_args()
    ingest_chroma("/content/drive/MyDrive/AMS 560 PROJECT", args.collection, args.persist_dir)
