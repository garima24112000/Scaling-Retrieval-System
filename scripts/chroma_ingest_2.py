import os, argparse, json
import numpy as np, pandas as pd
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTORS_ARE_NORMALIZED = True

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

    for col in ["domain", "source"]:
        if col in m.columns:
            m[col] = m[col].astype(str).str.strip().str.lower()

    for col in ["title", "url", "source_id", "id"]:
        if col in m.columns:
            m[col] = m[col].astype(str)

    if "chunk_id" in m.columns:
        m["chunk_id"] = m["chunk_id"].apply(lambda x: int(x) if pd.notna(x) else None)

    return m.to_dict(orient="records")

def ingest_chroma(meta_path: str, vec_path: str, collection_name: str, persist_dir: str):
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta parquet not found: {meta_path}")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"Vector .npy not found: {vec_path}")

    meta = pd.read_parquet(meta_path)
    vecs = np.load(vec_path).astype("float32")

    if len(meta) != len(vecs):
        raise ValueError(f"meta/vector length mismatch: {len(meta)} vs {len(vecs)}")

    os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
    os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))

    # Reset collection
    try:
        if collection_name in [c.name for c in client.list_collections()]:
            client.delete_collection(collection_name)
    except:
        pass

    coll = client.create_collection(name=collection_name)

    ids = _build_unique_ids(meta)

    print(f"Ingesting {len(meta):,} records into '{collection_name}' @ {persist_dir}")
    bs = 1000
    for i in tqdm(range(0, len(meta), bs)):
        batch = meta.iloc[i:i+bs]
        coll.add(
            ids=ids.iloc[i:i+bs].tolist(),
            embeddings=vecs[i:i+bs].tolist(),
            metadatas=_sanitize_metadata(batch)
        )

    manifest = {
        "collection": collection_name,
        "persist_dir": persist_dir,
        "n_records": len(meta),
        "fields": list(meta.columns),
        "id_format": "id::c<chunk_id> or id::row<i>",
        "model_name": MODEL_NAME,
        "vector_normed": VECTORS_ARE_NORMALIZED,
        "metric": "cosine"
    }

    man_path = os.path.join(persist_dir, f"{collection_name}_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("✓ Manifest created:", man_path)
    print("✓ Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--meta_path", required=True)
    ap.add_argument("--vec_path", required=True)
    ap.add_argument("--persist_dir", required=True)
    ap.add_argument("--collection", required=True)

    args = ap.parse_args()

    ingest_chroma(
        args.meta_path,
        args.vec_path,
        args.collection,
        args.persist_dir
    )
