from __future__ import annotations
from typing import Optional, List, Dict, Any

from pipeline.config import load_config, NormalizedIndexConfig
from pipeline.backends.base import VectorBackend
from pipeline.backends.chroma_backend import ChromaBackend
from pipeline.backends.faiss_backend import FaissBackend  # ensure this file exists

_REGISTRY = {
    "chroma": ChromaBackend,
    "faiss": FaissBackend,
}

class Pipeline:
    def __init__(self, backend: VectorBackend, nicfg: NormalizedIndexConfig):
        self.backend = backend
        self.nicfg = nicfg

    def search_one(self, query: str, top_k: int = 5, filter_source: str | None = None) -> List[Dict[str, Any]]:
        qvec = self.backend.encode(query)
        return self.backend.search(qvec, top_k=top_k, filter_source=filter_source)

    def search_many(self, queries: List[str], top_k: int = 5, filter_source: str | None = None) -> Dict[str, List[Dict[str, Any]]]:
        qvecs = self.backend.encode(queries)
        out = {}
        for i, q in enumerate(queries):
            out[q] = self.backend.search(qvecs[i:i+1], top_k=top_k, filter_source=filter_source)
        return out

    def close(self):
        self.backend.close()

def load_pipeline(cfg_path: str, index_key: Optional[str] = None, backend_override: Optional[str] = None) -> Pipeline:
    raw, nic = load_config(cfg_path, index_key=index_key)
    bkey = (backend_override or nic.backend or "faiss").lower()
    if bkey not in _REGISTRY:
        raise ValueError(f"Unknown backend '{bkey}'. Available: {list(_REGISTRY)}")
    backend = _REGISTRY[bkey]()  # type: ignore
    backend.load(nic)
    return Pipeline(backend, nic)
