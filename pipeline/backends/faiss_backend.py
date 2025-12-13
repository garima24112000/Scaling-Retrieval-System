from __future__ import annotations
import heapq, os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import faiss  # faiss-cpu
except Exception:  # pragma: no cover
    import faiss_cpu as faiss  # fallback if aliased

from sentence_transformers import SentenceTransformer
from pipeline.backends.base import VectorBackend, Result
from pipeline.config import NormalizedIndexConfig


class FaissBackend(VectorBackend):
    """
    FAISS backend supporting:
      • Single index (faiss_path + meta_path)
      • Sharded indices (index_paths + meta_paths) with global top-k merge
    """

    def __init__(self) -> None:
        self.cfg: Optional[NormalizedIndexConfig] = None
        self.model: Optional[SentenceTransformer] = None
        self.normalize_query: bool = True
        self.metric: str = "ip"
        self._higher_is_better: bool = True  # derived from index/metric (ip/cosine=True, l2=False)

        # single-index
        self.index = None
        self.meta_df: Optional[pd.DataFrame] = None

        # sharded
        self.shard_indices: List[Any] = []
        self.shard_metas: List[pd.DataFrame] = []
        self.sharded: bool = False

    # ---------- helpers ----------

    @staticmethod
    def _set_efsearch(index, ef: int) -> None:
        try:
            ps = faiss.ParameterSpace()
            ps.set_index_parameter(index, "efSearch", int(ef))
        except Exception:
            # Some FAISS builds or index types don't support this param (e.g., non-HNSW)
            pass

    @staticmethod
    def _set_nprobe(index, nprobe: Optional[int]) -> None:
        if nprobe is None:
            return
        try:
            ps = faiss.ParameterSpace()
            ps.set_index_parameter(index, "nprobe", int(nprobe))
        except Exception:
            # Silently ignore if index is not IVF or FAISS build lacks ParameterSpace
            pass

    @staticmethod
    def _norm_metric_name(name: Optional[str]) -> str:
        if not name:
            return ""
        n = str(name).strip().lower()
        if n in {"cos", "cosine", "cos_sim"}:
            return "cosine"
        if n in {"ip", "inner", "inner_product"}:
            return "ip"
        if n in {"l2", "euclidean"}:
            return "l2"
        return n

    def _derive_metric_direction(self) -> None:
        """
        Decide whether larger score = better. We normalize so that the internal 'score'
        always follows higher-is-better across metrics.
        """
        # Prefer the index's actual metric if provided; otherwise requested metric.
        m = self._norm_metric_name(getattr(self.cfg, "index_metric", None) or self.metric)
        self._higher_is_better = (m in {"ip", "cosine"})

    @staticmethod
    def _src_of_row(row: pd.Series) -> str:
        """Return a normalized (lowercased) source/domain string; empty if missing."""
        s = row.get("domain", "") or row.get("source", "") or ""
        return str(s).strip().lower()

    def _encode_one(self, text: str) -> np.ndarray:
        assert self.model is not None
        v = self.model.encode([text], normalize_embeddings=self.normalize_query)
        return np.asarray(v, dtype="float32")

    def _row_to_result(self, row: pd.Series, score: float, rank: int) -> Result:
        # choose best available text field
        txt = (row.get("chunk_text") or row.get("text") or row.get("preview") or row.get("content") or "")
        cid = row.get("chunk_id")
        try:
            cid = int(cid) if cid is not None and cid == cid else None
        except Exception:
            cid = None
        uid = f"{row.get('id','')}::c{cid if cid is not None else -1}"

        return {
            "rank": rank,
            "score": float(score),
            "id": row.get("id", ""),
            "uid": uid,
            "source": row.get("domain", "") or row.get("source", ""),
            "title": row.get("title", ""),
            "url": row.get("url", ""),
            "text": txt,
            "chunk_id": cid,
        }

    # ---------- VectorBackend API ----------

    def load(self, cfg: NormalizedIndexConfig) -> None:
        self.cfg = cfg
        self.normalize_query = bool(cfg.normalize_query)
        self.metric = cfg.metric
        self._derive_metric_direction()

        # threading control (optional knob)
        threads = int(cfg.extras.get("faiss_num_threads", os.cpu_count() or 8))
        try:
            faiss.omp_set_num_threads(threads)
            print(f"[FAISS] Using {threads} threads")
        except Exception:
            pass

        # model
        self.model = SentenceTransformer(cfg.model_name)

        # optional IVF knob
        nprobe = cfg.extras.get("nprobe")

        # load single or sharded indices
        if cfg.faiss_path and cfg.meta_path:
            self.sharded = False
            self.index = faiss.read_index(cfg.faiss_path)
            self._set_efsearch(self.index, cfg.efSearch)
            self._set_nprobe(self.index, nprobe)
            self.meta_df = pd.read_parquet(cfg.meta_path).reset_index(drop=True)
        else:
            self.sharded = True
            for ipath, mpath in zip(cfg.index_paths or [], cfg.meta_paths or []):
                idx = faiss.read_index(ipath)
                self._set_efsearch(idx, cfg.efSearch)
                self._set_nprobe(idx, nprobe)
                self.shard_indices.append(idx)
                self.shard_metas.append(pd.read_parquet(mpath).reset_index(drop=True))

        # sanity checks
        if not self.sharded:
            if self.index.ntotal != len(self.meta_df):
                raise ValueError(f"FAISS ntotal={self.index.ntotal} != meta rows={len(self.meta_df)}")
        else:
            nvec = sum(ix.ntotal for ix in self.shard_indices)
            nmeta = sum(len(md) for md in self.shard_metas)
            if nvec != nmeta:
                raise ValueError(f"[sharded] vectors {nvec} != meta rows {nmeta}")

    def encode(self, text_or_list: Any) -> np.ndarray:
        if isinstance(text_or_list, str):
            return self._encode_one(text_or_list)
        elif isinstance(text_or_list, list):
            assert self.model is not None
            v = self.model.encode(text_or_list, normalize_embeddings=self.normalize_query)
            return np.asarray(v, dtype="float32")
        raise TypeError("encode() expects str or list[str]")

    # ---- search helpers ----
    def _search_single(self, qvec: np.ndarray, top_k: int, filter_source: Optional[str]) -> List[Result]:
        norm_filter = filter_source.strip().lower() if filter_source else None
        dedup_on = bool(getattr(self.cfg, "extras", {}).get("dedup_by_uid", False))
        seen_uids: set[str] = set()

        overfetch = top_k * 4 if norm_filter else top_k
        D, I = self.index.search(qvec, overfetch)
        out: List[Result] = []
        taken = 0
        for idx, raw in zip(I[0], D[0]):
            if idx < 0:
                continue
            row = self.meta_df.iloc[idx]
            if norm_filter:
                if self._src_of_row(row) != norm_filter:
                    continue
            # Normalize score so "higher is better" regardless of metric.
            score = float(raw if self._higher_is_better else -raw)
            res = self._row_to_result(row, score, taken + 1)

            if dedup_on:
                uid = res["uid"]
                if uid in seen_uids:
                    continue
                seen_uids.add(uid)

            taken += 1
            out.append(res)
            if taken >= top_k:
                break
        return out

    def _search_sharded(self, qvec: np.ndarray, top_k: int, filter_source: Optional[str]) -> List[Result]:
        norm_filter = filter_source.strip().lower() if filter_source else None
        dedup_on = bool(getattr(self.cfg, "extras", {}).get("dedup_by_uid", False))
        seen_uids: set[str] = set()

        alpha = int(self.cfg.extras.get("alpha", 3))
        per = max(top_k * alpha, top_k)
        heap: List[Tuple[float, int, int, str]] = []

        for s_id, idx in enumerate(self.shard_indices):
            D, I = idx.search(qvec, per)
            for raw, li in zip(D[0], I[0]):
                if li < 0:
                    continue
                # Normalize score; store as a max-heap via negative key.
                score = float(raw if self._higher_is_better else -raw)
                uid = str(li)
                heapq.heappush(heap, (-score, s_id, int(li), uid))

        out: List[Result] = []
        seen = 0
        while heap and seen < top_k:
            neg, s_id, li, _ = heapq.heappop(heap)
            score = -neg
            row = self.shard_metas[s_id].iloc[li]
            if norm_filter:
                if self._src_of_row(row) != norm_filter:
                    continue

            res = self._row_to_result(row, score, seen + 1)

            if dedup_on:
                uid = res["uid"]
                if uid in seen_uids:
                    continue
                seen_uids.add(uid)

            seen += 1
            out.append(res)
        return out

    def search(self, qvec: Any, top_k: int = 5, filter_source: str | None = None) -> List[Result]:
        if not isinstance(qvec, np.ndarray):
            raise TypeError("search() expects numpy array (1,d)")
        if qvec.ndim != 2 or qvec.shape[0] != 1:
            raise ValueError("search() expects shape (1,d)")
        return (
            self._search_sharded(qvec, top_k, filter_source)
            if self.sharded
            else self._search_single(qvec, top_k, filter_source)
        )

    def close(self) -> None:
        self.index = None
        self.shard_indices.clear()
        self.shard_metas.clear()
        self.meta_df = None
        self.model = None
