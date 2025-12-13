"""
Chroma backend implementation matching VectorBackend interface.
- Converts Chroma cosine distance → similarity (score = 1 - distance)
- Validates collection/model/normalization parity via manifest
- Adaptive overfetch when filtering; de-dup by uid
- Stable uid mapping: returns both 'uid' and base 'id'
"""

from __future__ import annotations
import os, json
from typing import Any, List, Dict, Optional

import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.backends.base import VectorBackend, Result


class ChromaBackend(VectorBackend):
    def __init__(self):
        self.client = None
        self.collection = None
        self.model: Optional[SentenceTransformer] = None
        self.cfg = None
        self.text_fields_pref: List[str] = ["chunk_text", "text", "preview"]
        self.alpha: int = 3
        self.dedup_by_uid: bool = True
        self.collection_name: str = "multi_default"
        self.persist_dir: Optional[str] = None
        self.manifest: Dict[str, Any] = {}

    def _resolve_extras(self, cfg) -> dict:
        ex = getattr(cfg, "extras", {}) or {}
        if isinstance(ex, dict) and "persist_dir" not in ex and "extras" in ex and isinstance(ex["extras"], dict):
            ex = ex["extras"]
        return ex

    def _load_manifest_if_exists(self):
        # Try to read indices/<collection>_chroma_manifest.json for parity checks
        if not self.persist_dir:
            return
        man_path = os.path.join(os.path.dirname(self.persist_dir.rstrip("/")), f"{self.collection_name}_chroma_manifest.json")
        # If persist_dir is ".../indices/chroma_store", manifest path above points to ".../indices/multi_default_chroma_manifest.json"
        alt_path = os.path.join(self.persist_dir, f"{self.collection_name}_chroma_manifest.json")
        for p in [man_path, alt_path]:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        self.manifest = json.load(f)
                except Exception:
                    self.manifest = {}
                break

    def _assert_parity(self):
        # Only assert if we found a manifest
        if not self.manifest:
            return
        want_model = getattr(self.cfg, "model_name", None)
        have_model = self.manifest.get("model_name")
        if want_model and have_model and want_model != have_model:
            raise RuntimeError(
                f"Model mismatch for Chroma collection '{self.collection_name}': "
                f"config.model_name='{want_model}' vs manifest.model_name='{have_model}'. "
                f"Re-ingest with scripts/chroma_ingest.py or switch config."
            )
        # Vector normalization parity
        vec_normed = self.manifest.get("vector_normed")
        if vec_normed is not None and not bool(vec_normed):
            # You said your corpus is normalized; if manifest claims otherwise, warn/raise.
            raise RuntimeError(
                f"Manifest indicates vector_normed={vec_normed} but config expects normalized corpus vectors. "
                f"Rebuild or verify ingestion."
            )
        # Count sanity (best-effort)
        try:
            coll_count = self.collection.count()
            man_count = int(self.manifest.get("n_records", coll_count))
            if abs(coll_count - man_count) > 0:
                raise RuntimeError(
                    f"Collection count ({coll_count}) != manifest n_records ({man_count}). "
                    f"Re-ingest to synchronize."
                )
        except Exception:
            pass

    def load(self, cfg):
        self.cfg = cfg
        ex = self._resolve_extras(cfg)

        self.persist_dir = ex.get("persist_dir")
        if not self.persist_dir:
            raise RuntimeError("Chroma backend requires extras.persist_dir in config.")

        self.collection_name = ex.get("collection", "multi_default")
        self.alpha = int(ex.get("alpha", 3))
        self.dedup_by_uid = bool(ex.get("dedup_by_uid", True))
        # optional text field preference
        if isinstance(ex.get("text_fields"), list) and ex["text_fields"]:
            self.text_fields_pref = list(ex["text_fields"])

        # Persistent client
        os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
        self.client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings(anonymized_telemetry=False))

        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Chroma collection '{self.collection_name}' not found in '{self.persist_dir}'. "
                f"Run: python scripts/chroma_ingest.py --collection {self.collection_name} --persist_dir '{self.persist_dir}'"
            ) from e

        # Model
        self.model = SentenceTransformer(cfg.model_name)

        # Parity checks
        self._load_manifest_if_exists()
        self._assert_parity()

        print(f"Loaded Chroma collection: {self.collection_name} | count={self.collection.count()} | store={self.persist_dir}")

    def encode(self, text_or_list):
        if isinstance(text_or_list, str):
            texts = [text_or_list]
        else:
            texts = list(text_or_list)
        vecs = self.model.encode(
            texts,
            normalize_embeddings=getattr(self.cfg, "normalize_query", True)
        ).astype("float32")
        return vecs

    def _pick_text(self, meta: Dict[str, Any]) -> str:
        for k in self.text_fields_pref:
            if k in meta and meta[k]:
                return meta[k]
        return ""

    def _split_uid(self, uid: str) -> (str, Optional[int]):
        # uid format: "<id>::c<chunk_id>" or "<id>::row<i>"
        if "::c" in uid:
            base, suf = uid.split("::c", 1)
            try:
                return base, int(suf)
            except Exception:
                return base, None
        if "::row" in uid:
            base, _ = uid.split("::row", 1)
            return base, None
        return uid, None

    def search(self, qvec, top_k: int = 5, filter_source: str | None = None) -> List[Result]:
    # Chroma returns cosine distances; convert to similarity for consistency
      overfetch = top_k * (self.alpha if filter_source else 1)
      out: List[Result] = []
      seen_uid = set()

      # --- Normalize filter_source for case-insensitive matching ---
      fs = filter_source.strip().lower() if filter_source else None
      where = {"domain": fs} if fs else None

      query = qvec[0].tolist()
      results = self.collection.query(
          query_embeddings=[query],
          n_results=overfetch,
          where=where
      )

      ids = results.get("ids", [[]])[0]
      dists = results.get("distances", [[]])[0]  # cosine distance
      metas = results.get("metadatas", [[]])[0]

      for uid, dist, md in zip(ids, dists, metas):
          if uid is None:
              continue
          sim = 1.0 - float(dist)  # similarity = 1 - distance
          meta = md or {}

          # Extract source robustly and compare case-insensitively
          source = (meta.get("domain") or meta.get("source") or "").strip().lower()
          if fs and source != fs:
              continue

          if self.dedup_by_uid:
              if uid in seen_uid:
                  continue
              seen_uid.add(uid)

          base_id, chunk_id = self._split_uid(uid)

          out.append({
              "rank": len(out) + 1,
              "score": sim,                          # report similarity (higher is better)
              "id": base_id,                         # base document id
              "uid": uid,                            # stable chunk-level id
              "source": source,                      # already normalized lower-case
              "title": meta.get("title", ""),
              "url": meta.get("url", ""),
              "text": self._pick_text(meta),
              "chunk_id": chunk_id
          })
          if len(out) >= top_k:
              break
      return out


    def close(self):
        try:
            if self.client and hasattr(self.client, "reset"):
                self.client.reset()
        except Exception:
            pass
        self.client = None
        self.collection = None
        self.model = None
