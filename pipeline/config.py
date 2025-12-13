"""
Config loader for retrieval pipeline (scaling-ready).

- Supports JSON (.json) and YAML (.yml/.yaml)
- Validates index blocks (single-index or sharded) for FAISS backends
- Normalizes keys so backends see a consistent shape
- Adds schema versioning and metric/normalization semantics
- Supports remote paths (s3://, gs://, hdfs://) and optional path checks
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os

try:
    import yaml  # type: ignore
except Exception as e:
    raise ImportError(
        "pyyaml is required. Install with: pip install pyyaml"
    ) from e


class ConfigError(Exception):
    """Raised when configuration is missing/invalid."""


# Accepted config schema versions (simple gate for now)
_ACCEPTED_CONFIG_VERSIONS = {"1"}


@dataclass
class NormalizedIndexConfig:
    # core knobs
    backend: str
    model_name: str
    normalize_query: bool
    metric: str
    efSearch: int

    # union of single vs sharded (for FAISS backends)
    faiss_path: Optional[str] = None
    meta_path: Optional[str] = None
    index_paths: Optional[List[str]] = None
    meta_paths: Optional[List[str]] = None

    # semantics (for cross-backend parity & validation)
    index_metric: Optional[str] = None         # 'ip' | 'l2' | 'cosine'
    vector_normed: Optional[bool] = None       # True if stored vectors are unit-normalized

    # free-form passthrough (if backends need extras)
    extras: Dict[str, Any] = field(default_factory=dict)

    # loader context (useful for validation/telemetry)
    index_key: Optional[str] = None
    config_version: Optional[str] = None
    skip_path_checks: bool = False


def _load_raw_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise ConfigError(f"Config file not found: {path}")
    _, ext = os.path.splitext(path.lower())
    try:
        with open(path, "r", encoding="utf-8") as f:
            if ext == ".json":
                return json.load(f)
            if ext in (".yml", ".yaml"):
                return yaml.safe_load(f)
            # try JSON first then YAML if unknown extension
            try:
                f.seek(0)
                return json.load(f)
            except Exception:
                f.seek(0)
                return yaml.safe_load(f)
    except Exception as e:
        raise ConfigError(f"Failed to parse config {path}: {e}") from e


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required key '{key}' in {ctx}")
    return d[key]


def _as_bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    raise ConfigError(f"Expected boolean-like value, got: {v!r}")


def _as_int(v: Any, default: int) -> int:
    if v is None:
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        try:
            # tolerate "128 ", "1e2", etc.
            return int(float(s))
        except Exception:
            pass
    raise ConfigError(f"Expected integer-like value, got: {v!r}")


def _is_remote_path(p: str) -> bool:
    p = (p or "").lower()
    return p.startswith("s3://") or p.startswith("gs://") or p.startswith("hdfs://")


def _validate_paths_exist(paths: List[str], ctx: str, *, skip_checks: bool) -> None:
    if skip_checks:
        return
    missing = [
        p for p in paths
        if (not _is_remote_path(p)) and (not os.path.exists(p))
    ]
    if missing:
        raise ConfigError(f"{ctx}: missing files:\n  - " + "\n  - ".join(missing))


def _normalize_index_block(
    block: Dict[str, Any],
    ctx: str,
    index_key: str,
    top_level_skip_checks: bool,
) -> NormalizedIndexConfig:
    # knobs
    backend = (block.get("backend") or "faiss").lower()
    model_name = block.get("model_name", "BAAI/bge-small-en-v1.5")
    metric = (block.get("metric") or "ip").lower()
    efSearch = _as_int(block.get("efSearch"), 64)
    normalize_query = _as_bool(block.get("normalize_query"), True)

    # semantics (extra flags that help enforce parity across backends)
    index_metric = (block.get("index_metric") or metric or "").lower() or None
    vector_normed = block.get("vector_normed")
    if vector_normed is not None:
        vector_normed = _as_bool(vector_normed, False)

    # support single-index OR sharded-index (for FAISS backends)
    faiss_path = block.get("faiss_path")
    meta_path = block.get("meta_path")
    index_paths = block.get("index_paths")
    meta_paths = block.get("meta_paths")

    # per-block override for skipping path checks (e.g. cluster/remote mode)
    skip_path_checks = _as_bool(block.get("skip_path_checks"), top_level_skip_checks)

    # -------------------------------------------------------------
    # Only enforce single/sharded + path checks for FAISS backends.
    # For other backends (e.g., "chroma"), faiss_path/index_paths
    # are optional and can be omitted entirely.
    # -------------------------------------------------------------
    if backend == "faiss":
        single_ok = bool(faiss_path and meta_path)
        shard_ok = bool(index_paths and meta_paths)

        if single_ok and shard_ok:
            raise ConfigError(
                f"{ctx}: Provide EITHER single (faiss_path/meta_path) OR "
                f"sharded (index_paths/meta_paths), not both."
            )
        if not (single_ok or shard_ok):
            raise ConfigError(
                f"{ctx}: Must provide single (faiss_path+meta_path) OR "
                f"sharded (index_paths+meta_paths)."
            )

        # path existence checks
        if single_ok:
            _validate_paths_exist(
                [faiss_path, meta_path],
                f"{ctx} (single)",
                skip_checks=skip_path_checks,
            )
        else:
            if not isinstance(index_paths, list) or not isinstance(meta_paths, list):
                raise ConfigError(
                    f"{ctx}: index_paths/meta_paths must be lists for sharded mode."
                )
            if len(index_paths) != len(meta_paths):
                raise ConfigError(
                    f"{ctx}: index_paths and meta_paths length mismatch "
                    f"({len(index_paths)} vs {len(meta_paths)})."
                )
            _validate_paths_exist(
                index_paths,
                f"{ctx} index_paths",
                skip_checks=skip_path_checks,
            )
            _validate_paths_exist(
                meta_paths,
                f"{ctx} meta_paths",
                skip_checks=skip_path_checks,
            )

    # everything not core becomes extras
    extras = {
        k: v
        for k, v in block.items()
        if k
        not in {
            "backend",
            "model_name",
            "normalize_query",
            "metric",
            "efSearch",
            "faiss_path",
            "meta_path",
            "index_paths",
            "meta_paths",
            "index_metric",
            "vector_normed",
            "skip_path_checks",
        }
    }

    return NormalizedIndexConfig(
        backend=backend,
        model_name=model_name,
        normalize_query=normalize_query,
        metric=metric,
        efSearch=efSearch,
        faiss_path=faiss_path,
        meta_path=meta_path,
        index_paths=index_paths,
        meta_paths=meta_paths,
        index_metric=index_metric,
        vector_normed=vector_normed,
        extras=extras,
        index_key=index_key,
        config_version=None,         # filled by load_config
        skip_path_checks=skip_path_checks,
    )


def load_config(path: str, index_key: Optional[str] = None) -> Tuple[Dict[str, Any], NormalizedIndexConfig]:
    """
    Returns (raw_config_dict, normalized_index_cfg) for the chosen index key.
    - If index_key is None, uses raw['active_index'].
    - Validates presence and correctness of the selected index block.
    - Enforces schema version and provides remote-path/skip-check behavior.
    """
    raw = _load_raw_config(path)
    if not isinstance(raw, dict):
        raise ConfigError("Top-level config must be an object/dict.")

    # Schema versioning
    cfg_version = str(raw.get("config_version", "1")).strip()
    if cfg_version not in _ACCEPTED_CONFIG_VERSIONS:
        raise ConfigError(
            f"Unsupported config_version='{cfg_version}'. "
            f"Accepted versions: {_ACCEPTED_CONFIG_VERSIONS}"
        )

    indices = _require(raw, "indices", "root")
    if not isinstance(indices, dict) or not indices:
        raise ConfigError("'indices' must be a non-empty object.")

    if index_key is None:
        index_key = raw.get("active_index")
        if not index_key:
            raise ConfigError("Provide index_key or set 'active_index' in config.")

    if index_key not in indices:
        raise ConfigError(f"Index '{index_key}' not found in 'indices'.")

    block = indices[index_key]
    if not isinstance(block, dict):
        raise ConfigError(f"Index '{index_key}' must be an object.")

    # allow a top-level default for skipping path checks (e.g., cluster mode)
    top_level_skip_checks = _as_bool(raw.get("skip_path_checks"), False)

    nic = _normalize_index_block(
        block,
        f"index '{index_key}'",
        index_key,
        top_level_skip_checks,
    )
    # fill loader context fields
    nic.config_version = cfg_version

    if nic.index_metric and nic.metric and nic.index_metric != nic.metric:
        import warnings
        warnings.warn(
            f"Requested metric '{nic.metric}' differs from index_metric '{nic.index_metric}'. "
            "Ensure query normalization / FAISS setup matches."
        )

    return raw, nic
