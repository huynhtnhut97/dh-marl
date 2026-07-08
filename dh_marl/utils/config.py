"""YAML config loading with basic support for a `defaults:` include list.

We deliberately avoid Hydra so the runtime dependency footprint stays small and the
config semantics stay obvious. A config file may include:

    defaults:
      - default.yaml

and this loader will resolve them in declared order (later files override earlier).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    base_dir = path.parent
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    defaults = cfg.pop("defaults", []) or []
    resolved: dict[str, Any] = {}
    for d in defaults:
        d_path = base_dir / d
        resolved = _deep_merge(resolved, load_config(d_path))
    resolved = _deep_merge(resolved, cfg)
    return resolved


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
