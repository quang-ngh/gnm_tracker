"""Config loading with attribute access.

All thresholds / loss weights / schedule knobs live in `configs/*.yaml`
(Section 13.1). This module loads them into a nested namespace so code can do
`cfg.fit.stages.rigid.iters` while still allowing `cfg.get("a.b", default)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"


class Config:
    """Recursive attribute/dict view over a parsed YAML mapping."""

    def __init__(self, data: dict[str, Any]):
        self._data = data
        for key, value in data.items():
            setattr(self, key, _wrap(value))

    def get(self, dotted: str, default: Any = None) -> Any:
        """Look up a nested key like "fit.stages.rigid.iters"."""
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Config):
                node = getattr(node, part, None)
            elif isinstance(node, dict):
                node = node.get(part)
            else:
                return default
            if node is None:
                return default
        return node

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Config({self._data!r})"


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def repo_root() -> Path:
    return _REPO_ROOT


def resolve_path(path: str | Path) -> Path:
    """Resolve a possibly repo-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config (defaults to configs/default.yaml)."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data or {})
