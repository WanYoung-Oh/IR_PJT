from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(cfg: dict[str, Any], required: list[tuple[str, ...]]) -> None:
    """필수 키 경로 존재 여부를 검사. 누락 시 명확한 KeyError를 발생."""
    for key_path in required:
        node: Any = cfg
        for k in key_path:
            if not isinstance(node, dict) or k not in node:
                path_str = ".".join(key_path)
                raise KeyError(
                    f"Config missing required key: '{path_str}'. "
                    f"Check your config YAML."
                )
            node = node[k]


def resolve_config_path(root: Path, arg: str) -> Path:
    """CLI --config 인자를 절대 경로로 변환."""
    p = Path(arg)
    return p if p.is_absolute() else root / p


def repo_root_from(start: Path | None = None) -> Path:
    """Walk parents to find directory containing pyproject.toml."""
    cur = start or Path.cwd()
    for parent in [cur, *cur.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return cur
