# src/sdxl_custom_vae/utils/paths.py
from __future__ import annotations
from pathlib import Path

def find_repo_root(start: Path | None = None) -> Path:
    """
    repo root を探索して返す。
    条件: configs/ と src/ が存在するディレクトリを root とみなす
    """
    p = (start or Path(__file__).resolve())
    for d in [p] + list(p.parents):
        if (d / "configs").is_dir() and (d / "src").is_dir():
            return d
    raise RuntimeError("Repo root not found. Please run inside repository or check layout.")

def resolve_from_repo(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return find_repo_root() / p
