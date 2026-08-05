import os
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    if Path(path).is_absolute():
        return Path(path)
    return PROJECT_ROOT / path


def load_config(path: str | Path) -> Dict[str, Any]:
    path = resolve_path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("project_root", str(PROJECT_ROOT))
    cfg.setdefault("output_dir", str(PROJECT_ROOT / "outputs"))
    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "auto")
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def get_output_dir(cfg: Dict[str, Any], subdir: str = "") -> Path:
    base = resolve_path(cfg.get("output_dir", "outputs"))
    if subdir:
        base = base / subdir
    base.mkdir(parents=True, exist_ok=True)
    return base
