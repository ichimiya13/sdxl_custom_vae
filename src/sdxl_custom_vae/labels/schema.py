from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_label_schema(schema_path: str | Path) -> tuple[list[str], dict[str, list[str]], str, dict[str, Any]]:
    """
    schema yaml:
      classes: [...]
      label_groups: { group: [source1, source2, ...] }  (optional)
      group_reduce: "any" (default)
      mask: {...} (optional)

    Returns:
      classes, label_groups, group_reduce, mask_cfg
    """
    p = Path(schema_path)
    with p.open("r", encoding="utf-8") as f:
        s = yaml.safe_load(f)

    if not isinstance(s, dict):
        raise ValueError(f"Invalid label schema yaml: {p}")

    classes = s.get("classes", None)
    if not isinstance(classes, list) or len(classes) == 0:
        raise KeyError(f"'classes' not found or empty in label schema: {p}")
    classes = [str(x) for x in classes]

    label_groups = s.get("label_groups", {}) or {}
    if not isinstance(label_groups, dict):
        raise ValueError(f"'label_groups' must be dict in label schema: {p}")
    label_groups = {str(k): [str(vv) for vv in v] for k, v in label_groups.items()}

    group_reduce = str(s.get("group_reduce", "any")).lower()

    mask_cfg = s.get("mask", {}) or {}
    if not isinstance(mask_cfg, dict):
        raise ValueError(f"'mask' must be dict in label schema: {p}")

    return classes, label_groups, group_reduce, mask_cfg
