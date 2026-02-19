from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _is_pos(v: Any, thr: float) -> bool:
    try:
        return float(v) >= thr
    except Exception:
        return False


def should_drop_sample(label_dict: Dict[str, Any], mask_cfg: Dict[str, Any] | None) -> tuple[bool, Dict[str, Any]]:
    """
    Returns:
      drop (bool)
      info (dict):
        - reasons: list[str]
        - positive_labels: list[str]
        - threshold: float
    """
    if not mask_cfg:
        return False, {"reasons": [], "positive_labels": [], "threshold": 0.5}

    thr = float(mask_cfg.get("threshold", 0.5))

    # positive labels in raw yaml
    pos_labels: List[str] = []
    for k, v in (label_dict or {}).items():
        if _is_pos(v, thr):
            pos_labels.append(str(k))

    reasons: List[str] = []

    # Rule 1: drop if any of listed labels is positive
    drop_any = mask_cfg.get("drop_samples_if_any_positive", []) or []
    for name in drop_any:
        name = str(name)
        if name in pos_labels:
            reasons.append(f"any_positive:{name}")

    # Rule 2: coexist inconsistency (anchor positive + any other positive)
    ignore = set(str(x) for x in (mask_cfg.get("coexist_ignore_labels", []) or []))
    coexist_rules = mask_cfg.get("drop_samples_if_label_coexists", []) or []
    for rule in coexist_rules:
        if not isinstance(rule, dict):
            continue
        anchor = str(rule.get("label", ""))
        if not anchor:
            continue
        if anchor not in pos_labels:
            continue

        if bool(rule.get("with_any_other_positive", True)):
            others = [x for x in pos_labels if (x != anchor and x not in ignore)]
            if len(others) > 0:
                # 具体的な相手ラベルも残す（後で監査しやすい）
                reasons.append(f"coexist:{anchor}+{'+'.join(others[:8])}" + ("" if len(others) <= 8 else "+..."))

    drop = len(reasons) > 0
    info = {"reasons": reasons, "positive_labels": pos_labels, "threshold": thr}
    return drop, info
