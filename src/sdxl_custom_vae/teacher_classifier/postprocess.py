# src/sdxl_custom_vae/teacher_classifier/postprocess.py
from __future__ import annotations

from typing import Sequence, Mapping, Union, Any
import numpy as np
import torch


ThresholdSpec = Union[float, Sequence[float], Mapping[str, float]]


def _thresholds_to_array(
    thresholds: ThresholdSpec,
    class_names: Sequence[str],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    thresholds を (C,) の Tensor に変換する
    - float: 全クラス共通
    - list/tuple: クラス順に対応
    - dict: class_name -> threshold
    """
    C = len(class_names)

    if isinstance(thresholds, (float, int)):
        arr = torch.full((C,), float(thresholds), device=device, dtype=dtype or torch.float32)
        return arr

    if isinstance(thresholds, Mapping):
        vals = [float(thresholds[name]) for name in class_names]
        return torch.tensor(vals, device=device, dtype=dtype or torch.float32)

    # sequence
    if len(thresholds) != C:
        raise ValueError(f"thresholds length mismatch: expected {C}, got {len(thresholds)}")
    return torch.tensor([float(x) for x in thresholds], device=device, dtype=dtype or torch.float32)


@torch.no_grad()
def add_normal_if_none_positive(
    probs: torch.Tensor,
    class_names: Sequence[str],
    thresholds: ThresholdSpec = 0.5,
    normal_label: str = "異常なし",
) -> dict[str, Any]:
    """
    病変クラスの確率 probs (sigmoid後) に対して閾値判定し、
    どれも陽性でなければ normal_label を1にする。

    probs:
      - shape (C,) または (B, C)
      - torch.float

    return:
      - singleの場合: {"class_names": [...], "pred": Tensor[(C+1,)]}
      - batchの場合 : {"class_names": [...], "pred": Tensor[(B, C+1)]}

    predの先頭に normal_label を追加します。
    """
    if probs.ndim not in (1, 2):
        raise ValueError(f"probs must be 1D or 2D, got shape={tuple(probs.shape)}")

    device = probs.device
    thr = _thresholds_to_array(thresholds, class_names, device=device, dtype=probs.dtype)

    if probs.ndim == 1:
        # (C,)
        pos = probs >= thr
        any_pos = bool(pos.any().item())
        normal = torch.tensor([0.0 if any_pos else 1.0], device=device, dtype=probs.dtype)
        pred = torch.cat([normal, pos.float()], dim=0)  # (C+1,)
        return {"class_names": [normal_label] + list(class_names), "pred": pred}

    # (B, C)
    pos = probs >= thr.unsqueeze(0)  # broadcast
    any_pos = pos.any(dim=1)         # (B,)
    normal = (~any_pos).float().unsqueeze(1)  # (B,1)
    pred = torch.cat([normal, pos.float()], dim=1)  # (B, C+1)
    return {"class_names": [normal_label] + list(class_names), "pred": pred}


@torch.no_grad()
def probs_to_pred_dicts(
    probs: torch.Tensor,
    class_names: Sequence[str],
    thresholds: ThresholdSpec = 0.5,
    normal_label: str = "異常なし",
) -> list[dict[str, int]]:
    """
    add_normal_if_none_positive の結果を、人間が扱いやすい dict のリストに変換する。
    """
    out = add_normal_if_none_positive(probs, class_names, thresholds=thresholds, normal_label=normal_label)
    names = out["class_names"]
    pred = out["pred"]

    if pred.ndim == 1:
        pred = pred.unsqueeze(0)

    result = []
    for b in range(pred.shape[0]):
        d = {names[i]: int(pred[b, i].item() >= 0.5) for i in range(len(names))}
        result.append(d)
    return result
