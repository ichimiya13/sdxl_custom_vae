from __future__ import annotations

import argparse
import os
import json
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from src.sdxl_custom_vae.labels.schema import load_label_schema
from src.sdxl_custom_vae.labels.masking import should_drop_sample

import yaml
from tqdm import tqdm


# -------------------------
# config / io utils
# -------------------------
def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_visible_gpu(gpu_id: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def read_label_yaml(label_path: Path) -> dict[str, Any]:
    with label_path.open("r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        return {}
    return d


def _fmt(v: Any, prec: int = 3) -> Any:
    """CSV書き出し用：floatだけ小数点以下prec桁に丸めて文字列化"""
    try:
        import numpy as np
        if isinstance(v, (np.floating,)):
            v = float(v)
    except Exception:
        pass

    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.{prec}f}"
    return v


def write_csv(rows: list[dict[str, Any]], path: Path, float_precision: int = 3) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            rr = {k: _fmt(v, float_precision) for k, v in r.items()}
            w.writerow(rr)


# -------------------------
# PR / AP (no sklearn)
# -------------------------
def average_precision(y_true, y_score) -> float:
    """
    Average Precision (AUPRC) の標準的な定義（rankベース）。
    y_true: {0,1}
    y_score: float
    """
    import numpy as np

    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)

    pos = int(y_true.sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(-y_score)
    yt = y_true[order]

    tp = np.cumsum(yt == 1)
    rank = np.arange(1, yt.shape[0] + 1)
    prec = tp / rank

    ap = float(prec[yt == 1].sum() / pos)
    return ap


def safe_mean(x) -> float:
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


# -------------------------
# dataset list / inference
# -------------------------
def build_image_list(root: Path, split_filename: str, split_name: str) -> list[Path]:
    sp = root / split_filename
    if not sp.is_file():
        raise FileNotFoundError(sp)
    split_dict = yaml.safe_load(sp.read_text(encoding="utf-8"))
    if split_name not in split_dict:
        raise KeyError(f"split '{split_name}' not found in {sp}")
    files = split_dict[split_name]
    if not isinstance(files, list):
        raise ValueError(f"split '{split_name}' must be list")
    return [root / Path(fn) for fn in files]


def run_inference(
    model,
    image_paths: list[Path],
    transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device,
) -> tuple["np.ndarray", list[str]]:
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    class _DS(Dataset):
        def __init__(self, paths, tfm):
            self.paths = paths
            self.tfm = tfm
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, i):
            p = self.paths[i]
            with Image.open(p) as img:
                img = img.convert("RGB")
            x = self.tfm(img)
            return x, str(p)

    ds = _DS(image_paths, transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    probs_all = []
    paths_all: list[str] = []
    model.eval()

    with torch.no_grad():
        for x, paths in tqdm(loader, desc="infer", leave=False):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).detach().cpu()
            probs_all.append(probs)
            paths_all.extend(list(paths))

    probs_np = torch.cat(probs_all, dim=0).numpy().astype(np.float32)
    return probs_np, paths_all


def load_convnext_from_ckpt(ckpt_path: str, num_classes: int, device):
    import torch
    from src.sdxl_custom_vae.teacher_classifier import build_convnext_large

    model = build_convnext_large(num_classes=num_classes, pretrained=False).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=True)
    return model


# -------------------------
# group analysis helpers
# -------------------------
def aggregate_scores(p: "np.ndarray", idxs: list[int], method: str) -> "np.ndarray":
    import numpy as np
    sub = p[:, idxs]
    if method == "max":
        return np.max(sub, axis=1)
    if method == "noisy_or":
        return 1.0 - np.prod(1.0 - sub, axis=1)
    raise ValueError(f"Unknown agg method: {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    gpu_id = int(cfg.get("inference", {}).get("gpu_id", 0))
    set_visible_gpu(gpu_id)

    import numpy as np
    import torch
    from src.sdxl_custom_vae.teacher_classifier import build_teacher_transforms

    # paths & output
    exp_name = cfg.get("experiment_name", Path(args.config).stem)
    out_root = Path(cfg.get("output", {}).get("root_dir", "outputs/analysis/group_compare"))
    out_dir = out_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "used_config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    float_precision = int(cfg.get("output", {}).get("float_precision", 3))

    # dataset list
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    split_filename = str(data_cfg.get("split_filename", "default_split.yaml"))
    split_name = str(data_cfg.get("split", "test"))
    image_paths = build_image_list(root, split_filename, split_name)

    # read raw label dict once (for subtype slicing)
    label_suffix = str(data_cfg.get("label_suffix", ".yaml"))
    label_dicts: list[dict[str, Any]] = []
    for p in tqdm(image_paths, desc="load labels"):
        lp = p.with_suffix(label_suffix)
        label_dicts.append(read_label_yaml(lp))


    # transforms
    image_cfg = cfg.get("image", {})
    center_crop_size = int(image_cfg.get("center_crop_size", 3072))
    image_size = int(image_cfg.get("image_size", 1024))
    mean = tuple(data_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(data_cfg.get("std", [0.229, 0.224, 0.225]))

    tfm = build_teacher_transforms(
        center_crop_size=center_crop_size,
        image_size=image_size,
        mean=mean,
        std=std,
        train=False,
        augment=None,
    )

    # models
    models_cfg: dict[str, Any] = cfg["models"]
    model_names = list(models_cfg.keys())  # YAML順を維持

    # reference groups
    analysis_cfg = cfg.get("analysis", {}) or {}
    ref_model = str(analysis_cfg.get("reference_model", "p3"))
    if ref_model not in models_cfg:
        # fallback: 最初に label_groups を持つ schema を探す
        ref_model = model_names[0]

    # load schemas
    schemas: dict[str, dict[str, Any]] = {}
    for mn in model_names:
        schema_path = models_cfg[mn]["schema"]
        classes, groups, _group_reduce, mask_cfg = load_label_schema(schema_path)
        schemas[mn] = {
            "classes": classes,
            "groups": groups,
            "idx": {name: i for i, name in enumerate(classes)},
            "schema_path": str(schema_path),
            "mask": mask_cfg,
        }

    # reference_model の mask を適用（全モデル共通で同じデータ集合で比較するため）
    ref_mask = schemas[ref_model].get("mask", {}) or {}

    kept_paths = []
    kept_dicts = []
    dropped_rows = []
    dropped_counts = {}

    for p, d in zip(image_paths, label_dicts):
        # read_label_yaml が {} を返しているなら invalid 判定できないので、
        # ここでは dictとして扱いつつ、必要なら empty を invalid として落とす運用も可能
        # 今回は「invalidは除外したい」ので empty dict を invalid 扱いにするなら以下を有効化:
        # if d == {}:
        #     dropped_rows.append({"path": str(p), "reasons": "invalid_label_yaml"})
        #     dropped_counts["invalid_label_yaml"] = dropped_counts.get("invalid_label_yaml", 0) + 1
        #     continue

        drop, info = should_drop_sample(d, ref_mask)
        if drop:
            dropped_rows.append({
                "path": str(p),
                "reasons": ";".join(info.get("reasons", [])),
                "positive_labels": ";".join(info.get("positive_labels", [])),
            })
            for r in info.get("reasons", []):
                dropped_counts[r] = dropped_counts.get(r, 0) + 1
            continue

        kept_paths.append(p)
        kept_dicts.append(d)

    print(f"[mask] before={len(image_paths)} after={len(kept_paths)} dropped={len(dropped_rows)}", flush=True)
    if dropped_counts:
        print(f"[mask] dropped_counts={dropped_counts}", flush=True)

    # 置換
    image_paths = kept_paths
    label_dicts = kept_dicts

    # dropped一覧を保存
    write_csv(dropped_rows, out_dir / f"dropped_{split_name}.csv", float_precision=float_precision)

    # label_groups は reference_model のものを使う
    ref_groups = schemas[ref_model]["groups"]
    if not ref_groups:
        raise ValueError(f"reference_model='{ref_model}' schema has empty label_groups. Set analysis.reference_model to p3 etc.")

    groups_include = analysis_cfg.get("groups_include", None)
    if groups_include is None:
        group_names = sorted(list(ref_groups.keys()))
    else:
        group_names = [str(x) for x in groups_include]

    agg_methods = analysis_cfg.get("aggregate_methods", ["max", "noisy_or"])
    agg_methods = [str(x) for x in agg_methods]

    # inference
    infer_cfg = cfg.get("inference", {})
    bs = int(infer_cfg.get("batch_size", 16))
    nw = int(infer_cfg.get("num_workers", 8))
    pin = bool(infer_cfg.get("pin_memory", True))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    probs_by_model: dict[str, np.ndarray] = {}
    paths_ref: Optional[list[str]] = None

    for mn in tqdm(model_names, desc="models"):
        ckpt = str(models_cfg[mn]["checkpoint"])
        num_classes = len(schemas[mn]["classes"])
        model = load_convnext_from_ckpt(ckpt, num_classes=num_classes, device=device)
        probs, paths = run_inference(model, image_paths, tfm, bs, nw, pin, device)

        if paths_ref is None:
            paths_ref = paths
        else:
            assert paths_ref == paths, f"Path order mismatch for model {mn}"

        probs_by_model[mn] = probs

    assert paths_ref == [str(p) for p in image_paths], "inference paths mismatch with split order"

    # helper for subtype split
    def get_label(d: dict[str, Any], name: str) -> int:
        v = d.get(name, 0)
        try:
            return 1 if float(v) >= 0.5 else 0
        except Exception:
            return 0

    def subtype_masks(group: str, sources: list[str]):
        import numpy as np

        base_label = group if group in sources else None
        src_any = np.zeros((len(label_dicts),), dtype=np.int32)
        for s in sources:
            src_any |= np.array([get_label(d, s) for d in label_dicts], dtype=np.int32)

        if base_label:
            base = np.array([get_label(d, base_label) for d in label_dicts], dtype=np.int32)

            variant_any = np.zeros((len(label_dicts),), dtype=np.int32)
            for s in sources:
                if s == base_label:
                    continue
                variant_any |= np.array([get_label(d, s) for d in label_dicts], dtype=np.int32)

            neg = (src_any == 0)
            base_only = (base == 1) & (variant_any == 0)
            variant_only = (base == 0) & (variant_any == 1)
            both = (base == 1) & (variant_any == 1)
            return {
                "neg": neg,
                "overall_pos": (src_any == 1),
                "base_only": base_only,
                "variant_only": variant_only,
                "both": both,
                "base_label": base_label,
            }
        else:
            neg = (src_any == 0)
            return {
                "neg": neg,
                "overall_pos": (src_any == 1),
                "base_only": None,
                "variant_only": None,
                "both": None,
                "base_label": None,
            }

    rows_summary: list[dict[str, Any]] = []
    rows_source: list[dict[str, Any]] = []

    for g in group_names:
        if g not in ref_groups:
            continue
        sources = list(ref_groups[g])

        masks = subtype_masks(g, sources)
        neg_mask = masks["neg"]
        overall_pos = masks["overall_pos"]
        y_true_overall = overall_pos.astype(np.int32)

        # ---- overall: pos(OR) vs neg(OR==0) ----
        for mn in model_names:
            probs = probs_by_model[mn]
            idx = schemas[mn]["idx"]

            # 1) direct group score if model has that class name
            if g in idx:
                sc = probs[:, idx[g]].astype(float)
                apv = average_precision(y_true_overall, sc)
                rows_summary.append({
                    "group": g,
                    "subset": "overall",
                    "model": mn,
                    "score_type": "direct",
                    "n_labels_used": 1,
                    "used_labels": g,
                    "auprc": apv,
                    "n_pos": int(y_true_overall.sum()),
                    "n_neg": int((y_true_overall == 0).sum()),
                    "mean_score_pos": safe_mean(sc[y_true_overall == 1]),
                    "mean_score_neg": safe_mean(sc[y_true_overall == 0]),
                })

            # 2) aggregate from available source labels
            avail = [s for s in sources if s in idx]
            if len(avail) > 0:
                src_idxs = [idx[s] for s in avail]
                for m in agg_methods:
                    sc = aggregate_scores(probs, src_idxs, method=m).astype(float)
                    apv = average_precision(y_true_overall, sc)
                    rows_summary.append({
                        "group": g,
                        "subset": "overall",
                        "model": mn,
                        "score_type": f"agg_{m}",
                        "n_labels_used": len(avail),
                        "used_labels": ";".join(avail),
                        "auprc": apv,
                        "n_pos": int(y_true_overall.sum()),
                        "n_neg": int((y_true_overall == 0).sum()),
                        "mean_score_pos": safe_mean(sc[y_true_overall == 1]),
                        "mean_score_neg": safe_mean(sc[y_true_overall == 0]),
                    })

        # ---- subtype: (base_only / variant_only / both) vs neg only ----
        for subtype in ["base_only", "variant_only", "both"]:
            pos_mask = masks.get(subtype, None)
            if pos_mask is None:
                continue
            use = neg_mask | pos_mask
            y_sub = pos_mask[use].astype(np.int32)

            for mn in model_names:
                probs = probs_by_model[mn]
                idx = schemas[mn]["idx"]

                # direct
                if g in idx:
                    sc = probs[:, idx[g]].astype(float)
                    apv = average_precision(y_sub, sc[use])
                    rows_summary.append({
                        "group": g,
                        "subset": subtype,
                        "model": mn,
                        "score_type": "direct",
                        "n_labels_used": 1,
                        "used_labels": g,
                        "auprc": apv,
                        "n_pos": int(y_sub.sum()),
                        "n_neg": int((y_sub == 0).sum()),
                        "mean_score_pos": safe_mean(sc[use][y_sub == 1]),
                        "mean_score_neg": safe_mean(sc[use][y_sub == 0]),
                    })

                # aggregate
                avail = [s for s in sources if s in idx]
                if len(avail) > 0:
                    src_idxs = [idx[s] for s in avail]
                    for m in agg_methods:
                        sc = aggregate_scores(probs, src_idxs, method=m).astype(float)
                        apv = average_precision(y_sub, sc[use])
                        rows_summary.append({
                            "group": g,
                            "subset": subtype,
                            "model": mn,
                            "score_type": f"agg_{m}",
                            "n_labels_used": len(avail),
                            "used_labels": ";".join(avail),
                            "auprc": apv,
                            "n_pos": int(y_sub.sum()),
                            "n_neg": int((y_sub == 0).sum()),
                            "mean_score_pos": safe_mean(sc[use][y_sub == 1]),
                            "mean_score_neg": safe_mean(sc[use][y_sub == 0]),
                        })

        # ---- per-source label analysis ----
        # source label positive vs group-absent negative（= 全て0）
        for s in sources:
            y_s = np.array([get_label(d, s) for d in label_dicts], dtype=np.int32)
            use = neg_mask | (y_s == 1)
            y_bin = y_s[use].astype(np.int32)

            for mn in model_names:
                probs = probs_by_model[mn]
                idx = schemas[mn]["idx"]

                # source score (if exists)
                if s in idx:
                    sc_src = probs[:, idx[s]].astype(float)
                    apv = average_precision(y_bin, sc_src[use])
                    rows_source.append({
                        "group": g,
                        "source_label": s,
                        "model": mn,
                        "score_type": "source",
                        "n_labels_used": 1,
                        "used_labels": s,
                        "auprc": apv,
                        "n_pos": int(y_bin.sum()),
                        "n_neg": int((y_bin == 0).sum()),
                        "mean_score_pos": safe_mean(sc_src[use][y_bin == 1]),
                        "mean_score_neg": safe_mean(sc_src[use][y_bin == 0]),
                    })

                # group score via direct (if group exists)
                if g in idx:
                    sc_g = probs[:, idx[g]].astype(float)
                    apv = average_precision(y_bin, sc_g[use])
                    rows_source.append({
                        "group": g,
                        "source_label": s,
                        "model": mn,
                        "score_type": "group_direct",
                        "n_labels_used": 1,
                        "used_labels": g,
                        "auprc": apv,
                        "n_pos": int(y_bin.sum()),
                        "n_neg": int((y_bin == 0).sum()),
                        "mean_score_pos": safe_mean(sc_g[use][y_bin == 1]),
                        "mean_score_neg": safe_mean(sc_g[use][y_bin == 0]),
                    })

                # group score via aggregate of available source labels
                avail = [ss for ss in sources if ss in idx]
                if len(avail) > 0:
                    src_idxs = [idx[ss] for ss in avail]
                    for m in agg_methods:
                        sc = aggregate_scores(probs, src_idxs, method=m).astype(float)
                        apv = average_precision(y_bin, sc[use])
                        rows_source.append({
                            "group": g,
                            "source_label": s,
                            "model": mn,
                            "score_type": f"group_agg_{m}",
                            "n_labels_used": len(avail),
                            "used_labels": ";".join(avail),
                            "auprc": apv,
                            "n_pos": int(y_bin.sum()),
                            "n_neg": int((y_bin == 0).sum()),
                            "mean_score_pos": safe_mean(sc[use][y_bin == 1]),
                            "mean_score_neg": safe_mean(sc[use][y_bin == 0]),
                        })

    # save (CSVは3桁丸め)
    write_csv(rows_summary, out_dir / "group_auprc_summary.csv", float_precision=float_precision)
    write_csv(rows_source, out_dir / "group_sourcelabel_auprc.csv", float_precision=float_precision)

    # raw summary json（数値は丸めない）
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "models": {mn: schemas[mn]["schema_path"] for mn in model_names},
                "reference_model": ref_model,
                "num_images": len(image_paths),
                "rows_summary": len(rows_summary),
                "rows_source": len(rows_source),
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(f"Saved: {out_dir}/group_auprc_summary.csv")
    print(f"Saved: {out_dir}/group_sourcelabel_auprc.csv")


if __name__ == "__main__":
    main()
