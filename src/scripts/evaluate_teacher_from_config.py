from __future__ import annotations

import argparse
import os
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml
from tqdm import tqdm
from src.sdxl_custom_vae.labels.schema import load_label_schema


# -------------------------
# config helpers (torch import前)
# -------------------------
def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_gpu_ids(cfg: dict[str, Any]) -> list[int]:
    """
    優先順位:
      classifier.gpu_ids -> runtime.gpu_ids -> vae.gpu_ids -> [0]
    """
    for a, b in [("classifier", "gpu_ids"), ("runtime", "gpu_ids"), ("vae", "gpu_ids")]:
        v = cfg.get(a, {}).get(b, None)
        if isinstance(v, list) and len(v) > 0:
            return [int(x) for x in v]
    return [0]


def set_visible_gpus(gpu_ids: list[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)


# -------------------------
# metrics / threshold
# -------------------------
def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def compute_multilabel_metrics_bin(y_true, y_pred) -> dict[str, Any]:
    """
    y_true, y_pred: np.ndarray (N,C) with {0,1}
    出力:
      - accuracy_label: (TP+TN)/(N*C) （ラベル単位のaccuracy）
      - subset_accuracy: サンプル単位の完全一致率
      - precision_macro/micro
      - f1_macro/micro
      （参考として recall も返す）
    """
    import numpy as np

    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    N, C = y_true.shape
    tp_c = ((y_pred == 1) & (y_true == 1)).sum(axis=0).astype(np.float64)
    fp_c = ((y_pred == 1) & (y_true == 0)).sum(axis=0).astype(np.float64)
    fn_c = ((y_pred == 0) & (y_true == 1)).sum(axis=0).astype(np.float64)
    tn_c = ((y_pred == 0) & (y_true == 0)).sum(axis=0).astype(np.float64)

    precision_c = np.divide(tp_c, tp_c + fp_c, out=np.zeros_like(tp_c), where=(tp_c + fp_c) > 0)
    recall_c = np.divide(tp_c, tp_c + fn_c, out=np.zeros_like(tp_c), where=(tp_c + fn_c) > 0)
    f1_c = np.divide(2 * tp_c, 2 * tp_c + fp_c + fn_c, out=np.zeros_like(tp_c), where=(2 * tp_c + fp_c + fn_c) > 0)

    macro_precision = float(precision_c.mean()) if C > 0 else 0.0
    macro_recall = float(recall_c.mean()) if C > 0 else 0.0
    macro_f1 = float(f1_c.mean()) if C > 0 else 0.0

    tp = float(tp_c.sum())
    fp = float(fp_c.sum())
    fn = float(fn_c.sum())
    tn = float(tn_c.sum())

    micro_precision = _safe_div(tp, tp + fp)
    micro_recall = _safe_div(tp, tp + fn)
    micro_f1 = _safe_div(2 * tp, 2 * tp + fp + fn)

    accuracy_label = _safe_div(tp + tn, float(N * C))
    subset_accuracy = float(((y_pred == y_true).all(axis=1)).mean()) if N > 0 else 0.0

    return {
        "accuracy_label": accuracy_label,
        "subset_accuracy": subset_accuracy,
        "precision_macro": macro_precision,
        "precision_micro": micro_precision,
        "recall_macro": macro_recall,
        "recall_micro": micro_recall,
        "f1_macro": macro_f1,
        "f1_micro": micro_f1,
        "tp_micro": tp,
        "fp_micro": fp,
        "fn_micro": fn,
        "tn_micro": tn,
        "num_samples": int(N),
        "num_classes": int(C),
    }


def per_class_table(y_true, y_pred, class_names: list[str]) -> list[dict[str, Any]]:
    import numpy as np

    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    tp = ((y_pred == 1) & (y_true == 1)).sum(axis=0).astype(np.int64)
    fp = ((y_pred == 1) & (y_true == 0)).sum(axis=0).astype(np.int64)
    fn = ((y_pred == 0) & (y_true == 1)).sum(axis=0).astype(np.int64)
    tn = ((y_pred == 0) & (y_true == 0)).sum(axis=0).astype(np.int64)

    rows = []
    for i, name in enumerate(class_names):
        tp_i, fp_i, fn_i, tn_i = int(tp[i]), int(fp[i]), int(fn[i]), int(tn[i])
        prec = _safe_div(tp_i, tp_i + fp_i)
        rec = _safe_div(tp_i, tp_i + fn_i)
        f1 = _safe_div(2 * tp_i, 2 * tp_i + fp_i + fn_i)
        rows.append({
            "class": name,
            "support_pos": tp_i + fn_i,
            "support_neg": tn_i + fp_i,
            "tp": tp_i,
            "fp": fp_i,
            "fn": fn_i,
            "tn": tn_i,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })
    return rows


def choose_global_threshold_macro_f1(
    y_true, y_prob,
    grid_start: float = 0.01,
    grid_end: float = 0.99,
    grid_num: int = 199,
) -> dict[str, Any]:
    """
    VALの y_true/y_prob から、全クラス共通の閾値 t を macro_f1 最大で決定
    """
    import numpy as np

    y_true = y_true.astype(np.int32)
    ts = np.linspace(grid_start, grid_end, grid_num)

    best_t = None
    best_score = -1.0
    curve = []

    for t in ts:
        y_pred = (y_prob >= t).astype(np.int32)

        # macro f1（クラス平均）
        tp_c = ((y_pred == 1) & (y_true == 1)).sum(axis=0).astype(np.float64)
        fp_c = ((y_pred == 1) & (y_true == 0)).sum(axis=0).astype(np.float64)
        fn_c = ((y_pred == 0) & (y_true == 1)).sum(axis=0).astype(np.float64)

        f1_c = np.divide(
            2 * tp_c,
            2 * tp_c + fp_c + fn_c,
            out=np.zeros_like(tp_c),
            where=(2 * tp_c + fp_c + fn_c) > 0
        )
        score = float(f1_c.mean())

        curve.append({"threshold": float(t), "macro_f1": score})

        # tie-break: 同点なら高めの閾値（FP抑制寄り）を選ぶ
        if (score > best_score) or (abs(score - best_score) < 1e-12 and (best_t is None or t > best_t)):
            best_score = score
            best_t = float(t)

    return {
        "best_threshold": best_t,
        "best_macro_f1": best_score,
        "grid": {"start": grid_start, "end": grid_end, "num": int(grid_num)},
        "curve": curve,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if len(rows) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------
# DDP setup (optional)
# -------------------------
def ddp_setup(rank: int, world_size: int, backend: str, master_addr: str, master_port: int) -> None:
    import torch.distributed as dist
    init_method = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)


def ddp_cleanup() -> None:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# -------------------------
# worker
# -------------------------
def eval_worker(rank: int, world_size: int, cfg: dict[str, Any], config_path: str) -> None:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, DistributedSampler

    from src.sdxl_custom_vae.datasets.image_dataset import MultiLabelMedicalDataset
    from src.sdxl_custom_vae.teacher_classifier import build_convnext_large, build_teacher_transforms

    distributed_cfg = cfg.get("distributed", {}) or {}
    use_ddp = bool(distributed_cfg.get("enabled", False)) and world_size > 1 and torch.cuda.is_available()

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    if use_ddp:
        ddp_setup(
            rank=rank,
            world_size=world_size,
            backend=str(distributed_cfg.get("backend", "nccl")),
            master_addr=str(distributed_cfg.get("master_addr", "127.0.0.1")),
            master_port=int(distributed_cfg.get("master_port", 29510)),
        )

    data_cfg = cfg.get("data", {}) or {}
    image_cfg = cfg.get("image", {}) or {}
    infer_cfg = cfg.get("inference", {}) or {}
    thr_cfg = cfg.get("threshold", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    out_cfg = cfg.get("output", {}) or {}

    # label schema
    schema_path = data_cfg.get("label_schema_file", None)
    if not schema_path:
        raise KeyError("data.label_schema_file is required for evaluation.")
    class_names, label_groups, group_reduce, mask_cfg = load_label_schema(schema_path)
    C = len(class_names)

    # transforms（evalはtrain=Falseで固定）
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

    # splits
    splits = eval_cfg.get("splits", ["val"])
    if not isinstance(splits, list) or len(splits) == 0:
        splits = ["val"]

    split_name_to_key = {
        "train": data_cfg.get("train_split", "train"),
        "val": data_cfg.get("val_split", "val"),
        "test": data_cfg.get("test_split", "test"),
    }

    # output dir
    exp_name = cfg.get("experiment_name", Path(config_path).stem)
    out_root = Path(out_cfg.get("root_dir", "outputs/eval/teacher"))
    out_dir = out_root / exp_name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "used_config.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")

    # model load
    ckpt_path = cfg.get("checkpoint", {}).get("path", None)
    if not ckpt_path:
        raise KeyError("checkpoint.path is required.")
    ckpt_path = str(ckpt_path)

    model = build_convnext_large(num_classes=C, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # remove possible "module." prefix
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=True)
    model.eval()

    # inference settings
    bs = int(infer_cfg.get("batch_size", 16))
    num_workers = int(infer_cfg.get("num_workers", 8))
    pin_memory = bool(infer_cfg.get("pin_memory", True))

    save_predictions = bool(eval_cfg.get("save_predictions", True))
    save_paths = bool(eval_cfg.get("save_paths", False))  # 大きいならFalse推奨

    @torch.no_grad()
    def predict_on_dataset(ds: MultiLabelMedicalDataset) -> tuple[np.ndarray, np.ndarray, Optional[list[str]]]:
        if use_ddp:
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        else:
            sampler = None

        loader = DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        probs_list = []
        targets_list = []
        paths_list: list[str] = []

        for batch in tqdm(loader, desc=f"infer/{sp}", disable=use_ddp and rank != 0):
            x, y, paths = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            logits = model(x)
            probs = torch.sigmoid(logits)

            probs_list.append(probs.detach().cpu())
            targets_list.append(y.detach().cpu())
            if save_paths:
                paths_list.extend(list(paths))

        probs_np = torch.cat(probs_list, dim=0).numpy().astype(np.float32)
        targets_np = torch.cat(targets_list, dim=0).numpy().astype(np.float32)
        return probs_np, targets_np, (paths_list if save_paths else None)

    # run inference per split (each rank)
    local_results: dict[str, dict[str, Any]] = {}
    for sp in splits:
        sp_key = split_name_to_key.get(sp, sp)
        ds = MultiLabelMedicalDataset(
            root=data_cfg["root"],
            split=str(sp_key),
            classes=class_names,
            transform=tfm,
            center_crop_size=center_crop_size,
            image_size=image_size,
            split_filename=str(data_cfg.get("split_filename", "default_split.yaml")),
            label_groups=label_groups,
            group_reduce=group_reduce,
            mask=mask_cfg,
        )
        probs_np, targets_np, paths = predict_on_dataset(ds)
        local_results[sp] = {"probs": probs_np, "targets": targets_np, "paths": paths}
        if rank == 0:
            print(f"[mask] {sp}: kept={len(ds)} dropped={len(ds.dropped)}", flush=True)
            if ds.dropped_counts:
                print(f"[mask] {sp} dropped_counts={ds.dropped_counts}", flush=True)

    # gather to rank0
    if use_ddp:
        import torch.distributed as dist

        gathered: list[dict[str, Any]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered, local_results)

        if rank != 0:
            ddp_cleanup()
            return

        # rank0: concat for each split
        merged: dict[str, dict[str, Any]] = {}
        for sp in splits:
            probs_all = []
            targets_all = []
            paths_all: list[str] = []
            for gr in gathered:
                probs_all.append(gr[sp]["probs"])
                targets_all.append(gr[sp]["targets"])
                if save_paths and gr[sp]["paths"] is not None:
                    paths_all.extend(gr[sp]["paths"])
            merged[sp] = {
                "probs": np.concatenate(probs_all, axis=0),
                "targets": np.concatenate(targets_all, axis=0),
                "paths": (paths_all if save_paths else None),
            }
        results = merged
    else:
        results = local_results

    # -------------------------
    # threshold fit on VAL
    # -------------------------
    thr_mode = str(thr_cfg.get("mode", "global_from_val")).lower()
    metric_name = str(thr_cfg.get("metric", "macro_f1")).lower()

    if metric_name != "macro_f1":
        raise ValueError("This evaluator is configured for macro_f1 thresholding. Set threshold.metric: macro_f1")

    if "val" not in results:
        raise KeyError("VAL split is required to fit global threshold. Add 'val' to eval.splits.")

    if thr_mode == "fixed":
        best_t = float(thr_cfg.get("fixed_value", 0.5))
        thr_info = {"best_threshold": best_t, "mode": "fixed"}
        curve = None
    else:
        grid = thr_cfg.get("grid", {}) or {}
        grid_start = float(grid.get("start", 0.01))
        grid_end = float(grid.get("end", 0.99))
        grid_num = int(grid.get("num", 199))

        y_true_val = (results["val"]["targets"] >= 0.5).astype(np.int32)
        y_prob_val = results["val"]["probs"].astype(np.float32)

        fit = choose_global_threshold_macro_f1(
            y_true=y_true_val,
            y_prob=y_prob_val,
            grid_start=grid_start,
            grid_end=grid_end,
            grid_num=grid_num,
        )
        best_t = float(fit["best_threshold"])
        thr_info = {
            "mode": "global_from_val",
            "metric": "macro_f1",
            "best_threshold": best_t,
            "best_macro_f1": float(fit["best_macro_f1"]),
            "grid": fit["grid"],
        }
        curve = fit["curve"]

    # save threshold info
    (out_dir / "threshold.yaml").write_text(yaml.safe_dump(thr_info, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if curve is not None:
        write_csv(curve, out_dir / "threshold_curve.csv")

    print(f"[threshold] mode={thr_info.get('mode')} metric=macro_f1 best_t={best_t:.4f}", flush=True)

    # -------------------------
    # evaluate each split with best_t
    # -------------------------
    def eval_split(sp: str, probs: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
        y_true = (targets >= 0.5).astype(np.int32)
        y_pred = (probs >= best_t).astype(np.int32)

        m = compute_multilabel_metrics_bin(y_true, y_pred)

        # normal postprocess (異常なし): どれも陽性がなければ正常
        y_true_normal = (y_true.sum(axis=1) == 0).astype(np.int32)
        y_pred_normal = (y_pred.sum(axis=1) == 0).astype(np.int32)

        # binary metrics for normal
        tp = int(((y_pred_normal == 1) & (y_true_normal == 1)).sum())
        fp = int(((y_pred_normal == 1) & (y_true_normal == 0)).sum())
        fn = int(((y_pred_normal == 0) & (y_true_normal == 1)).sum())
        tn = int(((y_pred_normal == 0) & (y_true_normal == 0)).sum())

        normal_prec = _safe_div(tp, tp + fp)
        normal_f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
        normal_acc = _safe_div(tp + tn, tp + tn + fp + fn)

        m["normal/accuracy"] = normal_acc
        m["normal/precision"] = normal_prec
        m["normal/f1"] = normal_f1
        m["normal/tp"] = tp
        m["normal/fp"] = fp
        m["normal/fn"] = fn
        m["normal/tn"] = tn

        return m, y_true, y_pred

    all_metrics: dict[str, Any] = {"threshold": thr_info, "splits": {}}

    for sp in splits:
        probs = results[sp]["probs"]
        targets = results[sp]["targets"]
        m, y_true, y_pred = eval_split(sp, probs, targets)

        # print summary
        print(
            f"[{sp}] "
            f"accuracy_label={m['accuracy_label']:.4f} "
            f"precision_macro={m['precision_macro']:.4f} precision_micro={m['precision_micro']:.4f} "
            f"f1_macro={m['f1_macro']:.4f} f1_micro={m['f1_micro']:.4f} "
            f"subset_acc={m['subset_accuracy']:.4f} "
            f"normal_f1={m['normal/f1']:.4f}",
            flush=True
        )

        # save metrics json
        out_metrics_path = out_dir / f"metrics_{sp}.json"
        out_metrics_path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

        # per-class table
        rows = per_class_table(y_true, y_pred, class_names)
        write_csv(rows, out_dir / f"per_class_{sp}.csv")

        # optionally save predictions
        if save_predictions:
            npz_path = out_dir / f"preds_{sp}.npz"
            if save_paths and results[sp]["paths"] is not None:
                np.savez_compressed(npz_path, probs=probs, targets=targets, paths=np.array(results[sp]["paths"], dtype=object))
            else:
                np.savez_compressed(npz_path, probs=probs, targets=targets)

        all_metrics["splits"][sp] = m

    # optional wandb logging (rank0 only)
    wandb_cfg = cfg.get("wandb", {}) or {}
    if bool(wandb_cfg.get("enabled", False)):
        try:
            import wandb
            mode = str(wandb_cfg.get("mode", "online"))
            if mode.lower() not in ("disabled", "off", "false", "0"):
                run = wandb.init(
                    project=wandb_cfg.get("project", "uwf-teacher-eval"),
                    entity=wandb_cfg.get("entity", None),
                    name=wandb_cfg.get("name", exp_name),
                    group=wandb_cfg.get("group", None),
                    tags=wandb_cfg.get("tags", None),
                    notes=wandb_cfg.get("notes", None),
                    config=cfg,
                    mode=mode,
                )
                wandb.log({"threshold/best_t": best_t})
                for sp in splits:
                    mm = all_metrics["splits"][sp]
                    wandb.log({
                        f"{sp}/accuracy_label": mm["accuracy_label"],
                        f"{sp}/precision_macro": mm["precision_macro"],
                        f"{sp}/precision_micro": mm["precision_micro"],
                        f"{sp}/f1_macro": mm["f1_macro"],
                        f"{sp}/f1_micro": mm["f1_micro"],
                        f"{sp}/subset_accuracy": mm["subset_accuracy"],
                        f"{sp}/normal_f1": mm["normal/f1"],
                    })
                run.finish()
        except Exception as e:
            print(f"[wandb] skipped due to error: {e}", flush=True)

    # write combined summary
    (out_dir / "summary.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if use_ddp:
        ddp_cleanup()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    gpu_ids = get_gpu_ids(cfg)
    set_visible_gpus(gpu_ids)

    import torch
    import torch.multiprocessing as mp

    distributed_cfg = cfg.get("distributed", {}) or {}
    ddp_enabled = bool(distributed_cfg.get("enabled", False)) and len(gpu_ids) > 1 and torch.cuda.is_available()

    if ddp_enabled:
        world_size = len(gpu_ids)
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        mp.spawn(eval_worker, args=(world_size, cfg, args.config), nprocs=world_size, join=True)
    else:
        eval_worker(rank=0, world_size=1, cfg=cfg, config_path=args.config)


if __name__ == "__main__":
    main()
