"""Diagnose whether a VAE is a bottleneck for downstream classification.

This script follows the plan in `vae_bottleneck_experiments_v2.md`:

* Evaluate a teacher classifier on:
  - real images (orig)
  - VAE reconstructions (t=0)
  - latent-noised reconstructions (t-sweep)
* Report both GT metrics (AUROC/AUPRC) and agreement metrics
  between orig vs transformed inputs.

Outputs (per run):

outputs/diagnostics/vae_bottleneck/<exp_name>/
  config_used.yaml
  summary.json
  per_label.csv
  t_curve.csv
  worst_samples/
    <id>_orig.png
    <id>_recon_t0.png
    <id>_recon_tXXX.png (optional)
    <id>_diff_t0.png (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import csv
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_gpu_ids(cfg: dict[str, Any]) -> list[int]:
    """Priority: teacher.gpu_ids -> classifier.gpu_ids -> runtime.gpu_ids -> vae.gpu_ids -> [0]."""
    for a, b in [
        ("teacher", "gpu_ids"),
        ("classifier", "gpu_ids"),
        ("runtime", "gpu_ids"),
        ("vae", "gpu_ids"),
    ]:
        v = cfg.get(a, {}).get(b, None)
        if isinstance(v, list) and len(v) > 0:
            return [int(x) for x in v]
    return [0]


def set_visible_gpus(gpu_ids: list[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)


def select_indices(num_total: int, cfg_sampling: Dict[str, Any]) -> List[int]:
    import torch

    num_samples = cfg_sampling.get("num_samples", None)
    mode = cfg_sampling.get("mode", "first")
    seed = int(cfg_sampling.get("seed", 42))

    indices = list(range(num_total))
    if num_samples is None:
        return indices
    num_samples = min(int(num_samples), num_total)
    if mode == "first":
        return indices[:num_samples]
    if mode == "random":
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(num_total, generator=g).tolist()
        return perm[:num_samples]
    raise ValueError(f"Unknown sampling mode: {mode}")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def choose_global_threshold_macro_f1(
    y_true, y_prob,
    grid_start: float = 0.01,
    grid_end: float = 0.99,
    grid_num: int = 199,
) -> dict[str, Any]:
    """Choose a single global threshold that maximizes macro-F1 on real val."""
    import numpy as np

    y_true = y_true.astype(np.int32)
    ts = np.linspace(grid_start, grid_end, grid_num)

    best_t = None
    best_score = -1.0

    for t in ts:
        y_pred = (y_prob >= t).astype(np.int32)
        tp_c = ((y_pred == 1) & (y_true == 1)).sum(axis=0).astype(np.float64)
        fp_c = ((y_pred == 1) & (y_true == 0)).sum(axis=0).astype(np.float64)
        fn_c = ((y_pred == 0) & (y_true == 1)).sum(axis=0).astype(np.float64)
        f1_c = np.divide(
            2 * tp_c,
            2 * tp_c + fp_c + fn_c,
            out=np.zeros_like(tp_c),
            where=(2 * tp_c + fp_c + fn_c) > 0,
        )
        score = float(f1_c.mean())

        if (score > best_score) or (abs(score - best_score) < 1e-12 and (best_t is None or t > best_t)):
            best_score = score
            best_t = float(t)

    return {
        "best_threshold": float(best_t if best_t is not None else 0.5),
        "best_macro_f1": float(best_score),
        "grid": {"start": grid_start, "end": grid_end, "num": int(grid_num)},
    }


def bernoulli_kl(p: "np.ndarray", q: "np.ndarray", eps: float = 1e-6) -> "np.ndarray":
    """KL(Bern(p) || Bern(q)) elementwise."""
    import numpy as np

    p = np.clip(p, eps, 1.0 - eps)
    q = np.clip(q, eps, 1.0 - eps)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def sanitize_id(path: str, max_len: int = 120) -> str:
    """Make a filesystem-friendly id from a path."""
    s = path.replace("\\", "/")
    s = re.sub(r"[^0-9A-Za-z._/-]+", "_", s)
    s = s.strip("/_")
    if len(s) > max_len:
        s = s[-max_len:]
    s = s.replace("/", "__")
    return s


def main_worker(cfg: dict[str, Any], config_path: str | Path) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, Subset
    from torchvision import transforms as T
    from torchvision.utils import save_image
    from PIL import Image

    from src.sdxl_custom_vae.labels.schema import load_label_schema
    from src.sdxl_custom_vae.datasets.image_dataset import MultiLabelMedicalDataset
    from src.sdxl_custom_vae.teacher_classifier import build_convnext_large, build_teacher_transforms
    from src.sdxl_custom_vae.teacher_classifier.metrics import compute_multilabel_metrics
    from src.sdxl_custom_vae.sdxl.load_sdxl_vae import SDXLVAEConfig, load_sdxl_vae
    from src.sdxl_custom_vae.sdxl.noise import build_noise_scheduler

    exp_name = str(cfg.get("experiment_name", "diag_vae"))

    # -------------------------
    # output dir
    # -------------------------
    out_cfg = cfg.get("output", {}) or {}
    out_root = Path(out_cfg.get("root_dir", "outputs/diagnostics/vae_bottleneck"))
    out_dir = out_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # copy config
    config_path = Path(config_path)
    (out_dir / "config_used.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    # -------------------------
    # device / dtype
    # -------------------------
    vae_cfg = cfg.get("vae", {}) or {}
    device_str = str(vae_cfg.get("device", "cuda"))
    dtype_str = str(vae_cfg.get("dtype", "fp32")).lower()

    if torch.cuda.is_available() and device_str != "cpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if dtype_str == "fp16":
        torch_dtype = torch.float16
    elif dtype_str == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # seed
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -------------------------
    # label schema
    # -------------------------
    data_cfg = cfg.get("data", {}) or {}
    schema_path = data_cfg.get("label_schema_file", None)
    if not schema_path:
        raise KeyError("data.label_schema_file is required for diagnostic.")
    class_names, label_groups, group_reduce, mask_cfg = load_label_schema(schema_path)
    C = len(class_names)

    # -------------------------
    # transforms
    # -------------------------
    image_cfg = cfg.get("image", {}) or {}
    center_crop_size = int(image_cfg.get("center_crop_size", 3072))
    image_size = int(image_cfg.get("image_size", 1024))

    mean = tuple(data_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(data_cfg.get("std", [0.229, 0.224, 0.225]))

    teacher_tf = build_teacher_transforms(
        center_crop_size=center_crop_size,
        image_size=image_size,
        mean=mean,
        std=std,
        train=False,
        augment=None,
    )

    vae_tf = T.Compose([
        T.CenterCrop(center_crop_size),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Lambda(lambda x: x * 2.0 - 1.0),
    ])

    # for visualization: [0,1] without normalization
    vis_tf = T.Compose([
        T.CenterCrop(center_crop_size),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    # -------------------------
    # dataset
    # -------------------------
    base_ds = MultiLabelMedicalDataset(
        root=data_cfg["root"],
        split=str(data_cfg.get("split", "val")),
        classes=class_names,
        # We won't use base_ds.__getitem__, so any transform is OK.
        transform=lambda x: x,
        center_crop_size=center_crop_size,
        image_size=image_size,
        split_filename=str(data_cfg.get("split_filename", "default_split.yaml")),
        label_groups=label_groups,
        group_reduce=group_reduce,
        mask=mask_cfg,
    )

    class TwoViewDataset(Dataset):
        def __init__(self, base: MultiLabelMedicalDataset):
            self.base = base
            self.image_paths = base.image_paths
            self.labels = base.labels

        def __len__(self) -> int:
            return len(self.image_paths)

        def __getitem__(self, idx: int):
            path = self.image_paths[idx]
            label = self.labels[idx]
            with Image.open(path) as img:
                img = img.convert("RGB")
                x_teacher = teacher_tf(img)
                x_vae = vae_tf(img)
            return x_teacher, x_vae, label, str(path)

    ds = TwoViewDataset(base_ds)

    # optional sampling
    indices = select_indices(len(ds), cfg.get("sampling", {}) or {})
    ds = Subset(ds, indices)

    infer_cfg = cfg.get("inference", {}) or {}
    batch_size = int(infer_cfg.get("batch_size", 8))
    num_workers = int(infer_cfg.get("num_workers", 4))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # -------------------------
    # load teacher
    # -------------------------
    teacher_cfg = cfg.get("teacher", {}) or {}
    teacher_ckpt = teacher_cfg.get("checkpoint", None)
    if not teacher_ckpt:
        raise KeyError("teacher.checkpoint is required.")

    teacher = build_convnext_large(num_classes=C, pretrained=bool(teacher_cfg.get("imagenet_pretrained", True)))
    sd = torch.load(teacher_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] teacher checkpoint load: missing={len(missing)} unexpected={len(unexpected)}")
    teacher.to(device)
    teacher.eval()

    # optional embedding cosine
    agree_cfg = cfg.get("agreement", {}) or {}
    use_embedding_cos = bool(agree_cfg.get("embedding_cosine", False))

    def forward_teacher(x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Return (embedding, logits)."""
        if use_embedding_cos and hasattr(teacher, "features") and hasattr(teacher, "avgpool"):
            feat = teacher.features(x)
            feat = teacher.avgpool(feat)
            feat = torch.flatten(feat, 1)
            logits = teacher.classifier(feat)
            return feat, logits
        logits = teacher(x)
        return None, logits

    # -------------------------
    # load VAE
    # -------------------------
    vae_repo_id = str(vae_cfg.get("repo_id", "stabilityai/sdxl-vae"))
    vae = load_sdxl_vae(
        SDXLVAEConfig(
            repo_id=vae_repo_id,
            torch_dtype=torch_dtype,
            device="cuda" if device.type == "cuda" else "cpu",
        )
    )
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.13025))
    posterior_mode = str(vae_cfg.get("posterior", "mode")).lower()

    # -------------------------
    # noise scheduler
    # -------------------------
    noise_cfg = cfg.get("noise", {}) or {}
    timesteps: List[int] = [int(t) for t in (noise_cfg.get("timesteps", []) or [])]
    if not timesteps:
        # default to only t=0 (recon)
        timesteps = [0]
    if 0 not in timesteps:
        timesteps = [0] + timesteps
    timesteps = sorted(list(dict.fromkeys(timesteps)))

    scheduler = build_noise_scheduler(noise_cfg)
    noise_seed = int(noise_cfg.get("seed", 123))

    # single RNG stream for noise (reproducible, but not repeated every batch)
    noise_gen: Optional[torch.Generator] = None
    if len(timesteps) > 1:
        if device.type == "cuda":
            # For CUDA, rely on the CUDA RNG stream for reproducibility.
            torch.cuda.manual_seed_all(noise_seed)
            noise_gen = None
        else:
            noise_gen = torch.Generator()
            noise_gen.manual_seed(noise_seed)

    # -------------------------
    # helpers
    # -------------------------
    mean_t = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    def recon_to_teacher_input(x_recon: torch.Tensor) -> torch.Tensor:
        # x_recon: [-1,1]
        x01 = (x_recon.clamp(-1.0, 1.0) + 1.0) / 2.0
        x_norm = (x01 - mean_t) / std_t
        return x_norm

    # -------------------------
    # accumulation buffers
    # -------------------------
    y_true_all: list[np.ndarray] = []
    p_orig_all: list[np.ndarray] = []
    emb_orig_all: list[np.ndarray] = []

    p_by_t: dict[int, list[np.ndarray]] = {t: [] for t in timesteps}
    emb_by_t: dict[int, list[np.ndarray]] = {t: [] for t in timesteps}

    # top-k worst paths by recon(t=0) |Δp|
    topk = int(out_cfg.get("save_topk", 64))
    heap: list[Tuple[float, int, str]] = []
    uid = 0

    torch.set_grad_enabled(False)

    for batch in loader:
        x_teacher, x_vae, y, paths = batch
        x_teacher = x_teacher.to(device, dtype=torch.float32)
        x_vae = x_vae.to(device, dtype=torch_dtype)
        y = y.to(device, dtype=torch.float32)

        # teacher(orig)
        emb_o, logits_o = forward_teacher(x_teacher)
        p_orig = torch.sigmoid(logits_o).to(torch.float32)

        # encode
        posterior = vae.encode(x_vae).latent_dist
        if posterior_mode in ("mode", "mean", "deterministic"):
            latents = posterior.mode()
        else:
            latents = posterior.sample()

        # scale to diffusion latent space
        z0 = latents * scaling_factor

        # recon t=0 (decode z0)
        recon0 = vae.decode(z0 / scaling_factor).sample
        x_recon0_teacher = recon_to_teacher_input(recon0)
        emb_r0, logits_r0 = forward_teacher(x_recon0_teacher)
        p_r0 = torch.sigmoid(logits_r0).to(torch.float32)

        # update heap (worst by mean |Δp|)
        dp = (p_r0 - p_orig).abs().mean(dim=1)  # (B,)
        dp_cpu = dp.detach().cpu().numpy().astype(np.float64)
        for i, path in enumerate(paths):
            score = float(dp_cpu[i])
            if topk > 0:
                if len(heap) < topk:
                    heapq.heappush(heap, (score, uid, str(path)))
                else:
                    if score > heap[0][0]:
                        heapq.heapreplace(heap, (score, uid, str(path)))
            uid += 1

        # store
        y_true_all.append(y.detach().cpu().numpy().astype(np.int32))
        p_orig_all.append(p_orig.detach().cpu().numpy().astype(np.float32))
        if use_embedding_cos and emb_o is not None:
            emb_orig_all.append(emb_o.detach().cpu().numpy().astype(np.float32))
        p_by_t[0].append(p_r0.detach().cpu().numpy().astype(np.float32))
        if use_embedding_cos and emb_r0 is not None:
            emb_by_t[0].append(emb_r0.detach().cpu().numpy().astype(np.float32))

        # sweep t>0
        if len(timesteps) > 1:
            # Sample epsilon for this batch (reproducible across runs).
            if noise_gen is None:
                noise = torch.randn(z0.shape, device=device, dtype=torch_dtype)
            else:
                noise = torch.randn(z0.shape, generator=noise_gen, device=device, dtype=torch_dtype)

            for t in timesteps:
                if t == 0:
                    continue
                t_tensor = torch.full((z0.shape[0],), int(t), device=device, dtype=torch.long)
                zt = scheduler.add_noise(z0, noise, t_tensor)
                recon_t = vae.decode(zt / scaling_factor).sample
                x_recon_t_teacher = recon_to_teacher_input(recon_t)
                emb_t, logits_t = forward_teacher(x_recon_t_teacher)
                p_t = torch.sigmoid(logits_t).to(torch.float32)

                p_by_t[t].append(p_t.detach().cpu().numpy().astype(np.float32))
                if use_embedding_cos and emb_t is not None:
                    emb_by_t[t].append(emb_t.detach().cpu().numpy().astype(np.float32))

    # concat
    y_true = np.concatenate(y_true_all, axis=0)
    p_orig = np.concatenate(p_orig_all, axis=0)
    p_t_all: dict[int, np.ndarray] = {t: np.concatenate(v, axis=0) for t, v in p_by_t.items()}

    emb_orig = None
    emb_t_all: dict[int, Optional[np.ndarray]] = {t: None for t in timesteps}
    if use_embedding_cos and emb_orig_all:
        emb_orig = np.concatenate(emb_orig_all, axis=0)
        for t in timesteps:
            if emb_by_t[t]:
                emb_t_all[t] = np.concatenate(emb_by_t[t], axis=0)

    # threshold (real val)
    thr_cfg = cfg.get("threshold", {}) or {}
    thr_mode = str(thr_cfg.get("mode", "search_on_real_val")).lower()
    if thr_mode in ("fixed", "value"):
        threshold = float(thr_cfg.get("value", 0.5))
        thr_info = {"mode": "fixed", "best_threshold": threshold}
    else:
        # default: search on real val
        grid = thr_cfg.get("grid", {}) or {}
        thr_info = choose_global_threshold_macro_f1(
            y_true,
            p_orig,
            grid_start=float(grid.get("start", 0.01)),
            grid_end=float(grid.get("end", 0.99)),
            grid_num=int(grid.get("num", 199)),
        )
        threshold = float(thr_info["best_threshold"])

    # -------------------------
    # metrics
    # -------------------------
    # teacher metrics (GT)
    metrics_real = compute_multilabel_metrics(y_true, p_orig)
    metrics_by_t: dict[int, dict[str, Any]] = {t: compute_multilabel_metrics(y_true, p_t_all[t]) for t in timesteps}

    # agreement metrics
    def agreement(p_ref: np.ndarray, p_cmp: np.ndarray) -> dict[str, Any]:
        dp = np.abs(p_cmp - p_ref)
        mean_abs = float(dp.mean())
        mean_abs_per_class = dp.mean(axis=0).astype(np.float64)

        kl = bernoulli_kl(p_ref, p_cmp)
        mean_kl = float(kl.mean())
        mean_kl_per_class = kl.mean(axis=0).astype(np.float64)

        yhat_ref = (p_ref >= threshold).astype(np.int32)
        yhat_cmp = (p_cmp >= threshold).astype(np.int32)
        flip = (yhat_ref != yhat_cmp).astype(np.float32)
        flip_rate = float(flip.mean())
        flip_per_class = flip.mean(axis=0).astype(np.float64)

        out = {
            "mean_abs_dp": mean_abs,
            "mean_kl": mean_kl,
            "flip_rate": flip_rate,
            "mean_abs_dp_per_class": mean_abs_per_class,
            "mean_kl_per_class": mean_kl_per_class,
            "flip_rate_per_class": flip_per_class,
        }
        return out

    agree_by_t: dict[int, dict[str, Any]] = {t: agreement(p_orig, p_t_all[t]) for t in timesteps}

    # embedding cosine (optional)
    cos_by_t: dict[int, Optional[float]] = {t: None for t in timesteps}
    if use_embedding_cos and emb_orig is not None:
        for t in timesteps:
            emb_t = emb_t_all.get(t)
            if emb_t is None:
                continue
            # cosine per sample
            num = (emb_orig * emb_t).sum(axis=1)
            den = (np.linalg.norm(emb_orig, axis=1) * np.linalg.norm(emb_t, axis=1) + 1e-12)
            cos = num / den
            cos_by_t[t] = float(np.mean(cos))

    # -------------------------
    # write t_curve.csv
    # -------------------------
    t_rows: list[dict[str, Any]] = []
    for t in timesteps:
        m = metrics_by_t[t]
        a = agree_by_t[t]
        row = {
            "timestep": int(t),
            "macro_auroc": m.get("macro_auroc", None),
            "macro_auprc": m.get("macro_auprc", None),
            "mean_abs_dp": a.get("mean_abs_dp", None),
            "mean_kl": a.get("mean_kl", None),
            "flip_rate": a.get("flip_rate", None),
        }
        if use_embedding_cos:
            row["mean_cosine"] = cos_by_t.get(t, None)
        t_rows.append(row)
    write_csv(t_rows, out_dir / "t_curve.csv")

    # -------------------------
    # write per_label.csv (wide; includes t columns)
    # -------------------------
    per_label_rows: list[dict[str, Any]] = []
    # base per-class metrics
    auroc_real_pc = metrics_real.get("per_class_auroc", [None] * C)
    auprc_real_pc = metrics_real.get("per_class_auprc", [None] * C)

    # build t-specific per-class arrays
    auroc_pc_by_t = {t: metrics_by_t[t].get("per_class_auroc", [None] * C) for t in timesteps}
    auprc_pc_by_t = {t: metrics_by_t[t].get("per_class_auprc", [None] * C) for t in timesteps}

    # agreement per-class
    abs_pc_by_t = {t: agree_by_t[t]["mean_abs_dp_per_class"] for t in timesteps}
    kl_pc_by_t = {t: agree_by_t[t]["mean_kl_per_class"] for t in timesteps}
    flip_pc_by_t = {t: agree_by_t[t]["flip_rate_per_class"] for t in timesteps}

    for i, name in enumerate(class_names):
        row: dict[str, Any] = {
            "class": name,
            "auroc_real": auroc_real_pc[i],
            "auprc_real": auprc_real_pc[i],
        }
        # recon(t=0)
        row["auroc_t0"] = auroc_pc_by_t[0][i]
        row["auprc_t0"] = auprc_pc_by_t[0][i]
        row["mean_abs_dp_t0"] = float(abs_pc_by_t[0][i])
        row["mean_kl_t0"] = float(kl_pc_by_t[0][i])
        row["flip_rate_t0"] = float(flip_pc_by_t[0][i])

        # add all timesteps as columns
        for t in timesteps:
            if t == 0:
                continue
            row[f"auroc_t{t}"] = auroc_pc_by_t[t][i]
            row[f"auprc_t{t}"] = auprc_pc_by_t[t][i]
            row[f"mean_abs_dp_t{t}"] = float(abs_pc_by_t[t][i])
            row[f"mean_kl_t{t}"] = float(kl_pc_by_t[t][i])
            row[f"flip_rate_t{t}"] = float(flip_pc_by_t[t][i])
        per_label_rows.append(row)
    write_csv(per_label_rows, out_dir / "per_label.csv")

    # -------------------------
    # summary.json
    # -------------------------
    summary = {
        "experiment_name": exp_name,
        "num_samples": int(y_true.shape[0]),
        "num_classes": int(C),
        "threshold": {"value": float(threshold), **thr_info},
        "teacher_metrics_real": metrics_real,
        "teacher_metrics_recon_t0": metrics_by_t.get(0),
        "agreement_recon_t0": {
            "mean_abs_dp": float(agree_by_t[0]["mean_abs_dp"]),
            "mean_kl": float(agree_by_t[0]["mean_kl"]),
            "flip_rate": float(agree_by_t[0]["flip_rate"]),
            **({"mean_cosine": cos_by_t.get(0)} if use_embedding_cos else {}),
        },
        "t_curve": t_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # -------------------------
    # worst samples visualization
    # -------------------------
    if topk > 0 and heap:
        worst = sorted(heap, key=lambda x: x[0], reverse=True)
        worst_paths = [p for _, _, p in worst]
        worst_dir = out_dir / "worst_samples"
        worst_dir.mkdir(parents=True, exist_ok=True)

        # which timesteps to visualize (besides t=0)
        vis_steps: list[int] = [int(t) for t in (out_cfg.get("visualize_timesteps", []) or [])]
        vis_steps = [t for t in vis_steps if t in timesteps and t != 0]

        # helper to load a single image
        def load_pil(p: str) -> Image.Image:
            with Image.open(p) as img:
                return img.convert("RGB")

        for p in worst_paths:
            img = load_pil(p)
            x_vis = vis_tf(img)  # [0,1]
            x_vae1 = vae_tf(img).unsqueeze(0).to(device, dtype=torch_dtype)

            posterior = vae.encode(x_vae1).latent_dist
            lat = posterior.mode() if posterior_mode in ("mode", "mean", "deterministic") else posterior.sample()
            z = lat * scaling_factor

            recon0 = vae.decode(z / scaling_factor).sample.squeeze(0).to(torch.float32)
            recon0_01 = (recon0.clamp(-1, 1) + 1) / 2

            sid = sanitize_id(p)
            save_image(x_vis, worst_dir / f"{sid}_orig.png")
            save_image(recon0_01, worst_dir / f"{sid}_recon_t0.png")

            if bool(out_cfg.get("save_diff", True)):
                diff = (x_vis.to(torch.float32) - recon0_01).abs()
                # normalize for visibility
                dmax = float(diff.max().item())
                if dmax > 1e-8:
                    diff = diff / dmax
                save_image(diff, worst_dir / f"{sid}_diff_t0.png")

            # visualize extra timesteps
            if vis_steps:
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(noise_seed)
                    noise = torch.randn(z.shape, device=device, dtype=torch_dtype)
                else:
                    g = torch.Generator()
                    g.manual_seed(noise_seed)
                    noise = torch.randn(z.shape, generator=g, device=device, dtype=torch_dtype)
                for t in vis_steps:
                    t_tensor = torch.full((1,), int(t), device=device, dtype=torch.long)
                    zt = scheduler.add_noise(z, noise, t_tensor)
                    recon_t = vae.decode(zt / scaling_factor).sample.squeeze(0).to(torch.float32)
                    recon_t_01 = (recon_t.clamp(-1, 1) + 1) / 2
                    save_image(recon_t_01, worst_dir / f"{sid}_recon_t{t}.png")

    print(f"Done. Results saved to: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose VAE bottleneck using a teacher classifier.")
    p.add_argument("--config", type=str, required=True, help="Path to diagnostic YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    # Set visible GPUs early
    gpu_ids = get_gpu_ids(cfg)
    set_visible_gpus(gpu_ids)

    main_worker(cfg, args.config)


if __name__ == "__main__":
    main()
