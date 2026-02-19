# src/scripts/train_teacher_from_config.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml
from src.sdxl_custom_vae.labels.schema import load_label_schema

try:
    from tqdm.auto import tqdm  # progress bar
except Exception:
    tqdm = None  # fallback


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_gpu_ids(cfg: dict[str, Any]) -> list[int]:
    """
    優先順位：
      classifier.gpu_ids -> runtime.gpu_ids -> vae.gpu_ids -> [0]
    """
    for a, b in [("classifier", "gpu_ids"), ("runtime", "gpu_ids"), ("vae", "gpu_ids")]:
        v = cfg.get(a, {}).get(b, None)
        if isinstance(v, list) and len(v) > 0:
            return [int(x) for x in v]
    return [0]


def set_visible_gpus(gpu_ids: list[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)


def ddp_setup(
    rank: int,
    world_size: int,
    backend: str,
    master_addr: str,
    master_port: int,
    timeout_seconds: int | None = None,
) -> None:
    import datetime
    import torch.distributed as dist

    init_method = f"tcp://{master_addr}:{master_port}"

    kwargs = dict(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    if timeout_seconds is not None:
        kwargs["timeout"] = datetime.timedelta(seconds=int(timeout_seconds))

    dist.init_process_group(**kwargs)


def ddp_cleanup() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def train_worker(rank: int, world_size: int, cfg: dict[str, Any], config_path: str) -> None:
    # ---- warnings（邪魔なものだけ抑制）----
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"Grad strides do not match bucket view strides.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*torch\.cuda\.amp\.autocast.*",
    )

    # ---- torch import（CUDA_VISIBLE_DEVICES 設定後）----
    import numpy as np
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler

    from src.sdxl_custom_vae.datasets.image_dataset import MultiLabelMedicalDataset
    from src.sdxl_custom_vae.teacher_classifier import (
        build_convnext_large,
        AsymmetricLoss,
        build_teacher_transforms,
    )
    from src.sdxl_custom_vae.teacher_classifier.metrics import compute_multilabel_metrics

    distributed_cfg = cfg.get("distributed", {}) or {}
    use_ddp = bool(distributed_cfg.get("enabled", False)) and world_size > 1 and torch.cuda.is_available()

    # device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)  # rank は “見えているGPU列” のindex
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # seed（rankでずらす）
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)

    # DDP init（★timeoutを渡す）
    if use_ddp:
        timeout_seconds = distributed_cfg.get("timeout_seconds", None)
        # 複数ジョブ並列で遅くなるケースに備え、未指定なら長めを推奨
        if timeout_seconds is None:
            timeout_seconds = 7200  # 2h default safeguard

        ddp_setup(
            rank=rank,
            world_size=world_size,
            backend=str(distributed_cfg.get("backend", "nccl")),
            master_addr=str(distributed_cfg.get("master_addr", "127.0.0.1")),
            master_port=int(distributed_cfg.get("master_port", 29500)),
            timeout_seconds=int(timeout_seconds),
        )

    # tqdm設定
    train_cfg = cfg.get("train", {}) or {}
    use_pbar = bool(train_cfg.get("progress_bar", True)) and (tqdm is not None)
    pbar_disable = (rank != 0) or (not use_pbar)
    pbar_mininterval = float(train_cfg.get("tqdm_mininterval", 0.5))
    pbar_update_interval = int(train_cfg.get("tqdm_update_interval", 10))

    # autocast（FutureWarning対策：新APIを優先）
    def autocast_ctx(enabled: bool):
        if not torch.cuda.is_available():
            from contextlib import nullcontext
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", enabled=enabled)
        return torch.cuda.amp.autocast(enabled=enabled)

    # -------------------------
    # config
    # -------------------------
    data_cfg = cfg.get("data", {}) or {}

    schema_path = data_cfg.get("label_schema_file", None)
    if schema_path:
        classes, label_groups, group_reduce, mask_cfg = load_label_schema(schema_path)
    else:
        classes = data_cfg.get("classes", None)
        label_groups = data_cfg.get("label_groups", {}) or {}
        group_reduce = data_cfg.get("group_reduce", "any")
        mask_cfg = data_cfg.get("mask", {}) or {}
        if classes is None:
            raise KeyError(
                "Missing data.classes in config. "
                "Add data.classes or set data.label_schema_file."
            )

    # num_classes は classes 長に合わせる
    classifier_cfg = cfg.get("classifier", {}) or {}
    num_classes = len(classes)
    if "num_classes" in classifier_cfg and int(classifier_cfg["num_classes"]) != num_classes and rank == 0:
        print(
            f"[warn] classifier.num_classes({classifier_cfg['num_classes']}) != len(classes)({num_classes}). "
            f"Override to {num_classes}.",
            flush=True,
        )

    # image cfg
    image_cfg = cfg.get("image", {}) or {}
    center_crop_size = int(image_cfg.get("center_crop_size", 3072))
    image_size = int(image_cfg.get("image_size", 1024))

    # mean/std
    mean = tuple(data_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(data_cfg.get("std", [0.229, 0.224, 0.225]))

    # augmentation
    augment_cfg = cfg.get("augment", {}) or {}

    train_tf = build_teacher_transforms(
        center_crop_size=center_crop_size,
        image_size=image_size,
        mean=mean,
        std=std,
        train=True,
        augment=augment_cfg,
    )

    val_tf = build_teacher_transforms(
        center_crop_size=center_crop_size,
        image_size=image_size,
        mean=mean,
        std=std,
        train=False,
        augment=None,
    )

    # datasets（mask適用）
    train_ds = MultiLabelMedicalDataset(
        root=data_cfg["root"],
        split=data_cfg.get("train_split", "train"),
        classes=classes,
        transform=train_tf,
        center_crop_size=center_crop_size,
        image_size=image_size,
        split_filename=data_cfg.get("split_filename", "default_split.yaml"),
        label_groups=label_groups,
        group_reduce=group_reduce,
        mask=mask_cfg,
    )

    val_ds = MultiLabelMedicalDataset(
        root=data_cfg["root"],
        split=str(data_cfg.get("val_split", "val")),
        classes=classes,
        transform=val_tf,
        center_crop_size=center_crop_size,
        image_size=image_size,
        split_filename=str(data_cfg.get("split_filename", "default_split.yaml")),
        label_groups=label_groups,
        group_reduce=group_reduce,
        mask=mask_cfg,
    )

    if rank == 0:
        print(f"[mask] train kept={len(train_ds)} dropped={len(getattr(train_ds, 'dropped', []))}", flush=True)
        dc = getattr(train_ds, "dropped_counts", None)
        if dc:
            print(f"[mask] train dropped_counts={dc}", flush=True)

        print(f"[mask] val   kept={len(val_ds)} dropped={len(getattr(val_ds, 'dropped', []))}", flush=True)
        dc = getattr(val_ds, "dropped_counts", None)
        if dc:
            print(f"[mask] val dropped_counts={dc}", flush=True)

    # loaders
    batch_size = int(train_cfg.get("batch_size", 4))  # per-GPU
    num_workers = int(train_cfg.get("num_workers", 8))
    pin_memory = bool(train_cfg.get("pin_memory", True))

    if use_ddp:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # valはrank0 onlyで評価（推奨）
    eval_on_rank0_only = bool(distributed_cfg.get("eval_on_rank0_only", True))

    # model
    model = build_convnext_large(
        num_classes=num_classes,
        pretrained=bool(cfg.get("classifier", {}).get("pretrained", True)),
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # loss
    loss_cfg = cfg.get("loss", {}) or {}
    loss_fn = AsymmetricLoss(
        gamma_pos=float(loss_cfg.get("gamma_pos", 0.0)),
        gamma_neg=float(loss_cfg.get("gamma_neg", 4.0)),
        clip=float(loss_cfg.get("clip", 0.05)),
    ).to(device)

    # optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )

    # amp / accum
    amp = bool(train_cfg.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp) if torch.cuda.is_available() else None
    epochs = int(train_cfg.get("epochs", 20))
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    best_key = str(train_cfg.get("save_best_metric", "macro_auprc"))

    # output（rank0だけ）
    exp_name = cfg.get("experiment_name", Path(config_path).stem)
    out_root = Path(cfg.get("output", {}).get("root_dir", "outputs/checkpoints/teacher"))
    out_dir = out_root / exp_name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "used_config.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")

    # --- wandb setup (rank0 only) ---
    wandb_cfg = cfg.get("wandb", {}) or {}
    use_wandb = bool(wandb_cfg.get("enabled", False)) and (rank == 0)

    wandb_run = None
    if use_wandb:
        try:
            import wandb
        except Exception as e:
            print(f"[wandb] disabled because wandb import failed: {e}", flush=True)
            use_wandb = False
        else:
            mode = str(wandb_cfg.get("mode", "online"))
            if mode.lower() in ("disabled", "off", "false", "0"):
                use_wandb = False
            else:
                init_kwargs = dict(
                    project=wandb_cfg.get("project", None),
                    entity=wandb_cfg.get("entity", None),
                    name=wandb_cfg.get("name", exp_name),
                    group=wandb_cfg.get("group", None),
                    tags=wandb_cfg.get("tags", None),
                    notes=wandb_cfg.get("notes", None),
                    dir=str(wandb_cfg.get("dir", out_dir)),
                    config=cfg,
                    mode=mode,
                )
                init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
                wandb_run = wandb.init(**init_kwargs)

                wandb_run.summary["experiment_name"] = exp_name
                wandb_run.summary["output_dir"] = str(out_dir)
                wandb_run.summary["world_size"] = world_size

    best = None

    @torch.no_grad()
    def eval_rank0_full(epoch: int) -> dict[str, Any]:
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.eval()

        loader = DataLoader(
            val_ds,
            batch_size=int(train_cfg.get("eval_batch_size", batch_size)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        it = loader
        if tqdm is not None and use_pbar:
            it = tqdm(
                loader,
                desc=f"Val  {epoch}/{epochs}",
                disable=pbar_disable,
                dynamic_ncols=True,
                leave=False,
                mininterval=pbar_mininterval,
            )

        loss_sum = 0.0
        n_samples = 0
        all_logits = []
        all_targets = []

        for step, batch in enumerate(it, start=1):
            x, y, _path = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            logits = raw_model(x)
            loss_raw = loss_fn(logits, y)

            bs_ = int(x.shape[0])
            loss_sum += float(loss_raw.item()) * bs_
            n_samples += bs_

            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

            if tqdm is not None and use_pbar and (not pbar_disable) and (step % pbar_update_interval == 0):
                avg = loss_sum / max(n_samples, 1)
                it.set_postfix({"loss": f"{avg:.4f}"})  # type: ignore[attr-defined]

        logits_np = torch.cat(all_logits, 0).numpy()
        targets_np = torch.cat(all_targets, 0).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits_np))

        m = compute_multilabel_metrics(targets_np, probs)
        m["val_loss"] = loss_sum / max(n_samples, 1)
        return m

    log_interval = int(wandb_cfg.get("log_interval_steps", 50)) if use_wandb else 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        opt.zero_grad(set_to_none=True)

        it = train_loader
        if tqdm is not None and use_pbar:
            it = tqdm(
                train_loader,
                desc=f"Train {epoch}/{epochs}",
                disable=pbar_disable,
                dynamic_ncols=True,
                leave=False,
                mininterval=pbar_mininterval,
            )

        train_loss_sum = 0.0
        train_n_samples = 0

        for step, batch in enumerate(it, start=1):
            x, y, _path = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            with autocast_ctx(enabled=amp):
                logits = model(x)
                loss_raw = loss_fn(logits, y)
                loss = loss_raw / grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            bs_ = int(x.shape[0])
            train_loss_sum += float(loss_raw.item()) * bs_
            train_n_samples += bs_

            if step % grad_accum == 0:
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            if tqdm is not None and use_pbar and (not pbar_disable) and (step % pbar_update_interval == 0):
                avg = train_loss_sum / max(train_n_samples, 1)
                lr = opt.param_groups[0]["lr"]
                it.set_postfix({"loss": f"{avg:.4f}", "lr": f"{lr:.2e}"})  # type: ignore[attr-defined]

            global_step += 1
            if use_wandb and wandb_run is not None and log_interval > 0 and (global_step % log_interval == 0):
                lr = opt.param_groups[0]["lr"]
                train_avg = train_loss_sum / max(train_n_samples, 1)
                wandb_run.log(
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "train/loss_step": float(loss_raw.item()),
                        "train/loss_running": float(train_avg),
                        "train/lr": float(lr),
                    },
                    step=global_step,
                )

        # train loss: 全GPU平均
        train_loss_avg = train_loss_sum / max(train_n_samples, 1)
        if use_ddp:
            import torch.distributed as dist

            t_loss = torch.tensor(train_loss_sum, device=device, dtype=torch.float64)
            t_n = torch.tensor(train_n_samples, device=device, dtype=torch.float64)
            dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_n, op=dist.ReduceOp.SUM)
            train_loss_avg = float((t_loss / t_n.clamp(min=1.0)).item())

        # eval / save
        metrics = None
        if (not use_ddp) or (use_ddp and eval_on_rank0_only and rank == 0):
            metrics = eval_rank0_full(epoch)
            msg = (
                f"[epoch {epoch}] "
                f"train_loss={train_loss_avg:.4f} "
                f"val_loss={metrics['val_loss']:.4f} "
                f"macro_auroc={metrics['macro_auroc']} "
                f"macro_auprc={metrics['macro_auprc']}"
            )
            if tqdm is not None and use_pbar:
                tqdm.write(msg)
            else:
                print(msg, flush=True)

        if rank == 0:
            cur = metrics.get(best_key, None) if metrics is not None else None
            improved = cur is not None and (best is None or cur > best)
            if improved:
                best = float(cur)

            raw_model = model.module if isinstance(model, DDP) else model
            torch.save(
                {"model": raw_model.state_dict(), "cfg": cfg, "epoch": epoch, "metrics": metrics},
                out_dir / "last.pt",
            )
            if improved:
                torch.save(
                    {"model": raw_model.state_dict(), "cfg": cfg, "epoch": epoch, "metrics": metrics},
                    out_dir / "best.pt",
                )

        # ★ここで他rankが待つ。timeoutを十分大きくしておくのが今回のポイント
        if use_ddp:
            import torch.distributed as dist
            dist.barrier()

        if use_wandb and wandb_run is not None and metrics is not None and rank == 0:
            wandb_run.log(
                {
                    "global_step": global_step,
                    "epoch": epoch,
                    "train/loss_epoch": float(train_loss_avg),
                    "val/loss": float(metrics["val_loss"]),
                    "val/macro_auroc": metrics["macro_auroc"],
                    "val/macro_auprc": metrics["macro_auprc"],
                },
                step=global_step,
            )

    if rank == 0:
        print(f"Done. Best {best_key}={best}. Saved to: {out_dir}", flush=True)

    if use_ddp:
        ddp_cleanup()

    if use_wandb and wandb_run is not None:
        wandb_run.finish()


def _find_free_port() -> int:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


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

    # master_port を自動にしたい場合（任意）
    if ddp_enabled:
        mp_port = distributed_cfg.get("master_port", None)
        if mp_port in (None, 0, "auto"):
            p = _find_free_port()
            distributed_cfg["master_port"] = int(p)
            cfg["distributed"] = distributed_cfg
            print(f"[distributed] master_port auto-selected: {p}", flush=True)

    if ddp_enabled:
        world_size = len(gpu_ids)
        # mp.set_start_method("spawn") は不要（DataLoaderまでspawnになって重くなることがある）
        mp.spawn(train_worker, args=(world_size, cfg, args.config), nprocs=world_size, join=True)
    else:
        train_worker(rank=0, world_size=1, cfg=cfg, config_path=args.config)


if __name__ == "__main__":
    main()
