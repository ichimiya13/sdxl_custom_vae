"""Train / fine-tune a VAE (diffusers AutoencoderKL) on UWF images.

This repository already supports:
  - teacher classifier training/evaluation
  - VAE reconstruction (inference)

This script adds VAE training so we can compare bottleneck capacity
with latent channels C in {4, 8, 16} and different loss recipes.

Minimal baseline (Base-VAE): recon loss + KL.
Optional (Feat-VAE): add a frozen teacher feature matching loss.

Config-driven execution:

python -m src.scripts.train_vae_from_config --config configs/vae/train/vae_base_ft_4ch.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from tqdm.auto import tqdm  # progress bar
except Exception:
    tqdm = None  # fallback


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_gpu_ids(cfg: dict[str, Any]) -> list[int]:
    """Priority: vae.gpu_ids -> runtime.gpu_ids -> [0]."""
    for a, b in [("vae", "gpu_ids"), ("runtime", "gpu_ids")]:
        v = cfg.get(a, {}).get(b, None)
        if isinstance(v, list) and len(v) > 0:
            return [int(x) for x in v]
    return [0]


def set_visible_gpus(gpu_ids: list[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)


def build_vae_transform(center_crop_size: int, image_size: int, augment: Dict[str, Any] | None = None):
    from torchvision import transforms as T

    augment = augment or {}
    hflip_p = float(augment.get("hflip_p", 0.0))

    tfms = [
        T.CenterCrop(center_crop_size),
        T.Resize((image_size, image_size)),
    ]
    if hflip_p > 0:
        tfms.append(T.RandomHorizontalFlip(p=hflip_p))
    tfms += [
        T.ToTensor(),
        T.Lambda(lambda x: x * 2.0 - 1.0),  # [-1,1]
    ]
    return T.Compose(tfms)


def _kl_loss_from_posterior(posterior) -> "torch.Tensor":
    """Compute KL(q(z|x) || N(0,I)).

    Supports diffusers' DiagonalGaussianDistribution.
    """
    import torch

    if hasattr(posterior, "kl"):
        try:
            kl = posterior.kl()
            if isinstance(kl, torch.Tensor):
                return kl.mean() if kl.ndim > 0 else kl
        except Exception:
            pass

    # fallback: manual
    if not (hasattr(posterior, "mean") and hasattr(posterior, "logvar")):
        raise RuntimeError("posterior does not expose mean/logvar; cannot compute KL")

    mean = posterior.mean
    logvar = posterior.logvar
    kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)
    # sum over latent dims
    while kl.ndim > 1:
        kl = kl.sum(dim=-1)
    return kl.mean()


def train_from_config(cfg: dict[str, Any], config_path: str | Path) -> None:
    import sys
    import time
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from src.sdxl_custom_vae.labels.schema import load_label_schema
    from src.sdxl_custom_vae.datasets.image_dataset import MultiLabelMedicalDataset
    from src.sdxl_custom_vae.sdxl.vae_factory import build_autoencoder_kl
    from src.sdxl_custom_vae.teacher_classifier import build_convnext_large

    exp_name = str(cfg.get("experiment_name", "vae_train"))

    # -------------------------
    # output dir
    # -------------------------
    out_cfg = cfg.get("output", {}) or {}
    out_root = Path(out_cfg.get("root_dir", "outputs/checkpoints/vae"))
    out_dir = out_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")

    # -------------------------
    # device / dtype
    # -------------------------
    vae_cfg = cfg.get("vae", {}) or {}
    device_str = str(vae_cfg.get("device", "cuda"))
    dtype_str = str(vae_cfg.get("dtype", "fp16")).lower()

    if torch.cuda.is_available() and device_str != "cpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Keep master weights in fp32; use cfg dtype only for autocast compute
    param_dtype = torch.float32
    if dtype_str == "fp16":
        amp_dtype = torch.float16
    elif dtype_str == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None  # fp32 compute

    # -------------------------
    # seed
    # -------------------------
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -------------------------
    # data / schema (for masking)
    # -------------------------
    data_cfg = cfg.get("data", {}) or {}
    schema_path = data_cfg.get("label_schema_file", None)
    if schema_path:
        class_names, label_groups, group_reduce, mask_cfg = load_label_schema(schema_path)
    else:
        # fallback: explicit classes
        class_names = list(data_cfg.get("classes", []))
        if not class_names:
            raise KeyError("data.label_schema_file or data.classes is required")
        label_groups = data_cfg.get("label_groups", {}) or {}
        group_reduce = data_cfg.get("group_reduce", "any")
        mask_cfg = data_cfg.get("mask", {}) or {}

    image_cfg = cfg.get("image", {}) or {}
    center_crop_size = int(image_cfg.get("center_crop_size", 3072))
    image_size = int(image_cfg.get("image_size", 1024))

    aug_cfg = cfg.get("augment", {}) or {}
    train_tf = build_vae_transform(center_crop_size, image_size, augment=aug_cfg)
    val_tf = build_vae_transform(center_crop_size, image_size, augment=None)

    train_ds = MultiLabelMedicalDataset(
        root=data_cfg["root"],
        split=str(data_cfg.get("train_split", "train")),
        classes=class_names,
        transform=train_tf,
        center_crop_size=center_crop_size,
        image_size=image_size,
        split_filename=str(data_cfg.get("split_filename", "default_split.yaml")),
        label_groups=label_groups,
        group_reduce=group_reduce,
        mask=mask_cfg,
    )

    val_ds = MultiLabelMedicalDataset(
        root=data_cfg["root"],
        split=str(data_cfg.get("val_split", "val")),
        classes=class_names,
        transform=val_tf,
        center_crop_size=center_crop_size,
        image_size=image_size,
        split_filename=str(data_cfg.get("split_filename", "default_split.yaml")),
        label_groups=label_groups,
        group_reduce=group_reduce,
        mask=mask_cfg,
    )

    if len(train_ds) == 0:
        raise RuntimeError("No training samples after masking.")

    train_cfg = cfg.get("train", {}) or {}
    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 4))
    max_epochs = int(train_cfg.get("epochs", 10))
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    log_every = int(train_cfg.get("log_every", 50))
    amp = bool(train_cfg.get("amp", device.type == "cuda" and amp_dtype is not None))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # tqdm settings (epoch-level progress)
    use_pbar = bool(train_cfg.get("progress_bar", True)) and (tqdm is not None)
    pbar_disable = not use_pbar
    pbar_mininterval = float(train_cfg.get("tqdm_mininterval", 0.5))
    pbar_update_interval = int(train_cfg.get("tqdm_update_interval", 10))
    pbar_leave = bool(train_cfg.get("tqdm_leave", False))

    # --- wandb setup ---
    wandb_cfg = cfg.get("wandb", {}) or {}
    use_wandb = bool(wandb_cfg.get("enabled", False))

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

    wandb_log_interval = int(wandb_cfg.get("log_interval_steps", log_every)) if use_wandb else 0

    # autocast helper
    def autocast_ctx(enabled: bool):
        if not torch.cuda.is_available() or device.type != "cuda":
            from contextlib import nullcontext
            return nullcontext()
        if (not enabled) or (amp_dtype is None):
            from contextlib import nullcontext
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", enabled=True, dtype=amp_dtype)
        # fallback for older torch
        try:
            return torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype)
        except TypeError:
            return torch.cuda.amp.autocast(enabled=True)

    scaler = None
    if amp and device.type == "cuda" and amp_dtype == torch.float16:
        # Prefer the newer API when available (avoids FutureWarning)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = torch.cuda.amp.GradScaler()

    # -------------------------
    # build / resume VAE
    # -------------------------
    resume_dir = vae_cfg.get("resume_from", None)
    if resume_dir:
        from diffusers import AutoencoderKL  # type: ignore

        print(f"[resume] loading VAE from: {resume_dir}")
        vae = AutoencoderKL.from_pretrained(resume_dir, torch_dtype=param_dtype)
        vae.to(device)
        start_epoch = 0
        global_step = 0
        best_val = None
        state_path = Path(resume_dir) / "train_state.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu")
            start_epoch = int(state.get("epoch", 0))
            global_step = int(state.get("global_step", 0))
            best_val = state.get("best_val", None)
        else:
            state = None
    else:
        base_repo_id = str(vae_cfg.get("base_repo_id", vae_cfg.get("repo_id", "stabilityai/sdxl-vae")))
        latent_channels = int(vae_cfg.get("latent_channels", 4))
        init_from_pretrained = bool(vae_cfg.get("init_from_pretrained", True))
        vae = build_autoencoder_kl(
            base_repo_id=base_repo_id,
            latent_channels=latent_channels,
            torch_dtype=param_dtype,
            device=("cuda" if device.type == "cuda" else "cpu"),
            init_from_pretrained_if_possible=init_from_pretrained,
        )
        vae.train()
        start_epoch = 0
        global_step = 0
        best_val = None
        state = None

    # -------------------------
    # optimizer
    # -------------------------
    optim_cfg = cfg.get("optimizer", {}) or {}
    lr = float(optim_cfg.get("lr", 1e-4))
    wd = float(optim_cfg.get("weight_decay", 0.0))
    betas = optim_cfg.get("betas", [0.9, 0.999])
    betas = (float(betas[0]), float(betas[1]))

    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=wd, betas=betas)

    if resume_dir and state is not None:
        opt_state = state.get("optimizer", None)
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
            except Exception as e:
                print(f"[warn] failed to load optimizer state: {e}")
        if scaler is not None and state.get("scaler", None) is not None:
            try:
                scaler.load_state_dict(state["scaler"])
            except Exception as e:
                print(f"[warn] failed to load scaler state: {e}")

    # -------------------------
    # optional feature loss (frozen teacher)
    # -------------------------
    loss_cfg = cfg.get("loss", {}) or {}
    recon_type = str(loss_cfg.get("recon_type", "l1")).lower()
    w_recon = float(loss_cfg.get("recon_weight", 1.0))
    w_kl = float(loss_cfg.get("kl_weight", 1e-6))

    feat_cfg = loss_cfg.get("feature", {}) or {}
    use_feat = bool(feat_cfg.get("enabled", False)) and float(feat_cfg.get("weight", 0.0)) > 0
    w_feat = float(feat_cfg.get("weight", 0.0))
    feat_type = str(feat_cfg.get("type", "l2")).lower()

    teacher_feat = None
    if use_feat:
        ckpt = feat_cfg.get("teacher_checkpoint", None)
        if not ckpt:
            raise KeyError("loss.feature.teacher_checkpoint is required when feature loss is enabled")
        teacher_feat = build_convnext_large(num_classes=len(class_names), pretrained=bool(feat_cfg.get("imagenet_pretrained", True)))
        sd = torch.load(ckpt, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        teacher_feat.load_state_dict(sd, strict=False)
        teacher_feat.to(device)
        teacher_feat.eval()
        for p in teacher_feat.parameters():
            p.requires_grad_(False)

    # Teacher normalization for feature loss (no crop/resize needed; VAE tf already did it)
    mean = tuple(data_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(data_cfg.get("std", [0.229, 0.224, 0.225]))
    mean_t = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    def to_teacher_norm(x_m11: torch.Tensor) -> torch.Tensor:
        x01 = (x_m11.clamp(-1, 1) + 1) / 2
        return (x01 - mean_t) / std_t

    def forward_teacher_embedding(x_norm: torch.Tensor) -> torch.Tensor:
        # ConvNeXt embedding (flattened)
        assert teacher_feat is not None
        feat = teacher_feat.features(x_norm)
        feat = teacher_feat.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return feat

    # -------------------------
    # training loop
    # -------------------------
    best_dir = out_dir / "best"
    last_dir = out_dir / "last"
    best_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)

    def eval_val(epoch_idx: int) -> dict[str, float]:
        vae.eval()
        loss_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        n = 0
        with torch.no_grad():
            it_val = val_loader
            if tqdm is not None and use_pbar:
                it_val = tqdm(
                    val_loader,
                    desc=f"Val   {epoch_idx}/{max_epochs}",
                    disable=pbar_disable,
                    dynamic_ncols=True,
                    leave=False,
                    mininterval=pbar_mininterval,
                )

            for step_v, (x, _, _) in enumerate(it_val, start=1):
                x = x.to(device, dtype=param_dtype)
                posterior = vae.encode(x).latent_dist
                latents = posterior.mode() if hasattr(posterior, "mode") else posterior.sample()
                recon = vae.decode(latents).sample
                if recon_type == "mse":
                    recon_loss = F.mse_loss(recon, x)
                else:
                    recon_loss = F.l1_loss(recon, x)
                kl_loss = _kl_loss_from_posterior(posterior)
                total = w_recon * recon_loss + w_kl * kl_loss
                loss_sum += float(total.detach().cpu().item())
                recon_sum += float(recon_loss.detach().cpu().item())
                kl_sum += float(kl_loss.detach().cpu().item())
                n += 1

                if tqdm is not None and use_pbar and (not pbar_disable) and (step_v % pbar_update_interval == 0):
                    it_val.set_postfix({"loss": f"{loss_sum / max(n, 1):.4f}"})  # type: ignore[attr-defined]
        vae.train()
        return {
            "val_loss": loss_sum / max(n, 1),
            "val_recon": recon_sum / max(n, 1),
            "val_kl": kl_sum / max(n, 1),
        }

    start = time.time()
    try:
        for epoch in range(start_epoch, max_epochs):
            vae.train()
            running = 0.0
            running_recon = 0.0
            running_kl = 0.0
            running_feat = 0.0
            count = 0

            optimizer.zero_grad(set_to_none=True)

            it_train = train_loader
            if tqdm is not None and use_pbar:
                it_train = tqdm(
                    train_loader,
                    desc=f"Train {epoch+1}/{max_epochs}",
                    disable=pbar_disable,
                    dynamic_ncols=True,
                    leave=pbar_leave,
                    mininterval=pbar_mininterval,
                )

            for step_in_epoch, (x, _, _) in enumerate(it_train, start=1):
                x = x.to(device, dtype=param_dtype)

                with autocast_ctx(amp):
                    posterior = vae.encode(x).latent_dist
                    latents = posterior.sample() if hasattr(posterior, "sample") else posterior.latent_dist.sample()
                    recon = vae.decode(latents).sample

                    if recon_type == "mse":
                        recon_loss = F.mse_loss(recon, x)
                    else:
                        recon_loss = F.l1_loss(recon, x)

                    kl_loss = _kl_loss_from_posterior(posterior)

                    if use_feat and teacher_feat is not None:
                        x_n = to_teacher_norm(x.to(torch.float32))
                        r_n = to_teacher_norm(recon.to(torch.float32))
                        emb_x = forward_teacher_embedding(x_n)
                        emb_r = forward_teacher_embedding(r_n)
                        if feat_type in ("cos", "cosine"):
                            feat_loss = 1.0 - F.cosine_similarity(emb_x, emb_r, dim=1).mean()
                        else:
                            feat_loss = F.mse_loss(emb_r, emb_x)
                    else:
                        feat_loss = recon_loss.new_tensor(0.0)

                    loss_total = w_recon * recon_loss + w_kl * kl_loss + w_feat * feat_loss
                    # gradient accumulation
                    loss = loss_total / float(max(1, grad_accum))

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # logging (running averages)
                running += float(loss_total.detach().cpu().item())
                running_recon += float(recon_loss.detach().cpu().item())
                running_kl += float(kl_loss.detach().cpu().item())
                running_feat += float(feat_loss.detach().cpu().item())
                count += 1

                # progress bar update
                if tqdm is not None and use_pbar and (not pbar_disable) and (step_in_epoch % pbar_update_interval == 0):
                    lr_now = float(optimizer.param_groups[0]["lr"])
                    it_train.set_postfix(  # type: ignore[attr-defined]
                        {
                            "loss": f"{running / max(count, 1):.4f}",
                            "recon": f"{running_recon / max(count, 1):.4f}",
                            "kl": f"{running_kl / max(count, 1):.0f}",
                            "lr": f"{lr_now:.2e}",
                            "gs": global_step,
                        }
                    )

                # optimizer step
                if step_in_epoch % grad_accum == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    # step metrics -> wandb (preferred), otherwise fallback to stdout
                    if use_wandb and wandb_run is not None and wandb_log_interval > 0 and (global_step % wandb_log_interval == 0):
                        lr_now = float(optimizer.param_groups[0]["lr"])
                        elapsed = time.time() - start
                        payload = {
                            "global_step": global_step,
                            "epoch": epoch + 1,
                            "train/loss_step": float(loss_total.detach().cpu().item()),
                            "train/recon_step": float(recon_loss.detach().cpu().item()),
                            "train/kl_step": float(kl_loss.detach().cpu().item()),
                            "train/feat_step": float(feat_loss.detach().cpu().item()),
                            "train/loss_running": float(running / max(count, 1)),
                            "train/recon_running": float(running_recon / max(count, 1)),
                            "train/kl_running": float(running_kl / max(count, 1)),
                            "train/feat_running": float(running_feat / max(count, 1)),
                            "train/lr": lr_now,
                            "time/elapsed_min": float(elapsed / 60.0),
                        }
                        if scaler is not None:
                            try:
                                payload["train/grad_scale"] = float(scaler.get_scale())
                            except Exception:
                                pass
                        wandb_run.log(payload, step=global_step)
                    elif (not use_wandb) and log_every > 0 and (global_step % log_every == 0):
                        elapsed = time.time() - start
                        msg = (
                            f"[epoch {epoch+1}/{max_epochs}] step={global_step} "
                            f"loss={running / max(count, 1):.4f} "
                            f"recon={running_recon / max(count, 1):.4f} "
                            f"kl={running_kl / max(count, 1):.0f} "
                            f"feat={running_feat / max(count, 1):.4f} "
                            f"elapsed={elapsed/60:.1f}m"
                        )
                        if tqdm is not None and use_pbar:
                            tqdm.write(msg)
                        else:
                            print(msg, flush=True)

            # epoch end eval
            val_metrics = eval_val(epoch + 1)
            val_loss = float(val_metrics["val_loss"])
            msg = (
                f"[epoch {epoch+1}] "
                f"val_loss={val_loss:.6f} "
                f"val_recon={float(val_metrics['val_recon']):.6f} "
                f"val_kl={float(val_metrics['val_kl']):.2f}"
            )
            if tqdm is not None and use_pbar:
                tqdm.write(msg)
            else:
                print(msg, flush=True)

            # epoch metrics -> wandb
            if use_wandb and wandb_run is not None:
                lr_now = float(optimizer.param_groups[0]["lr"])
                wandb_run.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss_epoch": float(running / max(count, 1)),
                        "train/recon_epoch": float(running_recon / max(count, 1)),
                        "train/kl_epoch": float(running_kl / max(count, 1)),
                        "train/feat_epoch": float(running_feat / max(count, 1)),
                        "train/lr": lr_now,
                        "val/loss": val_loss,
                        "val/recon": float(val_metrics["val_recon"]),
                        "val/kl": float(val_metrics["val_kl"]),
                    },
                    step=global_step,
                )

            # save last
            try:
                vae.save_pretrained(last_dir)
            except Exception as e:
                print(f"[warn] save_pretrained(last) failed: {e}")

            train_state = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_val": best_val,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "val_metrics": val_metrics,
            }
            torch.save(train_state, last_dir / "train_state.pt")

            # save best
            improved = best_val is None or val_loss < float(best_val)
            if improved:
                best_val = val_loss
                try:
                    vae.save_pretrained(best_dir)
                except Exception as e:
                    print(f"[warn] save_pretrained(best) failed: {e}")
                torch.save({**train_state, "best_val": best_val}, best_dir / "train_state.pt")
                msg_best = f"[best] updated best_val={best_val:.6f}"
                if tqdm is not None and use_pbar:
                    tqdm.write(msg_best)
                else:
                    print(msg_best, flush=True)

                if use_wandb and wandb_run is not None:
                    try:
                        wandb_run.summary["best_val_loss"] = float(best_val)
                    except Exception:
                        pass
    finally:
        if use_wandb and wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass

    print(f"Done. Saved to: {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a custom SDXL VAE from YAML config")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    gpu_ids = get_gpu_ids(cfg)
    set_visible_gpus(gpu_ids)
    train_from_config(cfg, args.config)


if __name__ == "__main__":
    main()