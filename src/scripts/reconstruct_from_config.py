# src/scripts/reconstruct_from_config.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Any, Dict

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.utils import save_image

from src.sdxl_custom_vae.datasets.image_dataset import MultiLabelMedicalDataset
from src.sdxl_custom_vae.labels.schema import load_label_schema
from src.sdxl_custom_vae.sdxl.load_sdxl_vae import SDXLVAEConfig, load_sdxl_vae
from src.sdxl_custom_vae.sdxl.noise import build_noise_scheduler


class VAEReconstructionWrapper(nn.Module):
    """
    再構成専用のラッパーモデル。
    forward(x) で encode -> sample -> decode までやって recon を返す。
    """
    def __init__(self, vae: nn.Module, scaling_factor: float, posterior: str = "sample"):
        super().__init__()
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.posterior = str(posterior).lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [-1, 1] スケールの画像
        posterior = self.vae.encode(x)
        dist = posterior.latent_dist
        if self.posterior in ("mode", "mean", "deterministic") and hasattr(dist, "mode"):
            latents = dist.mode()
        else:
            latents = dist.sample()
        latents = latents * self.scaling_factor
        recon = self.vae.decode(latents / self.scaling_factor).sample  # [-1, 1]
        return recon


def build_vae_transform(center_crop_size: int, image_size: int):
    """
    SDXL VAE の前処理:
      - CenterCrop -> Resize -> ToTensor -> [-1, 1]
    """
    return T.Compose([
        T.CenterCrop(center_crop_size),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Lambda(lambda x: x * 2.0 - 1.0),
    ])


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def select_indices(num_total: int, cfg_sampling: Dict[str, Any]) -> List[int]:
    num_samples = cfg_sampling.get("num_samples", None)
    mode = cfg_sampling.get("mode", "first")
    seed = cfg_sampling.get("seed", 42)

    indices = list(range(num_total))

    if num_samples is None:
        return indices  # 全件

    num_samples = min(num_samples, num_total)

    if mode == "first":
        return indices[:num_samples]
    elif mode == "random":
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(num_total, generator=g).tolist()
        return perm[:num_samples]
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")


def reconstruct_from_config(cfg: Dict[str, Any], config_path: str | Path):
    exp_name = cfg["experiment_name"]

    # ===== Data & Dataset =====
    data_cfg = cfg["data"]
    data_root = data_cfg["root"]
    split = data_cfg["split"]

    # label schema is preferred; fallback to explicit classes for backward compatibility
    schema_path = data_cfg.get("label_schema_file", None)
    if schema_path:
        classes, label_groups, group_reduce, mask_cfg = load_label_schema(schema_path)
    else:
        classes = data_cfg.get("classes", None)
        if not classes:
            raise KeyError("data.classes or data.label_schema_file is required")
        label_groups = data_cfg.get("label_groups", {}) or {}
        group_reduce = data_cfg.get("group_reduce", "any")
        mask_cfg = data_cfg.get("mask", {}) or {}

    center_crop_size = cfg["image"]["center_crop_size"]
    image_size = cfg["image"]["image_size"]

    transform = build_vae_transform(center_crop_size, image_size)

    dataset = MultiLabelMedicalDataset(
        root=data_root,
        split=split,
        classes=classes,
        transform=transform,
        center_crop_size=center_crop_size,
        image_size=image_size,
        split_filename=str(data_cfg.get("split_filename", "default_split.yaml")),
        label_groups=label_groups,
        group_reduce=group_reduce,
        mask=mask_cfg,
    )

    indices = select_indices(len(dataset), cfg.get("sampling", {}))
    subset = Subset(dataset, indices)

    batch_size = cfg["inference"]["batch_size"]
    num_workers = cfg["inference"]["num_workers"]

    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ===== VAE (手動GPU選択。シングルGPUのみ) =====
    vae_cfg = cfg["vae"]
    device_str = vae_cfg.get("device", "cuda")
    dtype_str = vae_cfg.get("dtype", "fp16")
    gpu_ids = vae_cfg.get("gpu_ids", None)  # 例: [0], [1], ...

    if dtype_str == "fp16":
        torch_dtype = torch.float16
    elif dtype_str == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    # --- デバイス決定（シングルGPU or CPU） ---
    if device_str == "cpu" or not torch.cuda.is_available():
        main_device = torch.device("cpu")
        print("CUDA not available or device=cpu. Using CPU.")
    else:
        if not gpu_ids:
            gpu_ids = [0]
        main_device = torch.device(f"cuda:{gpu_ids[0]}")
        print(f"Using single GPU: cuda:{gpu_ids[0]}")

    # --- 元の VAE 本体をロード ---
    vae_core = load_sdxl_vae(
        SDXLVAEConfig(
            repo_id=vae_cfg.get("repo_id", "stabilityai/sdxl-vae"),
            torch_dtype=torch_dtype,
            device="cuda" if main_device.type == "cuda" else "cpu",
        )
    )

    # scaling_factor を取得
    scaling_factor = float(getattr(vae_core.config, "scaling_factor", 0.13025))

    # posterior mode / sample
    posterior_mode = str(vae_cfg.get("posterior", "sample")).lower()

    # ===== Output dir =====
    output_cfg = cfg["output"]
    out_root = Path(output_cfg["root_dir"])
    out_dir = out_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_side_by_side = bool(output_cfg.get("side_by_side", False))

    # dataset export options
    export_dataset = bool(output_cfg.get("export_dataset", False))
    preserve_relative_paths = bool(output_cfg.get("preserve_relative_paths", export_dataset))
    copy_labels = bool(output_cfg.get("copy_labels", export_dataset))
    write_split_yaml = bool(output_cfg.get("write_split_yaml", export_dataset))
    split_filename = str(output_cfg.get("split_filename", "default_split.yaml"))
    keep_original_name_for_t0 = bool(output_cfg.get("keep_original_name_for_t0", True))
    suffix_fmt = str(output_cfg.get("timestep_suffix_format", "__t{t:04d}"))

    # noise / t-curve saving (optional)
    noise_cfg = cfg.get("noise", {}) or {}
    timesteps = [int(t) for t in (noise_cfg.get("timesteps", []) or [])]
    if not timesteps:
        timesteps = [0]
    if 0 not in timesteps:
        timesteps = [0] + timesteps
    timesteps = sorted(list(dict.fromkeys(timesteps)))

    scheduler = build_noise_scheduler(noise_cfg)
    noise_seed = int(noise_cfg.get("seed", 123))
    noise_gen = None
    if len(timesteps) > 1:
        if main_device.type == "cuda":
            torch.cuda.manual_seed_all(noise_seed)
            noise_gen = None
        else:
            noise_gen = torch.Generator()
            noise_gen.manual_seed(noise_seed)

    # record saved relpaths for split yaml
    saved_relpaths: list[str] = []

    # 実行時の config もコピーしておく
    config_path = Path(config_path)
    config_copy_path = out_dir / "config_used.yaml"
    if config_path.is_file():
        config_copy_path.write_text(
            config_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    # ===== Reconstruction loop =====
    torch.set_grad_enabled(False)

    for batch_idx, (images, _, paths) in enumerate(dataloader):
        images = images.to(main_device, dtype=torch_dtype)  # [-1, 1]

        # encode once
        dist = vae_core.encode(images).latent_dist
        if posterior_mode in ("mode", "mean", "deterministic") and hasattr(dist, "mode"):
            latents = dist.mode()
        else:
            latents = dist.sample()

        z0 = latents * scaling_factor  # scaled latent

        # noise for this batch (shared across timesteps, so curve is smooth)
        eps = None
        if len(timesteps) > 1:
            if noise_gen is None:
                eps = torch.randn(z0.shape, device=main_device, dtype=torch_dtype)
            else:
                eps = torch.randn(z0.shape, generator=noise_gen, device=main_device, dtype=torch_dtype)

        originals_01 = None
        if save_side_by_side:
            originals_01 = (images.clamp(-1.0, 1.0) + 1.0) / 2.0  # [0, 1]

        for t in timesteps:
            if int(t) == 0:
                zt = z0
            else:
                assert eps is not None
                t_tensor = torch.full((z0.shape[0],), int(t), device=main_device, dtype=torch.long)
                zt = scheduler.add_noise(z0, eps, t_tensor)

            recon_t = vae_core.decode(zt / scaling_factor).sample  # [-1,1]
            recon_01 = (recon_t.clamp(-1.0, 1.0) + 1.0) / 2.0  # [0,1]

            for idx, path in enumerate(paths):
                recon_tensor = recon_01[idx]
                if save_side_by_side and originals_01 is not None and int(t) == 0:
                    orig_tensor = originals_01[idx]
                    tensor_to_save = torch.cat([orig_tensor, recon_tensor], dim=2)
                else:
                    tensor_to_save = recon_tensor
                tensor_to_save = tensor_to_save.to(torch.float32)

                p = Path(path)
                # relative path for dataset export
                if preserve_relative_paths:
                    try:
                        rel = p.relative_to(Path(data_root))
                    except Exception:
                        rel = Path(p.name)
                else:
                    rel = Path(p.name)

                if int(t) == 0 and keep_original_name_for_t0:
                    rel_out = rel
                else:
                    suf = suffix_fmt.format(t=int(t))
                    rel_out = rel.with_name(rel.stem + suf + rel.suffix)

                save_path = out_dir / rel_out
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(tensor_to_save.cpu(), save_path)

                saved_relpaths.append(rel_out.as_posix())

                if copy_labels:
                    label_src = p.with_suffix(".yaml")
                    label_dst = save_path.with_suffix(".yaml")
                    label_dst.parent.mkdir(parents=True, exist_ok=True)
                    if label_src.is_file():
                        try:
                            label_dst.write_text(label_src.read_text(encoding="utf-8"), encoding="utf-8")
                        except Exception:
                            # fallback: binary copy
                            import shutil

                            shutil.copy2(label_src, label_dst)

        print(
            f"[{exp_name}] batch {batch_idx + 1}/{len(dataloader)} saved {len(paths)} images"
        )

    # write split yaml if requested
    if write_split_yaml:
        split_key = str(split)
        split_path = out_dir / split_filename
        existing = {}
        if split_path.is_file():
            try:
                with split_path.open("r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}
            except Exception:
                existing = {}
        if not isinstance(existing, dict):
            existing = {}

        # replace the current split entry
        existing[split_key] = saved_relpaths
        split_path.write_text(yaml.safe_dump(existing, allow_unicode=True), encoding="utf-8")

    print(f"Done. Saved reconstructions to: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct images using config (pretrained SDXL VAE, 4ch)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    reconstruct_from_config(cfg, args.config)


if __name__ == "__main__":
    main()
