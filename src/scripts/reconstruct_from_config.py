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
from src.sdxl_custom_vae.sdxl.load_sdxl_vae import SDXLVAEConfig, load_sdxl_vae


class VAEReconstructionWrapper(nn.Module):
    """
    再構成専用のラッパーモデル。
    forward(x) で encode -> sample -> decode までやって recon を返す。
    """
    def __init__(self, vae: nn.Module, scaling_factor: float):
        super().__init__()
        self.vae = vae
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [-1, 1] スケールの画像
        posterior = self.vae.encode(x)
        latents = posterior.latent_dist.sample() * self.scaling_factor
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
    data_root = cfg["data"]["root"]
    split = cfg["data"]["split"]
    classes = cfg["data"]["classes"]

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
    scaling_factor = getattr(vae_core.config, "scaling_factor", 0.13025)

    # 再構成用ラッパーを作成
    model = VAEReconstructionWrapper(vae_core, scaling_factor)
    model.to(main_device)
    model.eval()

    # ===== Output dir =====
    output_cfg = cfg["output"]
    out_root = Path(output_cfg["root_dir"])
    out_dir = out_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_side_by_side = output_cfg.get("side_by_side", False)

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

        recon = model(images)  # [-1, 1]
        recon = (recon.clamp(-1.0, 1.0) + 1.0) / 2.0  # [0, 1]

        originals = None
        if save_side_by_side:
            originals = (images.clamp(-1.0, 1.0) + 1.0) / 2.0  # [0, 1]

        for idx, path in enumerate(paths):
            recon_tensor = recon[idx]
            if save_side_by_side and originals is not None:
                orig_tensor = originals[idx]
                tensor_to_save = torch.cat([orig_tensor, recon_tensor], dim=2)
            else:
                tensor_to_save = recon_tensor
            tensor_to_save = tensor_to_save.to(torch.float32)
            img_name = Path(path).name
            save_path = out_dir / img_name
            save_image(tensor_to_save.cpu(), save_path)

        print(
            f"[{exp_name}] batch {batch_idx + 1}/{len(dataloader)} "
            f"saved {len(paths)} images"
        )

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
