# src/scripts/reconstruct.py

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image

from src.sdxl_custom_vae.datasets.image_dataset import MultiLabelMedicalDataset
from src.sdxl_custom_vae.sdxl.load_sdxl_vae import load_sdxl_vae, SDXLVAEConfig


def build_vae_transform(
    center_crop_size: int = 3072,
    image_size: int = 1024,
):
    """
    SDXL VAE 向けの transform:
      - CenterCrop -> Resize -> ToTensor -> [-1, 1] へスケーリング
    """
    return T.Compose([
        T.CenterCrop(center_crop_size),
        T.Resize((image_size, image_size)),
        T.ToTensor(),               # [0, 1]
        T.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1]
    ])


def reconstruct_split(
    data_root: str | Path,
    split: str,
    classes: list[str],
    output_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    device: str = "cuda",
):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset（transform を VAE 用に上書き）
    transform = build_vae_transform()
    dataset = MultiLabelMedicalDataset(
        root=data_root,
        split=split,
        classes=classes,
        transform=transform,
        # center_crop_size/image_size は transform 側で決めるので適当でもOK
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # VAE ロード
    vae = load_sdxl_vae(
        SDXLVAEConfig(
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device=device,
        )
    )

    scaling_factor = getattr(vae.config, "scaling_factor", 0.13025)

    vae.to(device)
    vae.eval()

    torch.set_grad_enabled(False)

    for batch_idx, (images, _, paths) in enumerate(dataloader):
        images = images.to(device, dtype=vae.dtype)  # [-1, 1], float16/32

        # encode -> decode
        posterior = vae.encode(images)
        latents = posterior.latent_dist.sample() * scaling_factor
        recon = vae.decode(latents / scaling_factor).sample  # [-1, 1]

        # [-1, 1] -> [0, 1]
        recon = (recon.clamp(-1.0, 1.0) + 1.0) / 2.0

        # 1枚ずつ保存
        for img_tensor, path in zip(recon, paths):
            img_name = Path(path).name
            save_path = output_dir / img_name
            save_image(img_tensor, save_path)

        print(
            f"[{split}] batch {batch_idx + 1}/{len(dataloader)} "
            f"saved {len(paths)} images"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct images using pretrained SDXL VAE (4ch)."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to ../data/multilabel/MedicalCheckup/splitted",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        required=True,
        help="List of class names to use (must exist in YAML labels).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save reconstructed images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    reconstruct_split(
        data_root=args.data_root,
        split=args.split,
        classes=args.classes,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
