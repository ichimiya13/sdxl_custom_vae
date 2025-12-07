# src/sdxl/load_sdxl_vae.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from diffusers import AutoencoderKL
import torch


@dataclass
class SDXLVAEConfig:
    repo_id: str = "stabilityai/sdxl-vae"
    torch_dtype: torch.dtype = torch.float16
    device: Literal["cpu", "cuda"] = "cuda"


def load_sdxl_vae(config: SDXLVAEConfig) -> AutoencoderKL:
    """
    SDXL 用の VAE (stabilityai/sdxl-vae) を読み込んで返すヘルパー。
    今後カスタムVAEに差し替えるときも、この関数をいじるだけで済むようにする。
    """
    vae = AutoencoderKL.from_pretrained(
        config.repo_id,
        torch_dtype=config.torch_dtype,
    )
    vae.to(config.device)
    vae.eval()
    return vae