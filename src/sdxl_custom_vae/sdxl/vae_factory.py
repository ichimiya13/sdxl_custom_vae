"""Factory helpers for building AutoencoderKL variants.

We primarily rely on diffusers' `AutoencoderKL`.

Key use cases in this repository:

* Load a pretrained SDXL VAE (latent_channels=4)
* Train/fine-tune a VAE on UWF data
* Instantiate VAE variants with different latent channel sizes (4/8/16)

For latent_channels != 4, we typically create a new model from the
pretrained config, but with randomly initialized weights.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _clean_config_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Remove diffusers metadata keys that can break direct constructors."""
    return {k: v for k, v in dict(cfg).items() if not str(k).startswith("_")}


def instantiate_autoencoder_kl_from_config(config_dict: Dict[str, Any]):
    """Instantiate AutoencoderKL from a config dict.

    Prefer `AutoencoderKL.from_config` when available.
    """
    from diffusers import AutoencoderKL  # type: ignore

    cfg = _clean_config_dict(config_dict)
    # Some diffusers versions provide from_config
    if hasattr(AutoencoderKL, "from_config"):
        try:
            return AutoencoderKL.from_config(cfg)
        except Exception:
            pass
    return AutoencoderKL(**cfg)


def build_autoencoder_kl(
    *,
    base_repo_id: str,
    latent_channels: int,
    torch_dtype,
    device: str,
    init_from_pretrained_if_possible: bool = True,
):
    """Build an AutoencoderKL.

    If `latent_channels` matches the base model's latent_channels and
    `init_from_pretrained_if_possible` is True, we load pretrained weights.

    Otherwise we:
      1) load the base model only to read its config
      2) create a new AutoencoderKL with modified `latent_channels`
         (randomly initialized)
    """

    from diffusers import AutoencoderKL  # type: ignore
    import torch

    # Load base model config (CPU) to avoid wasting GPU memory.
    base = AutoencoderKL.from_pretrained(base_repo_id, torch_dtype=torch.float32)
    base_cfg = dict(getattr(base, "config", {}))

    base_latent = int(base_cfg.get("latent_channels", 4))
    if init_from_pretrained_if_possible and int(latent_channels) == int(base_latent):
        # reload on target dtype/device (pretrained weights)
        vae = AutoencoderKL.from_pretrained(base_repo_id, torch_dtype=torch_dtype)
        vae.to(device)
        return vae

    # build new model from config, but change latent channels
    new_cfg = _clean_config_dict(base_cfg)
    new_cfg["latent_channels"] = int(latent_channels)

    vae = instantiate_autoencoder_kl_from_config(new_cfg)
    vae.to(device)
    # cast parameters/buffers
    try:
        vae.to(dtype=torch_dtype)
    except Exception:
        # some modules may not support direct dtype cast
        pass
    return vae
