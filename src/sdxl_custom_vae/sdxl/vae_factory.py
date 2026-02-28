"""Factory helpers for building AutoencoderKL variants.

We primarily rely on diffusers' `AutoencoderKL`.

Key use cases in this repository:

* Load a pretrained SDXL VAE (latent_channels=4)
* Train/fine-tune a VAE on UWF data
* Instantiate VAE variants with different latent channel sizes (4/8/16)

For latent_channels != base_latent_channels, we create a new model from the
pretrained config with modified `latent_channels`, then **transplant all
shape-matching parameters** from the base model.

This is useful for experiments like latent_channels in {8, 16} where only a
small number of layers change shape (typically:
  - encoder.conv_out
  - quant_conv
  - post_quant_conv
  - decoder.conv_in
)

All other weights are copied when shapes match.
"""

from __future__ import annotations

from typing import Any, Dict, List


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
      1) load the base model on CPU (float32)
      2) create a new AutoencoderKL with modified `latent_channels`
      3) copy all parameters/buffers whose shapes match
      4) leave mismatched tensors randomly initialized
         (we also expose which parameters were skipped to support 2-stage
          training where we warm-start only the new layers)
    """

    from diffusers import AutoencoderKL  # type: ignore
    import torch

    # Load base model (CPU) to avoid wasting GPU memory.
    base = AutoencoderKL.from_pretrained(base_repo_id, torch_dtype=torch.float32)
    base_cfg = dict(getattr(base, "config", {}))

    base_latent = int(base_cfg.get("latent_channels", 4))
    if init_from_pretrained_if_possible and int(latent_channels) == int(base_latent):
        # reload on target dtype/device (pretrained weights)
        vae = AutoencoderKL.from_pretrained(base_repo_id, torch_dtype=torch_dtype)
        vae.to(device)
        # Attach a small init report for downstream training scripts.
        try:
            vae._sdxl_custom_vae_init = {
                "base_repo_id": base_repo_id,
                "base_latent_channels": int(base_latent),
                "target_latent_channels": int(latent_channels),
                "copied_state_keys": [],
                "skipped_state_keys": [],
                "random_init_param_names": [],
            }
        except Exception:
            pass
        return vae

    # build new model from config, but change latent channels
    new_cfg = _clean_config_dict(base_cfg)
    new_cfg["latent_channels"] = int(latent_channels)

    # Instantiate on CPU first; copy weights on CPU; then move to device.
    vae = instantiate_autoencoder_kl_from_config(new_cfg)

    # If caller does NOT want to use pretrained weights, keep full random init.
    if not init_from_pretrained_if_possible:
        try:
            vae._sdxl_custom_vae_init = {
                "base_repo_id": base_repo_id,
                "base_latent_channels": int(base_latent),
                "target_latent_channels": int(latent_channels),
                "init_type": "random_full",
                "copied_state_keys": [],
                "skipped_state_keys": [],
                "random_init_param_names": [],
            }
        except Exception:
            pass
    else:
        # --- transplant shape-matching tensors ---
        base_sd = base.state_dict()
        new_sd = vae.state_dict()

        copied: List[str] = []
        skipped: List[str] = []
        for k, v in new_sd.items():
            if k in base_sd and tuple(base_sd[k].shape) == tuple(v.shape):
                new_sd[k] = base_sd[k]
                copied.append(k)
            else:
                skipped.append(k)

        # Load the merged state dict. (strict=False because some buffers may differ
        # across diffusers versions, and we intentionally skip mismatched tensors.)
        vae.load_state_dict(new_sd, strict=False)

        # Expose which *parameters* were randomly initialized.
        try:
            named_params = dict(vae.named_parameters())
            random_param_names = sorted([k for k in skipped if k in named_params])

            # Sanity check: warn if we skipped unexpected params.
            expected_prefixes = (
                "encoder.conv_out",
                "quant_conv",
                "post_quant_conv",
                "decoder.conv_in",
            )
            unexpected = [
                n for n in random_param_names if not any(n.startswith(p) for p in expected_prefixes)
            ]
            if len(unexpected) > 0:
                print(
                    "[warn][vae_factory] unexpected randomly-initialized parameters: "
                    + ", ".join(unexpected[:50])
                    + (" ..." if len(unexpected) > 50 else "")
                )

            vae._sdxl_custom_vae_init = {
                "base_repo_id": base_repo_id,
                "base_latent_channels": int(base_latent),
                "target_latent_channels": int(latent_channels),
                "init_type": "transplant_matching_shapes",
                "copied_state_keys": copied,
                "skipped_state_keys": skipped,
                "random_init_param_names": random_param_names,
            }
        except Exception:
            pass

    # Move to device and cast parameters/buffers
    vae.to(device)
    try:
        vae.to(dtype=torch_dtype)
    except Exception:
        # some modules may not support direct dtype cast
        pass
    return vae
