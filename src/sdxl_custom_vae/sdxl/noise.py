"""Noise utilities for latent-space corruption (t-curve experiments).

This module provides a small wrapper so scripts can choose between:

* diffusers schedulers (recommended; e.g. DDPMScheduler)
* a lightweight custom DDPM-style scheduler (fallback)

The main interface is `build_noise_scheduler(cfg)` which returns an object
with an `add_noise(sample, noise, timesteps)` method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math
import torch


class NoiseSchedulerProtocol:
    """Minimal protocol used by our scripts."""

    def add_noise(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return a noised sample at given timesteps."""
        raise NotImplementedError


@dataclass
class CustomDDPMScheduler(NoiseSchedulerProtocol):
    """A tiny DDPM-style forward-process scheduler.

    This is intentionally minimal: it only supports `add_noise`.
    """

    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # "linear" or "scaled_linear"

    def __post_init__(self) -> None:
        if self.num_train_timesteps <= 1:
            raise ValueError("num_train_timesteps must be > 1")

        betas = self._make_betas(
            self.num_train_timesteps,
            self.beta_start,
            self.beta_end,
            self.beta_schedule,
        )
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alpha_cumprod))

    # ---- tiny buffer mechanism (so this works like an nn.Module without depending on it) ----
    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)

    @staticmethod
    def _make_betas(num_steps: int, beta_start: float, beta_end: float, schedule: str) -> torch.Tensor:
        schedule = str(schedule).lower()
        if schedule in ("linear",):
            return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        if schedule in ("scaled_linear", "scaled"):
            # matches common SD-style beta schedule (approx)
            return torch.linspace(math.sqrt(beta_start), math.sqrt(beta_end), num_steps, dtype=torch.float32) ** 2
        raise ValueError(f"Unknown beta_schedule: {schedule}")

    def add_noise(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """q(z_t | z_0) = sqrt(alpha_bar_t) z_0 + sqrt(1-alpha_bar_t) eps"""

        if timesteps.dtype not in (torch.int32, torch.int64):
            timesteps = timesteps.long()

        timesteps = timesteps.to(sample.device)

        # Ensure buffers on same device
        sqrt_ab = self.sqrt_alphas_cumprod.to(sample.device)
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod.to(sample.device)

        t = timesteps.clamp(0, self.num_train_timesteps - 1)

        a = sqrt_ab.gather(0, t)
        b = sqrt_1mab.gather(0, t)
        while a.ndim < sample.ndim:
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)

        return a * sample + b * noise


def build_noise_scheduler(noise_cfg: Dict[str, Any]) -> NoiseSchedulerProtocol:
    """Factory for noise schedulers.

    Parameters
    ----------
    noise_cfg:
        Example:
        {
          "scheduler": "ddpm",  # or "custom"
          "scheduler_kwargs": {"num_train_timesteps": 1000, ...}
        }
    """

    kind = str((noise_cfg or {}).get("scheduler", "ddpm")).lower()
    kwargs = (noise_cfg or {}).get("scheduler_kwargs", {}) or {}

    if kind in ("ddpm", "ddpmscheduler"):
        try:
            from diffusers import DDPMScheduler  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "diffusers is required for noise.scheduler=ddpm. "
                "Install diffusers or set noise.scheduler=custom."
            ) from e

        # Provide safe defaults if missing
        if "num_train_timesteps" not in kwargs:
            kwargs["num_train_timesteps"] = int((noise_cfg or {}).get("num_train_timesteps", 1000))
        return DDPMScheduler(**kwargs)

    if kind in ("custom", "simple"):
        # Use a small DDPM-style schedule as a fallback.
        return CustomDDPMScheduler(
            num_train_timesteps=int(kwargs.get("num_train_timesteps", 1000)),
            beta_start=float(kwargs.get("beta_start", 0.0001)),
            beta_end=float(kwargs.get("beta_end", 0.02)),
            beta_schedule=str(kwargs.get("beta_schedule", "linear")),
        )

    raise ValueError(f"Unknown noise.scheduler: {kind}")
