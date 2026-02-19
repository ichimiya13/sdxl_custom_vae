from __future__ import annotations
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        xs = torch.sigmoid(logits)
        xs_pos = xs
        xs_neg = 1.0 - xs

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss = -(
            targets * torch.log(xs_pos.clamp(min=self.eps)) +
            (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        )

        pt = xs_pos * targets + xs_neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        loss = loss * torch.pow((1.0 - pt), gamma)

        return loss.mean()
