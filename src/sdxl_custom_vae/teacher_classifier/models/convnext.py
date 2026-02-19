from __future__ import annotations

import torch.nn as nn
from torchvision.models import convnext_large, ConvNeXt_Large_Weights


def build_convnext_large(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
