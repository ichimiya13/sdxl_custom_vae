from __future__ import annotations

from typing import Any
from torchvision import transforms as T


def build_teacher_transforms(
    *,
    center_crop_size: int,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    train: bool,
    # 後方互換のため残す（augmentが無い場合に使用）
    hflip_p: float = 0.5,
    color_jitter: dict | None = None,
    # NEW: YAMLの augment をそのまま渡す
    augment: dict | None = None,
):
    """
    augment (YAML想定):
      augment:
        type: "basic" | "randaug" | "none"
        hflip_p: 0.5
        color_jitter: {brightness, contrast, saturation, hue}   # basic用
        randaug:
          num_ops: 2
          magnitude: 8
          num_magnitude_bins: 31   # optional

    方針:
      - train=False のときは augmentation を入れない（crop/resize + normalizeのみ）
      - randaug のときは基本 color_jitter は入れない（強すぎることがあるため）
    """
    augment = augment or {}
    aug_type = str(augment.get("type", "basic")).lower()

    # YAML側が指定されていればそれを優先
    hflip_p = float(augment.get("hflip_p", hflip_p))
    cj = augment.get("color_jitter", color_jitter)

    tfms: list[Any] = [
        T.CenterCrop(center_crop_size),
        T.Resize((image_size, image_size)),
    ]

    if train:
        if hflip_p > 0:
            tfms.append(T.RandomHorizontalFlip(p=hflip_p))

        if aug_type in ("none", "off", "false"):
            # 何もしない
            pass

        elif aug_type in ("randaug", "randaugment"):
            ra_cfg = augment.get("randaug", {}) or {}
            num_ops = int(ra_cfg.get("num_ops", 2))
            magnitude = int(ra_cfg.get("magnitude", 8))
            num_bins = int(ra_cfg.get("num_magnitude_bins", 31))

            # torchvision の RandAugment が必要
            if not hasattr(T, "RandAugment"):
                raise RuntimeError(
                    "torchvision.transforms.RandAugment is not available. "
                    "Please upgrade torchvision."
                )

            tfms.append(T.RandAugment(num_ops=num_ops, magnitude=magnitude, num_magnitude_bins=num_bins))

            # randaugとcolor_jitterの併用は強すぎになりやすいので、基本は無視
            # 併用したい場合は、YAML側で allow_color_jitter_with_randaug: true を付ける
            if cj and bool(augment.get("allow_color_jitter_with_randaug", False)):
                tfms.append(
                    T.ColorJitter(
                        brightness=float(cj.get("brightness", 0.0)),
                        contrast=float(cj.get("contrast", 0.0)),
                        saturation=float(cj.get("saturation", 0.0)),
                        hue=float(cj.get("hue", 0.0)),
                    )
                )

        else:
            # basic（現状互換）
            if cj:
                tfms.append(
                    T.ColorJitter(
                        brightness=float(cj.get("brightness", 0.0)),
                        contrast=float(cj.get("contrast", 0.0)),
                        saturation=float(cj.get("saturation", 0.0)),
                        hue=float(cj.get("hue", 0.0)),
                    )
                )

    tfms += [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    return T.Compose(tfms)
