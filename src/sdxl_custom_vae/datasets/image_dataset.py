from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional, Callable, Tuple, List, Dict, Any

import yaml
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.sdxl_custom_vae.labels.masking import should_drop_sample


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MultiLabelMedicalDataset(Dataset):
    """
    Multi-label medical image dataset.

    returns: (image, labels, path)
      image  : Tensor [C,H,W]
      labels : Tensor [num_classes] (float 0/1)
      path   : str
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        classes: Sequence[str],
        transform: Optional[Callable] = None,
        center_crop_size: int = 3072,
        image_size: int = 1024,
        split_filename: str = "default_split.yaml",
        label_suffix: str = ".yaml",
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD,
        # NEW: group labels
        label_groups: Dict[str, List[str]] | None = None,
        group_reduce: str = "any",
        # NEW: mask rules (from schema)
        mask: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.classes: List[str] = list(classes)
        self.num_classes = len(self.classes)
        self.label_suffix = label_suffix

        self.label_groups = label_groups or {}
        self.group_reduce = str(group_reduce).lower()
        self.mask = mask or {}

        if transform is None:
            self.transform = T.Compose([
                T.CenterCrop(center_crop_size),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transform

        # read split
        split_path = self.root / split_filename
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with split_path.open("r", encoding="utf-8") as f:
            split_dict = yaml.safe_load(f)

        if self.split not in split_dict:
            raise KeyError(f"Split '{self.split}' not found in {split_path}")

        file_list = split_dict[self.split]
        if not isinstance(file_list, list):
            raise ValueError(f"Expected a list of filenames for split '{self.split}'")

        self.image_paths: List[Path] = []
        self.labels: List[torch.Tensor] = []

        # dropped audit
        self.dropped: List[Dict[str, Any]] = []
        self.dropped_counts: Dict[str, int] = {}

        def _group_value(label_dict: Dict[str, Any], group_name: str) -> float:
            # sources for group
            srcs = self.label_groups.get(group_name, [])
            if not srcs:
                # group not defined -> fall back to direct
                if group_name not in label_dict:
                    raise KeyError(f"Class '{group_name}' not found in label yaml and not in label_groups.")
                return float(label_dict[group_name])

            vals = []
            for s in srcs:
                v = label_dict.get(s, 0.0)
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(0.0)

            if self.group_reduce in ("any", "or", "max"):
                return 1.0 if any(v >= 0.5 for v in vals) else 0.0
            if self.group_reduce in ("all", "and", "min"):
                return 1.0 if all(v >= 0.5 for v in vals) else 0.0

            raise ValueError(f"Unsupported group_reduce: {self.group_reduce}")

        # register samples
        for fname in file_list:
            img_path = self.root / fname
            if not img_path.is_file():
                raise FileNotFoundError(f"Image file not found: {img_path}")

            label_path = img_path.with_suffix(self.label_suffix)
            if not label_path.is_file():
                raise FileNotFoundError(f"Label YAML not found: {label_path}")

            with label_path.open("r", encoding="utf-8") as f:
                label_dict = yaml.safe_load(f)

            if not isinstance(label_dict, dict):
                # invalid label -> drop (扱いは好みだが、ここでは落とす)
                info = {"reasons": ["invalid_label_yaml"], "positive_labels": []}
                self.dropped.append({"path": str(img_path), **info})
                self.dropped_counts["invalid_label_yaml"] = self.dropped_counts.get("invalid_label_yaml", 0) + 1
                continue

            # NEW: mask filtering
            drop, info = should_drop_sample(label_dict, self.mask)
            if drop:
                self.dropped.append({"path": str(img_path), **info})
                for r in info.get("reasons", []):
                    self.dropped_counts[r] = self.dropped_counts.get(r, 0) + 1
                continue

            # build label vector
            label_vec = []
            for cls in self.classes:
                if cls in self.label_groups:
                    v = _group_value(label_dict, cls)
                else:
                    if cls not in label_dict:
                        raise KeyError(f"Class '{cls}' not found in label file: {label_path}")
                    v = float(label_dict[cls])
                label_vec.append(float(v))

            label_tensor = torch.tensor(label_vec, dtype=torch.float32)
            self.image_paths.append(img_path)
            self.labels.append(label_tensor)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No samples left for split '{self.split}' after masking. "
                f"Split file: {split_path}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label, str(img_path)
