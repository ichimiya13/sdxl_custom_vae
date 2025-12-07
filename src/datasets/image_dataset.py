# src/datasets/image_dataset.py

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional, Callable, Tuple, List

import yaml
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MultiLabelMedicalDataset(Dataset):
    """
    Multi-label medical image dataset for UWF images.

    前提:
      - 画像とラベルYAML、splitファイルは以下の構造で保存されている:
        ../data/multilabel/MedicalCheckup/splitted/
          ├── default_split.yaml
          ├── 05935945_20190618_111837.372_COLOR_R_1.png
          ├── 05935945_20190618_111837.372_COLOR_R_1.yaml
          ├── ...
      - default_split.yaml のフォーマット:
          test:
            - xxx.png
            - ...
          train:
            - ...
          val:
            - ...
      - 各画像に対して同名の .yaml ファイルがあり、内容は
        { class_name: 0 or 1 } の dict (最大 63 クラス)。
      - 実験に使うクラスは引数 classes で指定し、それ以外のクラスは無視する。
      - classes に含まれるクラス名が YAML に存在しない場合はエラーにする。

    __getitem__:
      returns (image, labels, path)
        image  : Tensor [C, H, W] (RGB, 正規化済み)
        labels : Tensor [num_classes] (float, 0/1)
        path   : str (画像へのパス)
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
    ) -> None:
        """
        Args:
            root:
                画像・ラベル・splitファイルが置かれているディレクトリ。
                例: "../data/multilabel/MedicalCheckup/splitted"
            split:
                "train", "val", "test" のいずれか。
            classes:
                使用するクラス名のリスト。順番がラベルベクトルの次元順になる。
            transform:
                追加の画像変換。None の場合は
                  CenterCrop(center_crop_size) -> Resize(image_size)
                  -> ToTensor() -> Normalize(mean, std)
                のデフォルトを使う。
                transform を渡した場合は、そちらが優先される（ユーザ側で
                ToTensor/Normalize を含める想定）。
            center_crop_size:
                センタークロップのサイズ（元画像が3072x3072想定）。
            image_size:
                最終的な入力解像度（例: 1024）。
            split_filename:
                split 情報を含む YAML ファイル名。デフォルト "default_split.yaml"。
            label_suffix:
                ラベルファイルの拡張子。デフォルト ".yaml"。
            mean, std:
                Normalize 用の平均・分散。デフォルトは ImageNet。
        """
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.classes: List[str] = list(classes)
        self.num_classes = len(self.classes)
        self.label_suffix = label_suffix

        if transform is None:
            # デフォルト: CenterCrop -> Resize -> ToTensor -> Normalize(ImageNet)
            self.transform = T.Compose([
                T.CenterCrop(center_crop_size),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
        else:
            # ユーザが完全に制御したい場合はこちらを使う
            self.transform = transform

        # split ファイルを読み込む
        split_path = self.root / split_filename
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with split_path.open("r", encoding="utf-8") as f:
            split_dict = yaml.safe_load(f)

        if self.split not in split_dict:
            raise KeyError(f"Split '{self.split}' not found in {split_path}")

        file_list = split_dict[self.split]
        if not isinstance(file_list, list):
            raise ValueError(
                f"Expected a list of filenames for split '{self.split}', "
                f"got {type(file_list)}"
            )

        self.image_paths: List[Path] = []
        self.labels: List[torch.Tensor] = []

        # すべてのサンプルを登録しつつ、クラス整合性をチェック
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
                raise ValueError(f"Invalid label YAML format: {label_path}")

            # ここで classes の各クラスが YAML に存在するかチェック
            label_vec = []
            for cls in self.classes:
                if cls not in label_dict:
                    raise KeyError(
                        f"Class '{cls}' not found in label file: {label_path}"
                    )
                v = label_dict[cls]
                # 0/1 以外が来た場合も一応 float にキャストしてしまう
                label_vec.append(float(v))

            label_tensor = torch.tensor(label_vec, dtype=torch.float32)

            self.image_paths.append(img_path)
            self.labels.append(label_tensor)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No samples found for split '{self.split}' "
                f"in split file {split_path}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 画像読み込み（RGB固定）
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # path は str で返す
        return img, label, str(img_path)
