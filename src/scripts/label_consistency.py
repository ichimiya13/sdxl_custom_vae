"""
ラベル整合性チェック用スクリプト。

前提ディレクトリ構造 (image_dataset.MultiLabelMedicalDataset と同じ):
  root/
    ├── default_split.yaml
    ├── <image>.png
    ├── <image>.yaml

チェック内容:
  - split ファイルに記載の全画像について、ラベル YAML の存在確認
  - ラベル YAML がマッピングであること
  - 想定クラス（--classes or config data.classes）すべてがキーとして存在すること
  - 値が数値であること（0/1 を想定だが厳密に 0/1 までは縛らず numeric を許容）
  - 想定外クラスが混入していれば報告（--allow-extra で許容可）

主な使い方:
  python -m src.scripts.label_consistency \
    --root ../data/uwf/multilabel/MedicalCheckup/splitted \
    --split-file default_split.yaml \
    --classes H1 H2 H3 H4 DME nAMD

  # config YAML から data.root / data.classes を流用
  python -m src.scripts.label_consistency \
    --config configs/vae/reconstruct/recon_pretrained_val_4ch.yaml

出力:
  - missing label files, missing class keys, non-numeric values, unexpected classes の件数と詳細（少数）
  - サマリテーブル（split ごとの件数）
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import yaml
from tqdm import tqdm


DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "../data/uwf/multilabel/MedicalCheckup/splitted"
)

# 明示的な親子関係マッピング（子 -> 親）
PARENT_MAP: Dict[str, str] = {
    "網膜裂孔PC後": "網膜裂孔",
    "網膜裂孔術後": "網膜裂孔",
    "網膜円孔PC後": "網膜円孔",
    "網膜萎縮性円孔": "網膜円孔",
    "lattice(PC後)": "lattice",
    "BRVO_PC後": "網膜静脈分枝閉塞",
    "ERM術後": "ERM",
    "黄斑偽円孔術後": "黄斑偽円孔",
    "黄斑円孔術後": "MH",
    "黄斑円孔PC後": "MH",
    "nAMD": "黄斑変性",
    "PCV": "黄斑変性",
    "mild_NPDR": "糖尿病網膜症",
    "moderate_NPDR": "糖尿病網膜症",
    "severe_NPDR": "糖尿病網膜症",
    "PDR": "糖尿病網膜症",
}

try:  # PyYAML C 実装を優先
    from yaml import CSafeLoader as YamlLoader  # type: ignore
except Exception:  # pragma: no cover
    from yaml import SafeLoader as YamlLoader


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=YamlLoader)


def normalize_split_dict(split_dict: Mapping[str, Iterable]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for key, value in split_dict.items():
        if not isinstance(value, Iterable):
            raise ValueError(f"Split '{key}' must be a list, got {type(value)}")
        normalized[key] = list(value)
    if not normalized:
        raise ValueError("Split file is empty. At least one split is required.")
    return normalized


def format_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*rows)]
    lines = []
    for r_idx, row in enumerate(rows):
        line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if r_idx == 0:
            sep = "-+-".join("-" * w for w in col_widths)
            lines.append(sep)
    return "\n".join(lines)


def resolve_config(args: argparse.Namespace) -> tuple[Path, Path, Optional[List[str]]]:
    cfg_root: Optional[Path] = None
    cfg_classes: Optional[List[str]] = None

    if args.config is not None:
        cfg = load_yaml(args.config)
        data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
        cfg_root_val = data_cfg.get("root")
        if cfg_root_val:
            cfg_root = Path(cfg_root_val)
        cfg_classes_val = data_cfg.get("classes")
        if cfg_classes_val:
            cfg_classes = list(cfg_classes_val)

    if args.root is not None:
        root = Path(args.root)
    elif cfg_root is not None:
        root = cfg_root
    else:
        root = DEFAULT_DATA_ROOT

    if args.split_file is not None:
        split_file = Path(args.split_file)
    else:
        split_file = root / "default_split.yaml"

    if args.classes:
        classes = list(args.classes)
    elif cfg_classes:
        classes = cfg_classes
    else:
        classes = None

    return root, split_file, classes


def check_consistency(
    root: Path,
    split_file: Path,
    classes: Optional[Sequence[str]] = None,
    label_suffix: str = ".yaml",
    allow_extra: bool = False,
    show_progress: bool = True,
    parent_check: bool = True,
    show_all_classes: bool = False,
):
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")

    split_dict_raw = load_yaml(split_file)
    if not isinstance(split_dict_raw, Mapping):
        raise ValueError(f"Invalid split file format: {split_file}")
    split_dict = normalize_split_dict(split_dict_raw)

    expected_classes = list(classes) if classes else None

    summary_missing_label = Counter()
    summary_invalid_yaml = Counter()
    summary_missing_class = Counter()
    summary_non_numeric = Counter()
    summary_unexpected_class = Counter()

    details_missing_label = defaultdict(list)
    details_invalid_yaml = defaultdict(list)
    details_missing_class = defaultdict(list)
    details_non_numeric = defaultdict(list)
    details_unexpected_class = defaultdict(list)
    summary_parent_missing = Counter()
    details_parent_missing = defaultdict(list)
    child_pos_total = Counter()
    child_parent_pos = Counter()

    # 収集用: 全ラベルに現れたクラス
    observed_class_counts = Counter()

    entries = [(s, f) for s, files in split_dict.items() for f in files]
    progress = tqdm(total=len(entries), disable=not show_progress, desc="Checking")

    for split_name, fname in entries:
        img_path = root / fname
        label_path = img_path.with_suffix(label_suffix)

        if not label_path.is_file():
            summary_missing_label[split_name] += 1
            if len(details_missing_label[split_name]) < 20:
                details_missing_label[split_name].append(str(label_path))
            progress.update(1)
            continue

        try:
            label_dict = load_yaml(label_path)
        except Exception as e:  # pragma: no cover
            summary_invalid_yaml[split_name] += 1
            if len(details_invalid_yaml[split_name]) < 20:
                details_invalid_yaml[split_name].append(f"{label_path} ({e})")
            progress.update(1)
            continue

        if not isinstance(label_dict, Mapping):
            summary_invalid_yaml[split_name] += 1
            if len(details_invalid_yaml[split_name]) < 20:
                details_invalid_yaml[split_name].append(
                    f"{label_path} (expected mapping, got {type(label_dict)})"
                )
            progress.update(1)
            continue

        label_keys = set(label_dict.keys())
        observed_class_counts.update(label_keys)

        # 想定クラスが指定されていれば存在を確認
        if expected_classes:
            for cls in expected_classes:
                if cls not in label_dict:
                    summary_missing_class[split_name] += 1
                    if len(details_missing_class[split_name]) < 20:
                        details_missing_class[split_name].append(
                                f"{label_path}: missing {cls}"
                        )
            if not allow_extra:
                for extra in sorted(label_keys - set(expected_classes)):
                    summary_unexpected_class[split_name] += 1
                    if len(details_unexpected_class[split_name]) < 20:
                            details_unexpected_class[split_name].append(
                                f"{label_path}: unexpected {extra}"
                            )

        # 値が数値か確認
        for cls, v in label_dict.items():
            try:
                float(v)
            except Exception:
                summary_non_numeric[split_name] += 1
                if len(details_non_numeric[split_name]) < 20:
                    details_non_numeric[split_name].append(
                            f"{label_path}: {cls} -> {v}"
                    )
                break

        # 親クラスチェック（例: 網膜裂孔PC後 -> 網膜裂孔 が1か）
        if parent_check:
            def infer_parent(name: str) -> Optional[str]:
                if name in PARENT_MAP:
                    return PARENT_MAP[name]
                suffixes = ["(PC後)", "PC後", "術後"]
                for s in suffixes:
                    if name.endswith(s):
                        base = name[: -len(s)]
                        base = base.rstrip(" _-（）()")
                        if base:
                            return base
                return None

            for cls, v in label_dict.items():
                try:
                    val = float(v)
                except Exception:
                    continue
                if val == 0:
                    continue
                parent = infer_parent(cls)
                if not parent:
                    continue
                child_pos_total[cls] += 1
                if parent in label_dict:
                    try:
                        parent_val = float(label_dict[parent])
                    except Exception:
                        continue
                    if parent_val != 0:
                        child_parent_pos[cls] += 1
                        continue
                    if parent_val == 0:
                        summary_parent_missing[split_name] += 1
                        if len(details_parent_missing[split_name]) < 20:
                            details_parent_missing[split_name].append(
                                f"{label_path}: {cls}=1 but {parent}=0"
                            )

        progress.update(1)

    progress.close()

    return {
        "summary_missing_label": summary_missing_label,
        "summary_invalid_yaml": summary_invalid_yaml,
        "summary_missing_class": summary_missing_class,
        "summary_non_numeric": summary_non_numeric,
        "summary_unexpected_class": summary_unexpected_class,
        "details_missing_label": details_missing_label,
        "details_invalid_yaml": details_invalid_yaml,
        "details_missing_class": details_missing_class,
        "details_non_numeric": details_non_numeric,
        "details_unexpected_class": details_unexpected_class,
        "observed_class_counts": observed_class_counts,
        "expected_classes": expected_classes,
        "show_all_classes": show_all_classes,
        "summary_parent_missing": summary_parent_missing,
        "details_parent_missing": details_parent_missing,
        "child_pos_total": child_pos_total,
        "child_parent_pos": child_parent_pos,
    }


def print_section(title: str, counter: Counter, details: Mapping[str, List[str]]):
    total = sum(counter.values())
    if total == 0:
        return
    print(f"=== {title} (total {total}) ===")
    for split, cnt in counter.items():
        print(f"[{split}] {cnt}")
        for item in details.get(split, []):
            print(f"  - {item}")
    print()


def print_summary(result: Mapping[str, object]):
    exp_classes = result["expected_classes"]
    if exp_classes:
        print("Expected classes:")
        for cls in exp_classes:
            print(f"  - {cls}")
        print()

    observed = result["observed_class_counts"]
    if observed:
        print("Observed classes (top 50 by count):")
        for cls, cnt in observed.most_common(50):
            print(f"  {cls}: {cnt}")
        if result.get("show_all_classes"):
            print("\nAll observed classes:")
            for cls, cnt in observed.most_common():
                print(f"  {cls}: {cnt}")
        print()

    print_section(
        "Missing label files",
        result["summary_missing_label"],
        result["details_missing_label"],
    )
    print_section(
        "Invalid label YAML",
        result["summary_invalid_yaml"],
        result["details_invalid_yaml"],
    )
    print_section(
        "Missing expected classes",
        result["summary_missing_class"],
        result["details_missing_class"],
    )
    print_section(
        "Non-numeric label values",
        result["summary_non_numeric"],
        result["details_non_numeric"],
    )
    print_section(
        "Unexpected classes",
        result["summary_unexpected_class"],
        result["details_unexpected_class"],
    )

    print_section(
        "Parent class missing when child=1",
        result.get("summary_parent_missing", Counter()),
        result.get("details_parent_missing", {}),
    )

    # 親子の1件数サマリ
    child_pos_total: Counter = result.get("child_pos_total", Counter())
    child_parent_pos: Counter = result.get("child_parent_pos", Counter())
    if child_pos_total:
        print("=== Parent-child positives summary ===")
        header = ["子クラス", "親クラス", "子=1件数", "親も1件数", "親も1率"]
        rows = [header]
        for child, parent in PARENT_MAP.items():
            child_cnt = child_pos_total.get(child, 0)
            parent_cnt = child_parent_pos.get(child, 0)
            ratio = (parent_cnt / child_cnt) if child_cnt else 0.0
            rows.append([
                child,
                parent,
                str(child_cnt),
                str(parent_cnt),
                f"{ratio:.4f}",
            ])
        print(format_table(rows))
        print()

    if all(sum(result[k].values()) == 0 for k in [
        "summary_missing_label",
        "summary_invalid_yaml",
        "summary_missing_class",
        "summary_non_numeric",
        "summary_unexpected_class",
    ]):
        print("All checks passed: no consistency issues found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label consistency checker")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "データセットのルートディレクトリ (画像/ラベル/default_split.yaml がある場所)。"
            f"未指定なら {DEFAULT_DATA_ROOT} を利用"
        ),
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="split YAML のパス。未指定なら <root>/default_split.yaml",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        help="期待するクラス名のリスト。未指定なら config か自動 (未指定の場合は存在チェックをスキップ)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="data.root / data.classes を含む設定 YAML (例: recon_pretrained_val_4ch.yaml)",
    )
    parser.add_argument(
        "--label-suffix",
        default=".yaml",
        help="ラベルファイルの拡張子 (デフォルト: .yaml)",
    )
    parser.add_argument(
        "--allow-extra",
        action="store_true",
        help="想定外クラスがあってもエラーとして扱わず、存在だけを集計",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="tqdm のプログレスバーを無効化",
    )
    parser.add_argument(
        "--show-all-classes",
        action="store_true",
        help="観測された全クラスと出現回数を表示 (デフォルトは上位50件)",
    )
    parser.add_argument(
        "--no-parent-check",
        action="store_true",
        help="PC後/術後 など子クラス=1時に親クラスも1かを確認するチェックを無効化",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root, split_file, classes = resolve_config(args)

    result = check_consistency(
        root=root,
        split_file=split_file,
        classes=classes,
        label_suffix=args.label_suffix,
        allow_extra=args.allow_extra,
        show_progress=not args.no_progress,
        parent_check=not args.no_parent_check,
        show_all_classes=args.show_all_classes,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
