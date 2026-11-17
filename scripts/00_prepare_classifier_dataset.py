"""Prepare the Kaggle Indonesian-cars 8-class dataset for a 3-class YOLO classify fine-tune.

Source (Apache 2.0): https://www.kaggle.com/datasets/muhammadluthfiarifin/indonesian-cars-classification-dataset
The source ships 8 car-type classes (hatchback, mpv, offroad, pickup, sedan, suv,
truck, van) either as flat class folders or as train/validation/test splits with
class folders. This script keeps only the configured body classes (MPV, Sedan,
Hatchback), copies images into an Ultralytics classify layout
(``train/<class>``, ``val/<class>``, ``test/<class>``), and writes a manifest +
summary for provenance. It never modifies the source tree.

Flat mode splits per class with a fixed seed; split mode preserves the source
split. Folder matching is case-insensitive; output class folders use the
canonical names from configs/bodytype_config.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VAL_DIR_NAMES = ("val", "valid", "validation")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "bodytype_config.json"


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if (Path.cwd().resolve() == ROOT.parent.resolve()
            and path.parts and path.parts[0].casefold() == ROOT.name.casefold()):
        return (Path.cwd() / path).resolve()
    return (ROOT / path).resolve()


def load_body_classes(config_path: Path) -> tuple[list[str], list[str]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return list(config["body_classes"]), list(config.get("dataset_source_classes", []))


def class_folders(directory: Path) -> dict[str, Path]:
    return {
        entry.name.casefold(): entry
        for entry in sorted(directory.iterdir())
        if entry.is_dir()
    }


def resolve_class_dirs(base: Path, source_classes: list[str]) -> dict[str, Path]:
    available = class_folders(base)
    resolved: dict[str, Path] = {}
    missing = []
    for name in source_classes:
        folder = available.get(name.casefold())
        if folder is None:
            missing.append(name)
        else:
            resolved[name.casefold()] = folder
    if missing:
        raise ValueError(
            f"Source class folder(s) not found under {base}: {missing}. "
            f"Available folders: {sorted(available)}"
        )
    return resolved


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )


def detect_split_layout(src: Path) -> tuple[dict[str, Path], str]:
    """Return ({split_name: dir}, mode) where mode is 'split' or 'flat'."""
    train_dir = src / "train"
    if train_dir.is_dir():
        splits: dict[str, Path] = {"train": train_dir}
        for candidate in VAL_DIR_NAMES:
            if (src / candidate).is_dir():
                splits["val"] = src / candidate
                break
        if (src / "test").is_dir():
            splits["test"] = src / "test"
        return splits, "split"
    return {"train": src}, "flat"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Dataset root (as downloaded from Kaggle)")
    parser.add_argument("--dst", type=Path, required=True, help="New output root for the 3-class classify dataset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--val-frac", type=float, default=0.1, help="Flat mode: validation fraction per class")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Flat mode: test fraction per class")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hash-manifest", action="store_true", help="Also record sha256 per copied file (slower)")
    args = parser.parse_args(argv)

    src = resolve_path(args.src)
    dst = resolve_path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"Source dataset folder not found: {src}")
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output folder: {dst}")
    if not 0 <= args.val_frac < 1 or not 0 <= args.test_frac < 1 or args.val_frac + args.test_frac >= 1:
        raise SystemExit("val-frac/test-frac must be in [0, 1) and sum to less than 1")

    body_classes, source_classes = load_body_classes(
        resolve_path(args.config)
    )
    if not source_classes:
        raise SystemExit("configs/bodytype_config.json has no dataset_source_classes")

    splits, mode = detect_split_layout(src)
    canonical_by_casefold = {name.casefold(): name for name in body_classes}
    source_by_casefold = {name.casefold() for name in source_classes}
    class_dirs_by_split = {
        split: resolve_class_dirs(directory, source_classes) for split, directory in splits.items()
    }

    rng = random.Random(args.seed)
    manifest_rows: list[dict[str, str]] = []
    counts: dict[str, dict[str, int]] = {canonical: {split: 0 for split in splits} for canonical in body_classes}
    selected_source_classes = {name.casefold() for name in source_classes}
    skipped_other_classes = sum(
        len(image_files(folder))
        for split_dir in splits.values()
        for name, folder in class_folders(split_dir).items()
        if name not in selected_source_classes
    )

    for split, directory in splits.items():
        class_dirs = class_dirs_by_split[split]
        for casefold, folder in class_dirs.items():
            canonical = canonical_by_casefold.get(casefold)
            if canonical is None or casefold not in source_by_casefold:
                continue
            files = image_files(folder)
            if mode == "flat":
                rng.shuffle(files)
                val_count = round(len(files) * args.val_frac)
                test_count = round(len(files) * args.test_frac)
                bucket = {
                    "val": files[:val_count],
                    "test": files[val_count : val_count + test_count],
                    "train": files[val_count + test_count :],
                }
                assigned: list[tuple[str, Path]] = [
                    (target_split, path) for target_split, paths in bucket.items() for path in paths
                ]
            else:
                assigned = [(split, path) for path in files]
            for target_split, path in assigned:
                target_dir = dst / target_split / canonical
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target_dir / path.name)
                counts[canonical][target_split] += 1
                row = {
                    "split": target_split,
                    "body_class": canonical,
                    "source_split": split if mode == "split" else "flat_root",
                    "source_relative_path": path.relative_to(src).as_posix(),
                    "file_name": path.name,
                    "bytes": str(path.stat().st_size),
                }
                if args.hash_manifest:
                    import hashlib

                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    row["sha256"] = digest
                manifest_rows.append(row)

    if not manifest_rows:
        raise SystemExit("No images were copied; check the source layout and class folder names")

    dst.mkdir(parents=True, exist_ok=True)
    with (dst / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as target:
        fieldnames = ["split", "body_class", "source_split", "source_relative_path", "file_name", "bytes"]
        if args.hash_manifest:
            fieldnames.append("sha256")
        writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest_rows)

    per_class_totals = {canonical: sum(split_counts.values()) for canonical, split_counts in counts.items()}
    smallest = min(per_class_totals.values())
    largest = max(per_class_totals.values())
    summary = {
        "schema": "car_classifier_dataset_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(src),
        "source_layout": mode,
        "output": str(dst),
        "body_classes": body_classes,
        "counts_by_class_by_split": counts,
        "counts_by_class_total": per_class_totals,
        "skipped_images_from_other_classes": skipped_other_classes,
        "imbalance_ratio_min_max": round(smallest / largest, 4) if largest else None,
        "warnings": [],
    }
    if largest and smallest / largest < 0.5:
        summary["warnings"].append(
            "Class imbalance: the smallest class has less than half the images of the largest class. "
            "Consider class-weighted loss or additional data before fine-tuning."
        )
    if mode == "flat" and (args.val_frac == 0 or args.test_frac == 0):
        summary["warnings"].append("A split fraction of 0 leaves that split empty; classifier training needs a validation split.")
    (dst / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
