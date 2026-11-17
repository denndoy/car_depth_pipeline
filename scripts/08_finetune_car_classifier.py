"""Fine-tune a three-class Ultralytics classifier on manually labeled car crops.

The checkpoint can be a .pt or the user's .pt.zip wrapper. Training data must
already be prepared by 07_build_reviewed_classifier_dataset.py and follow the
Ultralytics classify folder layout: train/<class>, val/<class>, optional
test/<class>. This script never uses pseudo-label predictions as ground truth.

Training is intentionally not started until corrected labels exist. The
recommended starting point is the existing three-class checkpoint so the
network can adapt from Kaggle imagery to local camera crops.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_HELPER = ROOT / "ultralytics_classifier_from_detection.py"
CLASSES = ("MPV", "Sedan", "Hatchback")


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if (Path.cwd().resolve() == ROOT.parent.resolve()
            and path.parts and path.parts[0].casefold() == ROOT.name.casefold()):
        return (Path.cwd() / path).resolve()
    return (ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_names(names: dict) -> dict[int, str]:
    mapping = {name.casefold(): name for name in CLASSES}
    result = {int(index): mapping.get(str(name).strip().casefold(), str(name).strip()) for index, name in names.items()}
    if set(result.values()) != set(CLASSES) or len(result) != len(CLASSES):
        raise ValueError(f"Checkpoint class names must be exactly {CLASSES}; got {names}")
    ordered = [result[index] for index in sorted(result)]
    dataset_order = sorted(CLASSES)
    if ordered != dataset_order:
        raise ValueError(
            f"Checkpoint class-index order {ordered} does not match Ultralytics' sorted folder order {dataset_order}; "
            "re-map/retrain the classifier head before fine-tuning to avoid label-index mismatch."
        )
    return result


def unwrap_checkpoint(checkpoint: Path, scratch: Path) -> Path:
    spec = importlib.util.spec_from_file_location("classifier_checkpoint_helper", CHECKPOINT_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import checkpoint ZIP helper: {CHECKPOINT_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_checkpoint(checkpoint, scratch)


def validate_dataset(dataset: Path) -> dict:
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    counts = {}
    for split in ("train", "val", "test"):
        split_dir = dataset / split
        if not split_dir.is_dir():
            continue
        split_counts = {}
        for class_name in CLASSES:
            class_dir = split_dir / class_name
            split_counts[class_name] = sum(
                1 for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ) if class_dir.is_dir() else 0
        counts[split] = split_counts
    if not counts.get("train") or any(counts["train"].get(name, 0) == 0 for name in CLASSES):
        raise ValueError(f"Training split must contain all 3 classes; got {counts.get('train')}")
    if not counts.get("val") or sum(counts["val"].values()) == 0:
        raise ValueError("Validation split is required; provide manually labeled held-out video crops")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Reviewed classify dataset from script 07")
    parser.add_argument("--weights", type=Path, required=True, help="3-class .pt or .pt.zip starting checkpoint")
    parser.add_argument("--out-dir", type=Path, required=True, help="New training output directory")
    parser.add_argument("--name", default="car_bodyclass_finetune")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu", help="cpu, 0 (GPU), or another Ultralytics device selector")
    parser.add_argument("--lr0", type=float, default=0.001, help="Fine-tuning learning rate (default matches the starting checkpoint's recorded lr0)")
    args = parser.parse_args(argv)
    if min(args.epochs, args.imgsz, args.batch) < 1 or args.patience < 0 or args.workers < 0:
        parser.error("epochs/imgsz/batch must be positive; patience/workers must be nonnegative")
    if args.lr0 <= 0:
        parser.error("--lr0 must be positive")

    dataset = resolve_path(args.dataset)
    checkpoint = resolve_path(args.weights)
    out_dir = resolve_path(args.out_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty training output: {out_dir}")
    dataset_counts = validate_dataset(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = unwrap_checkpoint(checkpoint, out_dir)
    model = YOLO(str(checkpoint_path), task="classify")
    source_names = dict(model.names)
    canonical_names(model.names)

    train_kwargs = {
        "data": str(dataset),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "device": args.device,
        "project": str(out_dir / "runs"),
        "name": args.name,
        "exist_ok": False,
    }
    train_kwargs["lr0"] = args.lr0

    print("Starting fine-tune with manually reviewed labels:")
    print(json.dumps({
        "dataset": str(dataset),
        "dataset_counts": dataset_counts,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_class_names": source_names,
        "train_args": train_kwargs,
    }, indent=2, ensure_ascii=False))
    model.train(**train_kwargs)

    run_dir = out_dir / "runs" / args.name
    best_path = run_dir / "weights" / "best.pt"
    if not best_path.is_file():
        candidates = list((out_dir / "runs").rglob("best.pt"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Could not uniquely locate best.pt under {out_dir / 'runs'}: {candidates}")
        best_path = candidates[0]
    fine_tuned = YOLO(str(best_path), task="classify")
    resulting_names = canonical_names(fine_tuned.names)
    summary = {
        "schema": "car_bodytype_classifier_finetune_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "dataset_counts": dataset_counts,
        "starting_checkpoint": str(checkpoint),
        "starting_checkpoint_sha256": sha256_file(checkpoint),
        "starting_checkpoint_names": source_names,
        "fine_tuned_best_pt": str(best_path),
        "fine_tuned_best_sha256": sha256_file(best_path),
        "fine_tuned_class_names": resulting_names,
        "train_args": train_kwargs,
        "interpretation": "Ultralytics validation metrics are model-selection metrics on the manually labeled val split; keep an untouched test video for final generalization claims.",
    }
    (out_dir / "finetune_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Fine-tune complete. Best weights: {best_path}")
    print(f"Summary: {out_dir / 'finetune_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
