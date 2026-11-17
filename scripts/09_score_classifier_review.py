"""Score classifier predictions against manually reviewed car body-type labels.

The corrected-label CSV must come from scripts/06_prepare_blind_crop_review.py
and must be completed by a human without seeing model predictions. Predictions
come from all_predictions.csv emitted by
ultralytics_classifier_from_detection.py generate. Only the requested split is
scored; rows marked Uncertain/Skip are reported as excluded.

Metrics are computed without scikit-learn: confusion matrix, accuracy,
per-class precision/recall/F1/support, macro-F1, and coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("MPV", "Sedan", "Hatchback")
SKIP_LABELS = {"uncertain", "skip", "ignore", "unknown", "other", "invalid", "reject", "not car", "not_car"}
SPLITS = {"train", "val", "test"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if (Path.cwd().resolve() == ROOT.parent.resolve()
            and path.parts and path.parts[0].casefold() == ROOT.name.casefold()):
        return (Path.cwd() / path).resolve()
    return (ROOT / path).resolve()


def canonical(raw: str) -> str | None:
    value = (raw or "").strip().casefold()
    if value in SKIP_LABELS:
        return None
    for name in CLASSES:
        if value == name.casefold():
            return name
    raise ValueError(f"Invalid class label {raw!r}; expected one of {CLASSES}, Uncertain, or Skip")


def read_video_split_map(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not {"source_video_id", "split"}.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} must contain source_video_id,split columns")
        result = {}
        for line_no, row in enumerate(reader, start=2):
            video_id = (row.get("source_video_id") or "").strip()
            split = (row.get("split") or "").strip().lower()
            if not video_id or video_id in result:
                raise ValueError(f"{path}:{line_no}: empty or duplicate source_video_id {video_id!r}")
            if split not in SPLITS | {"exclude"}:
                raise ValueError(f"{path}:{line_no}: split must be train/val/test/exclude, got {split!r}")
            result[video_id] = split
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-csv", type=Path, required=True, help="Human-completed CSV from the blind review page")
    parser.add_argument("--predictions-csv", type=Path, required=True, help="all_predictions.csv from the model being evaluated")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--video-split-map", type=Path, default=None,
                        help="Optional research-role map; if present, split is selected by source_video_id, not detection ZIP split")
    parser.add_argument("--confirm-detection-source-split", action="store_true",
                        help="Explicitly score the ZIP's original detection split")
    parser.add_argument("--out-dir", type=Path, required=True, help="New report directory")
    args = parser.parse_args(argv)

    reviewed_path = resolve_path(args.reviewed_csv)
    predictions_path = resolve_path(args.predictions_csv)
    out_dir = resolve_path(args.out_dir)
    if not reviewed_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(reviewed_path if not reviewed_path.is_file() else predictions_path)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty report directory: {out_dir}")
    if args.video_split_map is None and not args.confirm_detection_source_split:
        raise ValueError(
            "Specify --video-split-map for research roles, or explicitly confirm the detection ZIP split. "
            "The detection ZIP split may not match classifier train/val/test roles."
        )
    split_map_path = resolve_path(args.video_split_map) if args.video_split_map else None
    video_split_map = read_video_split_map(split_map_path) if split_map_path is not None else None

    reviewed = read_csv(reviewed_path)
    predictions = read_csv(predictions_path)
    for path, rows, required in (
        (reviewed_path, reviewed, {"crop_id", "source_split", "corrected_label"}),
        (predictions_path, predictions, {"crop_id", "source_split", "predicted_label", "confidence"}),
    ):
        fields = set(rows[0]) if rows else set()
        if not required.issubset(fields):
            raise ValueError(f"{path} is missing columns {sorted(required - fields)}")

    truth_by_id = {}
    for row in reviewed:
        crop_id = (row.get("crop_id") or "").strip()
        if not crop_id or crop_id in truth_by_id:
            raise ValueError(f"Empty or duplicate crop_id in {reviewed_path}: {crop_id!r}")
        truth_by_id[crop_id] = row
    predictions_by_id = {}
    for row in predictions:
        crop_id = (row.get("crop_id") or "").strip()
        if not crop_id or crop_id in predictions_by_id:
            raise ValueError(f"Empty or duplicate crop_id in {predictions_path}: {crop_id!r}")
        predictions_by_id[crop_id] = row

    selected = []
    for row in reviewed:
        source_split = (row.get("source_split") or "").strip().lower()
        if video_split_map is None:
            assigned_split = source_split
        else:
            video_id = (row.get("source_video_id") or "").strip()
            if not video_id or video_id not in video_split_map:
                raise ValueError(f"Reviewed crop {row.get('crop_id')} lacks a source_video_id in the split map")
            assigned_split = video_split_map[video_id]
        if assigned_split == args.split:
            selected.append(row)
    if not selected:
        raise ValueError(f"No reviewed crops for split={args.split}")
    scored = []
    excluded = 0
    missing_predictions = []
    blank_labels = []
    for row in selected:
        crop_id = row["crop_id"]
        raw_truth = (row.get("corrected_label") or "").strip()
        if not raw_truth:
            blank_labels.append(crop_id)
            continue
        truth = canonical(raw_truth)
        if truth is None:
            excluded += 1
            continue
        prediction = predictions_by_id.get(crop_id)
        if prediction is None:
            missing_predictions.append(crop_id)
            continue
        predicted = canonical(prediction.get("predicted_label", ""))
        if predicted is None:
            raise ValueError(f"Model prediction for {crop_id} is not a body class")
        scored.append({
            "crop_id": crop_id,
            "source_video_id": row.get("source_video_id", ""),
            "body_class_gt": truth,
            "predicted_label": predicted,
            "confidence": prediction.get("confidence", ""),
            "correct": truth == predicted,
        })
    if blank_labels:
        raise ValueError(
            f"{len(blank_labels)} {args.split} crop(s) have blank labels; first: {blank_labels[:5]}. "
            "Label all eligible samples or explicitly mark Uncertain/Skip."
        )
    if missing_predictions:
        raise ValueError(f"No model predictions for {len(missing_predictions)} reviewed crops; first: {missing_predictions[:5]}")
    if not scored:
        raise ValueError("No labeled, scoreable crops remain after excluding Uncertain/Skip")

    matrix = {truth: {pred: 0 for pred in CLASSES} for truth in CLASSES}
    for row in scored:
        matrix[row["body_class_gt"]][row["predicted_label"]] += 1
    per_class = {}
    f1_values = []
    for name in CLASSES:
        tp = matrix[name][name]
        support = sum(matrix[name].values())
        predicted_count = sum(matrix[truth][name] for truth in CLASSES)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "support": support,
            "predicted_count": predicted_count,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        f1_values.append(f1)
    correct = sum(row["correct"] for row in scored)
    summary = {
        "schema": "car_bodytype_classifier_review_metrics_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reviewed_csv": str(reviewed_path),
        "predictions_csv": str(predictions_path),
        "split_scored": args.split,
        "split_policy": "explicit research video split map" if video_split_map is not None else "explicitly confirmed original detection ZIP split",
        "review_split_count": len(selected),
        "scored_count": len(scored),
        "excluded_uncertain_or_skip": excluded,
        "accuracy": round(correct / len(scored), 6),
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_predicted": matrix,
        "interpretation": "Human-corrected labels are ground truth. Do not score on train for generalization claims; validation informs model selection, while final claims require a separate untouched test-video split.",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["crop_id", "source_video_id", "body_class_gt", "predicted_label", "confidence", "correct"]
    with (out_dir / "scored_predictions.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored)
    with (out_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.writer(target)
        writer.writerow(["body_class_gt", *CLASSES])
        for truth in CLASSES:
            writer.writerow([truth, *(matrix[truth][pred] for pred in CLASSES)])
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
