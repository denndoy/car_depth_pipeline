"""Add car body-type predictions (MPV/Sedan/Hatchback) to each tracked car segment.

Standalone car-pipeline adaptation of ``scripts/06_attribute_evaluation/classify_tracked_persons.py``.
Runs the fine-tuned Ultralytics classification model (trained with
``00_prepare_classifier_dataset.py`` output on the Kaggle Indonesian-cars
subset) after detector+ByteTrack. The raw detection files are never modified.

One best-size/high-confidence crop per local track segment is classified; the
result is joined to every detection row of that segment for stable track-level
labels. When the top-1 probability is below the configured fallback threshold
(default 0.6 from configs/bodytype_config.json), the segment is flagged
``bodyclass_fallback_to_global=TRUE``: the frozen evaluator must use the global
quartic for that track instead of a class-specific quartic.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "car_depth_pipeline" / "configs" / "bodytype_config.json"
DEFAULT_OUTPUT_ROOT = ROOT / "car_depth_pipeline" / "outputs" / "bodyclass"
TRACK_FIELDS_BASE = [
    "video_id", "pass_id", "track_id", "subject_id",
    "first_frame_idx", "last_frame_idx", "detection_rows", "sample_frame_idx",
    "sample_timestamp_s", "sample_width_px", "sample_height_px", "sample_confidence",
]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def square_pad_bgr(image: np.ndarray, fill: int = 114) -> np.ndarray:
    height, width = image.shape[:2]
    side = max(height, width)
    canvas = np.full((side, side, 3), fill, dtype=np.uint8)
    x = (side - width) // 2
    y = (side - height) // 2
    canvas[y:y + height, x:x + width] = image
    return canvas


def normalize_class_name(value: str) -> str:
    value = (value or "").strip()
    for canonical in ("MPV", "Sedan", "Hatchback"):
        if value.casefold() == canonical.casefold():
            return canonical
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Tracking run directory (01_track_cars.py output)")
    parser.add_argument("--video", type=Path, required=True, help="Source MP4 matching that run")
    parser.add_argument("--model", type=Path, required=True, help="Fine-tuned YOLO classify weights (3 body classes)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--labels", type=Path, default=None, help="Optional CSV video_id,subject_id,reference_body_class for calibration accuracy")
    parser.add_argument("--detections", type=Path, default=None, help="Defaults to detections_subject_mapped.csv if present, else detections.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=None, help="Defaults to classifier_imgsz from the config")
    parser.add_argument("--fallback-threshold", type=float, default=None, help="Defaults to fallback_threshold from the config")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads) < 1 or (args.imgsz is not None and args.imgsz < 1):
        parser.error("--batch-size, --threads, --imgsz must be positive")
    if args.fallback_threshold is not None and not 0 <= args.fallback_threshold <= 1:
        parser.error("--fallback-threshold must be within [0, 1]")

    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    body_classes: list[str] = [str(name) for name in config["body_classes"]]
    fallback_threshold = float(
        args.fallback_threshold if args.fallback_threshold is not None else config.get("fallback_threshold", 0.6)
    )
    imgsz = int(args.imgsz if args.imgsz is not None else config.get("classifier_imgsz", 224))

    run_dir = resolve(args.run_dir)
    video_path = resolve(args.video)
    model_path = resolve(args.model)
    detections_path = resolve(args.detections) if args.detections else (
        run_dir / "detections_subject_mapped.csv" if (run_dir / "detections_subject_mapped.csv").is_file()
        else run_dir / "detections.csv"
    )
    for path in (video_path, model_path, detections_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    references = {}
    if args.labels:
        labels_path = resolve(args.labels)
        if not labels_path.is_file():
            raise FileNotFoundError(labels_path)
        for row in read_csv(labels_path):
            references[(row["video_id"], row["subject_id"])] = normalize_class_name(row.get("reference_body_class", ""))

    output_dir = resolve(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / (
        f"track_bodyclass_{datetime.now():%Y%m%d_%H%M%S}"
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    detection_rows = read_csv(detections_path)
    by_track: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in detection_rows:
        video_id = row.get("video_id", "").strip()
        pass_id = row.get("pass_id", "").strip()
        track_id = row.get("track_id", "").strip()
        if track_id:
            by_track[(video_id, pass_id, track_id)].append(row)
    if not by_track:
        raise ValueError(f"No tracked detections found in {detections_path}")

    # A larger, higher-confidence crop tends to preserve more visible body cues.
    samples = []
    for key, rows in by_track.items():
        best = max(
            rows,
            key=lambda row: float(row["w_px"]) * float(row["h_px"]) * max(float(row["confidence"]), 0.01),
        )
        samples.append((key, rows, best))
    samples.sort(key=lambda item: (item[0][0], int(item[0][2])))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_cache: dict[int, np.ndarray] = {}
    crops = []
    candidate_rows = []
    try:
        for (video_id, pass_id, track_id), rows, best in samples:
            frame_idx = int(best["frame_idx"])
            if frame_idx not in frame_cache:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Could not read source frame {frame_idx}")
                frame_cache[frame_idx] = frame
            frame = frame_cache[frame_idx]
            x1 = max(0, min(frame_width, int(round(float(best["x1_px"])))))
            y1 = max(0, min(frame_height, int(round(float(best["y1_px"])))))
            x2 = max(0, min(frame_width, int(round(float(best["x2_px"])))))
            y2 = max(0, min(frame_height, int(round(float(best["y2_px"])))))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = square_pad_bgr(frame[y1:y2, x1:x2])
            crops.append(crop)
            subject_id = best.get("subject_id", "").strip()
            candidate_rows.append({
                "video_id": video_id,
                "pass_id": pass_id,
                "track_id": track_id,
                "subject_id": subject_id,
                "reference_body_class": references.get((video_id, subject_id), "") if subject_id else "",
                "first_frame_idx": str(min(int(r["frame_idx"]) for r in rows)),
                "last_frame_idx": str(max(int(r["frame_idx"]) for r in rows)),
                "detection_rows": str(len(rows)),
                "sample_frame_idx": str(frame_idx),
                "sample_timestamp_s": best.get("timestamp_s", ""),
                "sample_width_px": str(x2 - x1),
                "sample_height_px": str(y2 - y1),
                "sample_confidence": best.get("confidence", ""),
            })
    finally:
        cap.release()

    if not candidate_rows:
        raise RuntimeError("No valid tracked car crops could be extracted")

    torch.set_num_threads(args.threads)
    classifier = YOLO(str(model_path))
    class_names = {int(k): normalize_class_name(v) for k, v in classifier.names.items()}
    label_to_id = {label: index for index, label in class_names.items()}
    if set(label_to_id) != set(body_classes):
        raise ValueError(
            f"Classifier class names {sorted(label_to_id)} do not match configured body classes {sorted(body_classes)}. "
            "Retrain/export the classifier with exactly these class folders."
        )

    predictions = []
    for start in range(0, len(crops), args.batch_size):
        batch_crops = crops[start:start + args.batch_size]
        results = classifier.predict(
            source=batch_crops, batch=args.batch_size, imgsz=imgsz,
            device="cpu", verbose=False,
        )
        for result in results:
            if result.probs is None:
                raise ValueError("Classifier returned no classification probabilities; is this a YOLO classify model?")
            predictions.append(result.probs.data.cpu().numpy())
        print(f"Classified tracks {min(start + len(batch_crops), len(crops))}/{len(crops)}", flush=True)

    track_fields = list(TRACK_FIELDS_BASE) + [
        "reference_body_class", "bodyclass_predicted_model_class", "bodyclass_confidence_raw_softmax",
        "bodyclass_fallback_to_global",
    ] + [f"bodyclass_prob_{name}" for name in body_classes]
    track_rows = []
    pred_by_track: dict[tuple[str, str, str], dict] = {}
    for sample, scores in zip(candidate_rows, predictions):
        top_id = int(np.argmax(scores))
        pred = class_names[top_id]
        top_prob = float(scores[top_id])
        result = {
            **sample,
            "bodyclass_predicted_model_class": pred,
            "bodyclass_confidence_raw_softmax": f"{top_prob:.6f}",
            "bodyclass_fallback_to_global": "TRUE" if top_prob < fallback_threshold else "FALSE",
        }
        for name in body_classes:
            result[f"bodyclass_prob_{name}"] = f"{float(scores[label_to_id[name]]):.6f}"
        track_rows.append(result)
        pred_by_track[(sample["video_id"], sample["pass_id"], sample["track_id"])] = result

    detection_fields = list(detection_rows[0].keys()) + [
        "bodyclass_reference_model_class", "bodyclass_predicted_model_class", "bodyclass_confidence_raw_softmax",
        "bodyclass_fallback_to_global", "bodyclass_sample_frame_idx",
    ]
    detection_output = []
    for row in detection_rows:
        key = (row.get("video_id", "").strip(), row.get("pass_id", "").strip(), row.get("track_id", "").strip())
        pred = pred_by_track.get(key)
        detection_output.append({
            **row,
            "bodyclass_reference_model_class": pred["reference_body_class"] if pred else "",
            "bodyclass_predicted_model_class": pred["bodyclass_predicted_model_class"] if pred else "",
            "bodyclass_confidence_raw_softmax": pred["bodyclass_confidence_raw_softmax"] if pred else "",
            "bodyclass_fallback_to_global": pred["bodyclass_fallback_to_global"] if pred else "no_track_prediction",
            "bodyclass_sample_frame_idx": pred["sample_frame_idx"] if pred else "",
        })

    output_dir.mkdir(parents=True)
    write_csv(output_dir / "track_bodyclass_predictions.csv", track_rows, track_fields)
    write_csv(output_dir / "detections_with_bodyclass.csv", detection_output, detection_fields)

    reference_rows = [row for row in track_rows if row["reference_body_class"]]
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for row in reference_rows:
        confusion[(row["reference_body_class"], row["bodyclass_predicted_model_class"])] += 1
    matches = sum(
        1 for row in reference_rows if row["reference_body_class"] == row["bodyclass_predicted_model_class"]
    )
    summary = {
        "schema": "car_bodyclass_track_predictions_v1",
        "model": str(model_path),
        "model_classes": class_names,
        "config": {
            "fallback_threshold": fallback_threshold,
            "imgsz": imgsz,
            "batch_size": args.batch_size,
            "threads": args.threads,
        },
        "input_run": detections_path.relative_to(ROOT).as_posix() if detections_path.is_relative_to(ROOT) else str(detections_path),
        "video": str(video_path),
        "track_segments_classified": len(track_rows),
        "detection_rows_enriched": len(detection_output),
        "fallback_segments": sum(row["bodyclass_fallback_to_global"] == "TRUE" for row in track_rows),
        "reference_labeled_segments": len(reference_rows),
        "reference_top1_accuracy": round(matches / len(reference_rows), 4) if reference_rows else None,
        "reference_confusion_matrix": {
            f"{truth} -> {pred}": count for (truth, pred), count in sorted(confusion.items())
        },
        "note": (
            "One representative best-area/high-confidence crop is classified per ByteTrack segment; its body-type "
            "prediction is copied to every row of that segment. Segments below the fallback threshold are flagged "
            "for global-quartic use by the frozen evaluator. Scores are raw softmax values, not calibrated probabilities."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        """# Car body-type track enrichment

Post-tracking stage; leaves `detections.csv`/`detections_subject_mapped.csv` unchanged and writes a separate `detections_with_bodyclass.csv`.

- One representative, largest-area/high-confidence crop is classified per ByteTrack segment; the prediction is propagated to every detection row in that segment to avoid per-frame label flicker.
- `track_bodyclass_predictions.csv` retains per-segment predictions, raw softmax probabilities per class, and the fallback flag.
- `bodyclass_fallback_to_global=TRUE` means top-1 probability < fallback threshold; the frozen evaluator must use the global quartic for those tracks.
- Predictions are visual body-type estimates from the fine-tuned classifier, not manufacturer registrations.
- Source crops are read locally, square-padded in memory, and never written or uploaded by this script. Blur license plates in any media you share.
""",
        encoding="utf-8",
    )
    print(f"Wrote track and detection body-type attributes to {output_dir.relative_to(ROOT) if output_dir.is_relative_to(ROOT) else output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
