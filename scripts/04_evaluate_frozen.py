"""Frozen-inference evaluation for the car pipeline on Videos/Takes 2-4.

Loads ``fitted_models.json`` produced by ``03_fit_class_quartics.py`` (never
refits) and scores manually confirmed crossing events of unseen cars:

- ``end_to_end`` : the fine-tuned classifier (02_classify_car_tracks.py output)
  selects the body-class quartic per track; tracks below the fallback threshold
  or without a prediction use the global quartic. This is the deployable mode.
- ``oracle``     : the manually confirmed ground-truth body class selects the
  quartic. Upper bound that isolates regression error from classifier error.
- ``global``     : class-agnostic baseline quartic for every event.

Crossing events must be manually confirmed (same strict validation as the
person research). Generic helpers are imported from local vendored copies in
scripts/vendor_distance; the original research source tree is not required.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "vendor_distance"))

from polynomial_take_pipeline import (  # noqa: E402
    DEGREE,
    EXPECTED_MARKER_IDS,
    link_event_detections,
    load_tracking_metadata,
    normalize_track_id,
    read_confirmed_events,
    read_detection_files,
    read_marker_ground_truth,
    resolve_path,
    sha256_file,
    validate_tracking_metadata,
    workspace_path,
)
from monotone_quartic import predict  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "bodytype_config.json"
DEFAULT_MARKER_GT = ROOT / "inputs" / "marker_ground_truth.csv"
GLOBAL_MODEL_ID = "global_quartic_monotone_take1"
CLASS_MODELS_KEY = "body_class_quartic_monotone_take1"


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    classes = [str(name) for name in config.get("body_classes", [])]
    if len(classes) < 2:
        raise ValueError("Config must define at least two body classes")
    return config


def canonical_class(raw: str, body_classes: list[str]) -> str:
    by_casefold = {name.casefold(): name for name in body_classes}
    canonical = by_casefold.get((raw or "").strip().casefold())
    if canonical is None:
        raise ValueError(f"Unknown body_class_gt {raw!r}; allowed: {body_classes}")
    return canonical


def read_event_body_classes(paths: list[Path], body_classes: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            fields = set(reader.fieldnames or [])
            if not {"event_id", "body_class_gt"}.issubset(fields):
                raise ValueError(f"{path}: requires explicit unique 'event_id' and 'body_class_gt' columns")
            for row in reader:
                event_id = (row.get("event_id") or "").strip()
                if not event_id:
                    raise ValueError(f"{path}: empty event_id")
                if event_id in result:
                    raise ValueError(f"{path}: duplicate event_id {event_id!r}")
                result[event_id] = canonical_class(row.get("body_class_gt"), body_classes)
    return result


def load_classifier_predictions(paths: list[Path], body_classes: list[str]) -> tuple[dict, dict]:
    """Return {(video_id, normalized track): record, (video_id, subject): record} lookups."""
    by_track: dict[tuple[str, str], dict] = {}
    by_subject: dict[tuple[str, str], dict] = {}
    casefold = {name.casefold(): name for name in body_classes}
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            required = {"video_id", "bodyclass_predicted_model_class", "bodyclass_fallback_to_global"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"{path}: expected a 02_classify_car_tracks.py CSV with columns {sorted(required)}"
                )
            for row in reader:
                video_id = (row.get("video_id") or "").strip()
                track = normalize_track_id(row.get("track_id"))
                subject = (row.get("subject_id") or "").strip()
                predicted_raw = (row.get("bodyclass_predicted_model_class") or "").strip()
                predicted = casefold.get(predicted_raw.casefold(), predicted_raw)
                confidence_raw = (row.get("bodyclass_confidence_raw_softmax") or "").strip()
                try:
                    confidence_value = float(confidence_raw) if confidence_raw else None
                except ValueError:
                    confidence_value = None
                record = {
                    "predicted_class": predicted,
                    "confidence": confidence_raw,
                    "confidence_value": confidence_value,
                    "fallback": (row.get("bodyclass_fallback_to_global") or "").strip().upper() == "TRUE",
                    "source_csv": workspace_path(path),
                }
                if track:
                    by_track[(video_id, track)] = record
                if subject:
                    by_subject[(video_id, subject)] = record
    return by_track, by_subject


def metric_block(pred: np.ndarray, truth: np.ndarray) -> dict:
    if pred.size == 0:
        return {"N": 0, "MAE_m": None, "RMSE_m": None, "Bias_m": None, "AbsRel": None,
                "negative_prediction_count": 0}
    error = pred - truth
    return {
        "N": int(pred.size),
        "MAE_m": round(float(np.mean(np.abs(error))), 4),
        "RMSE_m": round(float(np.sqrt(np.mean(error**2))), 4),
        "Bias_m": round(float(np.mean(error)), 4),
        "AbsRel": round(float(np.mean(np.abs(error) / truth)), 4),
        "negative_prediction_count": int(np.sum(pred < 0)),
    }


def prepare_output_dir(path: Path) -> Path:
    path = resolve_path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="fitted_models.json from 03_fit_class_quartics.py")
    parser.add_argument("--events", type=Path, nargs="+", required=True, help="Confirmed test-take event CSV(s) with body_class_gt")
    parser.add_argument("--detections", type=Path, nargs="+", required=True, help="Detection CSV(s) of the test takes")
    parser.add_argument("--bodyclass-predictions", type=Path, nargs="+", default=None,
                        help="track_bodyclass_predictions.csv file(s) from 02_classify_car_tracks.py")
    parser.add_argument("--marker-ground-truth", type=Path, default=DEFAULT_MARKER_GT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory (never overwritten)")
    args = parser.parse_args(argv)

    config = load_config(resolve_path(args.config))
    body_classes = [str(name) for name in config["body_classes"]]

    model_path = resolve_path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    bundle = json.loads(model_path.read_text(encoding="utf-8"))
    if bundle.get("schema") != "car_bodyclass_and_global_monotone_quartic_v1":
        raise ValueError(f"Unsupported model bundle schema: {bundle.get('schema')!r}")
    if int(bundle.get("degree", -1)) != DEGREE:
        raise ValueError("Model bundle is not a degree-4 quartic bundle")
    global_model = bundle["models"][GLOBAL_MODEL_ID]
    class_models: dict = bundle["models"][CLASS_MODELS_KEY]
    for name in body_classes:
        if name not in class_models:
            raise ValueError(f"Model bundle lacks a quartic for body class {name}")

    marker_gt_path = resolve_path(args.marker_ground_truth)
    marker_gt = read_marker_ground_truth(marker_gt_path)
    if set(marker_gt) != set(EXPECTED_MARKER_IDS):
        raise ValueError(f"Marker table must contain exactly M01-M09; found {sorted(marker_gt)}")
    calibration_gt = {marker: float(value) for marker, value in bundle.get("marker_ground_truth_m", {}).items()}
    for marker_id, distance in marker_gt.items():
        if marker_id not in calibration_gt or abs(calibration_gt[marker_id] - distance) > 1e-6:
            raise ValueError(
                f"Marker ground truth changed after calibration for {marker_id}; the frozen model is not comparable"
            )

    event_paths = [resolve_path(path) for path in args.events]
    detection_paths = [resolve_path(path) for path in args.detections]
    events, event_sources = read_confirmed_events(event_paths, marker_gt)
    test_take_ids = sorted({event["take_id"] for event in events})
    overlap = sorted(set(bundle.get("calibration_take_ids", [])).intersection(test_take_ids))
    if overlap:
        raise ValueError(
            f"Test events include calibration take(s) {overlap}; use Videos 2-4 only so the frozen test stays independent"
        )

    body_class_by_event = read_event_body_classes(event_paths, body_classes)
    for event in events:
        event["body_class_gt"] = body_class_by_event[event["event_id"]]

    detections_by_frame, detection_sources = read_detection_files(detection_paths)
    linked = link_event_detections(events, detections_by_frame, require_detection=False)
    tracking_metadata, metadata_warnings = load_tracking_metadata(detection_paths)
    metadata_warnings.extend(validate_tracking_metadata(bundle.get("tracking_metadata", []), tracking_metadata))

    classifier_by_track, classifier_by_subject = ({}, {})
    fallback_policy = bundle.get("fallback_policy", {})
    if args.bodyclass_predictions:
        classifier_paths = [resolve_path(path) for path in args.bodyclass_predictions]
        classifier_by_track, classifier_by_subject = load_classifier_predictions(classifier_paths, body_classes)
    threshold = float(fallback_policy.get("threshold", config.get("fallback_threshold", 0.6)))

    prediction_fields = [
        "event_id", "take_id", "video_id", "subject_id", "marker_id", "frame_idx",
        "body_class_gt", "gt_distance_m", "track_id", "A_bb_px2", "confidence",
        "detection_found",
        "pred_end_to_end_m", "error_end_to_end_m", "model_used_end_to_end", "selection_reason_end_to_end",
        "pred_oracle_m", "error_oracle_m", "model_used_oracle",
        "pred_global_m", "error_global_m",
        "classifier_predicted_class", "classifier_confidence", "classifier_fallback_flag",
        "outside_model_area_range_end_to_end", "outside_model_area_range_oracle", "outside_model_area_range_global",
        "prediction_nonpositive_end_to_end", "prediction_nonpositive_oracle", "prediction_nonpositive_global",
    ]
    prediction_rows: list[dict] = []
    fallback_reason_counts: dict[str, int] = defaultdict(int)
    confusion_counts: dict[tuple[str, str], int] = defaultdict(int)

    def model_range(model: dict, area: float) -> tuple[bool, bool]:
        outside = area < float(model["training_area_min_px2"]) or area > float(model["training_area_max_px2"])
        return outside

    for row in linked:
        gt = float(row["gt_distance_m"])
        gt_class = row["body_class_gt"]
        event_id = row["event_id"]
        video_id = row["video_id"]
        subject_id = row.get("subject_id", "")
        track_id = row.get("track_id", "")
        base = {
            "event_id": event_id,
            "take_id": row["take_id"],
            "video_id": video_id,
            "subject_id": subject_id,
            "marker_id": row["marker_id"],
            "frame_idx": row["frame_idx"],
            "body_class_gt": gt_class,
            "gt_distance_m": gt,
            "track_id": track_id,
        }
        if row.get("_detection") is None:
            prediction_rows.append({
                **base,
                "A_bb_px2": "", "confidence": "", "detection_found": "FALSE",
                "pred_end_to_end_m": "", "error_end_to_end_m": "", "model_used_end_to_end": "",
                "selection_reason_end_to_end": "no_detection", "pred_oracle_m": "", "error_oracle_m": "",
                "model_used_oracle": "", "pred_global_m": "", "error_global_m": "",
                "classifier_predicted_class": "", "classifier_confidence": "", "classifier_fallback_flag": "",
                "outside_model_area_range_end_to_end": "", "outside_model_area_range_oracle": "",
                "outside_model_area_range_global": "", "prediction_nonpositive_end_to_end": "",
                "prediction_nonpositive_oracle": "", "prediction_nonpositive_global": "",
            })
            continue

        area = float(row["A_bb_px2"])
        confidence = row.get("confidence")
        pred_global = float(predict(np.asarray([area]), global_model)[0])
        oracle_model = class_models[gt_class]
        pred_oracle = float(predict(np.asarray([area]), oracle_model)[0])

        record = (
            classifier_by_track.get((video_id, normalize_track_id(track_id)))
            or classifier_by_subject.get((video_id, subject_id))
        )
        if record is None:
            used_class, reason = "GLOBAL", "no_classifier_prediction"
            classifier_class, classifier_conf, classifier_fallback = "", "", ""
        else:
            classifier_class = record["predicted_class"]
            classifier_conf = record["confidence"]
            # The frozen bundle owns the fallback policy: recompute from the raw
            # confidence against the bundle threshold instead of trusting the
            # classifier CSV flag, which may have been produced with another threshold.
            confidence_value = record.get("confidence_value")
            effective_fallback = bool(record["fallback"]) or (
                confidence_value is not None and confidence_value < threshold
            )
            classifier_fallback = "TRUE" if effective_fallback else "FALSE"
            if effective_fallback:
                used_class, reason = "GLOBAL", "classifier_fallback_low_confidence"
            elif classifier_class not in class_models:
                used_class, reason = "GLOBAL", "classifier_class_not_in_models"
            else:
                used_class, reason = classifier_class, "classifier_prediction"
        fallback_reason_counts[reason] += 1
        confusion_counts[(gt_class, classifier_class if classifier_class else "NO_PREDICTION")] += 1

        if used_class == "GLOBAL":
            e2e_model = global_model
        else:
            e2e_model = class_models[used_class]
        pred_e2e = float(predict(np.asarray([area]), e2e_model)[0])

        prediction_rows.append({
            **base,
            "A_bb_px2": area,
            "confidence": confidence if confidence is not None else "",
            "detection_found": "TRUE",
            "pred_end_to_end_m": round(pred_e2e, 6),
            "error_end_to_end_m": round(pred_e2e - gt, 6),
            "model_used_end_to_end": GLOBAL_MODEL_ID if used_class == "GLOBAL" else f"bodyclass_{used_class}_quartic_monotone_take1",
            "selection_reason_end_to_end": reason,
            "pred_oracle_m": round(pred_oracle, 6),
            "error_oracle_m": round(pred_oracle - gt, 6),
            "model_used_oracle": f"bodyclass_{gt_class}_quartic_monotone_take1",
            "pred_global_m": round(pred_global, 6),
            "error_global_m": round(pred_global - gt, 6),
            "classifier_predicted_class": classifier_class,
            "classifier_confidence": classifier_conf,
            "classifier_fallback_flag": classifier_fallback,
            "outside_model_area_range_end_to_end": "TRUE" if model_range(e2e_model, area) else "FALSE",
            "outside_model_area_range_oracle": "TRUE" if model_range(oracle_model, area) else "FALSE",
            "outside_model_area_range_global": "TRUE" if model_range(global_model, area) else "FALSE",
            "prediction_nonpositive_end_to_end": "TRUE" if pred_e2e <= 0 else "FALSE",
            "prediction_nonpositive_oracle": "TRUE" if pred_oracle <= 0 else "FALSE",
            "prediction_nonpositive_global": "TRUE" if pred_global <= 0 else "FALSE",
        })

    scored = [row for row in prediction_rows if row["detection_found"] == "TRUE"]
    variants = {
        "end_to_end": ("pred_end_to_end_m", "error_end_to_end_m"),
        "oracle": ("pred_oracle_m", "error_oracle_m"),
        "global": ("pred_global_m", "error_global_m"),
    }
    variant_arrays = {
        name: (
            np.asarray([row[pred_field] for row in scored], dtype=float),
            np.asarray([float(row["gt_distance_m"]) for row in scored], dtype=float),
        )
        for name, (pred_field, _error_field) in variants.items()
    }

    metric_fields = ["variant", "scope_type", "scope", "N", "MAE_m", "RMSE_m", "Bias_m", "AbsRel",
                     "negative_prediction_count", "events_without_detection"]
    metric_rows: list[dict] = []

    def add_metric_rows(scope_type: str, scope: str, rows_subset: list[dict]) -> None:
        has_detection = [row for row in rows_subset if row["detection_found"] == "TRUE"]
        for name, (pred_field, _error_field) in variants.items():
            pred = np.asarray([row[pred_field] for row in has_detection], dtype=float) if has_detection else np.asarray([])
            truth = np.asarray([float(row["gt_distance_m"]) for row in has_detection], dtype=float) if has_detection else np.asarray([])
            block = metric_block(pred, truth)
            metric_rows.append({
                "variant": name,
                "scope_type": scope_type,
                "scope": scope,
                **block,
                "events_without_detection": len(rows_subset) - len(has_detection),
            })

    add_metric_rows("all", "all_events", prediction_rows)
    for video_id in sorted({row["video_id"] for row in prediction_rows}):
        add_metric_rows("video", video_id, [row for row in prediction_rows if row["video_id"] == video_id])
    for body_class in body_classes:
        add_metric_rows("body_class_gt", body_class, [row for row in prediction_rows if row["body_class_gt"] == body_class])
    for marker_id in EXPECTED_MARKER_IDS:
        subset = [row for row in prediction_rows if row["marker_id"] == marker_id]
        if subset:
            add_metric_rows("marker", marker_id, subset)

    warnings: list[str] = list(metadata_warnings)
    if any(row["detection_found"] == "FALSE" for row in prediction_rows):
        warnings.append("Some confirmed crossing events have no matched detector box; they are counted in coverage, not distance error metrics.")
    outside_end_to_end = sum(row["outside_model_area_range_end_to_end"] == "TRUE" for row in scored)
    if outside_end_to_end:
        warnings.append(f"{outside_end_to_end} end-to-end predictions use areas outside the selected model's calibration range; they are extrapolations.")
    nonpositive = sum(
        row["prediction_nonpositive_end_to_end"] == "TRUE" or row["prediction_nonpositive_oracle"] == "TRUE"
        or row["prediction_nonpositive_global"] == "TRUE"
        for row in scored
    )
    if nonpositive:
        warnings.append(f"{nonpositive} events produced at least one nonpositive distance; no clipping was applied.")
    fallback_events = fallback_reason_counts.get("classifier_fallback_low_confidence", 0) + fallback_reason_counts.get("no_classifier_prediction", 0)
    if scored and fallback_events / max(len(scored), 1) > 0.2:
        warnings.append(
            f"{fallback_events}/{len(scored)} events fell back to the global quartic; check classifier quality or lower the threshold trade-off."
        )
    if not args.bodyclass_predictions:
        warnings.append("No --bodyclass-predictions provided: every end-to-end prediction used the global quartic (no_classifier_prediction).")

    output_dir = prepare_output_dir(args.output_dir)
    write_csv(output_dir / "distance_predictions.csv", prediction_rows, prediction_fields)
    write_csv(output_dir / "metrics_by_scope.csv", metric_rows, metric_fields)
    confusion_fields = ["body_class_gt", "classifier_predicted_class", "event_count"]
    confusion_rows = [
        {"body_class_gt": truth_class, "classifier_predicted_class": predicted_class, "event_count": count}
        for (truth_class, predicted_class), count in sorted(confusion_counts.items())
    ]
    write_csv(output_dir / "classifier_confusion.csv", confusion_rows, confusion_fields)

    summary = {
        "status": "evaluated_frozen_model",
        "evaluation_type": "video2_to_4_inference_test_unseen_vehicles",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_file": workspace_path(model_path),
        "model_sha256": sha256_file(model_path),
        "calibration_take_ids": bundle.get("calibration_take_ids", []),
        "test_take_ids": test_take_ids,
        "model_refit_during_evaluation": False,
        "model_degree": DEGREE,
        "body_classes": body_classes,
        "event_count": len(prediction_rows),
        "events_with_detection": len(scored),
        "events_without_detection": len(prediction_rows) - len(scored),
        "detection_coverage": round(len(scored) / len(prediction_rows), 4) if prediction_rows else None,
        "fallback_threshold": threshold,
        "selection_reason_counts": dict(fallback_reason_counts),
        "metrics": {
            name: metric_block(*variant_arrays[name]) for name in variants
        },
        "metrics_by_scope_csv": "metrics_by_scope.csv",
        "classifier_confusion_csv": "classifier_confusion.csv",
        "body_class_label_source_at_test": "manual ground truth (oracle) + classifier (end-to-end)",
        "event_csvs": event_sources,
        "detection_csvs": detection_sources,
        "classifier_prediction_csvs": [workspace_path(path) for path in (args.bodyclass_predictions or [])],
        "marker_ground_truth_source": workspace_path(marker_gt_path),
        "tracking_metadata": tracking_metadata,
        "interpretation": (
            "Metrics cover only manually confirmed events with a matched detector box. Vehicles in the test videos "
            "are disjoint from calibration vehicles, so these numbers measure generalization to unseen cars of the "
            "same body class, not cross-take consistency of known subjects. The end-to-end minus oracle gap is the "
            "classifier's contribution to distance error."
        ),
        "limitations": [
            "The target is longitudinal distance of the marker line; lateral lane offset is not corrected.",
            "30 fps recording: crossing-frame ambiguity is roughly half a frame of travel (~0.1-0.2 m at 20-30 km/h).",
            "Detection misses are reported through event coverage and excluded from conditional distance-error metrics.",
            "Classifier body-type labels are visual estimates; within-class size variation remains a model limitation.",
        ],
        "warnings": warnings,
    }
    write_json(output_dir / "evaluation_summary.json", summary)
    print(json.dumps({"output_dir": workspace_path(output_dir), "summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
