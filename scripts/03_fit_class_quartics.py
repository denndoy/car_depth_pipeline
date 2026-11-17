"""Fit frozen monotone quartic models for the car pipeline from Take 1 events.

Output model bundle (one file, reused verbatim by 04_evaluate_frozen.py):

  1. ``global_quartic_monotone_take1``    - one degree-4 curve for all events.
  2. ``body_class_quartic_monotone_take1`` - one degree-4 curve per body class
     (MPV / Sedan / Hatchback), fitted on all calibration vehicles of that class.

Design decisions locked with the researcher:

- Cars differ between calibration and test videos (disjoint vehicles), so the
  person-research "personal quartic per subject" becomes a "class-specific
  quartic per body type". Calibration needs >= 2 (recommended 3) vehicles per
  class to avoid overfitting to a single car's dimensions.
- Body-class labels at calibration come from MANUAL ground truth
  (``body_class_gt`` in the event CSV), never from classifier predictions.
  The fine-tuned classifier is only used at inference time to select a curve.
- Every curve is non-increasing in bounding-box area, verified analytically
  with the same constrained least-squares implementation as the person paper.
- Take 1 metrics are in-sample calibration fit error, NOT test accuracy.

Generic helpers (event validation, detection linking, tracking metadata
checks, monotone quartic fitting) are imported read-only from the person
research's scripts/04_distance_evaluation; no file of that research is edited.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "04_distance_evaluation"))

from polynomial_take_pipeline import (  # noqa: E402
    DEGREE,
    EXPECTED_MARKER_IDS,
    link_event_detections,
    load_tracking_metadata,
    read_confirmed_events,
    read_detection_files,
    read_marker_ground_truth,
    resolve_path,
    sha256_file,
    validate_tracking_metadata,
    workspace_path,
)
from evaluate_final_monotone_quartics import fit_monotone_quartic, predict  # noqa: E402

DEFAULT_CONFIG = ROOT / "car_depth_pipeline" / "configs" / "bodytype_config.json"
DEFAULT_MARKER_GT = ROOT / "car_depth_pipeline" / "inputs" / "marker_ground_truth.csv"


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    classes = [str(name) for name in config.get("body_classes", [])]
    if len(classes) < 2:
        raise ValueError("Config must define at least two body classes")
    return config


def canonical_classes(events_csv_paths: list[Path], body_classes: list[str]) -> dict[str, str]:
    """Read body_class_gt per event_id from the raw CSVs (read-only enrichment)."""
    by_casefold = {name.casefold(): name for name in body_classes}
    result: dict[str, str] = {}
    for path in events_csv_paths:
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            fields = set(reader.fieldnames or [])
            if "event_id" not in fields:
                raise ValueError(
                    f"{path}: this pipeline requires an explicit unique 'event_id' column "
                    "so body_class_gt can be joined unambiguously"
                )
            if "body_class_gt" not in fields:
                raise ValueError(f"{path}: missing required column 'body_class_gt'")
            for line_number, row in enumerate(reader, start=2):
                event_id = (row.get("event_id") or "").strip()
                raw_class = (row.get("body_class_gt") or "").strip()
                if not event_id:
                    raise ValueError(f"{path}:{line_number}: empty event_id")
                if event_id in result:
                    raise ValueError(f"{path}:{line_number}: duplicate event_id {event_id!r}")
                canonical = by_casefold.get(raw_class.casefold())
                if canonical is None:
                    raise ValueError(
                        f"{path}:{line_number}: unknown body_class_gt {raw_class!r}; "
                        f"allowed: {body_classes}"
                    )
                result[event_id] = canonical
    return result


def metric_block(predictions: np.ndarray, truth: np.ndarray) -> dict:
    if predictions.size == 0:
        return {"N": 0, "MAE_m": None, "RMSE_m": None, "Bias_m": None, "AbsRel": None,
                "negative_prediction_count": 0}
    error = predictions - truth
    return {
        "N": int(predictions.size),
        "MAE_m": round(float(np.mean(np.abs(error))), 4),
        "RMSE_m": round(float(np.sqrt(np.mean(error**2))), 4),
        "Bias_m": round(float(np.mean(error)), 4),
        "AbsRel": round(float(np.mean(np.abs(error) / truth)), 4),
        "negative_prediction_count": int(np.sum(predictions < 0)),
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
    parser.add_argument("--events", type=Path, nargs="+", required=True, help="Confirmed Take 1 event CSV(s) with body_class_gt")
    parser.add_argument("--detections", type=Path, nargs="+", required=True, help="Mapped/raw detection CSV(s) of the calibration take")
    parser.add_argument("--marker-ground-truth", type=Path, default=DEFAULT_MARKER_GT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-take", default="Take1", help="Expected calibration take_id; pass '' to disable")
    parser.add_argument("--allow-single-vehicle-per-class", action="store_true", help="Override the minimum-2-vehicles guard (NOT recommended)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory (never overwritten)")
    args = parser.parse_args(argv)

    config = load_config(resolve_path(args.config))
    body_classes = [str(name) for name in config["body_classes"]]
    recommended = int(config.get("recommended_calibration_vehicles_per_class", 3))
    minimum = int(config.get("minimum_calibration_vehicles_per_class", 2))

    marker_gt_path = resolve_path(args.marker_ground_truth)
    marker_gt = read_marker_ground_truth(marker_gt_path)
    if set(marker_gt) != set(EXPECTED_MARKER_IDS):
        raise ValueError(
            f"Marker table must contain exactly M01-M09; found {sorted(marker_gt)}. "
            "Build your own marker_ground_truth.csv from the template; do not mix person-research GT."
        )

    event_paths = [resolve_path(path) for path in args.events]
    detection_paths = [resolve_path(path) for path in args.detections]
    events, event_sources = read_confirmed_events(event_paths, marker_gt)
    take_ids = sorted({event["take_id"] for event in events})
    if len(take_ids) != 1:
        raise ValueError(f"Calibration must use only one take; found {take_ids}")
    expected_take = args.expected_take or None
    if expected_take and take_ids[0].casefold() != expected_take.casefold():
        raise ValueError(f"Expected calibration take {expected_take!r}, found {take_ids[0]!r}")

    body_class_by_event = canonical_classes(event_paths, body_classes)
    for event in events:
        event_id = event["event_id"]
        event["body_class_gt"] = body_class_by_event[event_id]

    detections_by_frame, detection_sources = read_detection_files(detection_paths)
    linked = link_event_detections(events, detections_by_frame, require_detection=True)
    tracking_metadata, metadata_warnings = load_tracking_metadata(detection_paths)
    metadata_warnings.extend(validate_tracking_metadata(tracking_metadata))

    areas = np.asarray([float(row["A_bb_px2"]) for row in linked], dtype=float)
    distances = np.asarray([float(row["gt_distance_m"]) for row in linked], dtype=float)
    warnings: list[str] = list(metadata_warnings)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in linked:
        by_class[row["body_class_gt"]].append(row)
    missing_classes = [name for name in body_classes if not by_class.get(name)]
    if missing_classes:
        raise ValueError(f"Calibration events are missing body classes: {missing_classes}")

    vehicles_per_class = {
        name: sorted({row["subject_id"] for row in by_class[name]}) for name in body_classes
    }
    for name in body_classes:
        count = len(vehicles_per_class[name])
        if count < minimum and not args.allow_single_vehicle_per_class:
            raise ValueError(
                f"Body class {name} has only {count} calibration vehicle(s); at least {minimum} are required so "
                "the class curve generalizes to unseen cars. Use --allow-single-vehicle-per-class only knowingly."
            )
        if count < recommended:
            warnings.append(
                f"Body class {name} has {count} calibration vehicle(s); {recommended} recommended to span "
                "within-class size variation."
            )
        class_areas = np.asarray([float(row["A_bb_px2"]) for row in by_class[name]], dtype=float)
        class_markers = {row["marker_id"] for row in by_class[name]}
        missing_markers = sorted(set(EXPECTED_MARKER_IDS) - class_markers)
        if missing_markers:
            raise ValueError(
                f"Body class {name} calibration must cover all markers M01-M09; missing {missing_markers}"
            )
        if len(np.unique(class_areas)) < DEGREE + 1:
            raise ValueError(
                f"Body class {name} needs at least {DEGREE + 1} distinct bbox areas for a degree-{DEGREE} fit"
            )

    global_model = fit_monotone_quartic(areas, distances)
    class_models: dict[str, dict] = {}
    for name in body_classes:
        class_rows = by_class[name]
        class_models[name] = fit_monotone_quartic(
            np.asarray([float(row["A_bb_px2"]) for row in class_rows], dtype=float),
            np.asarray([float(row["gt_distance_m"]) for row in class_rows], dtype=float),
        )

    for model_id, model in [("global_quartic_monotone_take1", global_model)] + [
        (f"bodyclass_{name}_quartic_monotone_take1", class_models[name]) for name in body_classes
    ]:
        if not model["monotonicity"]["verified_analytically"]:
            warnings.append(f"{model_id}: monotonicity verification did not converge as expected")
        if model["training_prediction_min_m"] < 0:
            warnings.append(f"{model_id}: raw curve produces negative distances inside the calibration area range; no clipping was applied")
        if model["max_positive_derivative_m_per_px2"] > 0:
            warnings.append(f"{model_id}: positive derivative detected; constraint violation")

    prediction_fields = [
        "event_id", "take_id", "video_id", "subject_id", "marker_id", "frame_idx",
        "body_class_gt", "gt_distance_m", "A_bb_px2", "w_px", "h_px", "confidence",
        "pred_global_m", "error_global_m", "pred_bodyclass_m", "error_bodyclass_m",
        "bodyclass_model_used",
    ]
    prediction_rows = []
    for row in linked:
        area = float(row["A_bb_px2"])
        gt = float(row["gt_distance_m"])
        body_class = row["body_class_gt"]
        pred_global = float(predict(np.asarray([area]), global_model)[0])
        pred_class = float(predict(np.asarray([area]), class_models[body_class])[0])
        prediction_rows.append({
            "event_id": row["event_id"],
            "take_id": row["take_id"],
            "video_id": row["video_id"],
            "subject_id": row["subject_id"],
            "marker_id": row["marker_id"],
            "frame_idx": row["frame_idx"],
            "body_class_gt": body_class,
            "gt_distance_m": gt,
            "A_bb_px2": area,
            "w_px": row.get("w_px", ""),
            "h_px": row.get("h_px", ""),
            "confidence": row.get("confidence", ""),
            "pred_global_m": round(pred_global, 6),
            "error_global_m": round(pred_global - gt, 6),
            "pred_bodyclass_m": round(pred_class, 6),
            "error_bodyclass_m": round(pred_class - gt, 6),
            "bodyclass_model_used": f"bodyclass_{body_class}_quartic_monotone_take1",
        })

    global_pred = np.asarray([row["pred_global_m"] for row in prediction_rows], dtype=float)
    class_pred = np.asarray([row["pred_bodyclass_m"] for row in prediction_rows], dtype=float)
    truth = np.asarray([row["gt_distance_m"] for row in prediction_rows], dtype=float)
    class_array = np.asarray([row["body_class_gt"] for row in prediction_rows])
    marker_array = np.asarray([row["marker_id"] for row in prediction_rows])

    metrics_by_class = {}
    for name in body_classes:
        mask = class_array == name
        metrics_by_class[name] = {
            "N": int(mask.sum()),
            "global_quartic": metric_block(global_pred[mask], truth[mask]),
            "body_class_quartic": metric_block(class_pred[mask], truth[mask]),
        }
    metrics_by_marker = {}
    for marker_id in EXPECTED_MARKER_IDS:
        mask = marker_array == marker_id
        if mask.any():
            metrics_by_marker[marker_id] = {
                "N": int(mask.sum()),
                "global_quartic": metric_block(global_pred[mask], truth[mask]),
                "body_class_quartic": metric_block(class_pred[mask], truth[mask]),
            }

    output_dir = prepare_output_dir(args.output_dir)
    model_bundle = {
        "schema": "car_bodyclass_and_global_monotone_quartic_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature": "A_bb_px2 (bounding-box area, source px squared)",
        "target": "gt_distance_m (longitudinal marker ground truth)",
        "degree": DEGREE,
        "body_classes": body_classes,
        "calibration_take_ids": take_ids,
        "calibration_event_count": len(linked),
        "events_per_body_class": {name: len(by_class[name]) for name in body_classes},
        "vehicles_per_body_class": {name: len(vehicles_per_class[name]) for name in body_classes},
        "vehicles_by_body_class": vehicles_per_class,
        "constraint": "distance estimate is non-increasing as bbox area increases",
        "constraint_domain": "observed Take 1 area range of each fitted model",
        "output_bounds": "none; no positivity constraint and no clipping",
        "inference_protocol": "freeze Take 1 coefficients; infer Videos 2-4 without refit",
        "body_class_label_source_at_calibration": "manual ground truth in event CSV (never classifier output)",
        "fallback_policy": {
            "enabled": True,
            "threshold": float(config.get("fallback_threshold", 0.6)),
            "behavior": config.get(
                "fallback_behavior", "use global quartic when classifier confidence is low or a prediction is missing"
            ),
        },
        "marker_ground_truth_m": marker_gt,
        "marker_ground_truth_source": workspace_path(marker_gt_path),
        "training_event_csvs": [
            {"path": workspace_path(path), "sha256": sha256_file(path)} for path in event_paths
        ],
        "training_detection_csvs": [
            {"path": workspace_path(path), "sha256": sha256_file(path)} for path in detection_paths
        ],
        "tracking_metadata": tracking_metadata,
        "models": {
            "global_quartic_monotone_take1": global_model,
            "body_class_quartic_monotone_take1": class_models,
        },
    }
    write_json(output_dir / "fitted_models.json", model_bundle)
    write_csv(output_dir / "calibration_predictions.csv", prediction_rows, prediction_fields)

    diagnostics_fields = [
        "model_id", "degree", "training_event_count", "training_area_min_px2", "training_area_max_px2",
        "training_distance_min_m", "training_distance_max_m", "training_mae_m", "training_rmse_m",
        "training_prediction_min_m", "training_prediction_max_m", "monotone_verified_analytically",
        "max_positive_derivative_m_per_px2", "verification_tolerance_m_per_normalized_area",
    ]
    diagnostics_rows = []
    for model_id, model in [("global_quartic_monotone_take1", global_model)] + [
        (f"bodyclass_{name}_quartic_monotone_take1", class_models[name]) for name in body_classes
    ]:
        diagnostics_rows.append({
            "model_id": model_id,
            "degree": model["degree"],
            "training_event_count": model["training_event_count"],
            "training_area_min_px2": model["training_area_min_px2"],
            "training_area_max_px2": model["training_area_max_px2"],
            "training_distance_min_m": model["training_distance_min_m"],
            "training_distance_max_m": model["training_distance_max_m"],
            "training_mae_m": model["training_mae_m"],
            "training_rmse_m": model["training_rmse_m"],
            "training_prediction_min_m": model["training_prediction_min_m"],
            "training_prediction_max_m": model["training_prediction_max_m"],
            "monotone_verified_analytically": model["monotonicity"]["verified_analytically"],
            "max_positive_derivative_m_per_px2": model["max_positive_derivative_m_per_px2"],
            "verification_tolerance_m_per_normalized_area": model["monotonicity"]["verification_tolerance_m_per_normalized_area"],
        })
    write_csv(output_dir / "monotonicity_diagnostics.csv", diagnostics_rows, diagnostics_fields)

    summary = {
        "status": "calibrated",
        "evaluation_type": "in_sample_calibration_fit",
        "take_id": take_ids[0],
        "model_file": "fitted_models.json",
        "event_count": len(linked),
        "events_per_body_class": {name: len(by_class[name]) for name in body_classes},
        "vehicles_per_body_class": {name: len(vehicles_per_class[name]) for name in body_classes},
        "metrics": {
            "global_quartic": metric_block(global_pred, truth),
            "body_class_quartic_all_events": metric_block(class_pred, truth),
        },
        "metrics_by_body_class": metrics_by_class,
        "metrics_by_marker": metrics_by_marker,
        "event_csvs": event_sources,
        "detection_csvs": detection_sources,
        "marker_ground_truth_source": workspace_path(marker_gt_path),
        "body_class_label_source": "manual ground truth; classifier is inference-only",
        "interpretation": (
            "These MAE/RMSE values are in-sample fit residuals on Take 1, the same events used to estimate "
            "coefficients. Use frozen-model Video 2-4 results from 04_evaluate_frozen.py as inference metrics."
        ),
        "warnings": warnings,
    }
    write_json(output_dir / "calibration_summary.json", summary)
    print(json.dumps({"output_dir": workspace_path(output_dir), "summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
