"""Fit the Take 1 quartic area-to-distance calibration and score frozen Take 2-4 runs.

The ground-truth target is the longitudinal distance assigned to each marker
(M01=10 m, ..., M09=90 m), independent of lane offset. Take 1 metrics are
in-sample calibration-fit metrics. Test evaluation never refits the model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKER_GT = ROOT / "inputs" / "marker_ground_truth.csv"
DEGREE = 4
EXPECTED_MARKER_IDS = tuple(f"M{index:02d}" for index in range(1, 10))
GROUND_TRUTH_DEFINITION = (
    "Longitudinal distance forward along the marker line; lateral lane offset is not corrected."
)
EVENT_REQUIRED_COLUMNS = (
    "take_id",
    "video_id",
    "subject_id",
    "marker_id",
    "frame_idx",
    "foot_crossing_confirmed",
    "annotation_source",
)
PREDICTION_FIELDS = (
    "event_id",
    "take_id",
    "video_id",
    "pass_id",
    "subject_id",
    "group_id",
    "activity",
    "lane",
    "marker_id",
    "frame_idx",
    "timestamp_s",
    "track_id",
    "gt_distance_m",
    "detection_found",
    "w_px",
    "h_px",
    "A_bb_px2",
    "confidence",
    "pred_distance_m",
    "error_m",
    "outside_calibration_area_range",
    "prediction_nonpositive",
    "metric_scope",
)
GROUP_DIMENSIONS = {
    "by_take": ("take_id",),
    "by_subject": ("subject_id",),
    "by_group": ("group_id",),
    "by_activity": ("activity",),
    "by_lane": ("lane",),
    "by_marker": ("marker_id",),
    "by_activity_and_lane": ("activity", "lane"),
    "by_take_activity_lane": ("take_id", "activity", "lane"),
}


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if (Path.cwd().resolve() == ROOT.parent.resolve()
            and path.parts and path.parts[0].casefold() == ROOT.name.casefold()):
        return (Path.cwd() / path).resolve()
    return (ROOT / path).resolve()


def workspace_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_dict_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader), fields


def read_marker_ground_truth(path: Path) -> dict[str, float]:
    rows, fields = read_dict_rows(path)
    required = {"marker_id", "gt_distance_m"}
    if not required.issubset(fields):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    marker_gt: dict[str, float] = {}
    for line_number, row in enumerate(rows, start=2):
        marker_id = (row.get("marker_id") or "").strip()
        if not marker_id:
            raise ValueError(f"Empty marker_id at {path}:{line_number}")
        if marker_id in marker_gt:
            raise ValueError(f"Duplicate marker_id {marker_id} in {path}")
        distance = float(row["gt_distance_m"])
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError(f"Invalid ground-truth distance for {marker_id}: {distance}")
        marker_gt[marker_id] = distance
    if not marker_gt:
        raise ValueError(f"No marker ground truth found in {path}")
    return marker_gt


def parse_frame_idx(value: str, context: str) -> int:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid frame_idx {value!r} in {context}") from None
    if result < 0:
        raise ValueError(f"Negative frame_idx in {context}: {result}")
    return result


def read_confirmed_events(
    paths: list[Path], marker_gt: dict[str, float]
) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    event_sources: list[str] = []
    seen_event_ids: set[str] = set()

    for path in paths:
        rows, fields = read_dict_rows(path)
        missing = set(EVENT_REQUIRED_COLUMNS) - set(fields)
        if missing:
            raise ValueError(f"{path} is missing event columns: {sorted(missing)}")
        if not rows:
            raise ValueError(f"No event rows in {path}")

        for line_number, row in enumerate(rows, start=2):
            context = f"{path}:{line_number}"
            confirmed = (row.get("foot_crossing_confirmed") or "").strip().upper()
            if confirmed != "TRUE":
                raise ValueError(
                    f"Only manually confirmed TRUE crossings are accepted ({context}; "
                    f"found {confirmed or 'blank'}). Keep unreviewed/false rows out of the confirmed-event CSV."
                )
            source = (row.get("annotation_source") or "").strip().lower()
            if (
                not source
                or not any(token in source for token in ("manual", "review", "user", "human"))
                or any(token in source for token in ("auto", "candidate", "prelabel", "unverified"))
            ):
                raise ValueError(
                    f"Event annotation_source must indicate reviewed/manual data and not an auto-candidate ({context})."
                )

            take_id = (row.get("take_id") or "").strip()
            video_id = (row.get("video_id") or "").strip()
            subject_id = (row.get("subject_id") or "").strip()
            marker_id = (row.get("marker_id") or "").strip()
            if not all((take_id, video_id, subject_id, marker_id)):
                raise ValueError(f"take_id, video_id, subject_id, and marker_id are required ({context})")
            if marker_id not in marker_gt:
                raise ValueError(f"Unknown marker {marker_id} in {context}; check {DEFAULT_MARKER_GT.name}")

            gt = marker_gt[marker_id]
            supplied_gt = (row.get("gt_distance_m") or "").strip()
            if supplied_gt:
                if abs(float(supplied_gt) - gt) > 1e-6:
                    raise ValueError(
                        f"GT mismatch for {marker_id} in {context}: event={supplied_gt}, marker table={gt}. "
                        "This experiment assigns longitudinal GT by marker, without lane-offset correction."
                    )

            frame_idx = parse_frame_idx(row.get("frame_idx", ""), context)
            pass_id = (row.get("pass_id") or "").strip()
            supplied_event_id = (row.get("event_id") or "").strip()
            event_id = supplied_event_id or ":".join(
                (take_id, video_id, pass_id or "pass-unspecified", subject_id, marker_id, str(frame_idx))
            )
            if event_id in seen_event_ids:
                raise ValueError(f"Duplicate event_id {event_id!r}; make event_id unique across input CSVs")
            seen_event_ids.add(event_id)

            timestamp = (row.get("timestamp_s") or "").strip()
            if timestamp:
                float(timestamp)  # validate, but preserve source formatting below
            event = {
                "event_id": event_id,
                "take_id": take_id,
                "video_id": video_id,
                "pass_id": pass_id,
                "subject_id": subject_id,
                "group_id": (row.get("group_id") or "").strip(),
                "activity": (row.get("activity") or "").strip(),
                "lane": (row.get("lane") or "").strip(),
                "marker_id": marker_id,
                "frame_idx": frame_idx,
                "timestamp_s": timestamp,
                "track_id": (row.get("track_id") or "").strip(),
                "gt_distance_m": gt,
                "annotation_source": source,
                "notes": (row.get("notes") or "").strip(),
                "_source_csv": workspace_path(path),
            }
            events.append(event)
        event_sources.append(workspace_path(path))

    if not events:
        raise ValueError("No confirmed events were loaded")
    return events, event_sources


def normalize_track_id(value: str | None) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return value


def read_detection_files(paths: list[Path]) -> tuple[dict[tuple[str, int], list[dict[str, str]]], list[str]]:
    by_frame: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    source_paths: list[str] = []
    for path in paths:
        rows, fields = read_dict_rows(path)
        required = {"video_id", "frame_idx"}
        if not required.issubset(fields):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for line_number, row in enumerate(rows, start=2):
            video_id = (row.get("video_id") or "").strip()
            if not video_id:
                raise ValueError(f"Empty video_id at {path}:{line_number}")
            frame_idx = parse_frame_idx(row.get("frame_idx", ""), f"{path}:{line_number}")
            row["_normalized_track_id"] = normalize_track_id(row.get("track_id"))
            row["_source_csv"] = workspace_path(path)
            by_frame[(video_id, frame_idx)].append(row)
        source_paths.append(workspace_path(path))
    if not by_frame:
        raise ValueError("No detections were loaded")
    return by_frame, source_paths


def detection_area(detection: dict[str, str], context: str) -> tuple[float, float, float]:
    try:
        width = float(detection.get("w_px", ""))
        height = float(detection.get("h_px", ""))
    except (TypeError, ValueError):
        width = height = math.nan
    area_value = (detection.get("A_bb_px2") or "").strip()
    if area_value:
        area = float(area_value)
    elif math.isfinite(width) and math.isfinite(height):
        area = width * height
    else:
        coords = [detection.get(key, "") for key in ("x1_px", "y1_px", "x2_px", "y2_px")]
        if not all(str(value).strip() for value in coords):
            raise ValueError(f"Detection lacks A_bb_px2 or box dimensions ({context})")
        x1, y1, x2, y2 = map(float, coords)
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        area = width * height
    if not math.isfinite(area) or area <= 0:
        raise ValueError(f"Nonpositive/nonfinite bounding-box area in {context}: {area}")
    if not math.isfinite(width) or width <= 0 or not math.isfinite(height) or height <= 0:
        if area_value:
            width = height = math.nan
        else:
            raise ValueError(f"Invalid box dimensions in {context}")
    elif area_value:
        computed_area = width * height
        tolerance = max(1.0, computed_area * 1e-4)
        if abs(area - computed_area) > tolerance:
            raise ValueError(
                f"A_bb_px2 is inconsistent with w_px*h_px in {context}: "
                f"A_bb_px2={area}, w*h={computed_area} (tolerance {tolerance})."
            )
    return width, height, area


def link_event_detections(
    events: list[dict[str, Any]],
    detections_by_frame: dict[tuple[str, int], list[dict[str, str]]],
    *,
    require_detection: bool,
) -> list[dict[str, Any]]:
    linked: list[dict[str, Any]] = []
    for event in events:
        context = f"event {event['event_id']} at {event['video_id']} frame {event['frame_idx']}"
        candidates = detections_by_frame.get((event["video_id"], event["frame_idx"]), [])
        event_track = normalize_track_id(event.get("track_id"))
        matches = []
        for detection in candidates:
            detection_track = detection.get("_normalized_track_id", "")
            detection_subject = (detection.get("subject_id") or "").strip()
            detection_pass = (detection.get("pass_id") or "").strip()
            if event_track and detection_track != event_track:
                continue
            if detection_subject and detection_subject != event["subject_id"]:
                continue
            if not event_track and not detection_subject:
                continue
            if event["pass_id"] and detection_pass and event["pass_id"] != detection_pass:
                continue
            matches.append(detection)

        if len(matches) > 1:
            raise ValueError(f"Expected one mapped detection for {context}; found {len(matches)}")
        if not matches:
            if require_detection:
                raise ValueError(
                    f"No matching detection for {context}. Check frame index, subject crosswalk, and event track_id."
                )
            linked.append({**event, "_detection": None})
            continue

        detection = matches[0]
        width, height, area = detection_area(detection, context)
        try:
            confidence_value = (detection.get("confidence") or "").strip()
            confidence = float(confidence_value) if confidence_value else None
            if confidence is not None and not math.isfinite(confidence):
                confidence = None
        except ValueError:
            confidence = None
        linked.append(
            {
                **event,
                "track_id": detection.get("track_id", "") or event.get("track_id", ""),
                "w_px": width,
                "h_px": height,
                "A_bb_px2": area,
                "confidence": confidence,
                "_detection": detection,
            }
        )
    return linked


def load_tracking_metadata(detection_paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    metadata: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_summaries: set[Path] = set()
    for path in detection_paths:
        summary_path = path.parent / "summary.json"
        if not summary_path.is_file():
            warnings.append(f"No tracking summary.json next to {workspace_path(path)}; detector settings cannot be verified.")
            continue
        summary_path = summary_path.resolve()
        if summary_path in seen_summaries:
            continue
        seen_summaries.add(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = {
            key: summary.get(key)
            for key in ("model", "weights_sha256", "imgsz", "conf", "tracker", "device")
            if summary.get(key) is not None
        }
        videos = {}
        for video in summary.get("videos", []):
            if video.get("video_id"):
                videos[video["video_id"]] = {
                    key: video.get(key)
                    for key in ("source_width", "source_height", "source_fps")
                    if video.get(key) is not None
                }
        metadata.append(
            {
                "summary_path": workspace_path(summary_path),
                "config": config,
                "videos": videos,
            }
        )
    return metadata, warnings


def validate_tracking_metadata(
    calibration_metadata: list[dict[str, Any]], test_metadata: list[dict[str, Any]] | None = None
) -> list[str]:
    warnings: list[str] = []
    all_sets = [("calibration", calibration_metadata)]
    if test_metadata is not None:
        all_sets.append(("test", test_metadata))
    comparable_keys = ("weights_sha256", "imgsz", "conf", "tracker", "device")
    reference: dict[str, Any] = {}
    for label, entries in all_sets:
        for entry in entries:
            config = entry.get("config", {})
            for key in comparable_keys:
                value = config.get(key)
                if value is None:
                    continue
                if key in reference and value != reference[key]:
                    raise ValueError(
                        f"Detector/tracker configuration mismatch for {key}: {reference[key]!r} vs {value!r} "
                        f"in {entry['summary_path']}. Use the same detector settings for calibration and inference."
                    )
                reference[key] = value
        if not entries:
            warnings.append(f"No tracking metadata available for {label}; verify detector settings manually.")

    if test_metadata is not None:
        calibration_resolutions = {
            (video.get("source_width"), video.get("source_height"))
            for entry in calibration_metadata
            for video in entry.get("videos", {}).values()
            if video.get("source_width") is not None and video.get("source_height") is not None
        }
        test_resolutions = {
            (video.get("source_width"), video.get("source_height"))
            for entry in test_metadata
            for video in entry.get("videos", {}).values()
            if video.get("source_width") is not None and video.get("source_height") is not None
        }
        if calibration_resolutions and test_resolutions and calibration_resolutions != test_resolutions:
            raise ValueError(
                f"Source video resolution mismatch: calibration={sorted(calibration_resolutions)}, "
                f"test={sorted(test_resolutions)}. Bounding-box pixel areas are not directly comparable."
            )
        if not calibration_resolutions or not test_resolutions:
            warnings.append("Source video resolution could not be verified from tracking summary metadata.")
    return warnings


def fit_quartic(area_px2: np.ndarray, distance_m: np.ndarray) -> dict[str, Any]:
    if len(area_px2) != len(distance_m) or len(area_px2) < DEGREE + 1:
        raise ValueError(f"A degree-{DEGREE} polynomial requires at least {DEGREE + 1} calibration events")
    if not np.all(np.isfinite(area_px2)) or not np.all(np.isfinite(distance_m)):
        raise ValueError("Calibration features/targets must all be finite")
    if len(np.unique(area_px2)) < DEGREE + 1:
        raise ValueError(f"At least {DEGREE + 1} distinct bounding-box areas are required to fit degree {DEGREE}")
    if len(np.unique(distance_m)) < DEGREE + 1:
        raise ValueError(
            f"Calibration must cover at least {DEGREE + 1} distinct marker distances for a degree-{DEGREE} model"
        )

    mean = float(np.mean(area_px2))
    scale = float(np.std(area_px2))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Calibration bounding-box areas have zero/invalid variance")
    z = (area_px2 - mean) / scale
    design = np.polynomial.polynomial.polyvander(z, DEGREE)
    coefficients, _, rank, _ = np.linalg.lstsq(design, distance_m, rcond=None)
    if rank < DEGREE + 1:
        raise ValueError(f"Quartic design matrix is rank-deficient ({rank}/{DEGREE + 1}); cannot fit a unique curve")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Quartic fit produced nonfinite coefficients")

    raw_coefficients = np.zeros(DEGREE + 1, dtype=float)
    for normalized_power, coefficient in enumerate(coefficients):
        for raw_power in range(normalized_power + 1):
            raw_coefficients[raw_power] += (
                coefficient
                * math.comb(normalized_power, raw_power)
                * (-mean) ** (normalized_power - raw_power)
                / (scale**normalized_power)
            )

    area_min, area_max = float(np.min(area_px2)), float(np.max(area_px2))
    grid_area = np.linspace(area_min, area_max, 501)
    grid_z = (grid_area - mean) / scale
    grid_distance = np.polynomial.polynomial.polyval(grid_z, coefficients)
    derivative_coefficients = np.polynomial.polynomial.polyder(coefficients)
    derivative_per_px2 = np.polynomial.polynomial.polyval(grid_z, derivative_coefficients) / scale
    monotonic_nonincreasing = bool(np.all(derivative_per_px2 <= 1e-12))
    positive_in_calibration_range = bool(np.all(grid_distance > 0))

    return {
        "degree": DEGREE,
        "feature_name": "A_bb_px2",
        "target_name": "gt_distance_m",
        "normalization_mean_px2": mean,
        "normalization_scale_px2": scale,
        "coefficients_ascending": [float(value) for value in coefficients],
        "normalized_feature_formula": "z = (A_bb_px2 - normalization_mean_px2) / normalization_scale_px2",
        "normalized_prediction_formula": "D_hat = beta0 + beta1*z + beta2*z^2 + beta3*z^3 + beta4*z^4",
        "coefficients_raw_area_ascending": [float(value) for value in raw_coefficients],
        "raw_area_coefficient_order": ["beta0", "beta1", "beta2", "beta3", "beta4"],
        "raw_area_coefficients_note": "Algebraically equivalent to the normalized polynomial; inference uses the normalized coefficients for numerical stability.",
        "calibration_area_min_px2": area_min,
        "calibration_area_max_px2": area_max,
        "calibration_distance_min_m": float(np.min(distance_m)),
        "calibration_distance_max_m": float(np.max(distance_m)),
        "design_matrix_rank": int(rank),
        "design_matrix_condition_number": float(np.linalg.cond(design)),
        "monotonic_nonincreasing_in_calibration_range": monotonic_nonincreasing,
        "positive_predictions_in_calibration_range": positive_in_calibration_range,
        "minimum_prediction_in_calibration_range_m": float(np.min(grid_distance)),
        "maximum_prediction_in_calibration_range_m": float(np.max(grid_distance)),
        "maximum_derivative_m_per_px2": float(np.max(derivative_per_px2)),
        "minimum_derivative_m_per_px2": float(np.min(derivative_per_px2)),
    }


def predict_distance(area_px2: float, model: dict[str, Any]) -> float:
    mean = float(model["normalization_mean_px2"])
    scale = float(model["normalization_scale_px2"])
    coefficients = np.asarray(model["coefficients_ascending"], dtype=float)
    normalized_area = (area_px2 - mean) / scale
    prediction = float(np.polynomial.polynomial.polyval(normalized_area, coefficients))
    if not math.isfinite(prediction):
        raise ValueError(f"Nonfinite distance prediction for area {area_px2}")
    return prediction


def finite_or_empty(value: Any) -> float | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def prediction_rows(
    linked_events: list[dict[str, Any]], model: dict[str, Any], *, fit_scope: str
) -> list[dict[str, Any]]:
    minimum = float(model["calibration_area_min_px2"])
    maximum = float(model["calibration_area_max_px2"])
    output: list[dict[str, Any]] = []
    for row in linked_events:
        detection = row.get("_detection")
        base = {
            key: row.get(key, "")
            for key in (
                "event_id", "take_id", "video_id", "pass_id", "subject_id", "group_id",
                "activity", "lane", "marker_id", "frame_idx", "timestamp_s", "track_id", "gt_distance_m",
            )
        }
        if detection is None:
            output.append(
                {
                    **base,
                    "detection_found": "FALSE",
                    "w_px": "",
                    "h_px": "",
                    "A_bb_px2": "",
                    "confidence": "",
                    "pred_distance_m": "",
                    "error_m": "",
                    "outside_calibration_area_range": "",
                    "prediction_nonpositive": "",
                    "metric_scope": fit_scope,
                }
            )
            continue

        area = float(row["A_bb_px2"])
        predicted = predict_distance(area, model)
        confidence = row.get("confidence")
        output.append(
            {
                **base,
                "detection_found": "TRUE",
                "w_px": finite_or_empty(row.get("w_px", "")),
                "h_px": finite_or_empty(row.get("h_px", "")),
                "A_bb_px2": area,
                "confidence": confidence if confidence is not None else "",
                "pred_distance_m": predicted,
                "error_m": predicted - float(row["gt_distance_m"]),
                "outside_calibration_area_range": "TRUE" if area < minimum or area > maximum else "FALSE",
                "prediction_nonpositive": "TRUE" if predicted <= 0 else "FALSE",
                "metric_scope": fit_scope,
            }
        )
    return output


def metric_values(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("detection_found") == "TRUE" and row.get("pred_distance_m") != ""]
    if not scored:
        return {"N": 0, "MAE_m": None, "RMSE_m": None, "Mean_Bias_m": None, "AbsRel": None}
    errors = np.asarray([float(row["pred_distance_m"]) - float(row["gt_distance_m"]) for row in scored])
    truth = np.asarray([float(row["gt_distance_m"]) for row in scored])
    return {
        "N": int(len(scored)),
        "MAE_m": round(float(np.mean(np.abs(errors))), 4),
        "RMSE_m": round(float(np.sqrt(np.mean(errors**2))), 4),
        "Mean_Bias_m": round(float(np.mean(errors)), 4),
        "AbsRel": round(float(np.mean(np.abs(errors) / truth)), 4),
    }


def summarize_group(rows: list[dict[str, Any]], dimensions: tuple[str, ...]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(dimension) or "not_recorded") for dimension in dimensions)
        groups[key].append(row)
    result: dict[str, Any] = {}
    for key, group_rows in sorted(groups.items()):
        key_name = " | ".join(key)
        found = sum(row.get("detection_found") == "TRUE" for row in group_rows)
        result[key_name] = {
            **{dimension: value for dimension, value in zip(dimensions, key)},
            "events": len(group_rows),
            "events_with_detection": found,
            "detection_coverage": round(found / len(group_rows), 4) if group_rows else None,
            "metrics_on_detected_events": metric_values(group_rows),
        }
    return result


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    found = sum(row.get("detection_found") == "TRUE" for row in rows)
    return {
        "event_count": len(rows),
        "events_with_detection": found,
        "events_without_detection": len(rows) - found,
        "detection_coverage": round(found / len(rows), 4) if rows else None,
        "metrics_on_detected_events": metric_values(rows),
        "grouped_metrics": {
            name: summarize_group(rows, dimensions)
            for name, dimensions in GROUP_DIMENSIONS.items()
        },
        "outside_calibration_area_range_count": sum(
            row.get("outside_calibration_area_range") == "TRUE" for row in rows
        ),
        "nonpositive_prediction_count": sum(row.get("prediction_nonpositive") == "TRUE" for row in rows),
    }


def prepare_output_dir(path: Path) -> Path:
    path = resolve_path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True)
    return path


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] = PREDICTION_FIELDS) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def model_for_prediction_file(model: dict[str, Any]) -> dict[str, Any]:
    return model["polynomial"]


def command_calibrate(args: argparse.Namespace) -> int:
    event_paths = [resolve_path(path) for path in args.events]
    detection_paths = [resolve_path(path) for path in args.detections]
    marker_gt_path = resolve_path(args.marker_ground_truth)
    marker_gt = read_marker_ground_truth(marker_gt_path)
    if set(marker_gt) != set(EXPECTED_MARKER_IDS):
        raise ValueError(
            "This calibration protocol requires the marker table to contain exactly M01-M09; "
            f"found {sorted(marker_gt)}"
        )
    events, event_sources = read_confirmed_events(event_paths, marker_gt)
    take_ids = sorted({event["take_id"] for event in events})
    if len(take_ids) != 1:
        raise ValueError(f"Calibration must use only Take 1 events; found take_id values {take_ids}")
    if args.expected_take and take_ids[0].casefold() != args.expected_take.casefold():
        raise ValueError(f"Expected calibration take {args.expected_take!r}, found {take_ids[0]!r}")

    detections_by_frame, detection_sources = read_detection_files(detection_paths)
    linked = link_event_detections(events, detections_by_frame, require_detection=True)
    tracking_metadata, metadata_warnings = load_tracking_metadata(detection_paths)
    metadata_warnings.extend(validate_tracking_metadata(tracking_metadata))

    areas = np.asarray([float(row["A_bb_px2"]) for row in linked], dtype=float)
    distances = np.asarray([float(row["gt_distance_m"]) for row in linked], dtype=float)
    polynomial = fit_quartic(areas, distances)
    warnings = list(metadata_warnings)
    expected_markers = set(EXPECTED_MARKER_IDS)
    observed_markers = {row["marker_id"] for row in linked}
    missing_markers = sorted(expected_markers - observed_markers)
    if missing_markers:
        raise ValueError(
            "Take 1 calibration must include confirmed events across all markers M01-M09; "
            f"missing: {', '.join(missing_markers)}"
        )
    markers_by_subject = {
        subject_id: sorted({row["marker_id"] for row in linked if row["subject_id"] == subject_id})
        for subject_id in sorted({row["subject_id"] for row in linked})
    }
    incomplete_subjects = {
        subject_id: sorted(expected_markers - set(markers))
        for subject_id, markers in markers_by_subject.items()
        if set(markers) != expected_markers
    }
    if incomplete_subjects:
        warnings.append(
            "Not every calibration subject has events at all nine markers: "
            + json.dumps(incomplete_subjects, ensure_ascii=False)
        )
    if not polynomial["monotonic_nonincreasing_in_calibration_range"]:
        warnings.append("Quartic prediction is not monotonically decreasing over the observed calibration-area range.")
    if not polynomial["positive_predictions_in_calibration_range"]:
        warnings.append("Quartic predicts nonpositive distances within the observed calibration-area range.")
    if polynomial["design_matrix_condition_number"] > 1e8:
        warnings.append("Quartic design matrix is poorly conditioned; interpret coefficients/predictions cautiously.")

    model = {
        "model_schema_version": 1,
        "model_type": "global_quartic_bbox_area_to_longitudinal_marker_distance",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "polynomial": polynomial,
        "ground_truth_definition": GROUND_TRUTH_DEFINITION,
        "calibration_take_ids": take_ids,
        "calibration_event_count": len(linked),
        "calibration_subject_ids": sorted({row["subject_id"] for row in linked}),
        "calibration_markers": sorted(observed_markers),
        "calibration_markers_by_subject": markers_by_subject,
        "marker_ground_truth_m": marker_gt,
        "marker_ground_truth_source": workspace_path(marker_gt_path),
        "training_event_csvs": [
            {"path": workspace_path(path), "sha256": sha256_file(path)} for path in event_paths
        ],
        "training_detection_csvs": [
            {"path": workspace_path(path), "sha256": sha256_file(path)} for path in detection_paths
        ],
        "tracking_metadata": tracking_metadata,
        "metric_interpretation": "in_sample_calibration_fit_error; these rows were also used to fit the coefficients",
        "lateral_offset_corrected": False,
    }

    calibration_predictions = prediction_rows(linked, polynomial, fit_scope="in_sample_calibration")

    output_dir = prepare_output_dir(args.output_dir)
    write_json(output_dir / "polynomial_model.json", model)
    write_csv(output_dir / "calibration_predictions.csv", calibration_predictions)
    summary = {
        "status": "calibrated",
        "evaluation_type": "in_sample_calibration_fit",
        "take_id": take_ids[0],
        "model_file": "polynomial_model.json",
        "fit": polynomial,
        **build_summary(calibration_predictions),
        "event_csvs": event_sources,
        "detection_csvs": detection_sources,
        "marker_ground_truth_source": workspace_path(marker_gt_path),
        "interpretation": (
            "These MAE/RMSE values describe fit residuals on Take 1, the same data used to estimate coefficients. "
            "Use frozen-model Take 2-4 results as inference/test metrics."
        ),
        "ground_truth_definition": GROUND_TRUTH_DEFINITION,
        "warnings": warnings,
    }
    write_json(output_dir / "calibration_summary.json", summary)
    print(json.dumps({"output_dir": workspace_path(output_dir), "summary": summary}, indent=2, ensure_ascii=False))
    return 0


def validate_model_and_marker_table(model: dict[str, Any], marker_gt: dict[str, float]) -> None:
    if model.get("model_type") != "global_quartic_bbox_area_to_longitudinal_marker_distance":
        raise ValueError("Unsupported model file/type")
    polynomial = model.get("polynomial", {})
    if polynomial.get("degree") != DEGREE or len(polynomial.get("coefficients_ascending", [])) != DEGREE + 1:
        raise ValueError("Model file does not contain the expected degree-4 polynomial")
    if float(polynomial.get("normalization_scale_px2", 0)) <= 0:
        raise ValueError("Model file has an invalid input-normalization scale")
    expected_gt = model.get("marker_ground_truth_m", {})
    for marker_id, distance in marker_gt.items():
        if marker_id not in expected_gt:
            raise ValueError(f"Marker {marker_id} was not present in the calibration ground-truth table")
        if abs(float(expected_gt[marker_id]) - distance) > 1e-6:
            raise ValueError(
                f"Marker ground truth changed after calibration for {marker_id}: "
                f"calibration={expected_gt[marker_id]}, evaluation={distance}"
            )


def command_evaluate(args: argparse.Namespace) -> int:
    model_path = resolve_path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    marker_gt_path = resolve_path(args.marker_ground_truth)
    marker_gt = read_marker_ground_truth(marker_gt_path)
    validate_model_and_marker_table(model, marker_gt)

    event_paths = [resolve_path(path) for path in args.events]
    detection_paths = [resolve_path(path) for path in args.detections]
    events, event_sources = read_confirmed_events(event_paths, marker_gt)
    test_take_ids = sorted({event["take_id"] for event in events})
    calibration_take_ids = set(model.get("calibration_take_ids", []))
    overlap = sorted(calibration_take_ids.intersection(test_take_ids))
    if overlap:
        raise ValueError(
            f"Test events include calibration take(s) {overlap}; use Take 2-4 events only so the frozen test is independent."
        )

    detections_by_frame, detection_sources = read_detection_files(detection_paths)
    linked = link_event_detections(events, detections_by_frame, require_detection=False)
    tracking_metadata, metadata_warnings = load_tracking_metadata(detection_paths)
    metadata_warnings.extend(
        validate_tracking_metadata(model.get("tracking_metadata", []), tracking_metadata)
    )

    predictions = prediction_rows(linked, model_for_prediction_file(model), fit_scope="take1_frozen_inference")
    warnings = list(metadata_warnings)
    if any(row["detection_found"] == "FALSE" for row in predictions):
        warnings.append("Some confirmed crossing events have no matched detector box; they are counted in coverage, not distance error metrics.")
    if any(row.get("outside_calibration_area_range") == "TRUE" for row in predictions):
        warnings.append("Some test bounding-box areas are outside Take 1 calibration range; these predictions are extrapolations.")
    if any(row.get("prediction_nonpositive") == "TRUE" for row in predictions):
        warnings.append("The frozen quartic produced one or more nonpositive distances; no clipping was applied.")
    if not model["polynomial"].get("monotonic_nonincreasing_in_calibration_range", False):
        warnings.append("Calibration curve was non-monotonic over its input range; interpret inference estimates cautiously.")

    output_dir = prepare_output_dir(args.output_dir)
    write_csv(output_dir / "distance_predictions.csv", predictions)
    summary = {
        "status": "evaluated_frozen_model",
        "evaluation_type": "take2_to_take4_inference_test",
        "model_file": workspace_path(model_path),
        "model_sha256": sha256_file(model_path),
        "calibration_take_ids": model.get("calibration_take_ids", []),
        "test_take_ids": test_take_ids,
        "model_refit_during_evaluation": False,
        "model_degree": DEGREE,
        "ground_truth_definition": GROUND_TRUTH_DEFINITION,
        **build_summary(predictions),
        "event_csvs": event_sources,
        "detection_csvs": detection_sources,
        "marker_ground_truth_source": workspace_path(marker_gt_path),
        "tracking_metadata": tracking_metadata,
        "interpretation": (
            "Metrics are computed only on manually confirmed events with a matched detector box. "
            "They evaluate the frozen Take 1 model on separate takes of the same subjects; they do not establish generalization to new subjects."
        ),
        "limitations": [
            "The target is longitudinal distance assigned to the marker line; lateral lane offset is not corrected.",
            "Take 2-4 use the same subjects, so results measure consistency across takes/conditions, not generalization to unseen people.",
            "Detection misses are reported through event coverage and excluded from conditional distance-error metrics.",
        ],
        "warnings": warnings,
    }
    write_json(output_dir / "evaluation_summary.json", summary)
    print(json.dumps({"output_dir": workspace_path(output_dir), "summary": summary}, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser("calibrate", help="Fit the global quartic using Take 1 only")
    calibrate.add_argument("--events", type=Path, nargs="+", required=True, help="Confirmed Take 1 event CSV file(s)")
    calibrate.add_argument(
        "--detections", type=Path, nargs="+", required=True,
        help="Mapped/raw detection CSV file(s); mapped subject detections are recommended",
    )
    calibrate.add_argument("--marker-ground-truth", type=Path, default=DEFAULT_MARKER_GT)
    calibrate.add_argument("--expected-take", default="Take1", help="Expected calibration take_id; use '' to disable")
    calibrate.add_argument("--output-dir", type=Path, required=True, help="New output directory; existing directories are never overwritten")
    calibrate.set_defaults(func=command_calibrate)

    evaluate = commands.add_parser("evaluate", help="Score Take 2-4 with a frozen Take 1 model; never refits")
    evaluate.add_argument("--model", type=Path, required=True, help="polynomial_model.json created by calibrate")
    evaluate.add_argument("--events", type=Path, nargs="+", required=True, help="Confirmed test event CSV file(s)")
    evaluate.add_argument(
        "--detections", type=Path, nargs="+", required=True,
        help="Mapped/raw detection CSV file(s); mapped subject detections are recommended",
    )
    evaluate.add_argument("--marker-ground-truth", type=Path, default=DEFAULT_MARKER_GT)
    evaluate.add_argument("--output-dir", type=Path, required=True, help="New output directory; existing directories are never overwritten")
    evaluate.set_defaults(func=command_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "expected_take", None) == "":
        args.expected_take = None
    try:
        return int(args.func(args))
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
