"""Apply the frozen Take-1 quartics to each tracked test-frame car detection.

Inputs:
- fitted_models.json from 03_fit_class_quartics.py
- detections_with_bodyclass.csv from 02_classify_car_tracks.py
- summary.json from the corresponding 01_track_cars.py run

Output is a visualization trajectory table for one video. Distance comes from
the frozen global or body-class quartic; low-confidence/missing classifier
labels use the global curve. Lateral position is a schematic mapping from
image x to road width, not calibrated camera geometry. Invalid/nonpositive
distances remain in the CSV for auditing but are not marked scene-eligible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("MPV", "Sedan", "Hatchback")
GLOBAL_KEY = "global_quartic_monotone_take1"
CLASS_KEY = "body_class_quartic_monotone_take1"
LENGTHS_M = {"MPV": 4.574, "Sedan": 4.674, "Hatchback": 4.064}
OUTPUT_FIELDS = [
    "video_id", "frame_idx", "timestamp_s", "source_fps", "track_id",
    "visual_body_class", "classifier_confidence", "classifier_used_fallback",
    "distance_model_used", "A_bb_px2", "pred_distance_m", "outside_model_area_range",
    "prediction_nonpositive", "center_x_px", "source_width", "source_height",
    "lateral_x_m_schematic", "approx_vehicle_length_m", "scene_eligible",
]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if (Path.cwd().resolve() == ROOT.parent.resolve()
            and path.parts and path.parts[0].casefold() == ROOT.name.casefold()):
        return (Path.cwd() / path).resolve()
    return (ROOT / path).resolve()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def canonical_class(value: str) -> str | None:
    for name in CLASSES:
        if (value or "").strip().casefold() == name.casefold():
            return name
    return None


def predict_area(area: float, model: dict[str, Any]) -> float:
    mean = float(model["normalization_mean_px2"])
    scale = float(model["normalization_scale_px2"])
    if scale <= 0:
        raise ValueError("Model normalization_scale_px2 must be positive")
    z = (area - mean) / scale
    return float(np.polynomial.polynomial.polyval(z, model["coefficients_ascending"]))


def find_video_metadata(summary_path: Path, video_id: str | None) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    videos = summary.get("videos", [])
    if not videos:
        raise ValueError(f"No video metadata in {summary_path}")
    if video_id:
        matches = [row for row in videos if row.get("video_id") == video_id]
    elif len(videos) == 1:
        matches = videos
    else:
        raise ValueError("--video-id is required because the tracking summary contains multiple videos")
    if len(matches) != 1:
        raise ValueError(f"Expected one matching video_id={video_id!r} in {summary_path}; found {len(matches)}")
    return {**matches[0], "tracking_summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True, help="fitted_models.json from script 03")
    parser.add_argument("--detections", type=Path, required=True, help="detections_with_bodyclass.csv from script 02")
    parser.add_argument("--tracking-summary", type=Path, required=True, help="summary.json from the matching script 01 run")
    parser.add_argument("--video-id", default=None, help="Required only if tracking summary has multiple videos")
    parser.add_argument("--road-width-m", type=float, default=6.0, help="Visualization-only lateral mapping span")
    parser.add_argument("--max-lateral-fraction", type=float, default=0.78, help="Clamp image-x mapping to this fraction of road half-width")
    parser.add_argument("--output", type=Path, required=True, help="New frame trajectory CSV")
    parser.add_argument("--summary", type=Path, default=None, help="Optional new JSON summary path")
    args = parser.parse_args(argv)

    model_path, detections_path, tracking_summary_path, output_path = map(
        resolve_path, (args.models, args.detections, args.tracking_summary, args.output)
    )
    summary_path = resolve_path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    for path in (model_path, detections_path, tracking_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite trajectory output or its summary")
    if args.road_width_m <= 0 or not 0 < args.max_lateral_fraction <= 1:
        parser.error("road-width-m must be positive and max-lateral-fraction must be in (0, 1]")

    model_bundle = json.loads(model_path.read_text(encoding="utf-8"))
    if model_bundle.get("schema") != "car_bodyclass_and_global_monotone_quartic_v1":
        raise ValueError(f"Unsupported frozen quartic model schema: {model_bundle.get('schema')!r}")
    global_model = model_bundle["models"][GLOBAL_KEY]
    class_models = model_bundle["models"][CLASS_KEY]
    threshold = float(model_bundle.get("fallback_policy", {}).get("threshold", 0.6))
    video_meta = find_video_metadata(tracking_summary_path, args.video_id)
    selected_video_id = str(video_meta["video_id"])
    width = int(video_meta["source_width"])
    height = int(video_meta["source_height"])
    fps = float(video_meta["source_fps"])
    detections = read_rows(detections_path)
    output_rows: list[dict[str, Any]] = []
    counts: Counter = Counter()
    seen_video_ids: set[str] = set()
    max_lateral = args.road_width_m * 0.5 * args.max_lateral_fraction

    for row in detections:
        video = (row.get("video_id") or "").strip()
        if not video:
            continue
        seen_video_ids.add(video)
        if video != selected_video_id:
            continue
        track_id = (row.get("track_id") or "").strip()
        if not track_id:
            counts["untracked_detections_skipped"] += 1
            continue
        area = finite_float(row.get("A_bb_px2"))
        if area is None:
            w, h = finite_float(row.get("w_px")), finite_float(row.get("h_px"))
            area = w * h if w is not None and h is not None and w > 0 and h > 0 else None
        if area is None or area <= 0:
            counts["invalid_area_skipped"] += 1
            continue

        visual_class = canonical_class(row.get("bodyclass_predicted_model_class", ""))
        confidence = finite_float(row.get("bodyclass_confidence_raw_softmax"))
        source_fallback = (row.get("bodyclass_fallback_to_global") or "").strip().upper() == "TRUE"
        fallback = source_fallback or confidence is None or confidence < threshold or visual_class is None
        selected_model = global_model if fallback else class_models[visual_class]
        model_used = GLOBAL_KEY if fallback else f"bodyclass_{visual_class}_quartic_monotone_take1"
        distance = predict_area(area, selected_model)
        area_min = float(selected_model["training_area_min_px2"])
        area_max = float(selected_model["training_area_max_px2"])
        outside_range = area < area_min or area > area_max
        nonpositive = distance <= 0
        center_x = finite_float(row.get("center_x_px"))
        if center_x is None:
            x1, x2 = finite_float(row.get("x1_px")), finite_float(row.get("x2_px"))
            center_x = (x1 + x2) / 2 if x1 is not None and x2 is not None else width / 2
        lateral = max(-max_lateral, min(max_lateral, (center_x / width - 0.5) * args.road_width_m))
        frame_idx = int(float(row.get("frame_idx", 0)))
        timestamp = finite_float(row.get("timestamp_s"))
        if timestamp is None:
            timestamp = frame_idx / fps
        approx_length = LENGTHS_M.get(visual_class, "")
        scene_eligible = bool(visual_class and not nonpositive)
        output_rows.append({
            "video_id": video,
            "frame_idx": frame_idx,
            "timestamp_s": round(timestamp, 6),
            "source_fps": fps,
            "track_id": track_id,
            "visual_body_class": visual_class or "UNKNOWN",
            "classifier_confidence": confidence if confidence is not None else "",
            "classifier_used_fallback": "TRUE" if fallback else "FALSE",
            "distance_model_used": model_used,
            "A_bb_px2": round(area, 3),
            "pred_distance_m": round(distance, 6),
            "outside_model_area_range": "TRUE" if outside_range else "FALSE",
            "prediction_nonpositive": "TRUE" if nonpositive else "FALSE",
            "center_x_px": round(center_x, 3),
            "source_width": width,
            "source_height": height,
            "lateral_x_m_schematic": round(lateral, 4),
            "approx_vehicle_length_m": approx_length,
            "scene_eligible": "TRUE" if scene_eligible else "FALSE",
        })
        counts["detections_written"] += 1
        counts[f"body_class_{visual_class or 'unknown'}"] += 1
        counts["fallback_global"] += int(fallback)
        counts["outside_model_area_range"] += int(outside_range)
        counts["nonpositive_predictions"] += int(nonpositive)
        counts["scene_ineligible"] += int(not scene_eligible)

    if len(seen_video_ids) > 1 and args.video_id is None:
        raise ValueError("Input detections contain multiple videos; pass --video-id explicitly")
    if not output_rows:
        raise ValueError(f"No trackable detections for video_id={selected_video_id}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    result_summary = {
        "schema": "car_frame_trajectory_visualization_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "video_id": selected_video_id,
        "source_width": width,
        "source_height": height,
        "source_fps": fps,
        "road_width_m_schematic": args.road_width_m,
        "lateral_position_method": "normalized center_x_px mapped and clamped to visualization road width; not calibrated camera geometry",
        "classifier_fallback_threshold": threshold,
        "model_file": str(model_path),
        "model_sha256": sha256_file(model_path),
        "detections_file": str(detections_path),
        "detections_sha256": sha256_file(detections_path),
        "tracking_summary": str(tracking_summary_path),
        "trajectory_csv": str(output_path),
        "trajectory_rows": len(output_rows),
        "counts": dict(counts),
        "limitations": [
            "3D lateral position is schematic and proportional to bbox center, not camera-calibrated world geometry.",
            "Quartic outputs are raw; no clipping is applied. Nonpositive estimates are excluded from scene objects.",
            "Track fragments are separate visual cars unless manually reconciled before trajectory export.",
            "Predictions outside the calibration area range are marked extrapolated and should be reviewed.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
