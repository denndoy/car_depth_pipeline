"""Synthetic end-to-end smoke test for car_depth_pipeline scripts 00 -> 03 -> 04.

No video, no YOLO weights, no real classifier is needed: the test builds a
temporary workspace with physically plausible synthetic data
(A_bb_px2 = (K_class / distance)^2 with small multiplicative noise) and:

1. Runs 00_prepare_classifier_dataset.py on a fake 8-class Kaggle-style layout
   (split mode) and asserts only MPV/Sedan/Hatchback are exported.
2. Runs 03_fit_class_quartics.py on Take 1 events (3 classes x 3 vehicles x 9
   markers = 81 events) and asserts per-class/global monotone quartics, event
   counts, and in-sample MAE sanity.
3. Runs 04_evaluate_frozen.py on Take 2 events (3 classes x 2 unseen vehicles
   x 9 markers = 54 events) with a synthetic classifier CSV containing one
   wrong-class prediction, one low-confidence fallback, and one missing
   prediction; asserts selection reasons, oracle vs end-to-end metrics, and
   coverage accounting.

Run with the project venv python:
  .venv/Scripts/python.exe car_depth_pipeline/tests/test_smoke_synthetic.py
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
ROOT = HERE.parents[1]
PY = sys.executable

MARKERS = {f"M{index:02d}": round(6.6 * index, 3) for index in range(1, 10)}
CLASS_SCALE = {"MPV": 3000.0, "Sedan": 2600.0, "Hatchback": 2200.0}
ASPECT_RATIO = 0.38  # h/w of a car box


def event_row(event_id: str, take_id: str, video_id: str, subject_id: str, marker_id: str,
              frame_idx: int, track_id: str, body_class: str) -> dict[str, str]:
    return {
        "event_id": event_id,
        "take_id": take_id,
        "video_id": video_id,
        "subject_id": subject_id,
        "marker_id": marker_id,
        "frame_idx": str(frame_idx),
        "foot_crossing_confirmed": "TRUE",
        "annotation_source": "manual_visual_review_user_confirmed",
        "gt_distance_m": "",
        "pass_id": "",
        "lane": "left",
        "timestamp_s": "",
        "track_id": track_id,
        "body_class_gt": body_class,
        "notes": "synthetic smoke test row",
    }


def detection_row(video_id: str, frame_idx: int, track_id: str, subject_id: str, area: float) -> dict[str, str]:
    width = math.sqrt(area / ASPECT_RATIO)
    height = width * ASPECT_RATIO
    x1, y1 = 1900.0 - width / 2, 1500.0 - height / 2
    return {
        "video_id": video_id,
        "pass_id": "",
        "frame_idx": str(frame_idx),
        "track_id": track_id,
        "subject_id": subject_id,
        "x1_px": f"{x1:.3f}",
        "y1_px": f"{y1:.3f}",
        "x2_px": f"{x1 + width:.3f}",
        "y2_px": f"{y1 + height:.3f}",
        "w_px": f"{width:.3f}",
        "h_px": f"{height:.3f}",
        "A_bb_px2": f"{width * height:.3f}",
        "confidence": "0.65",
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(script: Path, args: list[str]) -> None:
    result = subprocess.run(
        [PY, str(script), *args], capture_output=True, text=True, timeout=600, cwd=str(ROOT)
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{script.name} failed (exit {result.returncode}):\nSTDOUT:\n{result.stdout[-3000:]}\nSTDERR:\n{result.stderr[-3000:]}"
        )
    print(f"  ok: {script.name}")


def build_synthetic_dataset(rng: random.Random) -> tuple[list[dict], list[dict]]:
    events, detections = [], []
    frame_counter = 100
    classes = list(CLASS_SCALE)
    for class_index, body_class in enumerate(classes):
        for vehicle_index in range(3):
            subject_id = f"CAL_{body_class}_{vehicle_index + 1:02d}"
            for marker_id, distance in MARKERS.items():
                frame_counter += 1
                area = (CLASS_SCALE[body_class] / distance) ** 2 * rng.uniform(0.97, 1.03)
                track_id = f"T{frame_counter}"
                events.append(event_row(
                    f"TAKE1:{subject_id}:{marker_id}", "Take1", "take1_video", subject_id,
                    marker_id, frame_counter, track_id, body_class,
                ))
                detections.append(detection_row("take1_video", frame_counter, track_id, subject_id, area))
    return events, detections


def build_test_dataset(rng: random.Random) -> tuple[list[dict], list[dict], list[dict], dict[str, str]]:
    events, detections, classifier_rows = [], [], []
    plan: dict[str, str] = {}
    frame_counter = 500_000
    classes = list(CLASS_SCALE)
    special = {
        ("MPV", 0): "wrong_class",      # classifier predicts Sedan for an MPV
        ("Hatchback", 1): "fallback",   # correct class but confidence below threshold
        ("Sedan", 0): "no_prediction",  # track missing from classifier CSV
    }
    for body_class in classes:
        for vehicle_index in range(2):
            subject_id = f"T2_{body_class}_{vehicle_index + 1:02d}"
            for marker_id, distance in MARKERS.items():
                frame_counter += 1
                track_id = f"T{frame_counter}"
                scenario = special.get((body_class, vehicle_index))
                area = (CLASS_SCALE[body_class] / distance) ** 2 * rng.uniform(0.97, 1.03)
                events.append(event_row(
                    f"TAKE2:{subject_id}:{marker_id}", "Take2", "take2_video", subject_id,
                    marker_id, frame_counter, track_id, body_class,
                ))
                detections.append(detection_row("take2_video", frame_counter, track_id, "", area))
                plan[track_id] = scenario or "ok"
                predicted_class = body_class
                confidence = "0.90"
                fallback = "FALSE"
                if scenario == "wrong_class":
                    predicted_class = "Sedan" if body_class == "MPV" else "MPV"
                elif scenario == "fallback":
                    confidence = "0.45"
                    fallback = "TRUE"
                if scenario != "no_prediction":
                    classifier_rows.append({
                        "video_id": "take2_video",
                        "pass_id": "",
                        "track_id": track_id,
                        "subject_id": "",
                        "reference_body_class": "",
                        "first_frame_idx": "", "last_frame_idx": "", "detection_rows": "",
                        "sample_frame_idx": "", "sample_timestamp_s": "",
                        "sample_width_px": "", "sample_height_px": "", "sample_confidence": "",
                        "bodyclass_predicted_model_class": predicted_class,
                        "bodyclass_confidence_raw_softmax": confidence,
                        "bodyclass_fallback_to_global": fallback,
                        "bodyclass_prob_MPV": "", "bodyclass_prob_Sedan": "", "bodyclass_prob_Hatchback": "",
                    })
    return events, detections, classifier_rows, plan


def test_prepare_classifier_dataset(tmp: Path) -> None:
    src = tmp / "kaggle_like"
    for split in ("train", "validation", "test"):
        for class_name, count in (("mpv", 6), ("sedan", 5), ("hatchback", 4), ("suv", 3), ("truck", 2)):
            folder = src / split / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                (folder / f"{class_name}_{split}_{index}.jpg").write_bytes(b"fake-jpeg-bytes")
    dst = tmp / "classifier_dataset"
    run(SCRIPTS / "00_prepare_classifier_dataset.py", ["--src", str(src), "--dst", str(dst)])
    summary = json.loads((dst / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["counts_by_class_by_split"]) == {"MPV", "Sedan", "Hatchback"}, summary
    assert summary["counts_by_class_by_split"]["MPV"] == {"train": 6, "val": 6, "test": 6}, summary
    assert summary["counts_by_class_total"] == {"MPV": 18, "Sedan": 15, "Hatchback": 12}, summary
    assert (dst / "train" / "MPV").is_dir() and len(list((dst / "train" / "MPV").iterdir())) == 6
    assert len(list((dst / "val" / "Sedan").iterdir())) == 5
    assert not (dst / "train" / "suv").exists()
    manifest = list(csv.DictReader((dst / "manifest.csv").open(encoding="utf-8-sig")))
    assert len(manifest) == 45 and all(row["body_class"] in {"MPV", "Sedan", "Hatchback"} for row in manifest)
    print("  ok: 00_prepare_classifier_dataset (split mode, 3-of-8 classes, suv/truck ignored)")


def test_fit_and_evaluate(tmp: Path, rng: random.Random) -> None:
    inputs = tmp / "inputs"
    marker_gt = inputs / "marker_ground_truth.csv"
    write_csv(marker_gt, [
        {"marker_id": marker, "nominal_position_m": str(distance), "gt_distance_m": str(distance), "derivation": "synthetic"}
        for marker, distance in MARKERS.items()
    ])

    cal_events, cal_detections = build_synthetic_dataset(rng)
    write_csv(inputs / "take1_events.csv", cal_events)
    write_csv(inputs / "take1_detections_subject_mapped.csv", cal_detections)

    out_fit = tmp / "out_fit"
    run(SCRIPTS / "03_fit_class_quartics.py", [
        "--events", str(inputs / "take1_events.csv"),
        "--detections", str(inputs / "take1_detections_subject_mapped.csv"),
        "--marker-ground-truth", str(marker_gt),
        "--output-dir", str(out_fit),
    ])
    bundle = json.loads((out_fit / "fitted_models.json").read_text(encoding="utf-8"))
    assert bundle["schema"] == "car_bodyclass_and_global_monotone_quartic_v1"
    assert bundle["events_per_body_class"] == {"MPV": 27, "Sedan": 27, "Hatchback": 27}
    assert bundle["vehicles_per_body_class"] == {"MPV": 3, "Sedan": 3, "Hatchback": 3}
    class_models = bundle["models"]["body_class_quartic_monotone_take1"]
    for name, model in [("global", bundle["models"]["global_quartic_monotone_take1"]), *class_models.items()]:
        assert model["monotonicity"]["verified_analytically"] is True, f"{name} not monotone"
        assert model["max_positive_derivative_m_per_px2"] <= 1e-9, f"{name} positive derivative"
        assert model["training_prediction_min_m"] > -5, f"{name} implausible negative output"
    fit_summary = json.loads((out_fit / "calibration_summary.json").read_text(encoding="utf-8"))
    global_mae = fit_summary["metrics"]["global_quartic"]["MAE_m"]
    class_mae = fit_summary["metrics"]["body_class_quartic_all_events"]["MAE_m"]
    # Loose thresholds on purpose: a degree-4 polynomial in raw bbox area over a
    # 6.6-59.4 m range has large in-sample residuals (the person research's own
    # global Take-1 calibration showed in-sample MAE ~6.3 m on real data). The
    # smoke test validates plumbing, not curve quality.
    assert global_mae < 10.0 and class_mae < 10.0, f"in-sample MAE too large: global={global_mae}, class={class_mae}"
    assert class_mae <= global_mae + 0.5, "class-specific fit unexpectedly far worse than global"
    print(f"  ok: 03_fit_class_quartics (monotone verified; in-sample MAE global={global_mae} m, per-class={class_mae} m)")

    test_events, test_detections, classifier_rows, plan = build_test_dataset(rng)
    write_csv(inputs / "take2_events.csv", test_events)
    write_csv(inputs / "take2_detections.csv", test_detections)
    write_csv(inputs / "classifier_predictions.csv", classifier_rows)

    out_eval = tmp / "out_eval"
    run(SCRIPTS / "04_evaluate_frozen.py", [
        "--model", str(out_fit / "fitted_models.json"),
        "--events", str(inputs / "take2_events.csv"),
        "--detections", str(inputs / "take2_detections.csv"),
        "--bodyclass-predictions", str(inputs / "classifier_predictions.csv"),
        "--marker-ground-truth", str(marker_gt),
        "--output-dir", str(out_eval),
    ])
    summary = json.loads((out_eval / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["events_with_detection"] == 54 and summary["detection_coverage"] == 1.0
    reasons = summary["selection_reason_counts"]
    # wrong_class is still routed by the classifier (its error shows in the
    # confusion matrix), so classifier_prediction = 54 - 9 fallback - 9 missing.
    assert reasons.get("classifier_prediction") == 36, reasons
    assert reasons.get("classifier_fallback_low_confidence") == 9, reasons
    assert reasons.get("no_classifier_prediction") == 9, reasons
    assert reasons.get("classifier_class_not_in_models", 0) == 0
    assert set(summary["metrics"]) == {"end_to_end", "oracle", "global"}

    rows = list(csv.DictReader((out_eval / "distance_predictions.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 54
    global_model_id = "global_quartic_monotone_take1"
    by_track = {row["track_id"]: row for row in rows}
    for track_id, scenario in plan.items():
        row = by_track[track_id]
        if scenario == "fallback":
            assert row["model_used_end_to_end"] == global_model_id
            assert row["selection_reason_end_to_end"] == "classifier_fallback_low_confidence"
        elif scenario == "no_prediction":
            assert row["model_used_end_to_end"] == global_model_id
            assert row["selection_reason_end_to_end"] == "no_classifier_prediction"
        else:
            assert row["selection_reason_end_to_end"] == "classifier_prediction"
            expected_model = global_model_id if scenario == "wrong_class" else f"bodyclass_{row['body_class_gt']}_quartic_monotone_take1"
            if scenario != "wrong_class":
                assert row["model_used_end_to_end"] == expected_model, (track_id, row["model_used_end_to_end"])
        assert row["model_used_oracle"] == f"bodyclass_{row['body_class_gt']}_quartic_monotone_take1"
        assert row["detection_found"] == "TRUE"

    metrics = list(csv.DictReader((out_eval / "metrics_by_scope.csv").open(encoding="utf-8-sig")))
    overall = {row["variant"]: row for row in metrics if row["scope_type"] == "all"}
    for variant in ("end_to_end", "oracle", "global"):
        assert overall[variant]["N"] == "54", (variant, overall[variant])
    per_class = [row for row in metrics if row["scope_type"] == "body_class_gt"]
    assert {row["scope"] for row in per_class} == {"MPV", "Sedan", "Hatchback"}
    confusion = list(csv.DictReader((out_eval / "classifier_confusion.csv").open(encoding="utf-8-sig")))
    pairs = {(row["body_class_gt"], row["classifier_predicted_class"]): int(row["event_count"]) for row in confusion}
    assert pairs[("MPV", "Sedan")] == 9 and pairs[("MPV", "MPV")] == 9, pairs
    assert pairs[("Sedan", "NO_PREDICTION")] == 9, pairs
    print(f"  ok: 04_evaluate_frozen (fallback/oracle/wrong-class routing correct; metrics N=54 per variant)")


def test_enrich_events(tmp: Path) -> None:
    inputs = tmp / "inputs"
    src = inputs / "take2_events.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8-sig")))
    stripped = [{key: value for key, value in row.items() if key != "body_class_gt"} for row in rows]
    no_class = inputs / "take2_events_no_class.csv"
    write_csv(no_class, stripped)
    metadata = inputs / "vehicle_metadata.csv"
    write_csv(metadata, [
        {"vehicle_id": "T2_MPV_01", "body_class_gt": "MPV"},
        {"vehicle_id": "T2_MPV_02", "body_class_gt": "MPV"},
        {"vehicle_id": "T2_Sedan_01", "body_class_gt": "sedan"},
        {"vehicle_id": "T2_Sedan_02", "body_class_gt": "Sedan"},
        {"vehicle_id": "T2_Hatchback_01", "body_class_gt": "Hatchback"},
        {"vehicle_id": "T2_Hatchback_02", "body_class_gt": "HATCHBACK"},
    ])
    out = tmp / "out_enrich"
    run(SCRIPTS / "05_enrich_events_with_class.py", [
        "--events", str(no_class), "--vehicle-metadata", str(metadata), "--output-dir", str(out),
    ])
    enriched = list(csv.DictReader((out / "take2_events_no_class.csv").open(encoding="utf-8-sig")))
    original = {row["event_id"]: row["body_class_gt"] for row in rows}
    for row in enriched:
        assert row["body_class_gt"] == original[row["event_id"]], (row["event_id"], row["body_class_gt"])
    # Negative: unknown vehicle must be rejected.
    bad = inputs / "take2_events_bad.csv"
    broken = [dict(row, subject_id="UNKNOWN_CAR") for row in stripped[:3]]
    write_csv(bad, broken)
    bad_out = tmp / "out_enrich_bad"
    result = subprocess.run(
        [PY, str(SCRIPTS / "05_enrich_events_with_class.py"),
         "--events", str(bad), "--vehicle-metadata", str(metadata), "--output-dir", str(bad_out)],
        capture_output=True, text=True, timeout=120, cwd=str(ROOT),
    )
    assert result.returncode != 0 and "not in vehicle_metadata" in (result.stderr + result.stdout)
    print("  ok: 05_enrich_events_with_class (fill + casefold + unknown-vehicle rejection)")


def main() -> int:
    rng = random.Random(20261117)
    with tempfile.TemporaryDirectory(prefix="car_pipeline_smoke_") as raw_tmp:
        tmp = Path(raw_tmp)
        print("Smoke test 1/4: 00_prepare_classifier_dataset")
        test_prepare_classifier_dataset(tmp)
        print("Smoke test 2/4: 03_fit_class_quartics + 04_evaluate_frozen")
        test_fit_and_evaluate(tmp, rng)
        print("Smoke test 3/4: 05_enrich_events_with_class")
        test_enrich_events(tmp)
    print("SMOKE TEST PASS: 00, 03, 04, 05 berjalan dan routing fallback/oracle/global benar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
