"""Run pretrained YOLOv8n (COCO `car`, class id 2) + ByteTrack over source videos on CPU.

Standalone car-pipeline copy of ``scripts/03_tracking_crosswalk/full_video_tracking.py``
from the person research. Differences are limited to the detected class:

- ``classes=[2]`` (COCO car) instead of ``classes=[0]`` (person); trucks (5) and
  buses (7) stay excluded so the box area always describes a passenger car.
- Field/file names use ``car`` wording; ``subject_id`` columns are kept with the
  same schema so the original crosswalk and distance scripts remain compatible.

Everything else is intentionally identical: detections.csv/frames.csv geometry,
summary.json metadata (weights sha256, imgsz, conf, tracker, device, source
resolution) that the frozen-model evaluator validates, and the manual
track->vehicle crosswalk template. Keep detector settings (weights, imgsz, conf,
tracker) EXACTLY the same for the calibration take and all test takes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = ROOT / "data"
DEFAULT_WEIGHTS = ROOT / "models" / "yolov8n.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "tracking"
DETECTOR_CLASS_ID = 2
DETECTOR_CLASS_NAME = "car"
DETECTION_FIELDS = [
    "video_id",
    "pass_id",
    "frame_idx",
    "timestamp_s",
    "timestamp_source",
    "track_id",
    "subject_id",
    "class_id",
    "class_name",
    "x1_px",
    "y1_px",
    "x2_px",
    "y2_px",
    "w_px",
    "h_px",
    "A_bb_px2",
    "center_x_px",
    "center_y_px",
    "bottom_center_x_px",
    "bottom_center_y_px",
    "confidence",
    "imgsz",
    "model",
    "status",
]
FRAME_FIELDS = [
    "video_id",
    "pass_id",
    "frame_idx",
    "timestamp_s",
    "timestamp_source",
    "n_car_detections",
    "track_ids",
    "model_call_s",
    "status",
]
TRACK_FIELDS = [
    "video_id",
    "pass_id",
    "track_id",
    "first_frame_idx",
    "first_timestamp_s",
    "last_frame_idx",
    "last_timestamp_s",
    "detection_rows",
    "subject_id",
    "notes",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def make_track_thumbnail(frame, boxes, frame_idx: int, timestamp_s: float):
    thumb_width, thumb_height = 640, 360
    height, width = frame.shape[:2]
    thumb = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
    sx, sy = thumb_width / width, thumb_height / height
    palette = [
        (255, 80, 80), (80, 220, 80), (80, 160, 255), (220, 120, 255),
        (255, 210, 60), (70, 230, 230), (255, 140, 50), (180, 180, 255),
    ]
    for box, track_id, confidence in boxes:
        x1, y1, x2, y2 = box
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        color = palette[(track_id or 0) % len(palette)]
        cv2.rectangle(thumb, p1, p2, color, 2)
        cv2.putText(
            thumb,
            f"T{track_id if track_id is not None else '-'} {confidence:.2f}",
            (p1[0], max(16, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return thumb, f"frame {frame_idx} | {timestamp_s:.2f}s"


def write_contact_sheet(items: list[tuple[object, str]], output_path: Path) -> None:
    if not items:
        raise ValueError("No decoded preview frames; tracking output is incomplete")
    columns, tile_w, image_h, label_h = 4, 640, 360, 28
    rows = (len(items) + columns - 1) // columns
    sheet = 32 * __import__("numpy").ones((rows * (image_h + label_h), columns * tile_w, 3), dtype="uint8")
    for index, (image, label) in enumerate(items):
        x = (index % columns) * tile_w
        y = (index // columns) * (image_h + label_h)
        sheet[y : y + image_h, x : x + tile_w] = image
        cv2.rectangle(sheet, (x, y + image_h), (x + tile_w, y + image_h + label_h), (15, 15, 15), -1)
        cv2.putText(sheet, label, (x + 8, y + image_h + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not write contact sheet: {output_path}")


def select_videos(video_dir: Path, explicit: list[Path] | None = None) -> list[Path]:
    videos = [resolve_path(path) for path in explicit] if explicit else sorted(video_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No MP4 files found in {video_dir}")
    if len({path.resolve() for path in videos}) != len(videos):
        raise ValueError("Duplicate selected source videos")
    for path in videos:
        if not path.is_file() or path.suffix.lower() != ".mp4":
            raise FileNotFoundError(f"Selected source must be an existing MP4: {path}")
    return videos


def write_progress(run_dir: Path, status: str, **details) -> None:
    target = run_dir / "progress.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"status": status, "updated_at": datetime.now().isoformat(timespec="seconds"), **details}, indent=2), encoding="utf-8")
    temporary.replace(target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--video", type=Path, action="append", help="Process only this MP4 (repeatable); recommended per take")
    parser.add_argument("--weights", type=Path, required=True, help="Local COCO-pretrained YOLOv8n weights; keep identical across takes")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"yolov8n_car_imgsz1280_bytetrack_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="ByteTrack YAML: an installed default name or a path relative to the workspace root.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--preview-samples", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    video_dir = resolve_path(args.video_dir)
    weights_path = resolve_path(args.weights)
    output_root = resolve_path(args.output_root)
    videos = select_videos(video_dir, args.video)
    tracker_candidate = resolve_path(Path(args.tracker))
    tracker = str(tracker_candidate if tracker_candidate.is_file() else args.tracker)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if args.imgsz <= 0 or args.threads <= 0 or args.preview_samples <= 0 or not 0 <= args.conf <= 1:
        parser.error("imgsz, threads, preview-samples must be positive; conf must be in [0, 1]")

    run_dir = output_root / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output folder: {run_dir}")
    preview_dir = run_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    write_progress(run_dir, "loading", selected_videos=[path.name for path in videos], video_count=len(videos))

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.threads, 4)))
    except RuntimeError:
        pass

    summaries = []
    track_stats: dict[tuple[str, int], dict[str, int | float]] = {}
    detections_path = run_dir / "detections.csv"
    frames_path = run_dir / "frames.csv"

    with detections_path.open("w", newline="", encoding="utf-8-sig") as detections_file, frames_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as frames_file:
        detection_writer = csv.DictWriter(detections_file, fieldnames=DETECTION_FIELDS)
        frame_writer = csv.DictWriter(frames_file, fieldnames=FRAME_FIELDS)
        detection_writer.writeheader()
        frame_writer.writeheader()

        for video_path in videos:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0 or expected_frames <= 0:
                cap.release()
                raise RuntimeError(f"Invalid video metadata for {video_path.name}")

            model_load_start = time.perf_counter()
            model = YOLO(str(weights_path))
            model_load_s = time.perf_counter() - model_load_start
            video_start = time.perf_counter()
            frames_seen = 0
            no_detection_frames = 0
            car_rows = 0
            inference_times: list[float] = []
            prev_timestamp = -1.0
            timestamp_sources = Counter()
            preview_indices = {
                round(i * (expected_frames - 1) / (args.preview_samples - 1))
                for i in range(args.preview_samples)
            } if args.preview_samples > 1 else {expected_frames // 2}
            previews = []

            print(f"Starting {video_path.name}: {expected_frames} frames, {width}x{height}, imgsz={args.imgsz}", flush=True)
            write_progress(run_dir, "tracking", video_id=video_path.stem, frames_processed=0, expected_frames=expected_frames, selected_videos=[path.name for path in videos])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx = frames_seen
                timestamp_s = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                timestamp_source = "opencv_pos_msec"
                if not math.isfinite(timestamp_s) or timestamp_s < 0 or (frames_seen > 0 and timestamp_s <= prev_timestamp):
                    timestamp_s = max(frame_idx / fps, prev_timestamp + 1.0 / fps)
                    timestamp_source = "frame_idx_div_fps_fallback"
                prev_timestamp = timestamp_s
                timestamp_sources[timestamp_source] += 1

                inference_start = time.perf_counter()
                result = model.track(
                    frame,
                    persist=True,
                    tracker=tracker,
                    imgsz=args.imgsz,
                    classes=[DETECTOR_CLASS_ID],
                    conf=args.conf,
                    device="cpu",
                    verbose=False,
                )[0]
                model_call_s = time.perf_counter() - inference_start
                inference_times.append(model_call_s)
                boxes = result.boxes
                coords = boxes.xyxy.detach().cpu().tolist() if boxes is not None and len(boxes) else []
                confidences = boxes.conf.detach().cpu().tolist() if boxes is not None and len(boxes) else []
                ids_tensor = boxes.id if boxes is not None and len(boxes) else None
                track_ids = ids_tensor.detach().cpu().tolist() if ids_tensor is not None else [None] * len(coords)

                preview_boxes = []
                frame_track_ids = []
                for xyxy, confidence, raw_track_id in zip(coords, confidences, track_ids):
                    x1, y1, x2, y2 = [float(value) for value in xyxy]
                    box_width = max(0.0, x2 - x1)
                    box_height = max(0.0, y2 - y1)
                    track_id = int(raw_track_id) if raw_track_id is not None else None
                    if track_id is not None:
                        frame_track_ids.append(track_id)
                        key = (video_path.stem, track_id)
                        stats = track_stats.setdefault(
                            key,
                            {
                                "first_frame_idx": frame_idx,
                                "first_timestamp_s": timestamp_s,
                                "last_frame_idx": frame_idx,
                                "last_timestamp_s": timestamp_s,
                                "detection_rows": 0,
                            },
                        )
                        stats["last_frame_idx"] = frame_idx
                        stats["last_timestamp_s"] = timestamp_s
                        stats["detection_rows"] = int(stats["detection_rows"]) + 1
                    preview_boxes.append(([x1, y1, x2, y2], track_id, float(confidence)))
                    detection_writer.writerow(
                        {
                            "video_id": video_path.stem,
                            "pass_id": "",
                            "frame_idx": frame_idx,
                            "timestamp_s": f"{timestamp_s:.6f}",
                            "timestamp_source": timestamp_source,
                            "track_id": track_id,
                            "subject_id": "",
                            "class_id": DETECTOR_CLASS_ID,
                            "class_name": DETECTOR_CLASS_NAME,
                            "x1_px": f"{x1:.3f}",
                            "y1_px": f"{y1:.3f}",
                            "x2_px": f"{x2:.3f}",
                            "y2_px": f"{y2:.3f}",
                            "w_px": f"{box_width:.3f}",
                            "h_px": f"{box_height:.3f}",
                            "A_bb_px2": f"{box_width * box_height:.3f}",
                            "center_x_px": f"{(x1 + x2) / 2.0:.3f}",
                            "center_y_px": f"{(y1 + y2) / 2.0:.3f}",
                            "bottom_center_x_px": f"{(x1 + x2) / 2.0:.3f}",
                            "bottom_center_y_px": f"{y2:.3f}",
                            "confidence": f"{float(confidence):.6f}",
                            "imgsz": args.imgsz,
                            "model": weights_path.name,
                            "status": "detected",
                        }
                    )

                if not coords:
                    no_detection_frames += 1
                frame_writer.writerow(
                    {
                        "video_id": video_path.stem,
                        "pass_id": "",
                        "frame_idx": frame_idx,
                        "timestamp_s": f"{timestamp_s:.6f}",
                        "timestamp_source": timestamp_source,
                        "n_car_detections": len(coords),
                        "track_ids": ";".join(str(value) for value in frame_track_ids),
                        "model_call_s": f"{model_call_s:.6f}",
                        "status": "detected" if coords else "no_detection",
                    }
                )
                if frame_idx in preview_indices:
                    thumb, label = make_track_thumbnail(frame, preview_boxes, frame_idx, timestamp_s)
                    previews.append((thumb, label))

                frames_seen += 1
                car_rows += len(coords)
                if frames_seen % 300 == 0:
                    elapsed = time.perf_counter() - video_start
                    print(f"  {video_path.stem}: {frames_seen}/{expected_frames} frames, {frames_seen / elapsed:.2f} FPS", flush=True)
                    detections_file.flush()
                    frames_file.flush()
                    write_progress(run_dir, "tracking", video_id=video_path.stem, frames_processed=frames_seen,
                                   expected_frames=expected_frames, car_detection_rows=car_rows,
                                   elapsed_wall_s=round(elapsed, 3), end_to_end_fps=round(frames_seen / elapsed, 3))

            cap.release()
            metadata_mismatch_note = ""
            if frames_seen != expected_frames:
                # Phone containers often over-report the last frame; sequential
                # end-of-stream minus exactly one frame is accepted and recorded.
                if frames_seen == expected_frames - 1:
                    metadata_mismatch_note = (
                        "Container metadata reported one extra frame; all sequentially "
                        "decodable frames were processed (frames_seen < metadata count).")
                else:
                    write_progress(run_dir, "failed", video_id=video_path.stem, frames_processed=frames_seen,
                                   expected_frames=expected_frames, error="Incomplete sequential video decode; no complete run published")
                    raise RuntimeError(f"Incomplete decode of {video_path.name}: {frames_seen}/{expected_frames} frames")
            elapsed_s = time.perf_counter() - video_start
            preview_path = preview_dir / f"{video_path.stem}_tracks_contact.jpg"
            write_contact_sheet(previews, preview_path)
            summary = {
                "video_id": video_path.stem,
                "source": workspace_path(video_path),
                "source_sha256": sha256_file(video_path),
                "source_width": width,
                "source_height": height,
                "source_fps": fps,
                "metadata_frames": expected_frames,
                "frames_processed": frames_seen,
                "car_detection_rows": car_rows,
                "frames_without_detections": no_detection_frames,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "tracker": args.tracker,
                "device": "cpu",
                "model_load_s": round(model_load_s, 3),
                "elapsed_wall_s": round(elapsed_s, 3),
                "end_to_end_fps": round(frames_seen / elapsed_s, 3) if elapsed_s else 0.0,
                "inference_fps": round(frames_seen / sum(inference_times), 3) if inference_times else 0.0,
                "count_matches_metadata": frames_seen == expected_frames,
                "metadata_frame_count_note": metadata_mismatch_note,
                "frame_index_basis": "zero_based_sequential_decode",
                "timestamp_source_counts": dict(timestamp_sources),
                "track_contact_sheet": workspace_path(preview_path),
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)

    crosswalk_path = run_dir / "track_vehicle_crosswalk_template.csv"
    with crosswalk_path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=TRACK_FIELDS)
        writer.writeheader()
        for (video_id, track_id), stats in sorted(track_stats.items()):
            writer.writerow(
                {
                    "video_id": video_id,
                    "pass_id": "",
                    "track_id": track_id,
                    "first_frame_idx": stats["first_frame_idx"],
                    "first_timestamp_s": f"{float(stats['first_timestamp_s']):.6f}",
                    "last_frame_idx": stats["last_frame_idx"],
                    "last_timestamp_s": f"{float(stats['last_timestamp_s']):.6f}",
                    "detection_rows": stats["detection_rows"],
                    "subject_id": "",
                    "notes": "Manual match required; track IDs are local to this video/run and may switch. Column name 'subject_id' holds the vehicle ID for compatibility with the person-pipeline crosswalk scripts.",
                }
            )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": weights_path.name,
        "weights_path": workspace_path(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": str(torch.__version__),
        "opencv_version": cv2.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "device": "cpu",
        "torch_threads": args.threads,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "classes": {"2": "car"},
        "tracker": args.tracker,
        "tracker_sha256": sha256_file(tracker_candidate) if tracker_candidate.is_file() else None,
        "subject_ids": {},
        "note": "Vehicle IDs intentionally left blank in detections until track-to-vehicle matches are manually confirmed (calibration take). Test takes may keep blank vehicle IDs and rely on track_id. Track IDs restart per video.",
        "videos": summaries,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "README.md").write_text(
        f"# Full-video car tracking ({weights_path.name})\n\n"
        "YOLOv8n COCO car (class id 2) detections with ByteTrack on each source MP4, at the original frame scale and configured inference size. Trucks/buses are excluded by class filter.\n\n"
        "`detections.csv` contains one row per detected car with source-pixel box geometry. `frames.csv` contains every processed frame, including no-detection frames. `track_vehicle_crosswalk_template.csv` is intentionally unassigned pending manual confirmation for the calibration take. Track IDs restart for each video and can switch; they are not vehicle identities.\n\n"
        "Keep weights, imgsz, conf, tracker, device, and camera resolution identical between the calibration take and test takes.\n",
        encoding="utf-8",
    )
    print(f"Full-video car tracking complete: {run_dir}", flush=True)
    write_progress(run_dir, "complete", videos_processed=len(summaries),
                   frames_processed=sum(item["frames_processed"] for item in summaries),
                   car_detection_rows=sum(item["car_detection_rows"] for item in summaries),
                   mapping_status="awaiting_manual_vehicle_review", model_fitted=False)


if __name__ == "__main__":
    main()
