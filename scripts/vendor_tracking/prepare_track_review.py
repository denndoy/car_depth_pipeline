"""Offline, operator-confirmed track review; never infer subject identity or fit distances."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    "tracking_run_id", "detections_sha256", "video_id", "track_id", "start_frame_idx",
    "end_frame_idx", "subject_id", "pass_id", "mapping_status", "notes",
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def shown(path: Path) -> str:
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


def read_rows(path: Path, required=()) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing required columns {required}")
        return [{key: (value or "").strip() for key, value in row.items() if key is not None} for row in reader]


def write_rows(path: Path, rows, fields=FIELDS) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def frame_number(value) -> int:
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError("Frame/track index must be a nonnegative integer")
    return int(number)


def collect_tracks(rows: list[dict[str, str]], video_id: str):
    grouped = defaultdict(list)
    untracked = 0
    for row in rows:
        if row["video_id"] != video_id:
            raise ValueError("Review one selected video at a time; do not mix calibration and test takes")
        if not row.get("track_id"):
            untracked += 1
            continue
        track = str(frame_number(row["track_id"]))
        frame_number(row["frame_idx"])
        if not math.isfinite(float(row["A_bb_px2"])) or float(row["A_bb_px2"]) <= 0:
            raise ValueError("Invalid source bounding-box area")
        grouped[track].append(row)
    result = []
    for track, items in sorted(grouped.items(), key=lambda pair: int(pair[0])):
        items.sort(key=lambda row: int(row["frame_idx"]))
        if len({int(row["frame_idx"]) for row in items}) != len(items):
            raise ValueError(f"Duplicate source boxes for track {track} at one frame")
        best = max(items, key=lambda row: float(row["A_bb_px2"]) * float(row["confidence"]))
        samples = {int(row["frame_idx"]): row for row in (items[0], items[len(items) // 2], best, items[-1])}
        result.append({"video_id": video_id, "track_id": track, "first_frame": int(items[0]["frame_idx"]),
                       "last_frame": int(items[-1]["frame_idx"]), "first_time": float(items[0]["timestamp_s"]),
                       "last_time": float(items[-1]["timestamp_s"]), "detection_rows": len(items),
                       "samples": [samples[index] for index in sorted(samples)]})
    if not result:
        raise ValueError("No tracked detections to review; no identities were guessed")
    return result, untracked


def jpeg_data(image, width: int) -> str:
    if image.shape[1] > width:
        image = cv2.resize(image, (width, max(1, round(image.shape[0] * width / image.shape[1]))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Cannot encode review image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def source_crop(frame, row):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(row[key]) for key in ("x1_px", "y1_px", "x2_px", "y2_px"))
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid source-pixel box for review")
    pad = max(12, round(max(x2 - x1, y2 - y1) * .12))
    left, top = max(0, math.floor(x1) - pad), max(0, math.floor(y1) - pad)
    right, bottom = min(width, math.ceil(x2) + pad), min(height, math.ceil(y2) + pad)
    crop = frame[top:bottom, left:right].copy()
    if not crop.size:
        raise ValueError("Source box is outside the video frame")
    cv2.rectangle(crop, (round(x1) - left, round(y1) - top), (round(x2) - left, round(y2) - top), (0, 200, 255), 2)
    return crop


def read_subjects(path: Path):
    rows = read_rows(path, ("subject_id", "label"))
    if not rows or len({row["subject_id"] for row in rows}) != len(rows) or any(not row["subject_id"] for row in rows):
        raise ValueError("Subjects need unique nonempty anonymous IDs")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--html-name", default="REVIEW_TRACK_TAKE1_V2.html", help="Distinct local HTML filename, e.g. REVIEW_TRACK_TAKE1_CUT_V2.html")
    parser.add_argument("--download-name", default="take1_track_mapping_reviewed.csv", help="Distinct CSV filename for the selected source/run")
    parser.add_argument("--preview-video", action="store_true", help="Create a local 960px H.264 display preview; source and box areas unchanged")
    args = parser.parse_args(argv)
    for value, suffix in ((args.html_name, ".html"), (args.download_name, ".csv")):
        if Path(value).name != value or not value.lower().endswith(suffix):
            raise ValueError(f"Review filename must be a local {suffix} basename")
    run, video, subjects_path, out = map(resolve, (args.run_dir, args.video, args.subjects, args.output_dir))
    detections_path = run / "detections.csv"
    run_summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    videos = run_summary.get("videos", [])
    metadata = next((item for item in videos if item["video_id"] == video.stem), None)
    if metadata is None or metadata["frames_processed"] != metadata["metadata_frames"]:
        raise ValueError("Only complete tracking of the selected video can be reviewed")
    source_hash = sha256_file(video)
    if metadata.get("source_sha256") and source_hash != metadata["source_sha256"]:
        raise ValueError("Source video changed after tracking")
    subjects = read_subjects(subjects_path)
    raw = read_rows(detections_path, ("video_id", "frame_idx", "timestamp_s", "track_id", "A_bb_px2", "confidence", "x1_px", "y1_px", "x2_px", "y2_px"))
    tracks, untracked = collect_tracks(raw, video.stem)
    detection_hash = sha256_file(detections_path)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite review folder: {out}")
    out.mkdir(parents=True)
    selected = defaultdict(list)
    for track in tracks:
        for row in track["samples"]:
            selected[int(row["frame_idx"])].append((track, row))
        track["review_images"] = []
    contexts = []
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open {video}")
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        index, rendered = 0, set()
        while index <= max(selected):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Sequential decoding ended before selected review frame {max(selected)}")
            if index in selected:
                context = cv2.resize(frame, (1280, round(height * 1280 / width)))
                for track, row in selected[index]:
                    x1, y1, x2, y2 = (float(row[key]) for key in ("x1_px", "y1_px", "x2_px", "y2_px"))
                    scale = 1280 / width
                    cv2.rectangle(context, (round(x1 * scale), round(y1 * scale)), (round(x2 * scale), round(y2 * scale)), (0, 200, 255), 2)
                    cv2.putText(context, "T" + track["track_id"], (round(x1 * scale), max(15, round(y1 * scale) - 4)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 200, 255), 1, cv2.LINE_AA)
                context_index = len(contexts)
                contexts.append(jpeg_data(context, 1280))
                for track, row in selected[index]:
                    track["review_images"].append({"frame_idx": index, "timestamp_s": float(row["timestamp_s"]),
                                                   "source_width_px": float(row["x2_px"]) - float(row["x1_px"]),
                                                   "source_height_px": float(row["y2_px"]) - float(row["y1_px"]),
                                                   "confidence": float(row["confidence"]), "crop": jpeg_data(source_crop(frame, row), 420),
                                                   "context_index": context_index})
                rendered.add(index)
            index += 1
        if rendered != set(selected):
            raise ValueError("Some requested review frames were not decoded")
    finally:
        capture.release()
    for track in tracks:
        del track["samples"]
    if args.preview_video:
        import imageio_ffmpeg
        with (out / "preview.encode.log").open("wb") as log:
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-n", "-i", str(video), "-vf", "scale=960:-2", "-an",
                            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24", "-movflags", "+faststart", str(out / "preview.mp4")],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log, check=True)
    payload = {"run_id": run.name, "detections_sha256": detection_hash, "video_id": video.stem,
               "source_video": shown(video), "frame_count": metadata["frames_processed"], "source_size": [width, height],
               "model": run_summary["model"], "imgsz": run_summary["imgsz"], "conf": run_summary["conf"],
               "subjects": subjects, "tracks": tracks, "contexts": contexts, "fields": FIELDS,
               "untracked_rows": untracked, "preview_video": "preview.mp4" if args.preview_video else "",
               "download_name": args.download_name}
    template = Path(__file__).with_name("track_review.html").read_text(encoding="utf-8")
    html = template.replace("/*__TRACK_REVIEW_DATA__*/", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    target = out / args.html_name
    target.write_text(html, encoding="utf-8")
    draft = [{"tracking_run_id": run.name, "detections_sha256": detection_hash, "video_id": video.stem,
              "track_id": track["track_id"], "start_frame_idx": track["first_frame"], "end_frame_idx": track["last_frame"],
              "subject_id": "", "pass_id": "", "mapping_status": "", "notes": ""} for track in tracks]
    write_rows(out / "track_mapping_draft.csv", draft)
    report = {"status": "awaiting_manual_track_subject_mapping", "run_dir": shown(run), "detections_sha256": detection_hash,
              "source_video": shown(video), "source_video_sha256": source_hash, "html": shown(target),
              "tracks_to_review": len(tracks), "raw_rows_without_track_id": untracked, "sampled_source_frames": len(rendered),
              "sampling_basis": "zero_based_sequential_source_decode", "subject_mappings_confirmed": 0,
              "subjects_are_operator_metadata_not_model_predictions": True, "raw_detections_changed": False,
              "distance_model_fitted": False,
              "download_name": args.download_name,
              "note": "Track IDs may fragment or switch. Review images are display-only; source 4K box areas stay unchanged. Confirm identities/ranges/exclusions manually. Preview time seek is a review aid, not crossing-frame annotation. Keep crops and HTML private."}
    (out / "review_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Open locally: {shown(target)}")
    print(f"Review {len(tracks)} track segments, then download {args.download_name}. No automatic subject identity or GT was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
