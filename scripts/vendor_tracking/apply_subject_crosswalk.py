"""Apply confirmed manual track-to-subject mappings without editing raw detections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
def shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def frame_number(value: str, label: str) -> int:
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(number)


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mappings(path: Path, expected_source_hash: str | None = None) -> dict[tuple[str, str], list[dict]]:
    mapping = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"video_id", "track_id", "subject_id", "mapping_status"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Use a manually reviewed track-subject CSV, not a marker-point CSV")
        for row in reader:
            status = (row.get("mapping_status") or "").strip()
            if status in ("", "unreviewed"):
                continue
            if status not in ("confirmed_by_user", "excluded_by_user"):
                raise ValueError(f"Unsupported mapping_status: {status}")
            declared_hash = (row.get("detections_sha256") or "").strip()
            if declared_hash and expected_source_hash and declared_hash != expected_source_hash:
                raise ValueError("Reviewed CSV belongs to different raw detections; do not reuse IDs from another run")
            subject = (row.get("subject_id") or "").strip()
            if status == "confirmed_by_user" and not subject:
                raise ValueError("A confirmed mapping must have a subject_id")
            if status == "excluded_by_user" and subject:
                raise ValueError("An excluded track cannot also be a target subject")
            start, end = (row.get("start_frame_idx") or "").strip(), (row.get("end_frame_idx") or "").strip()
            if bool(start) != bool(end):
                raise ValueError("Fill both start_frame_idx and end_frame_idx, or leave both blank for a legacy whole-track mapping")
            lower = frame_number(start, "start_frame_idx") if start else 0
            upper = frame_number(end, "end_frame_idx") if end else None
            if upper is not None and upper < lower:
                raise ValueError("Mapping range end must not precede its start")
            video_id = (row.get("video_id") or "").strip()
            if not video_id:
                raise ValueError("A reviewed mapping needs video_id")
            key = (video_id, str(frame_number(row["track_id"], "track_id")))
            item = {"video_id": video_id, "track_id": key[1], "subject_id": subject, "mapping_status": status,
                    "start_frame_idx": lower, "end_frame_idx": upper, "pass_id": (row.get("pass_id") or "").strip()}
            ranges = mapping.setdefault(key, [])
            if item in ranges:
                continue
            for previous in ranges:
                if max(previous["start_frame_idx"], lower) <= min(previous["end_frame_idx"] if previous["end_frame_idx"] is not None else math.inf, upper if upper is not None else math.inf):
                    raise ValueError(f"Overlapping/conflicting reviewed ranges for {video_id}/track {key[1]}")
            ranges.append(item)
    if not mapping:
        raise ValueError("No manually confirmed/excluded mapping rows; raw detections remain unchanged")
    return mapping


def mapping_at(mapping: dict, video_id: str, track_id: str, frame_idx: int):
    if not track_id:
        return None
    key = (video_id, str(frame_number(track_id, "track_id")))
    for item in mapping.get(key, []):
        if item["start_frame_idx"] <= frame_idx and (item["end_frame_idx"] is None or frame_idx <= item["end_frame_idx"]):
            return item
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, default=None)
    parser.add_argument("--output-name", default="detections_subject_mapped.csv")
    args = parser.parse_args(argv)

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    crosswalk_path = args.crosswalk or (run_dir / "track_subject_crosswalk_confirmed.csv")
    if not crosswalk_path.is_absolute():
        crosswalk_path = ROOT / crosswalk_path
    source_path = run_dir / "detections.csv"
    output_path = run_dir / args.output_name
    if Path(args.output_name).name != args.output_name:
        raise ValueError("output-name must be a filename inside the run directory")
    summary_path = run_dir / "subject_mapping_summary.json"
    if not source_path.is_file() or not crosswalk_path.is_file():
        raise FileNotFoundError("Raw detection CSV or confirmed crosswalk is missing")
    if output_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing mapped output: {output_path}")

    detections_hash = source_hash(source_path)
    mapping = read_mappings(crosswalk_path, detections_hash)
    subject_counts = Counter()
    unmapped_counts = Counter()
    excluded_counts = Counter()
    seen_subject_frames = set()
    matched_ranges = Counter()
    output_rows = []
    with source_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        if not {"video_id", "track_id", "frame_idx", "subject_id"}.issubset(fields):
            raise ValueError("Raw detection CSV is missing video/track/frame/subject columns")
        if "subject_mapping_status" not in fields:
            fields.append("subject_mapping_status")
        if "pass_id" not in fields:
            fields.append("pass_id")
        for row in reader:
            track_id = row.get("track_id", "")
            index = frame_number(row["frame_idx"], "frame_idx")
            match = mapping_at(mapping, row["video_id"], track_id, index)
            if match:
                matched_ranges[id(match)] += 1
            subject_id = match["subject_id"] if match else ""
            row["subject_id"] = subject_id
            row.setdefault("pass_id", "")
            row["subject_mapping_status"] = match["mapping_status"] if match else "unmapped_track"
            if match and match["pass_id"]:
                row["pass_id"] = match["pass_id"]
            if subject_id:
                key = (row["video_id"], index, subject_id)
                if key in seen_subject_frames:
                    raise ValueError(f"Multiple boxes assigned to {subject_id} at {row['video_id']} frame {index}; resolve the duplicate before mapping")
                seen_subject_frames.add(key)
                subject_counts[(row["video_id"], subject_id)] += 1
            elif match and match["mapping_status"] == "excluded_by_user":
                excluded_counts[row["video_id"]] += 1
            else:
                unmapped_counts[row["video_id"]] += 1
            output_rows.append(row)
    for ranges in mapping.values():
        for item in ranges:
            if not matched_ranges[id(item)]:
                raise ValueError(f"Reviewed range has no matching detection rows: {item['video_id']}/track {item['track_id']}; verify the run and frame range")
    with output_path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "source_detections": shown(source_path),
        "confirmed_crosswalk": shown(crosswalk_path),
        "mapped_detections": shown(output_path),
        "source_detections_sha256": detections_hash,
        "confirmed_mapping_ranges": [item for key in sorted(mapping) for item in mapping[key]],
        "mapped_detection_rows_by_video_subject": {f"{video}/{subject}": count for (video, subject), count in sorted(subject_counts.items())},
        "unmapped_detection_rows_by_video": dict(unmapped_counts),
        "excluded_detection_rows_by_video": dict(excluded_counts),
        "note": "Raw detections.csv was not modified. Unmapped tracks remain blank for manual review.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
