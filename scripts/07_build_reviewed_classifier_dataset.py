"""Build an Ultralytics classify dataset from manually reviewed car crops.

Input is the CSV downloaded from scripts/06_prepare_blind_crop_review.py.
Only explicit manual labels MPV/Sedan/Hatchback are copied. Uncertain/Skip
rows are excluded; blank labels are rejected. The original train/val/test split
is retained exactly and source-video overlap across splits is rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("MPV", "Sedan", "Hatchback")
SKIP_LABELS = {"uncertain", "skip", "ignore", "unknown", "other", "invalid", "reject", "not car", "not_car"}
SPLITS = {"train", "val", "test"}


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if (Path.cwd().resolve() == ROOT.parent.resolve()
            and path.parts and path.parts[0].casefold() == ROOT.name.casefold()):
        return (Path.cwd() / path).resolve()
    return (ROOT / path).resolve()


def canonical_label(raw: str) -> str | None:
    normalized = (raw or "").strip().casefold()
    if normalized in SKIP_LABELS:
        return None
    for name in CLASSES:
        if normalized == name.casefold():
            return name
    raise ValueError(f"Invalid corrected_label {raw!r}; use MPV, Sedan, Hatchback, Uncertain, or Skip")


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
    parser.add_argument("--review-dir", type=Path, required=True, help="Folder produced by 06_prepare_blind_crop_review.py")
    parser.add_argument("--reviewed-csv", type=Path, required=True, help="Downloaded car_bodytype_reviewed.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="New Ultralytics classify dataset directory")
    parser.add_argument("--video-split-map", type=Path, default=None,
                        help="CSV source_video_id,split with research roles train/val/test/exclude; recommended")
    parser.add_argument("--confirm-detection-source-split", action="store_true",
                        help="Explicitly use the ZIP's original detection train/val split as classifier split")
    parser.add_argument("--allow-video-overlap", action="store_true", help="Override the train/val/test source-video leakage guard (not recommended)")
    args = parser.parse_args(argv)

    review_dir = resolve_path(args.review_dir)
    reviewed_csv = resolve_path(args.reviewed_csv)
    out_dir = resolve_path(args.out_dir)
    split_map_path = resolve_path(args.video_split_map) if args.video_split_map is not None else None
    if not review_dir.is_dir():
        raise FileNotFoundError(review_dir)
    if not reviewed_csv.is_file():
        raise FileNotFoundError(reviewed_csv)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty dataset directory: {out_dir}")
    if split_map_path is None and not args.confirm_detection_source_split:
        raise ValueError(
            "Choose the research-safe role for each source video with --video-split-map. "
            "The ZIP's train/val assignment was made for car detection and may leak classifier test videos. "
            "Use --confirm-detection-source-split only after verifying those roles are valid for this experiment."
        )
    video_split_map = read_video_split_map(split_map_path) if split_map_path is not None else None

    with reviewed_csv.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {
            "crop_id", "source_split", "source_video_id", "crop_path", "corrected_label", "notes",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{reviewed_csv} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Review CSV is empty")

    seen_ids: set[str] = set()
    counts: dict[str, Counter] = defaultdict(Counter)
    videos_by_split: dict[str, set[str]] = defaultdict(set)
    manifest: list[dict] = []
    validated_crops: list[tuple[Path, str, str, str]] = []
    blank_labels: list[str] = []
    ignored_rows = 0
    for line_no, row in enumerate(rows, start=2):
        crop_id = (row.get("crop_id") or "").strip()
        split = (row.get("source_split") or "").strip().lower()
        video_id = (row.get("source_video_id") or "").strip()
        if not crop_id or crop_id in seen_ids:
            raise ValueError(f"{reviewed_csv}:{line_no}: empty or duplicate crop_id {crop_id!r}")
        seen_ids.add(crop_id)
        if split not in SPLITS:
            raise ValueError(f"{reviewed_csv}:{line_no}: unsupported source_split {split!r}")
        if not video_id:
            raise ValueError(f"{reviewed_csv}:{line_no}: missing source_video_id")
        detection_source_split = split
        if video_split_map is not None:
            if video_id not in video_split_map:
                raise ValueError(f"{reviewed_csv}:{line_no}: {video_id!r} is absent from the video split map")
            split = video_split_map[video_id]
            if split == "exclude":
                ignored_rows += 1
                continue
        raw_label = (row.get("corrected_label") or "").strip()
        if not raw_label:
            blank_labels.append(crop_id)
            continue
        label = canonical_label(raw_label)
        if label is None:
            ignored_rows += 1
            continue
        relative_crop = Path(row["crop_path"])
        crop_path = (review_dir / relative_crop).resolve()
        try:
            crop_path.relative_to(review_dir.resolve())
        except ValueError:
            raise ValueError(f"Crop path escapes review directory at row {line_no}: {relative_crop}") from None
        if not crop_path.is_file():
            raise FileNotFoundError(f"Missing crop for {crop_id}: {crop_path}")
        validated_crops.append((crop_path, split, label, crop_id))
        counts[split][label] += 1
        videos_by_split[split].add(video_id)
        manifest.append({
            "crop_id": crop_id,
            "split": split,
            "detection_source_split": detection_source_split,
            "body_class": label,
            "source_video_id": video_id,
            "source_frame": row.get("source_frame", ""),
            "bbox_index": row.get("bbox_index", ""),
            "crop_path": row["crop_path"],
            "manual_notes": row.get("notes", ""),
            "human_verified": "true",
        })

    if blank_labels:
        raise ValueError(
            f"{len(blank_labels)} crops have blank corrected_label (first: {blank_labels[:5]}). "
            "Finish reviewing every crop or explicitly mark unusable samples Uncertain/Skip."
        )
    if not manifest:
        raise ValueError("No manually labeled crops remain after excluding Uncertain/Skip")

    split_names = sorted(videos_by_split)
    overlaps = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = sorted(videos_by_split[left] & videos_by_split[right])
            if overlap:
                overlaps[f"{left}__{right}"] = overlap
    if overlaps and not args.allow_video_overlap:
        raise ValueError(
            "Source-video leakage across splits detected: " + json.dumps(overlaps) +
            ". Rebuild the source split by whole video; do not randomly split adjacent frames."
        )

    missing_train = [name for name in CLASSES if counts["train"][name] == 0]
    if missing_train:
        raise ValueError(f"No manually labeled training crops for classes {missing_train}")
    if not counts["val"]:
        raise ValueError("No manually labeled validation crops; source val split must contain reviewed samples")
    missing_val = [name for name in CLASSES if counts["val"][name] == 0]
    warnings = []
    if missing_val:
        warnings.append(
            f"Validation split has no labeled crop(s) for {missing_val}; per-class validation metrics cannot cover all classes."
        )
    if not counts.get("test"):
        warnings.append(
            "No independent test split is available. Use val for model selection only; do not report it as an untouched final test."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in sorted(set(counts) | set(videos_by_split)):
        for class_name in CLASSES:
            (out_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    for crop_path, split, label, crop_id in validated_crops:
        destination = out_dir / split / label / f"{crop_id}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(crop_path, destination)
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as target:
        fields = ["crop_id", "split", "detection_source_split", "body_class", "source_video_id", "source_frame", "bbox_index", "crop_path", "manual_notes", "human_verified"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    summary = {
        "schema": "manually_labeled_car_classification_dataset_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_review_dir": str(review_dir),
        "source_review_csv": str(reviewed_csv),
        "output_dataset": str(out_dir),
        "split_policy": "explicit research video split map" if video_split_map is not None else "explicitly confirmed original detection ZIP split",
        "classes": list(CLASSES),
        "counts_by_split": {split: {name: int(counts[split][name]) for name in CLASSES} for split in sorted(set(counts) | set(videos_by_split))},
        "counts_by_class_total": {name: int(sum(counts[split][name] for split in counts)) for name in CLASSES},
        "excluded_uncertain_skip_or_video_policy": ignored_rows,
        "source_videos_by_split": {split: sorted(videos) for split, videos in videos_by_split.items()},
        "source_video_overlap": overlaps,
        "warnings": warnings,
        "label_source": "manual human review; no classifier predictions used as ground truth",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
