"""Filter the Kaggle 8-class ZIP to MPV/Sedan/Hatchback and remove exact leakage.

The source ZIP is read-only. Output is a new Ultralytics classify dataset with
train/val/test/<Hatchback|MPV|Sedan> folders, a kept-image manifest, a duplicate
audit, and a cross-class label-conflict report.

Cleaning policy:
1. Hash every source image (all 8 classes) so a target image duplicated under
   crossover/van/etc. is treated as a class-label conflict, not silently kept.
2. Exclude every target image in a hash group whose source class labels differ.
3. For same-label exact duplicates, keep one copy only. If copies occur across
   splits, keep the highest-priority split train > val > test to prevent exact
   image leakage into validation/test. Within a split, keep the lexicographically
   first source path.

Only exact byte-identical images are detected. Near-duplicates / same-car
different-frame images and vehicle identity are not inferred by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "bodytype_config.json"
DEFAULT_SOURCE = ROOT / "dataset-classification.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    # Accept both paths relative to the car_depth_pipeline folder and paths
    # prefixed with car_depth_pipeline when called from the parent project root.
    if (Path.cwd().resolve() == ROOT.parent.resolve()
            and path.parts and path.parts[0].casefold() == ROOT.name.casefold()):
        return (Path.cwd() / path).resolve()
    return (ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_classes(path: Path) -> tuple[list[str], dict[str, str]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    canonical = [str(item) for item in config["body_classes"]]
    return canonical, {name.casefold(): name for name in canonical}


def split_and_class(path: str) -> tuple[str | None, str]:
    parts = PurePosixPath(path).parts
    split = next((part.casefold() for part in parts if part.casefold() in SPLIT_PRIORITY), None)
    if split is None or len(parts) < 2:
        return split, ""
    return split, parts[-2].casefold()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True, help="New clean dataset folder; existing non-empty folder is never overwritten")
    args = parser.parse_args(argv)

    source_zip = resolve_path(args.source_zip)
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    classes, canonical_by_lower = load_classes(config_path)
    all_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_counts: Counter = Counter()
    total_images = 0
    total_image_bytes = 0

    with zipfile.ZipFile(source_zip, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Source ZIP CRC check failed at {bad_member}")
        for name in archive.namelist():
            if name.endswith("/") or PurePosixPath(name).suffix.casefold() not in IMAGE_EXTENSIONS:
                continue
            split, source_class = split_and_class(name)
            if split not in SPLIT_PRIORITY:
                continue
            data = archive.read(name)
            digest = hashlib.sha256(data).hexdigest()
            size = len(data)
            total_images += 1
            total_image_bytes += size
            raw_counts[(split, source_class)] += 1
            all_groups[digest].append({
                "sha256": digest,
                "source_split": split,
                "source_class": source_class,
                "source_path": name,
                "byte_size": size,
            })

    kept: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    conflict_groups: list[dict[str, Any]] = []
    conflict_file_count = 0
    same_label_duplicate_copies = 0
    target_exact_duplicate_groups = 0
    target_cross_split_duplicate_groups = 0
    clean_counts: Counter = Counter()

    for digest, entries in sorted(all_groups.items()):
        source_classes = sorted({entry["source_class"] for entry in entries})
        target_entries = [entry for entry in entries if entry["source_class"] in canonical_by_lower]
        if not target_entries:
            continue
        if len(entries) > 1:
            target_exact_duplicate_groups += 1
        if len({entry["source_split"] for entry in target_entries}) > 1:
            target_cross_split_duplicate_groups += 1

        if len(source_classes) > 1:
            conflict_file_count += len(target_entries)
            conflict_groups.append({
                "sha256": digest,
                "source_classes": ";".join(source_classes),
                "target_classes": ";".join(sorted({entry["source_class"] for entry in target_entries})),
                "target_file_count": len(target_entries),
                "all_source_paths_json": json.dumps([entry["source_path"] for entry in entries], ensure_ascii=False),
                "target_source_paths_json": json.dumps([entry["source_path"] for entry in target_entries], ensure_ascii=False),
            })
            for entry in target_entries:
                removed_rows.append({
                    **entry,
                    "action": "excluded_cross_class_label_conflict",
                    "kept_source_path": "",
                    "conflicting_source_classes": ";".join(source_classes),
                })
            duplicate_groups.append({
                "sha256": digest,
                "source_classes": ";".join(source_classes),
                "source_splits": ";".join(sorted({entry["source_split"] for entry in entries})),
                "files_in_zip": len(entries),
                "target_files": len(target_entries),
                "resolution": "excluded_target_copies_due_to_class_conflict",
                "kept_source_path": "",
                "source_paths_json": json.dumps([entry["source_path"] for entry in entries], ensure_ascii=False),
            })
            continue

        # All byte-identical entries have the same one of the selected classes.
        winner = min(
            target_entries,
            key=lambda entry: (SPLIT_PRIORITY[entry["source_split"]], entry["source_path"].casefold()),
        )
        canonical_class = canonical_by_lower[winner["source_class"]]
        kept_record = {
            **winner,
            "body_class": canonical_class,
            "output_path": f"{winner['source_split']}/{canonical_class}/{digest[:16]}{PurePosixPath(winner['source_path']).suffix.lower()}",
        }
        kept.append(kept_record)
        clean_counts[(winner["source_split"], canonical_class)] += 1

        if len(target_entries) > 1:
            same_label_duplicate_copies += len(target_entries) - 1
            duplicate_groups.append({
                "sha256": digest,
                "source_classes": ";".join(source_classes),
                "source_splits": ";".join(sorted({entry["source_split"] for entry in target_entries})),
                "files_in_zip": len(entries),
                "target_files": len(target_entries),
                "resolution": f"kept_one_in_{winner['source_split']}_split",
                "kept_source_path": winner["source_path"],
                "source_paths_json": json.dumps([entry["source_path"] for entry in target_entries], ensure_ascii=False),
            })
            for entry in target_entries:
                if entry["source_path"] == winner["source_path"]:
                    continue
                removed_rows.append({
                    **entry,
                    "action": "removed_exact_duplicate",
                    "kept_source_path": winner["source_path"],
                    "conflicting_source_classes": "",
                })

    if not kept:
        raise ValueError("No target-class images remain after exact-hash and label-conflict filtering")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_PRIORITY:
        for class_name in classes:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source_zip, "r") as archive:
        for record in kept:
            destination = output_dir / Path(record["output_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(record["source_path"]))

    manifest_rows = [
        {
            "split": row["source_split"],
            "body_class": row["body_class"],
            "source_class": row["source_class"],
            "source_path": row["source_path"],
            "output_path": row["output_path"],
            "sha256": row["sha256"],
            "bytes": row["byte_size"],
            "dedup_policy": "unique exact image hash across all source classes and splits",
        }
        for row in kept
    ]
    write_csv(output_dir / "manifest.csv", manifest_rows,
              ["split", "body_class", "source_class", "source_path", "output_path", "sha256", "bytes", "dedup_policy"])
    write_csv(output_dir / "duplicate_audit.csv", duplicate_groups,
              ["sha256", "source_classes", "source_splits", "files_in_zip", "target_files", "resolution", "kept_source_path", "source_paths_json"])
    write_csv(output_dir / "removed_or_conflicted_files.csv", removed_rows,
              ["sha256", "source_split", "source_class", "source_path", "byte_size", "action", "kept_source_path", "conflicting_source_classes"])
    write_csv(output_dir / "cross_class_conflicts.csv", conflict_groups,
              ["sha256", "source_classes", "target_classes", "target_file_count", "all_source_paths_json", "target_source_paths_json"])

    source_sha = file_sha256(source_zip)
    summary = {
        "schema": "deduplicated_car_bodytype_classification_dataset_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_zip": str(source_zip),
        "source_zip_sha256": source_sha,
        "source_zip_modified": False,
        "output_dir": str(output_dir),
        "source_image_count_all_classes": total_images,
        "source_image_bytes_scanned": total_image_bytes,
        "target_classes": classes,
        "target_raw_counts_by_split": {
            f"{split}/{source_class}": count
            for (split, source_class), count in sorted(raw_counts.items())
            if source_class in canonical_by_lower
        },
        "clean_counts_by_split": {
            f"{split}/{class_name}": int(clean_counts[(split, class_name)])
            for split in SPLIT_PRIORITY for class_name in classes
        },
        "clean_total": len(kept),
        "target_exact_duplicate_groups": target_exact_duplicate_groups,
        "target_groups_with_duplicates_across_splits": target_cross_split_duplicate_groups,
        "same_label_duplicate_copies_removed": same_label_duplicate_copies,
        "cross_class_exact_hash_conflict_groups": len(conflict_groups),
        "target_files_excluded_due_to_cross_class_conflict": conflict_file_count,
        "cleaning_policy": {
            "cross_class_conflict": "exclude all target-class copies when identical bytes occur under more than one of the source's eight class labels",
            "same_class_duplicate": "keep one exact hash globally, preferring train, then val, then test; tie-break by source path",
            "near_duplicates": "not detected; this audit is exact SHA-256 only",
        },
        "warnings": [
            "Image-level exact de-duplication does not guarantee vehicle-level separation; inspect same-vehicle/near-duplicate images before making generalization claims.",
            "The provided source splits are preserved except exact hashes moved to the highest-priority split. The test split remains untouched by training when used as test.",
            "Cross-class conflict groups were excluded; review cross_class_conflicts.csv if class semantics require manual adjudication.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Clean 3-class Indonesian car classifier dataset\n\n"
        "Filtered from the source 8-class Kaggle ZIP to MPV/Sedan/Hatchback. The source ZIP was not changed.\n\n"
        "- `train/`, `val/`, `test/`: Ultralytics classify folder layout.\n"
        "- `manifest.csv`: each retained image and its source path/hash.\n"
        "- `duplicate_audit.csv`, `removed_or_conflicted_files.csv`: exact duplicate resolution.\n"
        "- `cross_class_conflicts.csv`: identical bytes assigned to different source labels; all target copies excluded.\n\n"
        "Exact-hash cleaning does not detect near-duplicates or the same vehicle photographed differently. Review this limitation before reporting test metrics.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
