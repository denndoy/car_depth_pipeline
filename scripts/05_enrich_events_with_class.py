"""Attach the manual ground-truth body class to confirmed crossing event CSV(s).

`marker_line_tools.py export-confirmed` writes event CSVs (reviewed crossing
frames) but knows nothing about car body types. This helper joins
``vehicle_metadata.csv`` (vehicle_id -> body_class_gt) onto each event row and
writes a NEW CSV with a required ``body_class_gt`` column, without touching the
exported file.

Rules:
- Every event must have a non-empty ``subject_id`` that exists in the metadata.
- If the event CSV already has ``body_class_gt``, it must match the metadata
  (prevents silent label drift between calibration and oracle evaluation).
- Classes are canonicalized against configs/bodytype_config.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "car_depth_pipeline" / "configs" / "bodytype_config.json"


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        return list(reader), fields


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, nargs="+", required=True,
                        help="Exported event CSV(s) from marker_line_tools.py export-confirmed")
    parser.add_argument("--vehicle-metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory (never overwritten)")
    args = parser.parse_args(argv)

    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    body_classes = [str(name) for name in config["body_classes"]]
    by_casefold = {name.casefold(): name for name in body_classes}

    metadata_path = resolve(args.vehicle_metadata)
    metadata_rows, metadata_fields = read_rows(metadata_path)
    if not {"vehicle_id", "body_class_gt"}.issubset(set(metadata_fields)):
        raise ValueError(f"{metadata_path} must contain vehicle_id and body_class_gt columns")
    class_by_vehicle: dict[str, str] = {}
    for row in metadata_rows:
        vehicle_id = (row.get("vehicle_id") or "").strip()
        raw_class = (row.get("body_class_gt") or "").strip()
        canonical = by_casefold.get(raw_class.casefold())
        if not vehicle_id or canonical is None:
            raise ValueError(f"{metadata_path}: invalid vehicle row {vehicle_id!r}/{raw_class!r}")
        if vehicle_id in class_by_vehicle:
            raise ValueError(f"{metadata_path}: duplicate vehicle_id {vehicle_id!r}")
        class_by_vehicle[vehicle_id] = canonical

    output_dir = resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_event_ids: set[str] = set()
    for events_path in (resolve(path) for path in args.events):
        rows, fields = read_rows(events_path)
        if not {"event_id", "subject_id"}.issubset(set(fields)):
            raise ValueError(f"{events_path} must contain event_id and subject_id")
        for line_number, row in enumerate(rows, start=2):
            event_id = (row.get("event_id") or "").strip()
            subject_id = (row.get("subject_id") or "").strip()
            if not event_id or event_id in seen_event_ids:
                raise ValueError(f"{events_path}:{line_number}: empty/duplicate event_id {event_id!r}")
            seen_event_ids.add(event_id)
            expected = class_by_vehicle.get(subject_id)
            if expected is None:
                raise ValueError(
                    f"{events_path}:{line_number}: subject_id {subject_id!r} is not in {metadata_path.name}"
                )
            existing = (row.get("body_class_gt") or "").strip()
            if existing and by_casefold.get(existing.casefold()) != expected:
                raise ValueError(
                    f"{events_path}:{line_number}: body_class_gt {existing!r} conflicts with metadata {expected!r}"
                )
            row["body_class_gt"] = expected

        output_fields = fields if "body_class_gt" in fields else fields + ["body_class_gt"]
        output_path = output_dir / events_path.name
        if output_path.exists():
            raise FileExistsError(f"Output file already exists: {output_path}")
        with output_path.open("w", newline="", encoding="utf-8-sig") as target:
            writer = csv.DictWriter(target, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {output_path} ({len(rows)} events, body_class_gt filled from {metadata_path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
