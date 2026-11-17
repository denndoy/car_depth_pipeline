"""Local annotation, virtual-line previews and manually reviewed crossing events.

This helper does not estimate distances or fit models. Ground truth remains the
longitudinal marker table; the quartic area model is unchanged. For cars
approaching the camera, crossing suggestions use the bottom-center of the car
box as a front-bumper proxy and NEVER become confirmed ground truth without
human review.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from polynomial_take_pipeline import EXPECTED_MARKER_IDS, read_marker_ground_truth, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GT = ROOT / "inputs/marker_ground_truth.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
POINT_FIELDS = (
    "marker_id", "reference_path", "reference_frame_idx", "x_left_px", "y_left_px",
    "x_right_px", "y_right_px", "position_confirmed", "notes", "annotation_method",
)
ANNOTATION_SOURCES = {
    "cone-base": "manual_cone_ground_points_review",
    "estimated-line": "manual_estimated_crossing_line",
}
ESTIMATED_LINE_WARNING = (
    "Visually estimated transverse line, not two measured cone-base points. "
    "Crossing-line position/orientation uncertainty must be reported."
)
PROFILE_FIELDS = (
    "setup_id", "marker_id", "gt_distance_m", "x_left_px", "y_left_px", "x_right_px", "y_right_px",
    "source_width", "source_height", "lane_width_m", "reference_path", "reference_frame_idx",
    "reference_sha256", "position_confirmed", "annotation_source", "notes",
)
EVENT_FIELDS = (
    "event_id", "take_id", "video_id", "pass_id", "subject_id", "group_id", "activity", "lane",
    "marker_id", "frame_idx", "timestamp_s", "track_id", "gt_distance_m", "foot_crossing_confirmed",
    "annotation_source", "notes", "setup_id", "marker_profile_sha256",
)
CANDIDATE_FIELDS = (
    "event_id", "take_id", "video_id", "pass_id", "subject_id", "marker_id", "gt_distance_m",
    "vehicle_identity_source",
    "candidate_frame_idx", "candidate_timestamp_s", "candidate_track_id", "bumper_proxy_x_px",
    "bumper_proxy_y_px", "signed_distance_px", "segment_fraction", "bracket_start_frame",
    "bracket_end_frame", "sign_change_count", "samples_in_segment", "candidate_status",
    "review_image", "setup_id", "marker_profile_sha256",
)


def resolve(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_csv(path: Path, required: tuple[str, ...] = ()) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return [{key: (value or "").strip() for key, value in row.items() if key is not None} for row in reader]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def new_dir(path: Path) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {label}: {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {label}: {value!r}")
    return result


def integer(value: Any, label: str) -> int:
    result = number(value, label)
    if result < 0 or not result.is_integer():
        raise ValueError(f"{label} must be a nonnegative integer, found {value!r}")
    return int(result)


def save_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode image {path}")
    encoded.tofile(str(path))


def open_video(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"Video has invalid dimensions/fps/frame count: {path}")
    return capture, {"width": width, "height": height, "fps": fps, "frame_count": frame_count}


def video_frame(capture, frame_idx: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Cannot decode video frame {frame_idx}")
    return frame


def reference_frame(row: dict[str, str]) -> np.ndarray:
    if not row.get("reference_path"):
        raise ValueError(f"Fill reference_path for {row['marker_id']} before annotating")
    path = resolve(row["reference_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        if row.get("reference_frame_idx"):
            raise ValueError(f"Leave reference_frame_idx blank for a still image ({path})")
        frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot decode image {path}")
        return frame
    index = integer(row.get("reference_frame_idx", ""), f"reference_frame_idx for {row['marker_id']}")
    capture, info = open_video(path)
    try:
        if index >= info["frame_count"]:
            raise ValueError(f"Reference frame {index} is outside {path}")
        return video_frame(capture, index)
    finally:
        capture.release()


def manifest_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path, ("marker_id", "reference_path", "reference_frame_idx"))
    markers = [row["marker_id"] for row in rows]
    if len(markers) != 9 or set(markers) != set(EXPECTED_MARKER_IDS):
        raise ValueError("Reference manifest must contain each of M01-M09 exactly once")
    return sorted(rows, key=lambda row: row["marker_id"])


def annotation_method(row: dict[str, Any]) -> str:
    # Older nine-column point CSVs retain the original cone-base interpretation.
    method = row.get("annotation_method", "").strip() or "cone-base"
    if method not in ANNOTATION_SOURCES:
        raise ValueError(f"Unsupported annotation_method for {row['marker_id']}: {method}")
    return method


def estimated_lines(profile: list[dict[str, Any]]) -> bool:
    return any(row["annotation_source"] == ANNOTATION_SOURCES["estimated-line"] for row in profile)


def checked_line(row: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
    coords = tuple(number(row.get(key), key) for key in ("x_left_px", "y_left_px", "x_right_px", "y_right_px"))
    x1, y1, x2, y2 = coords
    if not (0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height):
        raise ValueError(f"{row['marker_id']}: endpoint outside source frame {width}x{height}")
    if x2 <= x1:
        raise ValueError(f"{row['marker_id']}: left/right endpoints must have x_left_px < x_right_px")
    if math.hypot(x2 - x1, y2 - y1) < 1:
        raise ValueError(f"{row['marker_id']}: line segment is too short")
    return coords


def line_position(line: dict[str, Any], x: float, y: float) -> tuple[float, float]:
    """Signed perpendicular distance and projection fraction on the measured segment."""
    dx = float(line["x_right_px"]) - float(line["x_left_px"])
    dy = float(line["y_right_px"]) - float(line["y_left_px"])
    px, py = x - float(line["x_left_px"]), y - float(line["y_left_px"])
    length = math.hypot(dx, dy)
    return (dx * py - dy * px) / length, (dx * px + dy * py) / length**2


def load_profile(path: Path, gt: dict[str, float]) -> list[dict[str, Any]]:
    rows = read_csv(path, PROFILE_FIELDS)
    if len(rows) != 9 or {row["marker_id"] for row in rows} != set(EXPECTED_MARKER_IDS):
        raise ValueError("Profile must contain M01-M09 exactly once")
    sizes, setups, widths = set(), set(), set()
    for row in rows:
        marker = row["marker_id"]
        if marker not in gt or abs(number(row["gt_distance_m"], "GT") - gt[marker]) > 1e-6:
            raise ValueError(f"Profile GT does not match marker table for {marker}")
        if not row["setup_id"] or row["position_confirmed"].upper() != "TRUE":
            raise ValueError("Profile needs setup_id and manually confirmed endpoints")
        if not manual_source(row["annotation_source"]):
            raise ValueError("Profile annotation_source must indicate human/manual review")
        w = integer(row["source_width"], "source_width")
        h = integer(row["source_height"], "source_height")
        if w == 0 or h == 0:
            raise ValueError("Profile dimensions must be positive")
        sizes.add((w, h))
        setups.add(row["setup_id"])
        if row["lane_width_m"]:
            lane_width = number(row["lane_width_m"], "lane_width_m")
            if lane_width <= 0:
                raise ValueError("lane_width_m must be positive")
            widths.add(lane_width)
        else:
            widths.add(None)
        for key, value in zip(("x_left_px", "y_left_px", "x_right_px", "y_right_px"), checked_line(row, w, h)):
            row[key] = value
        row.update(gt_distance_m=gt[marker], source_width=w, source_height=h)
    if len(sizes) != 1 or len(setups) != 1 or len(widths) != 1:
        raise ValueError("A profile must describe one setup, source resolution and lane width")
    return sorted(rows, key=lambda row: row["marker_id"])


def manual_source(value: str) -> bool:
    value = value.strip().lower()
    return any(token in value for token in ("manual", "human", "user", "review")) and not any(
        token in value for token in ("auto", "candidate", "prelabel", "unverified", "unreviewed")
    )


def validate_view(profile: list[dict[str, Any]], info: dict[str, Any], same_view_confirmed: bool) -> None:
    if not same_view_confirmed:
        raise ValueError(
            "Confirm unchanged camera position/zoom/crop with --same-camera-view-confirmed. "
            "Matching resolution alone does not prove the same view."
        )
    if (profile[0]["source_width"], profile[0]["source_height"]) != (info["width"], info["height"]):
        raise ValueError("Profile/video source resolution mismatch; do not silently rescale a ground-truth profile")


def draw_lines(image: np.ndarray, profile: list[dict[str, Any]], scale: float = 1, active: str = "") -> None:
    for row in profile:
        p1 = (round(row["x_left_px"] * scale), round(row["y_left_px"] * scale))
        p2 = (round(row["x_right_px"] * scale), round(row["y_right_px"] * scale))
        color = (0, 255, 255) if row["marker_id"] == active else (100, 220, 100)
        cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
        cv2.circle(image, p1, 3, color, -1)
        cv2.circle(image, p2, 3, color, -1)
        label = f"{row['marker_id']} {row['gt_distance_m']:g}m"
        cv2.putText(image, label, (p1[0] + 3, max(14, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


class PointViewport:
    """Exact source/display mapping, including zoom/pan for small distant cones."""

    def __init__(self, source_width: int, source_height: int, width: int, height: int):
        self.source_width, self.source_height = source_width, source_height
        self.width, self.height = width, height
        self.fit()

    def fit(self) -> None:
        self.scale = min(self.width / self.source_width, self.height / self.source_height)
        self.origin_x = (self.source_width - self.width / self.scale) / 2
        self.origin_y = (self.source_height - self.height / self.scale) / 2

    def source_point(self, x: float, y: float) -> tuple[float, float]:
        return self.origin_x + x / self.scale, self.origin_y + y / self.scale

    def screen_point(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.origin_x) * self.scale, (y - self.origin_y) * self.scale

    def zoom(self, factor: float, x: float, y: float) -> None:
        anchor_x, anchor_y = self.source_point(x, y)
        fit_scale = min(self.width / self.source_width, self.height / self.source_height)
        self.scale = min(max(self.scale * factor, fit_scale), fit_scale * 32)
        self.origin_x, self.origin_y = anchor_x - x / self.scale, anchor_y - y / self.scale

    def render(self, frame: np.ndarray) -> np.ndarray:
        matrix = np.array([[self.scale, 0, -self.origin_x * self.scale], [0, self.scale, -self.origin_y * self.scale]])
        return cv2.warpAffine(frame, matrix, (self.width, self.height), flags=cv2.INTER_LINEAR)


def pick_points(frame: np.ndarray, marker: str, width: int, height: int, initial: list[tuple[float, float]]):
    window = "Marker cone ground points"
    viewport = PointViewport(frame.shape[1], frame.shape[0], width, height)
    points = initial.copy()
    cursor = [width / 2, height / 2]

    def mouse(event, x, y, flags, _):
        if y >= height:
            return
        cursor[:] = [x, y]
        if event == cv2.EVENT_MOUSEWHEEL:
            viewport.zoom(1.5 if flags > 0 else 1 / 1.5, x, y)
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            sx, sy = viewport.source_point(x, y)
            if 0 <= sx < frame.shape[1] and 0 <= sy < frame.shape[0]:
                points.append((sx, sy))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            panel = viewport.render(frame)
            screen_points = [tuple(round(v) for v in viewport.screen_point(*p)) for p in points]
            if len(screen_points) == 2:
                cv2.line(panel, screen_points[0], screen_points[1], (0, 255, 255), 2)
            for index, point in enumerate(screen_points):
                cv2.circle(panel, point, 5, (0, 255, 255), 2)
                cv2.putText(panel, str(index + 1), point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            footer = np.zeros((64, width, 3), dtype=np.uint8)
            cv2.putText(footer, f"{marker}: click 2 cone BASE points. ENTER confirm | U undo | R clear | S skip | Q stop", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(footer, "+/- or wheel: zoom at cursor | I/J/K/L: pan | SPACE: fit | left/right sorted on save", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow(window, np.vstack((panel, footer)))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return "stop", []
            if key == ord("s"):
                return "skip", []
            if key in (10, 13) and len(points) == 2:
                return "confirm", sorted(points, key=lambda p: p[0])
            if key == ord("u") and points:
                points.pop()
            if key == ord("r"):
                points.clear()
            if key in (ord("+"), ord("="), ord("-")):
                viewport.zoom(1.5 if key != ord("-") else 1 / 1.5, *cursor)
            if key == 32:
                viewport.fit()
            if key in (ord("i"), ord("j"), ord("k"), ord("l")):
                viewport.origin_x += (1 if key == ord("l") else -1 if key == ord("j") else 0) * width / viewport.scale / 8
                viewport.origin_y += (1 if key == ord("k") else -1 if key == ord("i") else 0) * height / viewport.scale / 8
    finally:
        cv2.destroyWindow(window)


def assemble_profile(rows: list[dict[str, str]], args) -> tuple[list[dict[str, Any]], np.ndarray]:
    gt = read_marker_ground_truth(resolve(args.marker_ground_truth))
    if set(gt) != set(EXPECTED_MARKER_IDS):
        raise ValueError("Marker GT table must contain exactly M01-M09")
    if not args.setup_id.strip():
        raise ValueError("setup_id cannot be blank")
    if args.lane_width_m is not None and (not math.isfinite(args.lane_width_m) or args.lane_width_m <= 0):
        raise ValueError("lane-width-m must be positive and finite")
    dimensions = None
    cache: dict[Path, str] = {}
    profile = []
    first_frame = None
    for row in rows:
        method = annotation_method(row)
        if method == "estimated-line" and not getattr(args, "allow_estimated_lines", False):
            raise ValueError("Estimated lines need explicit --allow-estimated-lines; they are not measured cone-base references")
        if row.get("position_confirmed", "").upper() != "TRUE":
            raise ValueError(f"{row['marker_id']}: manually confirm endpoints before building a profile")
        frame = reference_frame(row)
        h, w = frame.shape[:2]
        if dimensions is not None and dimensions != (w, h):
            raise ValueError("All reference frames must share the same source resolution and camera view")
        dimensions = (w, h)
        if first_frame is None:
            first_frame = frame
        coords = checked_line(row, w, h)
        path = resolve(row["reference_path"])
        if path not in cache:
            cache[path] = sha256_file(path)
        profile.append({
            "setup_id": args.setup_id.strip(), "marker_id": row["marker_id"], "gt_distance_m": gt[row["marker_id"]],
            **dict(zip(("x_left_px", "y_left_px", "x_right_px", "y_right_px"), coords)),
            "source_width": w, "source_height": h, "lane_width_m": args.lane_width_m if args.lane_width_m is not None else "",
            "reference_path": shown(path), "reference_frame_idx": row.get("reference_frame_idx", ""),
            "reference_sha256": cache[path], "position_confirmed": "TRUE",
            "annotation_source": ANNOTATION_SOURCES[method], "notes": row.get("notes", ""),
        })
    return profile, first_frame


def save_profile(out_dir: Path, profile: list[dict[str, Any]], frame: np.ndarray, gt_path: Path) -> None:
    target = out_dir / "marker_lines_setup.csv"
    write_csv(target, profile, PROFILE_FIELDS)
    scale = min(1, 1280 / frame.shape[1])
    overview = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
    draw_lines(overview, profile, scale)
    save_image(out_dir / "profile_overview.jpg", overview)
    write_json(out_dir / "profile_summary.json", {
        "status": "manual_estimated_line_profile" if estimated_lines(profile) else "manual_profile_ready",
        "setup_id": profile[0]["setup_id"], "markers": list(EXPECTED_MARKER_IDS),
        "profile": shown(target), "profile_sha256": sha256_file(target), "marker_ground_truth": shown(gt_path),
        "annotation_sources": sorted({row["annotation_source"] for row in profile}),
        "contains_estimated_lines": estimated_lines(profile),
        "warnings": [ESTIMATED_LINE_WARNING] if estimated_lines(profile) else [],
        "camera_view_confirmed_by_operator": True, "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Coordinate reuse assumes unchanged camera/zoom/crop, planar straight crossing lines and negligible uncorrected distortion. This is not a distance estimator.",
    })


def write_browser_annotator(out_dir: Path, rows: list[dict[str, str]], args) -> None:
    """Bundle full-source frames into a self-contained, network-free annotation page."""
    template = Path(__file__).with_name("marker_line_annotator.html")
    frames, images, cache = [], [], {}
    for row in rows:
        key = (str(resolve(row["reference_path"])), row.get("reference_frame_idx", ""))
        if key not in cache:
            frame = reference_frame(row)
            ok, encoded = cv2.imencode(".png", frame)
            if not ok:
                raise RuntimeError(f"Cannot encode reference {row['marker_id']}")
            cache[key] = (len(images), frame.shape[1], frame.shape[0])
            images.append("data:image/png;base64," + base64.b64encode(encoded).decode("ascii"))
        image_index, width, height = cache[key]
        frames.append({"row": row, "width": width, "height": height, "image_index": image_index})
    methods = sorted({annotation_method(row) for row in rows})
    gt = read_marker_ground_truth(resolve(args.marker_ground_truth))
    view_confirmed = bool(getattr(args, "same_camera_view_confirmed", False))
    warnings = [ESTIMATED_LINE_WARNING] if "estimated-line" in methods else []
    if not view_confirmed:
        warnings.append("Camera-view alignment is not confirmed; this page is a reference-point draft, not a ready take profile.")
    payload = json.dumps({"setup_id": args.setup_id, "fields": POINT_FIELDS, "frames": frames,
                          "images": images, "marker_ground_truth_m": gt,
                          "camera_view_confirmed_by_operator": view_confirmed}, ensure_ascii=False).replace("<", "\\u003c")
    html = template.read_text(encoding="utf-8").replace("/*__REFERENCE_DATA__*/", payload)
    target = out_dir / "annotate_points.html"
    target.write_text(html, encoding="utf-8")
    write_json(out_dir / "annotation_summary.json", {
        "status": "awaiting_manual_browser_annotation", "setup_id": args.setup_id,
        "annotator": shown(target), "references": shown(resolve(args.references)),
        "annotation_methods": methods, "unique_embedded_frames": len(images),
        "camera_view_confirmed_by_operator": view_confirmed, "warnings": warnings,
        "note": "Open HTML locally, click/confirm endpoints and export the reviewed CSV. No ready profile or crossing GT is created here. Estimated lines retain a separate annotation method. No uploads or external assets; keep embedded frames private.",
    })
    print(f"Open locally in your browser: {shown(target)}")
    filename = "manual_lines_reviewed.csv" if "estimated-line" in methods else "points_reviewed.csv"
    print(f"Export {filename} after manually reviewing each marker. Estimated lines require explicit acceptance when building/using a profile.")


def command_profile(args) -> None:
    browser_draft = args.command == "annotate" and args.interface == "browser"
    if not args.same_camera_view_confirmed and not browser_draft:
        raise ValueError("Use --same-camera-view-confirmed only after verifying the reference camera was unchanged")
    if not args.setup_id.strip():
        raise ValueError("setup_id cannot be blank")
    if args.lane_width_m is not None and (not math.isfinite(args.lane_width_m) or args.lane_width_m <= 0):
        raise ValueError("lane-width-m must be positive and finite")
    rows = manifest_rows(resolve(args.references))
    if args.command == "build-profile":
        profile, frame = assemble_profile(rows, args)
        out_dir = new_dir(args.output_dir)
    else:
        # Verify references/resolution before creating the GUI output folder.
        sizes, seen_references = set(), set()
        for row in rows:
            previous_method = annotation_method(row)
            method = args.annotation_method or previous_method
            if method != previous_method:
                row["position_confirmed"] = ""
            row["annotation_method"] = method
            if method == "estimated-line" and args.interface != "browser":
                raise ValueError("Estimated-line annotation currently uses the offline browser interface")
            key = (str(resolve(row["reference_path"])), row.get("reference_frame_idx", ""))
            if key in seen_references:
                continue
            seen_references.add(key)
            frame = reference_frame(row)
            sizes.add(frame.shape[:2])
        if len(sizes) != 1:
            raise ValueError("Reference images/videos must share source resolution; view alignment still needs human verification")
        out_dir = new_dir(args.output_dir)
        write_csv(out_dir / "points_draft.csv", rows, POINT_FIELDS)
        if args.interface == "browser":
            write_browser_annotator(out_dir, rows, args)
            return
        for row in rows:
            frame = reference_frame(row)
            initial = []
            if row.get("position_confirmed", "").upper() == "TRUE":
                x1, y1, x2, y2 = checked_line(row, frame.shape[1], frame.shape[0])
                initial = [(x1, y1), (x2, y2)]
            action, points = pick_points(frame, row["marker_id"], args.window_width, args.window_height, initial)
            if action == "stop":
                print(f"Stopped. Resume from {shown(out_dir / 'points_draft.csv')} into a new output directory.")
                return
            if action == "confirm":
                row.update(dict(zip(("x_left_px", "y_left_px", "x_right_px", "y_right_px"), (*points[0], *points[1]))))
                row["position_confirmed"] = "TRUE"
            else:
                row["position_confirmed"] = ""
            write_csv(out_dir / "points_draft.csv", rows, POINT_FIELDS)
        if any(row.get("position_confirmed", "").upper() != "TRUE" for row in rows):
            print(f"Incomplete: points_draft.csv saved in {shown(out_dir)}; no ready profile produced.")
            return
        profile, frame = assemble_profile(rows, args)
    save_profile(out_dir, profile, frame, resolve(args.marker_ground_truth))
    print(f"Profile saved: {shown(out_dir / 'marker_lines_setup.csv')}")


def detection_rows(path: Path, video_id: str, start: int, end: int, pass_id: str = "") -> list[dict[str, Any]]:
    rows = read_csv(path, ("video_id", "frame_idx", "track_id"))
    if rows and video_id not in {row["video_id"] for row in rows}:
        raise ValueError(f"No video_id={video_id!r} in detections; check filename or --video-id")
    result = []
    seen = set()
    for row in rows:
        if row["video_id"] != video_id:
            continue
        vehicle_id = (row.get("subject_id") or "").strip()
        identity_source = "manual_vehicle_id"
        if not vehicle_id:
            vehicle_id = (row.get("track_id") or "").strip()
            identity_source = "track_id_pseudo_vehicle_id"
        if not vehicle_id:
            continue
        row["subject_id"] = vehicle_id  # legacy event schema column; semantically vehicle_id here
        row["_vehicle_identity_source"] = identity_source
        index = integer(row["frame_idx"], "frame_idx")
        if not start <= index <= end:
            continue
        row_pass = row.get("pass_id", "")
        if pass_id and row_pass and pass_id != row_pass:
            continue
        row["pass_id"] = row_pass or pass_id
        key = (vehicle_id, row["pass_id"], index)
        if key in seen:
            raise ValueError(f"Ambiguous duplicate vehicle detection at frame {index}: {key}")
        seen.add(key)
        row["frame_idx"] = index
        if row.get("timestamp_s") and number(row["timestamp_s"], "timestamp_s") < 0:
            raise ValueError("Detection timestamps must be nonnegative")
        if row.get("bottom_center_x_px") and row.get("bottom_center_y_px"):
            row["_bumper_x"] = number(row["bottom_center_x_px"], "bottom_center_x_px")
            row["_bumper_y"] = number(row["bottom_center_y_px"], "bottom_center_y_px")
        else:
            row["_bumper_x"] = (number(row.get("x1_px"), "x1_px") + number(row.get("x2_px"), "x2_px")) / 2
            row["_bumper_y"] = number(row.get("y2_px"), "y2_px")
        result.append(row)
    return sorted(result, key=lambda row: row["frame_idx"])


def crossing_suggestion(rows: list[dict[str, Any]], line: dict[str, Any], gap: int, direction: str) -> dict[str, Any]:
    samples = []
    for row in rows:
        distance, fraction = line_position(line, row["_bumper_x"], row["_bumper_y"])
        if -1e-9 <= fraction <= 1 + 1e-9:
            samples.append((row, distance, fraction))
    if not samples:
        return {"sample": None, "candidate_status": "no_in_segment_samples", "sign_change_count": 0, "samples_in_segment": 0, "bracket": None}
    brackets = []
    for left, right in zip(samples, samples[1:]):
        if right[0]["frame_idx"] - left[0]["frame_idx"] > gap:
            continue
        if direction == "negative-to-positive":
            crossing = left[1] < 0 < right[1]
        elif direction == "positive-to-negative":
            crossing = left[1] > 0 > right[1]
        else:
            crossing = left[1] * right[1] < 0
        if crossing:
            brackets.append((left, right))
    if brackets:
        bracket = min(brackets, key=lambda pair: (max(abs(pair[0][1]), abs(pair[1][1])), pair[0][0]["frame_idx"]))
        chosen = min(bracket, key=lambda sample: (abs(sample[1]), sample[0]["frame_idx"]))
        status = "sign_change_needs_review"
    else:
        bracket = None
        chosen = min(samples, key=lambda sample: (abs(sample[1]), sample[0]["frame_idx"]))
        status = "on_line_proxy_needs_review" if abs(chosen[1]) < 1e-9 else "nearest_only_no_sign_change"
    return {"sample": chosen, "candidate_status": status, "sign_change_count": len(brackets), "samples_in_segment": len(samples), "bracket": bracket}


def bounds(args, info) -> tuple[int, int]:
    start = args.start_frame
    end = info["frame_count"] - 1 if args.end_frame is None else args.end_frame
    if start < 0 or not start <= end < info["frame_count"]:
        raise ValueError(f"Frame range must satisfy 0 <= start <= end < {info['frame_count']}")
    return start, end


def video_profile(args):
    path = resolve(args.video)
    profile = load_profile(resolve(args.profile), read_marker_ground_truth(resolve(args.marker_ground_truth)))
    if args.command != "preview" and estimated_lines(profile) and not args.allow_estimated_lines:
        raise ValueError("Using estimated lines for crossing events requires --allow-estimated-lines and reporting annotation uncertainty")
    capture, info = open_video(path)
    try:
        validate_view(profile, info, args.same_camera_view_confirmed)
        bounds(args, info)
    except Exception:
        capture.release()
        raise
    return path, profile, capture, info


def command_preview(args) -> None:
    path, profile, capture, info = video_profile(args)
    writer = None
    try:
        start, end = bounds(args, info)
        out_dir = new_dir(args.output_dir)
        scale = min(1, args.preview_width / info["width"])
        width = max(2, round(info["width"] * scale) // 2 * 2)
        height = max(2, round(info["height"] * width / info["width"]) // 2 * 2)
        scale_x, scale_y = width / info["width"], height / info["height"]
        # Anisotropic rounding below changes only display, not profile coordinates.
        display_profile = [{**row, "x_left_px": row["x_left_px"] * scale_x, "x_right_px": row["x_right_px"] * scale_x,
                            "y_left_px": row["y_left_px"] * scale_y, "y_right_px": row["y_right_px"] * scale_y} for row in profile]
        output = out_dir / "virtual_lines_preview.mp4"
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (width, height))
        if not writer.isOpened():
            raise RuntimeError("Cannot open MP4 video writer")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for index in range(start, end + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video decoding stopped early at frame {index}; preview is incomplete")
            preview = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            draw_lines(preview, display_profile)
            cv2.putText(preview, f"REVIEW ONLY | {profile[0]['setup_id']} | source frame {index}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(preview)
            if index == start:
                save_image(out_dir / "virtual_lines_snapshot.jpg", preview)
        write_json(out_dir / "preview_summary.json", {
            "video": shown(path), "profile": shown(resolve(args.profile)), "profile_sha256": sha256_file(resolve(args.profile)),
            "setup_id": profile[0]["setup_id"], "source_dimensions": [info["width"], info["height"]],
            "start_frame": start, "end_frame": end, "frames_written": end - start + 1,
            "contains_estimated_lines": estimated_lines(profile),
            "warnings": [ESTIMATED_LINE_WARNING] if estimated_lines(profile) else [],
            "note": "Display-only virtual lines; source video not changed. MP4 preview has no audio. View consistency is operator-confirmed, not automatically verified.",
        })
        print(f"Preview saved: {shown(output)}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def contact_sheet(capture, candidate, profile, by_frame, frame_count: int, step: int) -> np.ndarray:
    panels = []
    for offset in (-2, -1, 0, 1, 2):
        index = max(0, min(frame_count - 1, candidate["candidate_frame_idx"] + offset * step))
        frame = video_frame(capture, index)
        draw_lines(frame, profile, active=candidate["marker_id"])
        matches = [row for row in by_frame.get(index, []) if row["subject_id"] == candidate["subject_id"] and row["pass_id"] == candidate["pass_id"]]
        det = matches[0] if matches else None
        if det:
            fx, fy = det["_bumper_x"], det["_bumper_y"]
            if all(det.get(k) for k in ("x1_px", "y1_px", "x2_px", "y2_px")):
                x1, y1, x2, y2 = (round(number(det[k], k)) for k in ("x1_px", "y1_px", "x2_px", "y2_px"))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 255), 2)
            cv2.circle(frame, (round(fx), round(fy)), 4, (0, 150, 255), -1)
        else:
            fx, fy = candidate["bumper_proxy_x_px"], candidate["bumper_proxy_y_px"]
        full = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
        # A source-pixel crop makes distant bumper boxes visible without changing GT coordinates.
        crop_w, crop_h = min(600, frame.shape[1]), min(400, frame.shape[0])
        x0 = max(0, min(frame.shape[1] - crop_w, round(fx) - crop_w // 2))
        y0 = max(0, min(frame.shape[0] - crop_h, round(fy) - crop_h // 2))
        crop = cv2.resize(frame[y0:y0 + crop_h, x0:x0 + crop_w], (480, 270))
        strip = np.zeros((44, 480, 3), dtype=np.uint8)
        label = f"{candidate['subject_id']} {candidate['marker_id']} f{index}"
        cv2.putText(strip, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(strip, f"UNVERIFIED | track {det['track_id'] if det else 'missing'} | bottom-center PROXY", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        panels.append(np.vstack((strip, full, crop)))
    return np.hstack(panels)


def command_candidates(args) -> None:
    path, profile, capture, info = video_profile(args)
    try:
        start, end = bounds(args, info)
        video_id = args.video_id or path.stem
        detections = detection_rows(resolve(args.detections), video_id, start, end, args.pass_id)
        trajectories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in detections:
            if args.subjects and row["subject_id"] not in args.subjects:
                continue
            if not (0 <= row["_bumper_x"] < info["width"] and 0 <= row["_bumper_y"] < info["height"]):
                if getattr(args, "clip_offscreen_bumper", False):
                    # Car detector boxes can extend past the frame edge
                    # is partially out of view. Clipping keeps the sample usable
                    # for suggestion lookup; review images show the true box.
                    row["_bumper_x"] = min(max(row["_bumper_x"], 0), info["width"] - 1)
                    row["_bumper_y"] = min(max(row["_bumper_y"], 0), info["height"] - 1)
                    row["bumper_proxy_clipped"] = True
                else:
                    raise ValueError("Detection front-bumper proxy is outside video; check source-pixel scale")
            trajectories[(row["subject_id"], row["pass_id"])].append(row)
            by_frame[row["frame_idx"]].append(row)
        if args.subjects:
            for subject in args.subjects:
                if not any(key[0] == subject for key in trajectories):
                    trajectories[(subject, args.pass_id)] = []
        if not trajectories:
            raise ValueError("No mapped vehicles for this video; provide a confirmed vehicle crosswalk or --subjects to report missing vehicles")
        metadata = {}
        if args.subject_metadata:
            for row in read_csv(resolve(args.subject_metadata), ("subject_id", "group_id", "activity", "lane")):
                if not row["subject_id"] or row["subject_id"] in metadata:
                    raise ValueError("Vehicle metadata must have unique nonempty subject_id (vehicle ID) values")
                metadata[row["subject_id"]] = row
        profile_hash = sha256_file(resolve(args.profile))
        candidates, reviews = [], []
        for (subject, pass_id), rows in sorted(trajectories.items()):
            for line in profile:
                suggestion = crossing_suggestion(rows, line, args.max_frame_gap, args.direction)
                event_id = ":".join((args.take_id, video_id, pass_id or "pass-unspecified", subject, line["marker_id"]))
                candidate = dict.fromkeys(CANDIDATE_FIELDS, "")
                identity_source = rows[0].get("_vehicle_identity_source", "manual_vehicle_id") if rows else "unmapped_expected_vehicle"
                candidate.update(event_id=event_id, take_id=args.take_id, video_id=video_id, pass_id=pass_id,
                                 subject_id=subject, marker_id=line["marker_id"], gt_distance_m=line["gt_distance_m"],
                                 vehicle_identity_source=identity_source,
                                 setup_id=line["setup_id"], marker_profile_sha256=profile_hash,
                                 candidate_status=suggestion["candidate_status"], sign_change_count=suggestion["sign_change_count"],
                                 samples_in_segment=suggestion["samples_in_segment"])
                sample = suggestion["sample"]
                if sample:
                    det, distance, fraction = sample
                    candidate.update(candidate_frame_idx=det["frame_idx"], candidate_track_id=det["track_id"],
                                     candidate_timestamp_s=det.get("timestamp_s") or f"{det['frame_idx'] / info['fps']:.6f}",
                                     bumper_proxy_x_px=det["_bumper_x"], bumper_proxy_y_px=det["_bumper_y"],
                                     signed_distance_px=distance, segment_fraction=fraction)
                if suggestion["bracket"]:
                    candidate["bracket_start_frame"] = suggestion["bracket"][0][0]["frame_idx"]
                    candidate["bracket_end_frame"] = suggestion["bracket"][1][0]["frame_idx"]
                meta = metadata.get(subject, {})
                review = {**candidate, "frame_idx": candidate["candidate_frame_idx"], "timestamp_s": candidate["candidate_timestamp_s"],
                          "track_id": candidate["candidate_track_id"], "foot_crossing_confirmed": "",
                          "annotation_source": "auto_candidate_unverified", "notes": "Review actual front-bumper contact and vehicle track; correct frame_idx and enter TRUE/FALSE plus a manual annotation_source.",
                           **{key: meta.get(key, "") for key in ("group_id", "activity", "lane")}}
                if identity_source == "track_id_pseudo_vehicle_id":
                    review["notes"] += " Pseudo vehicle ID comes from ByteTrack; manually reconcile fragments if one car has multiple track IDs."
                if line["annotation_source"] == ANNOTATION_SOURCES["estimated-line"]:
                    review["notes"] += " | " + ESTIMATED_LINE_WARNING
                candidates.append(candidate)
                reviews.append(review)
        out_dir = new_dir(args.output_dir)
        if not args.no_contact_sheets:
            (out_dir / "review_images").mkdir()
            for candidate in candidates:
                if candidate["candidate_frame_idx"] == "":
                    continue
                safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", candidate["event_id"])[-100:]
                suffix = hashlib.sha256(candidate["event_id"].encode()).hexdigest()[:8]
                output = out_dir / "review_images" / f"{safe_id}_{suffix}.jpg"
                sheet = contact_sheet(capture, candidate, profile, by_frame, info["frame_count"], args.review_step)
                save_image(output, sheet)
                candidate["review_image"] = shown(output)
        write_csv(out_dir / "crossing_candidates.csv", candidates, CANDIDATE_FIELDS)
        write_csv(out_dir / "marker_events_to_review.csv", reviews, EVENT_FIELDS)
        write_json(out_dir / "candidate_summary.json", {
            "status": "unverified_candidates_only", "video": shown(path), "take_id": args.take_id,
            "profile": shown(resolve(args.profile)), "profile_sha256": profile_hash,
            "contains_estimated_lines": estimated_lines(profile),
            "detections": shown(resolve(args.detections)), "start_frame": start, "end_frame": end,
            "events_to_review": len(candidates), "events_with_suggestion": sum(row["candidate_frame_idx"] != "" for row in candidates),
            "vehicle_ids": sorted({key[0] for key in trajectories}), "max_frame_gap": args.max_frame_gap,
            "direction": args.direction, "multiple_sign_change_events": sum(row["sign_change_count"] > 1 for row in candidates),
            "vehicle_identity_source_counts": {
                source: sum(row.get("vehicle_identity_source") == source for row in candidates)
                for source in sorted({row.get("vehicle_identity_source", "") for row in candidates})
            },
            "clipped_offscreen_bumper_samples": sum(bool(row.get("bumper_proxy_clipped")) for rows_ in trajectories.values() for row in rows_),
            "warnings": ["Bottom-center is a car-box front-bumper proxy, not a measured bumper keypoint.",
                         "No automatic TRUE ground truth was created. Review actual front-bumper crossing, not the box proxy alone.",
                         "For multiple passes, set pass_id and frame range or provide pass_id in detections. Multiple sign changes can be jitter or repeated passes.",
                         "Provide --subjects with all expected vehicle IDs so completely missing tracks are still reported.",
                         "View consistency is operator-confirmed; resolution alone does not validate camera alignment."]
                        + ([ESTIMATED_LINE_WARNING] if estimated_lines(profile) else []),
        })
        print(f"Unverified events for manual review: {shown(out_dir / 'marker_events_to_review.csv')}")
    finally:
        capture.release()


def command_export(args) -> None:
    path, profile, capture, info = video_profile(args)
    capture.release()
    video_id = args.video_id or path.stem
    start, end = bounds(args, info)
    profile_hash = sha256_file(resolve(args.profile))
    gt = {row["marker_id"]: row["gt_distance_m"] for row in profile}
    estimated_markers = {row["marker_id"] for row in profile if row["annotation_source"] == ANNOTATION_SOURCES["estimated-line"]}
    reviewed = read_csv(resolve(args.reviewed_events), EVENT_FIELDS)
    detections = detection_rows(resolve(args.detections), video_id, start, end)
    by_subject_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in detections:
        by_subject_frame[(row["subject_id"], row["frame_idx"])].append(row)
    confirmed, seen_ids, seen_keys = [], set(), set()
    missing_box_count = 0
    for row in reviewed:
        status = row["foot_crossing_confirmed"].upper()
        if status not in ("", "TRUE", "FALSE"):
            raise ValueError("foot_crossing_confirmed must be blank, TRUE or FALSE")
        if status != "TRUE":
            continue
        if not manual_source(row["annotation_source"]):
            raise ValueError(f"{row['event_id']}: replace auto source with a human/manual annotation_source after review")
        if row["video_id"] != video_id or row["setup_id"] != profile[0]["setup_id"] or row["marker_profile_sha256"] != profile_hash:
            raise ValueError("Reviewed event video/setup/profile hash mismatch")
        if not row["event_id"] or not row["take_id"] or not row["subject_id"]:
            raise ValueError("Confirmed events require event_id, take_id and subject_id")
        marker = row["marker_id"]
        if marker not in gt or abs(number(row["gt_distance_m"], "event GT") - gt[marker]) > 1e-6:
            raise ValueError(f"Reviewed GT mismatch for {marker}")
        index = integer(row["frame_idx"], "reviewed frame_idx")
        if not start <= index <= end:
            raise ValueError(f"Reviewed frame {index} is outside the declared pass/video range")
        key = (row["take_id"], video_id, row["pass_id"], row["subject_id"], marker)
        if row["event_id"] in seen_ids or key in seen_keys:
            raise ValueError("Duplicate confirmed event; use a distinct pass_id for repeated crossings")
        seen_ids.add(row["event_id"])
        seen_keys.add(key)
        matches = [det for det in by_subject_frame.get((row["subject_id"], index), [])
                   if not row["pass_id"] or not det["pass_id"] or det["pass_id"] == row["pass_id"]]
        if len(matches) > 1:
            raise ValueError("Ambiguous detection at corrected crossing frame")
        if matches:
            row["track_id"] = matches[0]["track_id"]
            row["timestamp_s"] = matches[0].get("timestamp_s") or f"{index / info['fps']:.6f}"
        else:
            missing_box_count += 1
            row["track_id"] = ""
            row["timestamp_s"] = f"{index / info['fps']:.6f}"
            row["notes"] += " | No subject-mapped box at confirmed frame; retained for detection coverage."
        row["foot_crossing_confirmed"] = "TRUE"
        row["frame_idx"] = index
        row["gt_distance_m"] = gt[marker]
        if marker in estimated_markers and ESTIMATED_LINE_WARNING not in row["notes"]:
            row["notes"] += " | " + ESTIMATED_LINE_WARNING
        confirmed.append(row)
    if not confirmed:
        raise ValueError("No manually confirmed TRUE events; no confirmed output created")
    output = resolve(args.output)
    report = output.with_suffix(".summary.json")
    if output.exists() or report.exists():
        raise FileExistsError(f"Refusing to overwrite confirmed events/report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, confirmed, EVENT_FIELDS)
    write_json(report, {"status": "manually_confirmed_export", "source_review": shown(resolve(args.reviewed_events)),
                        "confirmed_events": len(confirmed), "unconfirmed_or_false_rows": len(reviewed) - len(confirmed),
                         "confirmed_events_without_subject_box": missing_box_count,
                         "contains_estimated_lines": estimated_lines(profile),
                         "warnings": [ESTIMATED_LINE_WARNING] if estimated_lines(profile) else [],
                        "profile_sha256": profile_hash, "note": "Frames chosen by human review. Track IDs refreshed from mapped detections at corrected frames. Timestamps use detection time when available, otherwise nominal frame_idx/fps (not verified PTS)."})
    print(f"Confirmed events ready for the polynomial pipeline: {shown(output)}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    for name in ("annotate", "build-profile"):
        command = commands.add_parser(name, help="Click cone base points" if name == "annotate" else "Build profile from manually entered/confirmed points (no GUI)")
        command.add_argument("--references", type=Path, required=True, help="Reference manifest or saved points_draft.csv; paths relative to repo root")
        command.add_argument("--setup-id", required=True)
        command.add_argument("--lane-width-m", type=float)
        command.add_argument("--same-camera-view-confirmed", action="store_true")
        command.add_argument("--marker-ground-truth", type=Path, default=DEFAULT_GT)
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "annotate":
            command.add_argument("--annotation-method", choices=tuple(ANNOTATION_SOURCES), help="Default: keep the manifest method or cone-base; estimated-line records visual line placement, not two measured cones")
            command.add_argument("--interface", choices=("browser", "opencv"), default="browser", help="Default: offline browser HTML; OpenCV interface requires a HighGUI-enabled installation")
            command.add_argument("--window-width", type=int, default=1280)
            command.add_argument("--window-height", type=int, default=640)
        else:
            command.add_argument("--allow-estimated-lines", action="store_true", help="Explicitly accept/report uncertainty for visually estimated crossing lines")
        command.set_defaults(func=command_profile)
    for name in ("preview", "candidates", "export-confirmed"):
        command = commands.add_parser(name)
        command.add_argument("--profile", type=Path, required=True)
        command.add_argument("--video", type=Path, required=True)
        command.add_argument("--same-camera-view-confirmed", action="store_true")
        command.add_argument("--marker-ground-truth", type=Path, default=DEFAULT_GT)
        command.add_argument("--start-frame", type=int, default=0)
        command.add_argument("--end-frame", type=int)
        if name != "preview":
            command.add_argument("--allow-estimated-lines", action="store_true", help="Explicitly accept/report estimated-line uncertainty before creating crossing events")
            command.add_argument("--video-id", help="Defaults to video filename stem")
            command.add_argument("--detections", type=Path, required=True, help="Subject-mapped detections in source pixels")
        if name == "preview":
            command.add_argument("--preview-width", type=int, default=1280)
            command.add_argument("--output-dir", type=Path, required=True)
            command.set_defaults(func=command_preview)
        elif name == "candidates":
            command.add_argument("--take-id", required=True)
            command.add_argument("--pass-id", default="")
            command.add_argument("--subjects", nargs="+", help="All expected vehicle IDs (legacy subject_id), including those missed by the detector")
            command.add_argument("--subject-metadata", type=Path)
            command.add_argument("--clip-offscreen-feet", "--clip-offscreen-bumper", dest="clip_offscreen_bumper", action="store_true", help="Clamp front-bumper proxies that fall outside the frame edge (detector boxes partially offscreen) instead of aborting; clipped samples stay marked for review")
            command.add_argument("--direction", choices=("either", "negative-to-positive", "positive-to-negative"), default="either")
            command.add_argument("--max-frame-gap", type=int, default=5)
            command.add_argument("--review-step", type=int, default=1)
            command.add_argument("--no-contact-sheets", action="store_true")
            command.add_argument("--output-dir", type=Path, required=True)
            command.set_defaults(func=command_candidates)
        else:
            command.add_argument("--reviewed-events", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.set_defaults(func=command_export)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        for key in ("window_width", "window_height", "preview_width", "max_frame_gap", "review_step"):
            if hasattr(args, key) and getattr(args, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if hasattr(args, "take_id") and not args.take_id.strip():
            raise ValueError("take_id cannot be blank")
        args.func(args)
        return 0
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
