"""Generate a Blender urban-street scene from frozen car trajectories.

Run inside Blender 5.2 LTS, not system Python:

  blender --background --python scripts/12_generate_urban_car_scene.py -- \
    --trajectories outputs/3d/Video2/frame_trajectories.csv \
    --assets-dir "3d model" --output-blend outputs/3d/Video2/urban_scene.blend \
    --output-glb outputs/3d/Video2/urban_scene.glb

The composition is intentionally distinct from the person project: a city
street with buildings/facades, sidewalks, curbs, and lane markings. It does not
reuse the roadside trees/grass-verge scene. Car assets are supplied .blend
files; this script does not modify them. Lateral position is schematic, and
all geometry is for visualization rather than surveyed reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
ASSET_FOR_CLASS = {
    "Hatchback": "1990s-hatchback.blend",
    "MPV": "1990s-minivan.blend",
    "Sedan": "1990s-sedan.blend",
}
DEFAULT_ROAD_WIDTH_M = 6.0


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories", type=Path, required=True, help="frame_trajectories.csv from script 11")
    parser.add_argument("--assets-dir", type=Path, required=True, help="Directory containing the three allowed .blend car assets")
    parser.add_argument("--output-blend", type=Path, required=True, help="New .blend scene path")
    parser.add_argument("--output-glb", type=Path, default=None, help="Optional .glb export path")
    parser.add_argument("--road-width-m", type=float, default=DEFAULT_ROAD_WIDTH_M)
    parser.add_argument("--camera-offset-y-m", type=float, default=-8.0)
    parser.add_argument("--building-spacing-m", type=float, default=20.0)
    parser.add_argument("--building-setback-m", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resolution-x", type=int, default=1280)
    parser.add_argument("--resolution-y", type=int, default=720)
    parser.add_argument("--render-preview", action="store_true", help="Render one temporary camera view to output-blend's parent")
    return parser.parse_args(argv)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if (Path.cwd().resolve() == ROOT.parent.resolve()
            and path.parts and path.parts[0].casefold() == ROOT.name.casefold()):
        return (Path.cwd() / path).resolve()
    return (ROOT / path).resolve()


def read_trajectory_rows(path: Path) -> tuple[list[dict], float, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"video_id", "frame_idx", "track_id", "visual_body_class", "pred_distance_m", "lateral_x_m_schematic", "scene_eligible"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing trajectory columns: {sorted(missing)}")
        rows = list(reader)
    eligible = []
    for row in rows:
        if (row.get("scene_eligible") or "").strip().upper() != "TRUE":
            continue
        body_class = (row.get("visual_body_class") or "").strip()
        if body_class not in ASSET_FOR_CLASS:
            continue
        try:
            frame = int(float(row["frame_idx"]))
            distance = float(row["pred_distance_m"])
            lateral = float(row["lateral_x_m_schematic"])
        except (ValueError, TypeError):
            continue
        if not math.isfinite(distance) or distance <= 0 or not math.isfinite(lateral):
            continue
        row["_frame"] = frame
        row["_distance"] = distance
        row["_lateral"] = lateral
        eligible.append(row)
    video_ids = sorted({(row.get("video_id") or "").strip() for row in eligible})
    if len(video_ids) != 1:
        raise ValueError(f"Trajectories must represent exactly one video after filtering; found {video_ids}")
    fps_values = {round(float(row.get("source_fps") or 30), 3) for row in eligible}
    if len(fps_values) > 1:
        raise ValueError(f"Mixed source FPS values in trajectory file: {sorted(fps_values)}")
    return eligible, fps_values.pop() if fps_values else 30.0, video_ids[0]


def make_material(name: str, color: tuple[float, float, float, float], *, metallic: float = 0.0, roughness: float = 0.7):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF") if material.node_tree else None
    if principled:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Metallic"].default_value = metallic
        principled.inputs["Roughness"].default_value = roughness
    return material


def create_box(name: str, location, dimensions, material, collection=None):
    dx, dy, dz = [float(v) for v in dimensions]
    hx, hy, hz = dx / 2, dy / 2, dz / 2
    verts = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
    ]
    faces = [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    (collection or bpy.context.scene.collection).objects.link(obj)
    obj.location = location
    obj.dimensions = (dx, dy, dz)
    return obj


def create_windows_mesh(name: str, x: float, windows: list[tuple[float, float, float, float]], material, collection):
    """Create many simple window boxes as one mesh object to keep GLB node count moderate."""
    vertices = []
    faces = []
    for y, z, width, height in windows:
        hx, hy, hz = 0.035, width / 2, height / 2
        base = len(vertices)
        vertices.extend([
            (x - hx, y - hy, z - hz), (x + hx, y - hy, z - hz),
            (x + hx, y + hy, z - hz), (x - hx, y + hy, z - hz),
            (x - hx, y - hy, z + hz), (x + hx, y - hy, z + hz),
            (x + hx, y + hy, z + hz), (x - hx, y + hy, z + hz),
        ])
        faces.extend([
            (base + 0, base + 3, base + 2, base + 1), (base + 4, base + 5, base + 6, base + 7),
            (base + 0, base + 1, base + 5, base + 4), (base + 1, base + 2, base + 6, base + 5),
            (base + 2, base + 3, base + 7, base + 6), (base + 3, base + 0, base + 4, base + 7),
        ])
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def build_city_environment(scene, *, road_width: float, camera_y: float, max_distance: float,
                           building_spacing: float, building_setback: float, rng: random.Random):
    road_collection = bpy.data.collections.new("Urban_Road_Infrastructure")
    city_collection = bpy.data.collections.new("Urban_Buildings")
    scene.collection.children.link(road_collection)
    scene.collection.children.link(city_collection)

    asphalt = make_material("City_Asphalt", (0.12, 0.14, 0.17, 1), roughness=0.92)
    concrete = make_material("Sidewalk_Concrete", (0.38, 0.39, 0.40, 1), roughness=0.9)
    curb_mat = make_material("Curb_Stone", (0.56, 0.57, 0.57, 1), roughness=0.85)
    line_mat = make_material("Road_Marking_White", (0.9, 0.9, 0.82, 1), roughness=0.7)
    ground_mat = make_material("City_Ground", (0.25, 0.27, 0.28, 1), roughness=0.95)
    glass_mat = make_material("Urban_Window_Glass", (0.13, 0.23, 0.30, 1), metallic=0.15, roughness=0.3)
    facade_colors = [
        (0.62, 0.55, 0.45, 1), (0.49, 0.52, 0.53, 1),
        (0.72, 0.69, 0.61, 1), (0.42, 0.46, 0.48, 1),
        (0.68, 0.60, 0.52, 1), (0.55, 0.57, 0.56, 1),
    ]
    facade_materials = [make_material(f"Urban_Facade_{i:02d}", color, roughness=0.9) for i, color in enumerate(facade_colors)]

    margin = 15.0
    road_start = camera_y - 15.0
    road_end = camera_y + max_distance + margin
    road_length = road_end - road_start
    road_center_y = (road_start + road_end) / 2
    create_box("Urban_Ground_Base", (0, road_center_y, -0.28), (road_width + 42, road_length + 30, 0.35), ground_mat, road_collection)
    create_box("Urban_Asphalt_Road", (0, road_center_y, -0.08), (road_width, road_length, 0.16), asphalt, road_collection)

    sidewalk_width = 1.8
    for side in (-1, 1):
        sidewalk_x = side * (road_width / 2 + sidewalk_width / 2)
        create_box(f"Sidewalk_{'L' if side < 0 else 'R'}", (sidewalk_x, road_center_y, -0.015),
                   (sidewalk_width, road_length, 0.18), concrete, road_collection)
        curb_x = side * (road_width / 2 + 0.08)
        create_box(f"Curb_{'L' if side < 0 else 'R'}", (curb_x, road_center_y, 0.04),
                   (0.16, road_length, 0.28), curb_mat, road_collection)
        edge_x = side * (road_width / 2 - 0.18)
        create_box(f"Road_Edge_Line_{'L' if side < 0 else 'R'}", (edge_x, road_center_y, 0.012),
                   (0.07, road_length, 0.015), line_mat, road_collection)

    # Broken center divider for a two-way urban street.
    dash_y = road_start + 4.0
    dash_index = 0
    while dash_y < road_end - 2:
        create_box(f"Center_Dash_{dash_index:03d}", (0, dash_y, 0.014), (0.10, 2.8, 0.02), line_mat, road_collection)
        dash_y += 5.8
        dash_index += 1

    # A few crosswalks make the road read as urban infrastructure, not a rural track.
    crosswalk_y = camera_y + 16.0
    crosswalk_index = 0
    while crosswalk_y < road_end - 12.0:
        stripe_count = 7
        stripe_width = road_width / (stripe_count * 1.7)
        for stripe_idx in range(stripe_count):
            x = -road_width / 2 + (stripe_idx + 0.5) * road_width / stripe_count
            create_box(f"Crosswalk_{crosswalk_index:02d}_{stripe_idx:02d}", (x, crosswalk_y, 0.015),
                       (stripe_width, 2.4, 0.02), line_mat, road_collection)
        crosswalk_y += 28.0
        crosswalk_index += 1

    y = camera_y + 6.0
    building_id = 0
    while y < road_end - 6.0:
        for side in (-1, 1):
            length = rng.uniform(14.0, min(27.0, building_spacing + 7.0))
            depth = rng.uniform(8.0, 13.0)
            floors = rng.randint(3, 8)
            height = floors * rng.uniform(2.9, 3.35)
            center_x = side * (road_width / 2 + sidewalk_width + building_setback + depth / 2)
            center_y = y + rng.uniform(-1.5, 1.5)
            facade = rng.choice(facade_materials)
            create_box(f"City_Building_{building_id:03d}", (center_x, center_y, height / 2),
                       (depth, length, height), facade, city_collection)

            # Road-facing windows; the car scene uses architecture rather than trees.
            road_face_x = center_x - side * (depth / 2 + 0.045)
            windows = []
            columns = max(3, int(length // 3.0))
            window_width = min(1.35, (length - 2.0) / columns * 0.62)
            for floor_index in range(floors):
                z = 1.8 + floor_index * 3.0
                if z + 0.9 >= height:
                    continue
                for col in range(columns):
                    wy = center_y - length / 2 + (col + 0.5) * length / columns
                    windows.append((wy, z, window_width, 1.25))
            create_windows_mesh(f"City_Windows_{building_id:03d}", road_face_x, windows, glass_mat, city_collection)
            building_id += 1
        y += building_spacing + rng.uniform(-2.0, 3.0)

    return road_start, road_end, building_id


def load_vehicle_template(asset_path: Path, template_collection):
    with bpy.data.libraries.load(str(asset_path), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    objects = [obj for obj in data_to.objects if obj is not None]
    for obj in objects:
        if len(obj.users_collection) == 0:
            template_collection.objects.link(obj)
    roots = [obj for obj in objects if obj.type == "EMPTY" and obj.parent is None]
    root = roots[0] if roots else next((obj for obj in objects if obj.parent is None), None)
    if root is None:
        raise ValueError(f"No root object found in vehicle asset {asset_path}")
    child_objects = [obj for obj in objects if obj != root]
    if not any(obj.type == "MESH" for obj in child_objects):
        raise ValueError(f"Vehicle asset contains no mesh objects: {asset_path}")
    y_values = []
    for obj in child_objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world_local = obj.matrix_local @ Vector(corner)
            y_values.append(float(world_local.y))
    length = max(y_values) - min(y_values) if y_values else float(root.dimensions.y)
    if length <= 0:
        raise ValueError(f"Could not measure longitudinal asset length: {asset_path}")
    return {"root": root, "objects": objects, "children": child_objects, "length_m": length}


def duplicate_vehicle(template: dict, track_id: str, body_class: str, vehicle_collection):
    source_root = template["root"]
    new_root = source_root.copy()
    new_root.name = f"Car_{body_class}_{re.sub(r'[^A-Za-z0-9_-]+', '_', track_id)}"
    vehicle_collection.objects.link(new_root)
    mapping = {source_root: new_root}
    for source in template["children"]:
        obj = source.copy()
        if source.data is not None:
            obj.data = source.data
        obj.name = f"{new_root.name}_{source.name.split('_')[-1]}"
        vehicle_collection.objects.link(obj)
        obj.parent = mapping.get(source.parent, new_root)
        obj.matrix_parent_inverse = source.matrix_parent_inverse.copy()
        obj.matrix_basis = source.matrix_basis.copy()
        mapping[source] = obj
    new_root.location = (0, 0, 0)
    new_root.rotation_euler = (0, 0, 0)
    new_root.scale = (1, 1, 1)
    return new_root


def add_camera_and_lighting(scene, camera_y: float, max_distance: float, resolution_x: int, resolution_y: int):
    camera_data = bpy.data.cameras.new("Urban_Camera")
    camera = bpy.data.objects.new("Urban_Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, camera_y - 9.0, 3.4)
    target = Vector((0.0, camera_y + max(8.0, max_distance * 0.58), 1.0))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 42
    scene.camera = camera

    sun_data = bpy.data.lights.new("City_Sun", type="SUN")
    sun = bpy.data.objects.new("City_Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.data.energy = 2.0
    sun.rotation_euler = (math.radians(28), math.radians(-20), math.radians(-25))

    fill_data = bpy.data.lights.new("Street_Fill", type="AREA")
    fill = bpy.data.objects.new("Street_Fill", fill_data)
    scene.collection.objects.link(fill)
    fill.location = (0, camera_y + 18, 18)
    fill.data.energy = 3000
    fill.data.shape = "DISK"
    fill.data.size = 20
    fill.rotation_euler = (Vector((0, camera_y + 30, 0)) - fill.location).to_track_quat("-Z", "Y").to_euler()

    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100


def main() -> None:
    args = parse_args()
    trajectories_path = resolve_path(args.trajectories)
    assets_dir = resolve_path(args.assets_dir)
    blend_path = resolve_path(args.output_blend)
    glb_path = resolve_path(args.output_glb) if args.output_glb else None
    if not trajectories_path.is_file():
        raise FileNotFoundError(trajectories_path)
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"3D asset directory not found: {assets_dir}")
    for output in (blend_path, glb_path):
        if output and output.exists():
            raise FileExistsError(f"Refusing to overwrite existing 3D output: {output}")
    if args.road_width_m <= 0 or args.building_spacing_m <= 0:
        raise ValueError("road-width-m and building-spacing-m must be positive")

    rows, source_fps, video_id = read_trajectory_rows(trajectories_path)
    max_distance = max(row["_distance"] for row in rows)
    frame_start = min(row["_frame"] for row in rows)
    frame_end = max(row["_frame"] for row in rows)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = f"Urban_Car_{video_id}"
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.fps = max(1, int(round(source_fps)))
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.world = bpy.data.worlds.new("Urban_Overcast_World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background:
        background.inputs["Color"].default_value = (0.42, 0.48, 0.55, 1)
        background.inputs["Strength"].default_value = 0.45

    vehicle_collection = bpy.data.collections.new("Tracked_Cars")
    template_collection = bpy.data.collections.new("Hidden_Asset_Templates")
    scene.collection.children.link(vehicle_collection)
    scene.collection.children.link(template_collection)
    template_collection.hide_render = True
    template_collection.hide_viewport = True

    templates = {}
    for body_class, filename in ASSET_FOR_CLASS.items():
        asset_path = assets_dir / filename
        if not asset_path.is_file():
            raise FileNotFoundError(f"Missing model asset for {body_class}: {asset_path}")
        templates[body_class] = load_vehicle_template(asset_path, template_collection)

    rng = random.Random(args.seed)
    road_start, road_end, building_count = build_city_environment(
        scene,
        road_width=args.road_width_m,
        camera_y=args.camera_offset_y_m,
        max_distance=max_distance,
        building_spacing=args.building_spacing_m,
        building_setback=args.building_setback_m,
        rng=rng,
    )
    add_camera_and_lighting(scene, args.camera_offset_y_m, max_distance, args.resolution_x, args.resolution_y)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["track_id"])].append(row)
    car_roots = {}
    keyframe_count = 0
    for track_id, track_rows in sorted(grouped.items()):
        labels = Counter(row["visual_body_class"] for row in track_rows)
        body_class = labels.most_common(1)[0][0]
        if len(labels) > 1:
            raise ValueError(f"Track {track_id} changes body class across rows: {dict(labels)}")
        root = duplicate_vehicle(templates[body_class], track_id, body_class, vehicle_collection)
        car_roots[track_id] = root
        length = float(templates[body_class]["length_m"])
        for row in sorted(track_rows, key=lambda item: item["_frame"]):
            # Asset front faces -Y. Therefore front bumper y is root_y - length/2.
            root.location = (
                row["_lateral"],
                args.camera_offset_y_m + row["_distance"] + length / 2,
                0.0,
            )
            root.keyframe_insert(data_path="location", frame=row["_frame"])
            keyframe_count += 1

    # Use linear interpolation to preserve the measured frame-to-frame trajectory.
    for root in car_roots.values():
        action = root.animation_data.action if root.animation_data else None
        if action and hasattr(action, "fcurves"):
            for curve in action.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"

    scene["video_id"] = video_id
    scene["visualization_note"] = "Urban buildings street; lateral position schematic; distance is frozen quartic inference."
    scene["trajectory_csv"] = str(trajectories_path)
    scene.frame_set(frame_start)

    blend_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(blend_path.with_name(blend_path.stem + "_preview.png"))
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    export_result = None
    if glb_path:
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        export_result = bpy.ops.export_scene.gltf(
            filepath=str(glb_path),
            export_format="GLB",
            use_selection=False,
            export_yup=True,
            export_animations=True,
            export_frame_range=True,
            export_nla_strips=True,
            export_optimize_animation_size=True,
            export_anim_single_armature=True,
            export_reset_pose_bones=True,
            export_hierarchy_full_collections=False,
            export_attributes=False,
        )
        if "FINISHED" not in export_result or not glb_path.is_file():
            raise RuntimeError(f"GLB export failed: {export_result}")
    if args.render_preview:
        scene.render.resolution_percentage = 60
        scene.render.filepath = str(blend_path.with_name(blend_path.stem + "_preview.png"))
        bpy.ops.render.render(write_still=True)

    report = {
        "schema": "urban_car_scene_generation_v1",
        "video_id": video_id,
        "source_fps": source_fps,
        "frame_range": [frame_start, frame_end],
        "trajectory_rows": len(rows),
        "tracked_cars": len(car_roots),
        "classes_in_scene": {name: sum(1 for _track, group in grouped.items() if group[0]["visual_body_class"] == name) for name in ASSET_FOR_CLASS},
        "keyframes_written": keyframe_count,
        "road_width_m_schematic": args.road_width_m,
        "road_longitudinal_range_m": [road_start - args.camera_offset_y_m, road_end - args.camera_offset_y_m],
        "urban_building_count": building_count,
        "tree_objects_created": 0,
        "blend_output": str(blend_path),
        "glb_output": str(glb_path) if glb_path else None,
        "preview_output": str(blend_path.with_name(blend_path.stem + "_preview.png")) if args.render_preview else None,
        "style": "urban street with buildings/facades, sidewalks, curbs, and road markings; no rural tree/verge scene",
        "limitations": [
            "Lateral car placement is schematic, derived from normalized image x and configured road width.",
            "Car assets are illustrative 1990s models, not the exact vehicles from the videos.",
            "Track fragments render as separate cars unless reconciled upstream.",
            "Wheel meshes are static; only the root vehicle translation is animated.",
            "Building facades are procedural generic city blocks, not surveyed architecture.",
        ],
    }
    report_path = blend_path.with_name(blend_path.stem + "_scene_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
