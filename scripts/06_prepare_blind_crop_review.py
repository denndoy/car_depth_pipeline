"""Crop COCO-car detections and create an offline, blind body-type review page.

The input ZIP contains full-frame images and YOLO detection labels (usually
class 0 = car), not MPV/Sedan/Hatchback labels. This script extracts one crop
per car box and makes a local HTML annotation page. It deliberately does NOT
load or display classifier predictions, avoiding pseudo-label/confirmation
bias while creating manual ground truth.

Output includes 224x224 square-padded crops, small full-frame context images
with the selected box highlighted, a blank review_template.csv, and
review.html. Review labels are MPV, Sedan, Hatchback, Uncertain, or Skip.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = ROOT / "ultralytics_classifier_from_detection.py"
CLASS_NAMES = ("MPV", "Sedan", "Hatchback")
OUTPUT_FIELDS = [
    "crop_id", "source_split", "source_video_id", "source_frame", "bbox_index",
    "x1", "y1", "x2", "y2", "source_width", "source_height", "box_width",
    "box_height", "touches_edge", "too_small", "crop_path", "context_path",
    "corrected_label", "notes",
]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if (Path.cwd().resolve() == ROOT.parent.resolve()
            and path.parts and path.parts[0].casefold() == ROOT.name.casefold()):
        return (Path.cwd() / path).resolve()
    return (ROOT / path).resolve()


def load_crop_helpers():
    if not SOURCE_SCRIPT.is_file():
        raise FileNotFoundError(f"Crop helper script not found: {SOURCE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("ultralytics_classifier_from_detection", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load crop helper from {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_video_id(source_frame: str) -> str:
    stem = PurePosixPath(source_frame).stem
    return re.sub(r"_f\d+$", "", stem, flags=re.IGNORECASE) or stem


def save_context_frames(detection_zip: Path, out_dir: Path, crop_helpers) -> tuple[dict[str, dict], dict[str, str]]:
    frame_meta: dict[str, dict] = {}
    context_paths: dict[str, str] = {}
    with zipfile.ZipFile(detection_zip, "r") as archive:
        for split, image_name, _label_name in crop_helpers.find_detection_pairs(archive):
            image_bytes = archive.read(image_name)
            with Image.open(__import__("io").BytesIO(image_bytes)) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            source_width, source_height = image.size
            small = image.copy()
            small.thumbnail((1440, 1440), Image.Resampling.LANCZOS)
            digest = hashlib.sha1(image_name.encode("utf-8")).hexdigest()[:10]
            relative = PurePosixPath("contexts") / split / f"{digest}.jpg"
            target = out_dir.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            small.save(target, format="JPEG", quality=82, optimize=True)
            frame_meta[image_name] = {
                "source_width": source_width,
                "source_height": source_height,
                "context_path": relative.as_posix(),
            }
            context_paths[image_name] = relative.as_posix()
    return frame_meta, context_paths


def write_review_html(out_dir: Path, rows: list[dict]) -> None:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Review Manual Kelas Mobil</title>
<style>
:root{font:16px/1.45 system-ui,Segoe UI,sans-serif;color:#172033;background:#f3f5f8}*{box-sizing:border-box}body{margin:0}.top{position:sticky;top:0;background:#fff;padding:12px 18px;box-shadow:0 2px 8px #0002;z-index:3;display:flex;gap:12px;align-items:center;flex-wrap:wrap}.top strong{font-size:18px}.bar{height:9px;background:#e5e9f0;border-radius:9px;flex:1;min-width:140px;overflow:hidden}.bar span{display:block;height:100%;background:#1677ff;width:0}.layout{display:grid;grid-template-columns:minmax(270px,38%) 1fr;gap:16px;padding:16px;max-width:1600px;margin:auto}.card{background:white;border:1px solid #e0e5ed;border-radius:12px;padding:14px;box-shadow:0 2px 5px #0001}.image-title{font-size:14px;font-weight:650;margin:0 0 8px}.context-wrap{position:relative;width:100%;line-height:0}.context-wrap img{display:block;width:100%;height:auto;border-radius:6px}.box-overlay{position:absolute;border:3px solid #ff362f;box-shadow:0 0 0 9999px #0002;pointer-events:none}.crop{display:block;max-width:min(100%,420px);max-height:420px;object-fit:contain;margin:auto;background:#d9dde5;border-radius:8px}.meta{font-size:13px;color:#46536a;overflow-wrap:anywhere}.meta b{color:#172033}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.btn{border:1px solid #cbd3df;background:#fff;padding:9px 13px;border-radius:8px;font-size:15px;cursor:pointer}.btn:hover{background:#eef4ff}.btn.active{outline:3px solid #b9d4ff}.btn.MPV{border-color:#087a4b}.btn.Sedan{border-color:#1c63cc}.btn.Hatchback{border-color:#9c48be}.btn.Uncertain{border-color:#a86b00}.btn.Skip{border-color:#a32929}.primary{background:#1769d2;color:#fff;border-color:#1769d2}.note{width:100%;padding:9px;border:1px solid #cbd3df;border-radius:8px}.status{font-weight:650}.hint{font-size:13px;color:#68758b}.warning{background:#fff8e8;border:1px solid #f0d493;padding:10px;border-radius:8px;margin:10px 0}.spacer{flex:1}@media(max-width:850px){.layout{grid-template-columns:1fr}.top{position:relative}}
</style></head><body>
<div class="top"><strong>Review manual kelas mobil</strong><span id="counter"></span><div class="bar"><span id="progress"></span></div><button class="btn primary" onclick="downloadCsv()">Unduh CSV hasil review</button></div>
<main class="layout"><section class="card"><p class="image-title">Konteks frame — kotak merah adalah mobil yang dinilai</p><div class="context-wrap"><img id="context"><div id="overlay" class="box-overlay"></div></div><p class="hint">Periksa bentuk bodi pada crop dan konteks. Jika tampilan tidak cukup jelas untuk menentukan tipe, pilih Uncertain; jangan menebak.</p></section>
<section class="card"><p class="image-title">Crop mobil (tanpa prediksi classifier)</p><img id="crop" class="crop"><div id="meta" class="meta"></div><div class="warning">Ground truth kelas body type belum tersedia di ZIP deteksi. Label yang Anda pilih di sini menjadi anotasi manual untuk supervised fine-tuning.</div><div class="controls" id="labels"></div><label class="meta" for="note">Catatan (opsional)</label><input id="note" class="note" placeholder="contoh: hanya bagian depan terlihat; sulit membedakan hatchback/sedan"><div class="controls"><button class="btn" onclick="move(-1)">← Sebelumnya</button><button class="btn" onclick="move(1)">Berikutnya →</button><span class="spacer"></span><button class="btn primary" onclick="downloadCsv()">Simpan/unduh progres CSV</button></div><p class="hint">Shortcut: 1 = MPV · 2 = Sedan · 3 = Hatchback · 4 = Uncertain · 5 = Skip · ←/→ berpindah crop. Progres juga tersimpan lokal di browser; unduh CSV berkala sebagai cadangan.</p></section></main>
<script>
const rows=__ROWS_JSON__;
const labels=['MPV','Sedan','Hatchback','Uncertain','Skip'];
const key='car-body-review:'+location.pathname;
let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
let idx=0;
const $=id=>document.getElementById(id);
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function save(){try{localStorage.setItem(key,JSON.stringify(state))}catch(e){}}
function render(){if(!rows.length)return;const r=rows[idx],v=state[r.crop_id]||{};$('counter').textContent=`Crop ${idx+1} / ${rows.length}`;$('progress').style.width=(100*Object.keys(state).filter(k=>state[k].label).length/rows.length)+'%';$('crop').src=r.crop_path;$('context').src=r.context_path;$('overlay').style.left=(100*r.x1/r.source_width)+'%';$('overlay').style.top=(100*r.y1/r.source_height)+'%';$('overlay').style.width=(100*(r.x2-r.x1)/r.source_width)+'%';$('overlay').style.height=(100*(r.y2-r.y1)/r.source_height)+'%';$('meta').innerHTML=`<p><b>Video:</b> ${esc(r.source_video_id)} &nbsp; <b>Split sumber:</b> ${esc(r.source_split)}<br><b>Frame:</b> ${esc(r.source_frame)}<br><b>Box:</b> ${r.box_width}×${r.box_height}px ${r.touches_edge==='true'?'· menyentuh tepi':''} ${r.too_small==='true'?'· crop kecil':''}</p><p><b>ID crop:</b> ${esc(r.crop_id)}</p>`;$('note').value=v.notes||'';$('labels').innerHTML=labels.map((name,i)=>`<button class="btn ${name} ${v.label===name?'active':''}" onclick="choose('${name}')">${i+1}. ${name}</button>`).join('')}
function choose(label){state[rows[idx].crop_id]={label,notes:$('note').value};save();render()}
function move(delta){state[rows[idx].crop_id]={...(state[rows[idx].crop_id]||{}),notes:$('note').value};save();idx=Math.max(0,Math.min(rows.length-1,idx+delta));render()}
function csvCell(v){return '"'+String(v??'').replace(/"/g,'""')+'"'}
function downloadCsv(){state[rows[idx].crop_id]={...(state[rows[idx].crop_id]||{}),notes:$('note').value};save();const cols=Object.keys(rows[0]).concat(['corrected_label','notes']);const lines=[cols.map(csvCell).join(',')];for(const r of rows){const v=state[r.crop_id]||{};lines.push(cols.map(c=>csvCell(c==='corrected_label'?v.label||'':c==='notes'?v.notes||'':r[c])).join(','))}const blob=new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='car_bodytype_reviewed.csv';a.click();URL.revokeObjectURL(a.href)}
document.addEventListener('keydown',e=>{if(['1','2','3','4','5'].includes(e.key)){choose(labels[Number(e.key)-1])}else if(e.key==='ArrowRight'){move(1)}else if(e.key==='ArrowLeft'){move(-1)}});
render();
</script></body></html>'''
    (out_dir / "review.html").write_text(template.replace("__ROWS_JSON__", payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--crop-padding", type=float, default=0.04)
    parser.add_argument("--edge-margin", type=float, default=0.01)
    parser.add_argument("--min-width", type=int, default=48)
    parser.add_argument("--min-height", type=int, default=32)
    args = parser.parse_args(argv)
    detection_zip = resolve_path(args.detection_zip)
    out_dir = resolve_path(args.out_dir)
    if not detection_zip.is_file():
        raise FileNotFoundError(detection_zip)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty review folder: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_helpers = load_crop_helpers()
    rows = crop_helpers.create_crops(
        detection_zip, out_dir, imgsz=args.imgsz, padding=args.crop_padding,
        edge_margin=args.edge_margin, min_width=args.min_width, min_height=args.min_height,
    )
    if not rows:
        raise ValueError("No car boxes found; check ZIP images/labels and class_id=0")
    frame_meta, context_paths = save_context_frames(detection_zip, out_dir, crop_helpers)

    output_rows = []
    for row in rows:
        source_frame = row["source_frame"]
        meta = frame_meta.get(source_frame)
        if meta is None:
            raise ValueError(f"Missing context metadata for {source_frame}")
        enriched = {
            "crop_id": row["crop_id"],
            "source_split": row["source_split"],
            "source_video_id": safe_video_id(source_frame),
            "source_frame": source_frame,
            "bbox_index": row["bbox_index"],
            "x1": row["x1"], "y1": row["y1"], "x2": row["x2"], "y2": row["y2"],
            "source_width": meta["source_width"], "source_height": meta["source_height"],
            "box_width": row["box_width"], "box_height": row["box_height"],
            "touches_edge": row["touches_edge"], "too_small": row["too_small"],
            "crop_path": row["crop_path"],
            "context_path": context_paths[source_frame],
            "corrected_label": "",
            "notes": "",
        }
        output_rows.append(enriched)

    with (out_dir / "review_template.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    write_review_html(out_dir, output_rows)

    split_counts = {}
    video_by_split: dict[str, set[str]] = {}
    for row in output_rows:
        split = row["source_split"]
        split_counts[split] = split_counts.get(split, 0) + 1
        video_by_split.setdefault(split, set()).add(row["source_video_id"])
    overlap = {
        split: sorted(video_by_split.get(split, set()) & video_by_split.get(other, set()))
        for split in video_by_split for other in video_by_split if split < other
        and video_by_split.get(split, set()) & video_by_split.get(other, set())
    }
    summary = {
        "schema": "blind_car_bodytype_crop_review_v1",
        "source_detection_zip": str(detection_zip),
        "source_detection_zip_sha256": crop_helpers.hashlib.sha256(detection_zip.read_bytes()).hexdigest(),
        "crop_count": len(output_rows),
        "image_count": len(frame_meta),
        "crop_count_by_source_split": split_counts,
        "source_videos_by_split": {key: sorted(value) for key, value in video_by_split.items()},
        "video_overlap_between_splits": overlap,
        "body_type_predictions_used": False,
        "body_classes_to_annotate": list(CLASS_NAMES),
        "review_options": [*CLASS_NAMES, "Uncertain", "Skip"],
        "notes": [
            "Original YOLO labels only mean car/non-car and are not body-type ground truth.",
            "The classifier is intentionally not loaded; annotation is blind to its predictions.",
            "source_split is the original car-detection split, not automatically a research-safe classifier split.",
            "Assign train/val/test/exclude by the study protocol in inputs_templates/classifier_video_split_TEMPLATE.csv before building the classifier dataset.",
            "All crops marked Uncertain/Skip are excluded from the supervised dataset builder.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "Open review.html in a browser. Label each car crop as MPV, Sedan, or Hatchback; use Uncertain/Skip when not visually reliable.\n"
        "Download car_bodytype_reviewed.csv before closing the browser. This page does not show or use classifier predictions.\n"
        "The ZIP's train/val split was created for object detection, not necessarily for the research experiment.\n"
        "Before training, fill inputs_templates/classifier_video_split_TEMPLATE.csv with research roles train/val/test/exclude.\n"
        "If Videos 2-4 are depth-test videos, do not train or tune the classifier using those crops.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Review page: {out_dir / 'review.html'}")
    print(f"Review CSV template: {out_dir / 'review_template.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
