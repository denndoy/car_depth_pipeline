#!/usr/bin/env python3
"""Crop YOLO detection ZIP, pseudo-label with trained Ultralytics image classifier,
then optionally build human-reviewed classification dataset for Ultralytics Platform.

Usage:
  python ultralytics_classifier_from_detection.py generate \
    --detection-zip detection.zip --classifier-weights best.pt --out-dir output
  python ultralytics_classifier_from_detection.py finalize \
    --out-dir output --corrections-csv output/review_template.csv \
    --accept-high-confidence-train
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile
from collections import Counter

from PIL import Image, ImageOps

CLASS_NAMES = ('Hatchback', 'MPV', 'Sedan')
VALID_CORRECTIONS = {x.lower(): x for x in CLASS_NAMES}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def prepare_checkpoint(path: Path, workdir: Path) -> Path:
    """Torch checkpoints are themselves ZIP containers, including .pt.zip uploads."""
    if not path.is_file():
        raise FileNotFoundError(f'Classifier checkpoint not found: {path}')
    if path.suffix.lower() == '.pt':
        return path
    if not zipfile.is_zipfile(path):
        raise ValueError(f'Unsupported classifier weights: {path}. Expected .pt or .pt.zip')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        # A PyTorch .pt file itself is a ZIP with a data.pkl record.
        if any(name.endswith('/data.pkl') or name == 'data.pkl' for name in names):
            dest = workdir / 'uploaded_classifier.pt'
            shutil.copy2(path, dest)
            return dest
        files = [name for name in names if name.lower().endswith('.pt') and not name.endswith('/')]
        if len(files) != 1:
            raise ValueError(f'Expected exactly one .pt inside normal ZIP, found {len(files)}')
        dest = workdir / 'uploaded_classifier.pt'
        with z.open(files[0]) as src, dest.open('wb') as out:
            shutil.copyfileobj(src, out)
        return dest


def get_video_split(name: str) -> str | None:
    parts = PurePosixPath(name).parts
    for part in parts:
        if part.lower() in ('train', 'val', 'test'):
            return part.lower()
    return None


def find_detection_pairs(z: zipfile.ZipFile) -> list[tuple[str, str, str]]:
    members = set(z.namelist())
    images = [n for n in members if '/images/' in n and PurePosixPath(n).suffix.lower() in IMAGE_EXTS]
    if not images:
        raise ValueError('No images under train/images or val/images detected in ZIP.')
    pairs = []
    for image_name in sorted(images):
        split = get_video_split(image_name)
        if split is None:
            continue
        label_name = image_name.replace('/images/', '/labels/', 1)
        label_name = str(PurePosixPath(label_name).with_suffix('.txt'))
        pairs.append((split, image_name, label_name if label_name in members else ''))
    if not pairs:
        raise ValueError('No valid train/images or val/images pairs found.')
    return pairs


def parse_yolo_boxes(text: str, width: int, height: int, class_id: int = 0):
    for idx, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        toks = line.split()
        if len(toks) != 5:
            continue
        try:
            cls, cx, cy, bw, bh = map(float, toks)
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in (cls, cx, cy, bw, bh)):
            continue
        if int(cls) != class_id or bw <= 0 or bh <= 0:
            continue
        left = max(0.0, min(width, (cx - bw / 2) * width))
        right = max(0.0, min(width, (cx + bw / 2) * width))
        top = max(0.0, min(height, (cy - bh / 2) * height))
        bottom = max(0.0, min(height, (cy + bh / 2) * height))
        if right - left < 2 or bottom - top < 2:
            continue
        yield idx, left, top, right, bottom


def pad_square(im: Image.Image, size: int, fill=(114, 114, 114)) -> Image.Image:
    im = im.convert('RGB')
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (size, size), fill)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def create_crops(zippath: Path, out: Path, *, imgsz: int = 224,
                 padding: float = .04, edge_margin: float = .01,
                 min_width: int = 48, min_height: int = 32) -> list[dict]:
    crop_root = out / 'all_crops'
    crop_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(zippath, 'r') as z:
        for split, image_name, label_name in find_detection_pairs(z):
            with Image.open(io.BytesIO(z.read(image_name))) as im0:
                im = ImageOps.exif_transpose(im0).convert('RGB')
            width, height = im.size
            label_text = z.read(label_name).decode('utf-8-sig', errors='replace') if label_name else ''
            for bbox_index, x1, y1, x2, y2 in parse_yolo_boxes(label_text, width, height):
                bw, bh = x2 - x1, y2 - y1
                # Flags reflect original detector box, not the augmented crop.
                touches_edge = (x1 <= width * edge_margin or x2 >= width * (1 - edge_margin)
                                or y1 <= height * edge_margin or y2 >= height * (1 - edge_margin))
                too_small = bw < min_width or bh < min_height
                cx1 = max(0, int(math.floor(x1 - bw * padding)))
                cy1 = max(0, int(math.floor(y1 - bh * padding)))
                cx2 = min(width, int(math.ceil(x2 + bw * padding)))
                cy2 = min(height, int(math.ceil(y2 + bh * padding)))
                crop = im.crop((cx1, cy1, cx2, cy2))
                square = pad_square(crop, imgsz)
                image_hash = hashlib.sha1(image_name.encode('utf-8')).hexdigest()[:8]
                stem = PurePosixPath(image_name).stem
                crop_id = f'{split}__{stem}__{image_hash}__b{bbox_index:03d}'
                crop_file = crop_root / split / f'{crop_id}.jpg'
                crop_file.parent.mkdir(parents=True, exist_ok=True)
                square.save(crop_file, format='JPEG', quality=94)
                rows.append({
                    'crop_id': crop_id,
                    'source_split': split,
                    'source_frame': image_name,
                    'bbox_index': bbox_index,
                    'x1': int(round(x1)), 'y1': int(round(y1)),
                    'x2': int(round(x2)), 'y2': int(round(y2)),
                    'box_width': round(bw, 2), 'box_height': round(bh, 2),
                    'touches_edge': str(touches_edge).lower(),
                    'too_small': str(too_small).lower(),
                    'crop_path': str(crop_file.relative_to(out).as_posix()),
                    'predicted_label': '', 'confidence': '', 'margin': '',
                    'p_Hatchback': '', 'p_MPV': '', 'p_Sedan': '',
                    'status': '', 'review_reason': '',
                    'corrected_label': '',
                })
    return rows


def canonical_name(raw: str) -> str:
    name = str(raw).strip().lower()
    if name not in VALID_CORRECTIONS:
        raise ValueError(f'Classifier must have exactly Hatchback/MPV/Sedan, got class {raw!r}')
    return VALID_CORRECTIONS[name]


def predict_rows(rows: list[dict], checkpoint: Path, out: Path, *, batch: int,
                 imgsz: int, min_conf: float, min_margin: float,
                 device: str = '', model_override=None):
    if model_override is None:
        from ultralytics import YOLO
        pt = prepare_checkpoint(checkpoint, out)
        model = YOLO(str(pt), task='classify')
    else:
        model = model_override
    names = model.names
    classmap = {int(idx): canonical_name(name) for idx, name in names.items()}
    if len(classmap) != 3 or set(classmap.values()) != set(CLASS_NAMES):
        raise ValueError(f'Classifier classes expected Hatchback/MPV/Sedan; got {names}')
    if not rows:
        raise ValueError('No detection boxes found in input ZIP; cannot classify.')
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        paths = [str(out / r['crop_path']) for r in chunk]
        kwargs = {'source': paths, 'imgsz': imgsz, 'batch': batch, 'verbose': False}
        if device:
            kwargs['device'] = device
        predictions = model.predict(**kwargs)
        if len(predictions) != len(chunk):
            raise RuntimeError(f'Expected {len(chunk)} predictions; got {len(predictions)}')
        for row, pred in zip(chunk, predictions):
            if pred.probs is None:
                raise ValueError('Model prediction has no .probs: use YOLO classification model')
            values = pred.probs.data.detach().cpu().tolist()
            scores = {classmap[i]: float(values[i]) for i in classmap}
            ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            label, confidence = ordered[0]
            margin = confidence - ordered[1][1]
            reasons = []
            if row['too_small'] == 'true': reasons.append('small_crop')
            if row['touches_edge'] == 'true': reasons.append('truncated_at_edge')
            if confidence < min_conf: reasons.append('low_confidence')
            if margin < min_margin: reasons.append('ambiguous_prediction')
            row['predicted_label'] = label
            row['confidence'] = f'{confidence:.6f}'
            row['margin'] = f'{margin:.6f}'
            for cls in CLASS_NAMES:
                row[f'p_{cls}'] = f'{scores[cls]:.6f}'
            row['status'] = 'REVIEW' if reasons else 'HIGH_CONFIDENCE_UNVERIFIED'
            row['review_reason'] = ';'.join(reasons)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def zip_folder(folder: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(folder).as_posix())


def prepare_candidate_dataset(rows: list[dict], out: Path):
    candidates = out / 'pseudo_dataset'
    review = out / 'review_queue'
    if candidates.exists(): shutil.rmtree(candidates)
    if review.exists(): shutil.rmtree(review)
    for r in rows:
        src = out / r['crop_path']
        if r['status'] == 'HIGH_CONFIDENCE_UNVERIFIED':
            dst = candidates / r['source_split'] / r['predicted_label'] / src.name
        else:
            dst = review / 'images' / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    write_csv(review / 'review_queue.csv', [r for r in rows if r['status'] == 'REVIEW'])
    zip_folder(candidates, out / 'classification_pseudo_candidates.zip')
    zip_folder(review, out / 'classification_review_queue.zip')


def generate(args, *, model_override=None):
    input_zip, checkpoint, out = map(Path, (args.detection_zip, args.classifier_weights, args.out_dir))
    if not input_zip.is_file(): raise FileNotFoundError(input_zip)
    out.mkdir(parents=True, exist_ok=True)
    rows = create_crops(input_zip, out, imgsz=args.imgsz, padding=args.crop_padding,
                        edge_margin=args.edge_margin,
                        min_width=args.min_width, min_height=args.min_height)
    if not rows: raise ValueError('No car boxes available. Check ZIP input and labels.')
    predict_rows(rows, checkpoint, out, batch=args.batch, imgsz=args.imgsz,
                 min_conf=args.min_conf, min_margin=args.min_margin,
                 device=args.device, model_override=model_override)
    write_csv(out / 'all_predictions.csv', rows)
    # All rows are reviewable; only explicit human correction creates ground truth.
    review_cols = ['crop_id', 'source_split', 'source_frame', 'predicted_label',
                   'confidence', 'margin', 'status', 'review_reason', 'corrected_label']
    write_csv(out / 'review_template.csv', rows, review_cols)
    prepare_candidate_dataset(rows, out)
    zip_folder(out / 'all_crops', out / 'classification_all_crops.zip')
    stats = {'total_crops': len(rows),
             'split_counts': dict(Counter(r['source_split'] for r in rows)),
             'prediction_counts': dict(Counter(r['predicted_label'] for r in rows)),
             'review_status_counts': dict(Counter(r['status'] for r in rows)),
             'warning': 'Pseudo-labels are NOT ground truth. Validate val/test manually before accuracy claims.'}
    (out / 'summary.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))
    print('Outputs:', out / 'classification_pseudo_candidates.zip',
          out / 'classification_review_queue.zip', out / 'review_template.csv')
    return rows


def finalize(args):
    out = Path(args.out_dir)
    predictions_path = out / 'all_predictions.csv'
    if not predictions_path.is_file():
        raise FileNotFoundError(f'Missing {predictions_path}; run generate first')
    with predictions_path.open(newline='', encoding='utf-8-sig') as f:
        original = list(csv.DictReader(f))
    with Path(args.corrections_csv).open(newline='', encoding='utf-8-sig') as f:
        correction_rows = list(csv.DictReader(f))
    correction_map = {}
    for r in correction_rows:
        crop_id = str(r.get('crop_id') or '').strip()
        if not crop_id: continue
        correction_map[crop_id] = str(r.get('corrected_label') or '').strip()
    dest = out / 'classification_curated'
    if dest.exists(): shutil.rmtree(dest)
    records = []
    for row in original:
        raw_label = correction_map.get(row['crop_id'], '')
        explicitly_verified = bool(raw_label)
        if raw_label.lower() in ('skip', 'ignore', 'unknown', 'other', 'invalid', 'reject'):
            continue
        if explicitly_verified:
            if raw_label.lower() not in VALID_CORRECTIONS:
                raise ValueError(f'Invalid corrected_label={raw_label!r} for {row["crop_id"]}')
            label = VALID_CORRECTIONS[raw_label.lower()]
        elif args.accept_high_confidence_train and row['source_split'] == 'train' and row['status'] == 'HIGH_CONFIDENCE_UNVERIFIED':
            label = row['predicted_label']
        else:
            continue
        source = out / row['crop_path']
        if not source.is_file():
            raise FileNotFoundError(f'Missing crop {source}')
        target = dest / row['source_split'] / label / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({**row, 'final_label': label, 'human_verified': explicitly_verified})
    if not records:
        print('No reviewed/candidate crops selected; fill corrected_label first or opt into high-confidence TRAIN pseudo-labels.')
        return []
    write_csv(out / 'curated_manifest.csv', records)
    zip_folder(dest, out / 'classification_curated_for_ultralytics.zip')
    val_count = sum(r['source_split'] == 'val' for r in records)
    print('Curated dataset:', len(records), 'crops; verified validation:', val_count)
    print('Zip:', out / 'classification_curated_for_ultralytics.zip')
    if val_count == 0:
        print('WARNING: No reviewed val samples yet. Verify val images before training/evaluation.')
    return records


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate', help='Auto-crop and predict body type from YOLO detection ZIP')
    g.add_argument('--detection-zip', required=True)
    g.add_argument('--classifier-weights', required=True)
    g.add_argument('--out-dir', default='classifier_output')
    g.add_argument('--imgsz', type=int, default=224)
    g.add_argument('--batch', type=int, default=16)
    g.add_argument('--device', default='', help='Empty auto, 0 for first GPU or cpu')
    g.add_argument('--min-conf', type=float, default=.80)
    g.add_argument('--min-margin', type=float, default=.15)
    g.add_argument('--min-width', type=int, default=48)
    g.add_argument('--min-height', type=int, default=32)
    g.add_argument('--edge-margin', type=float, default=.01)
    g.add_argument('--crop-padding', type=float, default=.04)
    f = sub.add_parser('finalize', help='Export human-reviewed class folders (val only from manual corrections)')
    f.add_argument('--out-dir', default='classifier_output')
    f.add_argument('--corrections-csv', required=True)
    f.add_argument('--accept-high-confidence-train', action='store_true',
                   help='Also include unverified high-confidence TRAIN predictions; never auto-accept val')
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.command == 'generate':
        generate(args)
    elif args.command == 'finalize':
        finalize(args)


if __name__ == '__main__':
    main()
