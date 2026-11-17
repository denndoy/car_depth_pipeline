# Car Depth Pipeline — Estimasi Jarak Monokular Mobil via Kuartik Luas Box

Pipeline penelitian rekan: **object mobil** (MPV / Sedan / Hatchback), detektor
**YOLOv8n COCO pretrained** (class `car` id 2), pembeda kelas **classifier
fine-tune** (dataset Kaggle Indonesian Cars, 8 kelas → dipakai 3), dan **kuartik
monoton per kelas body type** yang dikalibrasi di Take 1 terhadap marker
M01–M09 lalu dibekukan untuk Video 2–4.

Folder ini **mandiri** dan tidak mengubah satu baris pun source code penelitian
person. Skrip generik penelitian person (anotasi marker, kandidat crossing,
review, crosswalk) dipakai ulang **read-only** lewat perintah CLI.

Dokumen metodologi lengkap: [PROTOKOL_PENELITIAN_MOBIL.md](PROTOKOL_PENELITIAN_MOBIL.md).

## Keputusan desain terkunci

| # | Keputusan | Nilai |
|---|---|---|
| D1 | Unit kuartik | **Per kelas body type** (MPV/Sedan/Hatchback) — mobil antar video berbeda, personal-per-kendaraan mustahil |
| D2 | Definisi crossing | **Bumper depan menyentuh garis marker**; mobil **selalu mendekat ke kamera** |
| D3 | Classifier | Fine-tune dari [Kaggle Indonesian Cars Classification](https://www.kaggle.com/datasets/muhammadluthfiarifin/indonesian-cars-classification-dataset) (mpv/sedan/hatchback dari 8 kelas) |
| D4 | Video | 30 fps — mitigasi lewat aturan review frame & dilaporkan sebagai limitasi |
| D5 | Fallback | Confidence classifier < threshold (default 0.6, dibekukan di bundle model) → **kuartik global** |
| D6 | Label kelas kalibrasi | **Ground truth manual**, classifier hanya untuk inference |
| D7 | Mobil kalibrasi per kelas | Minimal 2 (rekomendasi 3) — guard di `03_fit_class_quartics.py` |

## Struktur folder

```
car_depth_pipeline/
├── README.md                         <- file ini
├── PROTOKOL_PENELITIAN_MOBIL.md      <- protokol lengkap (metodologi & lapangan)
├── configs/bodytype_config.json      <- kelas, threshold fallback, guard kendaraan
├── inputs_templates/                 <- template CSV + kamus kolom (README.md di dalamnya)
├── scripts/
│   ├── 00_prepare_classifier_dataset.py   Kaggle 8 kelas -> 3 kelas Ultralytics classify
│   ├── 01_track_cars.py                   YOLOv8n COCO car + ByteTrack per take
│   ├── 02_classify_car_tracks.py          body-type per track + flag fallback
│   ├── 03_fit_class_quartics.py           kuartik global + per kelas (monoton) Take 1
│   ├── 04_evaluate_frozen.py              evaluasi beku Video 2-4 (end-to-end/oracle/global)
│   └── 05_enrich_events_with_class.py     isi body_class_gt dari vehicle_metadata
└── tests/test_smoke_synthetic.py     smoke test sintetis 00->03->04->05
```

Semua output ditulis ke `car_depth_pipeline/outputs/` (tracking, marker_lines,
events, quartic_fit, quartic_eval, bodyclass) agar tidak bercampur artefak
penelitian person di `outputs/` lama.

## Alur eksekusi

### Fase A — Dataset & fine-tune classifier

```powershell
# 1. Susun dataset 3 kelas dari dataset Kaggle 8 kelas (tanpa mengubah sumber)
.\.venv\Scripts\python.exe car_depth_pipeline\scripts\00_prepare_classifier_dataset.py `
  --src <path-folder-dataset-kaggle> `
  --dst car_depth_pipeline\datasets\car3

# 2. Fine-tune (rekan sudah menjalankan; perintah acuan Ultralytics)
.\.venv\Scripts\python.exe -m ultralytics classify train `
  data=car_depth_pipeline\datasets\car3 model=yolov8n-cls.pt `
  epochs=50 imgsz=224 batch=16 project=car_depth_pipeline\outputs\classifier_training name=car3_v1
```

Validasi classifier **pada crop asli video penelitian** (domain nyata), bukan
hanya split dataset web — buat label manual kecil dari crop Take 1 dan bandingkan.

### Fase B — Setup fisik & marker line (reuse skrip person, read-only)

Isi `car_depth_pipeline/inputs/` dari `inputs_templates/` (lihat kamus kolom di
`inputs_templates/README.md`), lalu:

```powershell
# Anotasi dasar cone di browser offline (download points_reviewed.csv dari halaman)
.\.venv\Scripts\python.exe scripts\04_distance_evaluation\marker_line_tools.py annotate `
  --references car_depth_pipeline\inputs\marker_line_references.csv `
  --marker-ground-truth car_depth_pipeline\inputs\marker_ground_truth.csv `
  --setup-id CAR_SETUP --lane-width-m 6 --annotation-method cone-base `
  --output-dir car_depth_pipeline\outputs\marker_lines\annotation_01

# Bangun profil 9 garis crossing (setelah koordinat dikonfirmasi operator)
.\.venv\Scripts\python.exe scripts\04_distance_evaluation\marker_line_tools.py build-profile `
  --references car_depth_pipeline\inputs\marker_line_points_reviewed.csv `
  --marker-ground-truth car_depth_pipeline\inputs\marker_ground_truth.csv `
  --setup-id CAR_SETUP --lane-width-m 6 --same-camera-view-confirmed `
  --output-dir car_depth_pipeline\outputs\marker_lines\profile_01

# QC overlay garis vs take
.\.venv\Scripts\python.exe scripts\04_distance_evaluation\marker_line_tools.py preview `
  --profile car_depth_pipeline\outputs\marker_lines\profile_01\marker_lines_setup.csv `
  --video <take1.mp4> --same-camera-view-confirmed `
  --marker-ground-truth car_depth_pipeline\inputs\marker_ground_truth.csv `
  --output-dir car_depth_pipeline\outputs\marker_lines\preview_01
```

### Fase C — Tracking mobil per take

```powershell
.\.venv\Scripts\python.exe car_depth_pipeline\scripts\01_track_cars.py `
  --video <take1.mp4> --run-id take1_car_01 --imgsz 1280 --conf 0.05 --threads 4
```

Pertahankan bobot/imgsz/conf/tracker **identik** untuk semua take.

### Fase D — Mapping track→kendaraan (Take 1 saja) & event crossing

```powershell
# Review track->kendaraan (halaman HTML; kendaraan dari review_subjects.csv)
.\.venv\Scripts\python.exe scripts\03_tracking_crosswalk\prepare_track_review.py `
  --run-dir car_depth_pipeline\outputs\tracking\take1_car_01 --video <take1.mp4> `
  --subjects car_depth_pipeline\inputs\review_subjects.csv `
  --output-dir car_depth_pipeline\outputs\review\take1_track_01 `
  --html-name REVIEW_TRACK_CAR_TAKE1.html --download-name take1_track_mapping_reviewed.csv

# Terapkan hasil review manual
.\.venv\Scripts\python.exe scripts\03_tracking_crosswalk\apply_subject_crosswalk.py `
  --run-dir car_depth_pipeline\outputs\tracking\take1_car_01 `
  --crosswalk car_depth_pipeline\inputs\take1_track_mapping_reviewed.csv

# Kandidat crossing (mobil mendekat kamera; mulai --direction either, lalu dikunci)
.\.venv\Scripts\python.exe scripts\04_distance_evaluation\marker_line_tools.py candidates `
  --profile car_depth_pipeline\outputs\marker_lines\profile_01\marker_lines_setup.csv `
  --video <take1.mp4> --same-camera-view-confirmed `
  --marker-ground-truth car_depth_pipeline\inputs\marker_ground_truth.csv `
  --detections car_depth_pipeline\outputs\tracking\take1_car_01\detections_subject_mapped.csv `
  --take-id Take1 --subjects CAL_MPV_01 CAL_MPV_02 CAL_MPV_03 CAL_SEDAN_01 CAL_SEDAN_02 CAL_SEDAN_03 CAL_HATCHBACK_01 CAL_HATCHBACK_02 CAL_HATCHBACK_03 `
  --clip-offscreen-feet --output-dir car_depth_pipeline\outputs\events\candidates_take1_01

# Review crossing di HTML -> download take1_events_reviewed.csv -> export-confirmed
.\.venv\Scripts\python.exe scripts\04_distance_evaluation\marker_line_tools.py export-confirmed `
  --profile car_depth_pipeline\outputs\marker_lines\profile_01\marker_lines_setup.csv `
  --video <take1.mp4> --same-camera-view-confirmed `
  --marker-ground-truth car_depth_pipeline\inputs\marker_ground_truth.csv `
  --detections car_depth_pipeline\outputs\tracking\take1_car_01\detections_subject_mapped.csv `
  --reviewed-events car_depth_pipeline\inputs\take1_events_reviewed.csv `
  --output car_depth_pipeline\inputs\take1_events_confirmed_raw.csv

# Lengkapi kolom kelas ground truth (wajib untuk kalibrasi & oracle)
.\.venv\Scripts\python.exe car_depth_pipeline\scripts\05_enrich_events_with_class.py `
  --events car_depth_pipeline\inputs\take1_events_confirmed_raw.csv `
  --vehicle-metadata car_depth_pipeline\inputs\vehicle_metadata.csv `
  --output-dir car_depth_pipeline\inputs\enriched
```

Ulangi Fase C-D untuk Take 2–4. Untuk take uji, mapping track→kendaraan tidak
wajib: event boleh memakai pseudo-ID kendaraan (mis. `T2_MPV_01`) dan pencocokan
deteksi berjalan lewat `track_id`. Tetap isi `body_class_gt` (label manual) di
event CSV uji untuk evaluasi oracle & confusion matrix.

### Fase E — Kalibrasi kuartik kelas (Take 1)

```powershell
.\.venv\Scripts\python.exe car_depth_pipeline\scripts\03_fit_class_quartics.py `
  --events car_depth_pipeline\inputs\enriched\take1_events_confirmed_raw.csv `
  --detections car_depth_pipeline\outputs\tracking\take1_car_01\detections_subject_mapped.csv `
  --marker-ground-truth car_depth_pipeline\inputs\marker_ground_truth.csv `
  --output-dir car_depth_pipeline\outputs\quartic_fit\take1_01
```

Output kunci: `fitted_models.json` (1 global + 3 kelas, semuanya monoton
terverifikasi), `calibration_summary.json` (MAE/RMSE in-sample),
`monotonicity_diagnostics.csv`. Peringatan in-sample besar adalah perilaku
normal kuartik pada luas mentah di rentang jarak lebar — angka resmi adalah
hasil evaluasi beku, bukan in-sample.

### Fase F — Classifier per track + evaluasi beku (Take 2–4)

```powershell
# Klasifikasi body type per track untuk tiap take uji
.\.venv\Scripts\python.exe car_depth_pipeline\scripts\02_classify_car_tracks.py `
  --run-dir car_depth_pipeline\outputs\tracking\take2_car_01 --video <take2.mp4> `
  --model car_depth_pipeline\models\car_bodyclass_best.pt `
  --output-dir car_depth_pipeline\outputs\bodyclass\take2_01

# Evaluasi beku: end-to-end (classifier pilih kuartik) + oracle (kelas GT) + global
.\.venv\Scripts\python.exe car_depth_pipeline\scripts\04_evaluate_frozen.py `
  --model car_depth_pipeline\outputs\quartic_fit\take1_01\fitted_models.json `
  --events car_depth_pipeline\inputs\enriched\take2_events_confirmed_raw.csv `
            car_depth_pipeline\inputs\enriched\take3_events_confirmed_raw.csv `
            car_depth_pipeline\inputs\enriched\take4_events_confirmed_raw.csv `
  --detections car_depth_pipeline\outputs\tracking\take2_car_01\detections.csv `
               car_depth_pipeline\outputs\tracking\take3_car_01\detections.csv `
               car_depth_pipeline\outputs\tracking\take4_car_01\detections.csv `
  --bodyclass-predictions car_depth_pipeline\outputs\bodyclass\take2_01\track_bodyclass_predictions.csv `
                          car_depth_pipeline\outputs\bodyclass\take3_01\track_bodyclass_predictions.csv `
                          car_depth_pipeline\outputs\bodyclass\take4_01\track_bodyclass_predictions.csv `
  --marker-ground-truth car_depth_pipeline\inputs\marker_ground_truth.csv `
  --output-dir car_depth_pipeline\outputs\quartic_eval\take2to4_01
```

Output kunci: `distance_predictions.csv` (prediksi per event + alasan pemilihan
model), `metrics_by_scope.csv` (per video/kelas/marker untuk ketiga varian),
`classifier_confusion.csv` (kontribusi error classifier), `evaluation_summary.json`.

## Skrip person yang dipakai ulang read-only

| Skrip penelitian person | Peran di pipeline mobil |
|---|---|
| `scripts/04_distance_evaluation/marker_line_tools.py` | anotate / build-profile / preview / candidates / export-confirmed |
| `scripts/03_tracking_crosswalk/prepare_track_review.py`, `apply_subject_crosswalk.py` | review track→kendaraan Take 1 (kolom `subject_id` dipakai sebagai ID kendaraan) |
| `scripts/04_distance_evaluation/polynomial_take_pipeline.py` | sumber validator event/deteksi/metadata yang diimpor `03` dan `04` |
| `scripts/04_distance_evaluation/evaluate_final_monotone_quartics.py` | sumber `fit_monotone_quartic` (constraint monoton + verifikasi analitik) yang diimpor `03` dan `04` |
| `models/yolov8n.pt` | bobot detektor COCO pretrained (class car) |

Tidak ada file di atas yang diubah — `git status` hanya menampilkan folder baru ini.

## Smoke test

```powershell
.\.venv\Scripts\python.exe car_depth_pipeline\tests\test_smoke_synthetic.py
```

Menjalankan 00 → 03 → 04 → 05 pada data sintetis (81 event kalibrasi, 54 event
uji, classifier sintetis dengan kasus salah-kelas / fallback / tanpa prediksi)
dan memverifikasi routing model, monotonicity, coverage, dan confusion matrix.

## Etika data

Blur **plat nomor** pada semua foto/video/crop sebelum dibagikan atau diunggah;
media tetap lokal, hanya CSV parameter dan metrik yang di-share.
