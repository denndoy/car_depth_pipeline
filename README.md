# Car Depth Pipeline — Estimasi Jarak Monokular Mobil via Kuartik Luas Box

Pipeline penelitian rekan: **object mobil** (MPV / Sedan / Hatchback), detektor
**YOLOv8n COCO pretrained** (class `car` id 2), pembeda kelas **classifier
fine-tune** (dataset Kaggle Indonesian Cars, 8 kelas → dipakai 3), dan **kuartik
monoton per kelas body type** yang dikalibrasi di Take 1 terhadap marker
M01–M09 lalu dibekukan untuk Video 2–4.

Folder ini **mandiri** dan tidak mengubah satu baris pun source code penelitian
person. Skrip generik marker, crossing, review track, dan validasi kuartik yang
dibutuhkan sudah **disalin ke `scripts/vendor_*`** agar repository ini tidak
mengimpor file dari luar saat dijalankan. Source asli penelitian person tidak
diubah.

**Perintah pada README diasumsikan dijalankan dari root repository ini.**
Dataset/video dan checkpoint `.pt` tidak ikut dilacak Git. Aset kendaraan CC0
ada di `3d model/`. `requirements.txt` memasang dependensi Python; Blender 5.2
LTS dipasang terpisah untuk tahap 3D.

Dokumen metodologi lengkap: [PROTOKOL_PENELITIAN_MOBIL.md](PROTOKOL_PENELITIAN_MOBIL.md).

## Setup standalone

Di Windows, jalankan dari root hasil clone repository:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Blender 5.2 LTS diperlukan hanya untuk `scripts/12_generate_urban_car_scene.py`.
Siapkan secara lokal (tidak masuk Git):

- `dataset-classification.zip` — unduh dari Kaggle URL di tabel keputusan.
- `data/` — MP4 penelitian; blur plat untuk materi yang dibagikan.
- `models/yolov8n.pt` dan classifier `best.pt`/`.pt.zip` — bobot tidak disimpan di repo.
- Aset `.blend` sudah tersedia di `3d model/`; metadata embedded menyatakan
  Grab3D / Cozy Retro Cars Kit, CC0-1.0. Lihat `3d model/README.md`.

`.gitignore` mencegah zip media/dataset, weights, hasil bersih, video, dan hasil
generasi 3D ikut ter-commit secara tidak sengaja; tiga aset sumber CC0 tetap
termasuk dalam repo.

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
├── requirements.txt                  <- dependensi Python
├── .gitignore                        <- cegah data/weights/output lokal masuk Git
├── configs/bodytype_config.json      <- kelas, threshold fallback, guard kendaraan
├── inputs_templates/                 <- template CSV + kamus kolom (README.md di dalamnya)
├── scripts/
│   ├── 00_prepare_classifier_dataset.py   Kaggle 8 kelas -> 3 kelas Ultralytics classify
│   ├── 01_track_cars.py                   YOLOv8n COCO car + ByteTrack per take
│   ├── 02_classify_car_tracks.py          body-type per track + flag fallback
│   ├── 03_fit_class_quartics.py           kuartik global + per kelas (monoton) Take 1
│   ├── 04_evaluate_frozen.py              evaluasi beku Video 2-4 (end-to-end/oracle/global)
│   ├── 05_enrich_events_with_class.py     isi body_class_gt dari vehicle_metadata
│   ├── 06_prepare_blind_crop_review.py    crop box + label manual tanpa prediksi classifier
│   ├── 07_build_reviewed_classifier_dataset.py susun train/val dari label manual
│   ├── 08_finetune_car_classifier.py      fine-tune checkpoint classifier 3 kelas
│   ├── 09_score_classifier_review.py      metrik classifier vs label manual
│   ├── 10_prepare_deduplicated_classifier_dataset.py filter 3 kelas + hash audit/dedup
│   ├── 11_export_frame_trajectories.py    frozen quartic + classifier -> per-frame car trajectories
│   ├── 12_generate_urban_car_scene.py     Blender urban scene + GLB from tracks and car assets
│   ├── vendor_tracking/                   manual track/vehicle crosswalk helpers
│   └── vendor_distance/                   marker line + frozen quartic helpers
├── tests/test_smoke_synthetic.py     smoke test sintetis 00->03->04->05->07->09->11
```

Keluaran eksperimen ada di `outputs/`; cleaned classifier dataset ada di
`datasets/`. Keduanya sengaja masuk `.gitignore` agar media/hasil lokal tidak
ikut terunggah.

## Alur eksekusi

### Fase A — Dataset & fine-tune classifier

```powershell
# 1. Filter 3 kelas dan buang duplikat persis/conflict label (ZIP asli read-only)
.\.venv\Scripts\python.exe scripts\10_prepare_deduplicated_classifier_dataset.py `
  --source-zip dataset-classification.zip `
  --output-dir datasets\indonesian_cars_3class_deduplicated_v1

# 2a. Jika belum punya checkpoint 3-kelas: fine-tune dari YOLOv8n-cls pretrained
.\.venv\Scripts\yolo.exe classify train `
  model=yolov8n-cls.pt `
  data=datasets\indonesian_cars_3class_deduplicated_v1 `
  epochs=50 imgsz=224 batch=16 patience=10 `
  project=outputs\classifier_training name=car3_clean_v1

# 2b. Jika checkpoint 3-kelas sudah ada: lanjutkan/fine-tune checkpoint tersebut
.\.venv\Scripts\python.exe scripts\08_finetune_car_classifier.py `
  --dataset datasets\indonesian_cars_3class_deduplicated_v1 `
  --weights <path-to-three-class-best.pt-or-pt.zip> `
  --out-dir outputs\classifier_finetune_kaggle_clean_v1 `
  --name car3_clean --epochs 40 --patience 10 --imgsz 224 --batch 16 --device cpu
```

ZIP bersih saat ini berisi **6.190 gambar**: train 5.050, val 580, test 560.
Sebanyak 53 grup hash memiliki label konflik antar-kelas sumber (537 salinan
target dikeluarkan); 836 salinan duplikat lain dihapus dengan prioritas train →
val → test. Lihat `summary.json`, `duplicate_audit.csv`, dan
`cross_class_conflicts.csv`. Dedup berbasis byte persis tidak mendeteksi
near-duplicate/satu kendaraan dari foto berbeda.

Model yang sedang dilanjutkan sudah 3-kelas; metadata checkpoint mencatat 50
epoch. Karena provenance dataset tepat checkpoint belum terverifikasi, jangan
anggap Kaggle test split untouched untuk checkpoint tersebut. Video penelitian
2–4 tetap harus disimpan sebagai test domain-lokal setelah classifier dibekukan.

### Fase B — Setup fisik & marker line (helper vendored)

Isi `inputs/` dari `inputs_templates/` (lihat kamus kolom di
`inputs_templates/README.md`), lalu:

```powershell
# Anotasi dasar cone di browser offline (download points_reviewed.csv dari halaman)
.\.venv\Scripts\python.exe scripts\vendor_distance\marker_line_tools.py annotate `
  --references inputs\marker_line_references.csv `
  --marker-ground-truth inputs\marker_ground_truth.csv `
  --setup-id CAR_SETUP --lane-width-m 6 --annotation-method cone-base `
  --output-dir outputs\marker_lines\annotation_01

# Bangun profil 9 garis crossing (setelah koordinat dikonfirmasi operator)
.\.venv\Scripts\python.exe scripts\vendor_distance\marker_line_tools.py build-profile `
  --references inputs\marker_line_points_reviewed.csv `
  --marker-ground-truth inputs\marker_ground_truth.csv `
  --setup-id CAR_SETUP --lane-width-m 6 --same-camera-view-confirmed `
  --output-dir outputs\marker_lines\profile_01

# QC overlay garis vs take
.\.venv\Scripts\python.exe scripts\vendor_distance\marker_line_tools.py preview `
  --profile outputs\marker_lines\profile_01\marker_lines_setup.csv `
  --video <take1.mp4> --same-camera-view-confirmed `
  --marker-ground-truth inputs\marker_ground_truth.csv `
  --output-dir outputs\marker_lines\preview_01
```

### Fase C — Tracking mobil per take

```powershell
.\.venv\Scripts\python.exe scripts\01_track_cars.py `
  --video data\take1.mp4 --weights models\yolov8n.pt `
  --run-id take1_car_01 --imgsz 1280 --conf 0.05 --threads 4
```

Pertahankan bobot/imgsz/conf/tracker **identik** untuk semua take.

### Fase D — Mapping track→kendaraan (Take 1 saja) & event crossing

```powershell
# Review track->kendaraan (halaman HTML; kendaraan dari review_subjects.csv)
.\.venv\Scripts\python.exe scripts\vendor_tracking\prepare_track_review.py `
  --run-dir outputs\tracking\take1_car_01 --video <take1.mp4> `
  --subjects inputs\review_subjects.csv `
  --output-dir outputs\review\take1_track_01 `
  --html-name REVIEW_TRACK_CAR_TAKE1.html --download-name take1_track_mapping_reviewed.csv

# Terapkan hasil review manual
.\.venv\Scripts\python.exe scripts\vendor_tracking\apply_subject_crosswalk.py `
  --run-dir outputs\tracking\take1_car_01 `
  --crosswalk inputs\take1_track_mapping_reviewed.csv

# Kandidat crossing (mobil mendekat kamera; mulai --direction either, lalu dikunci)
.\.venv\Scripts\python.exe scripts\vendor_distance\marker_line_tools.py candidates `
  --profile outputs\marker_lines\profile_01\marker_lines_setup.csv `
  --video <take1.mp4> --same-camera-view-confirmed `
  --marker-ground-truth inputs\marker_ground_truth.csv `
  --detections outputs\tracking\take1_car_01\detections_subject_mapped.csv `
  --take-id Take1 --subjects CAL_MPV_01 CAL_MPV_02 CAL_MPV_03 CAL_SEDAN_01 CAL_SEDAN_02 CAL_SEDAN_03 CAL_HATCHBACK_01 CAL_HATCHBACK_02 CAL_HATCHBACK_03 `
  --clip-offscreen-bumper --output-dir outputs\events\candidates_take1_01

# Review crossing di HTML -> download take1_events_reviewed.csv -> export-confirmed
.\.venv\Scripts\python.exe scripts\vendor_distance\marker_line_tools.py export-confirmed `
  --profile outputs\marker_lines\profile_01\marker_lines_setup.csv `
  --video <take1.mp4> --same-camera-view-confirmed `
  --marker-ground-truth inputs\marker_ground_truth.csv `
  --detections outputs\tracking\take1_car_01\detections_subject_mapped.csv `
  --reviewed-events inputs\take1_events_reviewed.csv `
  --output inputs\take1_events_confirmed_raw.csv

# Lengkapi kolom kelas ground truth (wajib untuk kalibrasi & oracle)
.\.venv\Scripts\python.exe scripts\05_enrich_events_with_class.py `
  --events inputs\take1_events_confirmed_raw.csv `
  --vehicle-metadata inputs\vehicle_metadata.csv `
  --output-dir inputs\enriched
```

Ulangi Fase C-D untuk Take 2–4. Pada helper crossing mobil, jika `subject_id`
kosong maka `track_id` dipakai sebagai pseudo vehicle ID untuk membuat kandidat
event; gunakan raw `detections.csv` dan isi `body_class_gt` secara manual pada
event CSV (atau metadata kendaraan) untuk oracle/confusion matrix. Jika ByteTrack
memecah satu kendaraan ke beberapa segmen, satukan segmen secara manual sebelum
menetapkan ID kendaraan yang sama.

### Fase E — Kalibrasi kuartik kelas (Take 1)

```powershell
.\.venv\Scripts\python.exe scripts\03_fit_class_quartics.py `
  --events inputs\enriched\take1_events_confirmed_raw.csv `
  --detections outputs\tracking\take1_car_01\detections_subject_mapped.csv `
  --marker-ground-truth inputs\marker_ground_truth.csv `
  --output-dir outputs\quartic_fit\take1_01
```

Output kunci: `fitted_models.json` (1 global + 3 kelas, semuanya monoton
terverifikasi), `calibration_summary.json` (MAE/RMSE in-sample),
`monotonicity_diagnostics.csv`. Peringatan in-sample besar adalah perilaku
normal kuartik pada luas mentah di rentang jarak lebar — angka resmi adalah
hasil evaluasi beku, bukan in-sample.

### Fase F — Classifier per track + evaluasi beku (Take 2–4)

```powershell
# Klasifikasi body type per track untuk tiap take uji
.\.venv\Scripts\python.exe scripts\02_classify_car_tracks.py `
  --run-dir outputs\tracking\take2_car_01 --video data\take2.mp4 `
  --model outputs\classifier_training\car3_clean_v1\weights\best.pt `
  --output-dir outputs\bodyclass\take2_01

# Evaluasi beku: end-to-end (classifier pilih kuartik) + oracle (kelas GT) + global
.\.venv\Scripts\python.exe scripts\04_evaluate_frozen.py `
  --model outputs\quartic_fit\take1_01\fitted_models.json `
  --events inputs\enriched\take2_events_confirmed_raw.csv `
            inputs\enriched\take3_events_confirmed_raw.csv `
            inputs\enriched\take4_events_confirmed_raw.csv `
  --detections outputs\tracking\take2_car_01\detections.csv `
               outputs\tracking\take3_car_01\detections.csv `
               outputs\tracking\take4_car_01\detections.csv `
  --bodyclass-predictions outputs\bodyclass\take2_01\track_bodyclass_predictions.csv `
                          outputs\bodyclass\take3_01\track_bodyclass_predictions.csv `
                          outputs\bodyclass\take4_01\track_bodyclass_predictions.csv `
  --marker-ground-truth inputs\marker_ground_truth.csv `
  --output-dir outputs\quartic_eval\take2to4_01
```

Output kunci: `distance_predictions.csv` (prediksi per event + alasan pemilihan
model), `metrics_by_scope.csv` (per video/kelas/marker untuk ketiga varian),
`classifier_confusion.csv` (kontribusi error classifier), `evaluation_summary.json`.

### Fase G — Trajektori frame dan generasi 3D urban

Jalankan setelah ada model kuartik Take 1, tracking, dan classifier per track
untuk video target:

```powershell
.\.venv\Scripts\python.exe scripts\11_export_frame_trajectories.py `
  --models outputs\quartic_fit\take1_01\fitted_models.json `
  --detections outputs\bodyclass\take2_01\detections_with_bodyclass.csv `
  --tracking-summary outputs\tracking\take2_car_01\summary.json `
  --video-id take2 --road-width-m 6 `
  --output outputs\3d\take2\frame_trajectories.csv

blender --background --python scripts\12_generate_urban_car_scene.py -- `
  --trajectories outputs\3d\take2\frame_trajectories.csv `
  --assets-dir "3d model" `
  --output-blend outputs\3d\take2\urban_car_scene.blend `
  --output-glb outputs\3d\take2\urban_car_scene.glb --render-preview
```

Scene: jalan kota, trotoar, marka, dan **gedung-gedung di kedua sisi**; tidak
menyalin scene person yang memakai pepohonan/verge. Model `1990s-minivan.blend`
dipakai sebagai representasi visual MPV. Hasil lateral bersifat skematis dari
posisi x bounding box; jarak longitudinal berasal dari kuartik beku. Aset
kendaraan tetap lokal sampai sumber/lisensinya dapat dicatat.

## Vendored helper scripts

Repository ini berdiri sendiri dari source tree person; salinan helper yang
dibutuhkan ada di:

| File di repo ini | Peran |
|---|---|
| `scripts/vendor_distance/marker_line_tools.py` | anotasi marker, profile, preview, kandidat bumper, export event manual |
| `scripts/vendor_tracking/prepare_track_review.py` | review offline track→kendaraan Take 1 |
| `scripts/vendor_tracking/apply_subject_crosswalk.py` | terapkan mapping manual tanpa menimpa raw detections |
| `scripts/vendor_distance/polynomial_take_pipeline.py` | validator GT/event/detection/metadata dan fungsi prediksi |
| `scripts/vendor_distance/evaluate_final_monotone_quartics.py` | fit kuartik monoton + verifikasi turunan secara analitik |

Kelima helper disalin dari modul generik penelitian person; berkas aslinya
tidak diubah. `scripts/vendor_tracking/README.md` dan
`scripts/vendor_distance/README.md` mencatat fungsinya.

## Smoke test

```powershell
.\.venv\Scripts\python.exe tests\test_smoke_synthetic.py
```

Menjalankan 00 dan 10, 03 → 04 → 05, 07 → 09 pada data sintetis, memverifikasi
hash de-dup/conflict handling, quartic monotonicity, classifier routing/fallback,
coverage, manual split-map/video-leakage guard, confusion matrix, track-id
pseudo-vehicle mapping, serta export trajektori script 11.
Script 12 sudah diuji lokal dengan aset kendaraan yang tersedia, data sintetis,
Blender 5.2.2, GLB export, dan render preview urban.

## Etika data

Blur **plat nomor** pada semua foto/video/crop sebelum dibagikan atau diunggah;
media tetap lokal, hanya CSV parameter dan metrik yang di-share.

## Fine-tune classifier dari crop video berlabel manual

`ultralytics_detection_4videos.zip` berisi **frame penuh + YOLO box kelas car**,
bukan crop beratribut MPV/Sedan/Hatchback. Gunakan urutan ini untuk membuat
ground truth manual terlebih dahulu. Script review tidak memuat/menampilkan
prediksi model agar label tidak bias oleh tebakan checkpoint.

> **Pencegahan data leakage:** split `train` (Video 1–3) / `val` (Video 4) di
> ZIP dibuat untuk object detection. Itu **belum tentu** split yang benar untuk
> classifier penelitian. Jika Video 2–4 akan menjadi video uji estimasi jarak,
> jangan gunakan crop Video 2–4 untuk fine-tuning atau memilih threshold
> classifier. Sebelum menyusun dataset, isi
> `inputs_templates/classifier_video_split_TEMPLATE.csv` dengan peran video
> penelitian: `train`, `val`, `test`, atau `exclude`. Video test tidak boleh
> ikut fine-tuning/early stopping. Jika tidak tersedia video val non-test,
> gunakan validation split eksternal Kaggle atau rekam video tambahan.

Salin template split map dan tentukan peran sebelum menyusun dataset:

```powershell
New-Item -ItemType Directory -Force inputs | Out-Null
Copy-Item inputs_templates\classifier_video_split_TEMPLATE.csv `
  inputs\classifier_video_split.csv
```

Isi satu baris per video. Bila Video 2–4 adalah held-out untuk penelitian
jarak, jangan masukkan video tersebut ke `train` atau `val`. Jika pemetaan itu
membuat tidak ada validation video, **jangan** memakai test video sebagai
pengganti; siapkan validation eksternal dari Kaggle atau rekam video val
terpisah sebelum menjalankan fine-tune.

### 1. Buat crop dan buka halaman anotasi blind

```powershell
.\.venv\Scripts\python.exe scripts\06_prepare_blind_crop_review.py `
  --detection-zip ultralytics_detection_4videos.zip `
  --out-dir outputs\classifier_manual_review_01
```

Buka `outputs\classifier_manual_review_01\review.html` di
browser. ZIP saat ini menghasilkan 255 crop dari 43 frame. Kolom split yang
ditampilkan adalah split **deteksi sumber**, bukan keputusan train/val/test
classifier. Label crop yang masuk sesuai protokol sebagai `MPV`, `Sedan`, atau
`Hatchback`; gunakan `Uncertain`/`Skip` untuk mobil yang terlalu kecil,
terpotong, atau tidak cukup terlihat untuk dipastikan. **Jangan mengisi kelas
berdasarkan model prediction.** Klik **Unduh CSV hasil review** secara berkala;
simpan hasilnya sebagai:

```text
outputs/classifier_manual_review_01/car_bodytype_reviewed.csv
```

### 2. Ukur baseline checkpoint saat ini (opsional, disarankan)

Setelah label manual selesai, prediksi ulang crop dengan checkpoint awal,
kemudian hitung metrik pada split penelitian yang dipilih. Ini mengukur
performa domain awal sebelum fine-tune. Bandingkan checkpoint awal dan hasil
fine-tune pada `val`; simpan `test` untouched sampai keputusan model/threshold
sudah dibekukan.

```powershell
.\.venv\Scripts\python.exe ultralytics_classifier_from_detection.py generate `
  --detection-zip ultralytics_detection_4videos.zip `
  --classifier-weights yolov8ncls.pt.zip `
  --out-dir outputs\classifier_baseline_01 --device cpu

.\.venv\Scripts\python.exe scripts\09_score_classifier_review.py `
  --reviewed-csv outputs\classifier_manual_review_01\car_bodytype_reviewed.csv `
  --predictions-csv outputs\classifier_baseline_01\all_predictions.csv `
  --video-split-map inputs\classifier_video_split.csv `
  --split val --out-dir outputs\classifier_baseline_metrics_01
```

`generate` sendiri hanya membuat pseudo-label; `09_score_classifier_review.py`
baru menghitung metrik dengan membandingkannya ke label manual.

### 3. Susun dataset training dari label manual

```powershell
.\.venv\Scripts\python.exe scripts\07_build_reviewed_classifier_dataset.py `
  --review-dir outputs\classifier_manual_review_01 `
  --reviewed-csv outputs\classifier_manual_review_01\car_bodytype_reviewed.csv `
  --video-split-map inputs\classifier_video_split.csv `
  --out-dir outputs\classifier_dataset_manual_01
```

Builder menyalin **hanya label manual**, menolak crop yang belum diberi label,
mengeluarkan Uncertain/Skip, dan mengikuti split penelitian per video. File
`classifier_video_split.csv` harus berisi satu baris per `source_video_id`
dengan split `train`, `val`, `test`, atau `exclude`. Crop dari satu video tidak
boleh dibagi acak antar-split. Jangan sekadar menyalin train/val sumber ZIP
jika itu memasukkan video uji ke train/val.

### 4. Fine-tune checkpoint 3-kelas yang sudah ada

```powershell
.\.venv\Scripts\python.exe scripts\08_finetune_car_classifier.py `
  --dataset outputs\classifier_dataset_manual_01 `
  --weights yolov8ncls.pt.zip `
  --out-dir outputs\classifier_finetune_01 `
  --name local_video_adaptation --epochs 40 --patience 10 --imgsz 224 --batch 16 --device cpu
```

Model terbaik akan ada di
`outputs\classifier_finetune_01\runs\local_video_adaptation\weights\best.pt`.
Jumlah contoh lokal saat ini kecil, jadi cek kurva train/val dan jangan menaikkan
epoch hanya karena training accuracy terus naik.

### 5. Bandingkan classifier awal vs hasil fine-tune

Ulangi `ultralytics_classifier_from_detection.py generate` dengan
`--classifier-weights` mengarah ke `best.pt` (gunakan output-dir baru), lalu
jalankan `09_score_classifier_review.py` terhadap CSV label manual dan
`classifier_video_split.csv` yang sama.
Bandingkan macro-F1, confusion matrix, serta precision/recall per kelas pada
split val. Setelah model dan threshold dibekukan, evaluasi split `test` satu
kali untuk angka generalisasi final. Jangan memakai test untuk early stopping,
pemilihan threshold, atau retraining; jika metrik test dilihat lalu dipakai
untuk mengubah model, test itu tidak lagi untouched.

### Batasan penting

- Inferensi awal pada model yang diberikan menghasilkan 255 crop: prediksi
  137 MPV, 106 Sedan, 12 Hatchback. Ini **distribusi prediksi saja**, bukan
  distribusi ground truth atau bukti kualitas.
- Pada output tersebut 92/255 crop masuk antrean review karena confidence
  rendah, crop kecil, terpotong tepi, atau prediksi ambigu. Tinjau kondisi ini
  sebelum memutuskan label; jangan memaksa kelas jika tipe bodi tidak terlihat.
- Detection ZIP saat ini menempatkan Video 1–3 di train (180 crop) dan Video 4
  di val (75 crop), tetapi split tersebut **tidak otomatis sah** untuk desain
  penelitian mobil. Tetapkan peran penelitian sebelum label masuk ke fine-tune.
- Checkpoint awal memiliki 3 kelas yang benar; metadata-nya mencatat 50 epoch,
  `imgsz=224`, `batch=1024`, `lr0=0.001`, dan basis `yolov8n-cls`. Path dataset
  di checkpoint bersifat internal; provenance Kaggle perlu dicatat dari sumber
  training yang sebenarnya.
- Tidak ada training yang dijalankan otomatis oleh langkah persiapan ini.
  `08_finetune_car_classifier.py` baru dijalankan setelah label manual selesai.
