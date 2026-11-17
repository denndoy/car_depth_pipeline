# Template Input `inputs/`

Salin (jangan pindahkan) template ke `inputs/`, isi datanya,
dan **jangan mengubah nama kolom** — skrip memvalidasi kolom ini.

| Template | Disalin menjadi | Dipakai oleh |
|---|---|---|
| `marker_ground_truth_TEMPLATE.csv` | `marker_ground_truth.csv` | Semua perintah `--marker-ground-truth` |
| `marker_line_references_TEMPLATE.csv` | `marker_line_references.csv` | `marker_line_tools.py annotate` |
| `vehicle_metadata_TEMPLATE.csv` | `vehicle_metadata.csv` | Dokumentasi kendaraan (provenance, bukan input skrip wajib) |
| `review_subjects_TEMPLATE.csv` | `review_subjects.csv` | `prepare_track_review.py --subjects` (Take 1) |
| `events_confirmed_TEMPLATE.csv` | `take1_events_confirmed.csv`, `take2_events_confirmed.csv`, dst. | `03_fit_class_quartics.py --events`, `04_evaluate_frozen.py --events` |
| `classifier_video_split_TEMPLATE.csv` | `classifier_video_split.csv` | `07_build_reviewed_classifier_dataset.py`, `09_score_classifier_review.py` |

## `marker_ground_truth.csv` — jarak marker (GT)

| Kolom | Arti |
|---|---|
| `marker_id` | `M01`–`M09` (wajib persis id ini; skrip menolak set lain) |
| `nominal_position_m` | Posisi nominal (opsional) |
| `gt_distance_m` | **Jarak longitudinal terukur dari titik 0 sejajar kamera** — hasil pengukuran lapangan, bukan hasil gambar |
| `derivation` | Catatan cara pengukuran |

Aturan: semua take memakai tabel ini; **jangan mencampur** dengan tabel GT
penelitian person (10–90 m / 6,6–59,4 m). GT tidak pernah dikoreksi lateral.

## `marker_line_references.csv` — manifest referensi marker

| Kolom | Arti |
|---|---|
| `reference_path` | Foto atau klip referensi per marker (relatif root repo) |
| `reference_frame_idx` | Indeks frame sumber (0-based) bila referensi berupa klip; kosong bila foto |
| `x_left_px`, `y_left_px`, `x_right_px`, `y_right_px` | Dua dasar cone kiri-kanan di px sumber — diisi lewat halaman anotasi browser, bukan diketik manual |
| `position_confirmed` | `TRUE` hanya setelah review operator |
| `notes` | Catatan |

## `vehicle_metadata.csv` — metadata kendaraan

| Kolom | Arti |
|---|---|
| `vehicle_id` | ID anonim kendaraan, mis. `CAL_MPV_01` (kalibrasi) / `T2_MPV_01` (video uji) |
| `body_class_gt` | Kelas ground truth: `MPV` / `Sedan` / `Hatchback` |
| `make_model`, `length_m`, `width_m`, `height_m`, `color` | Dokumentasi; berguna untuk analisis variasi dalam kelas |
| `plate_action` | Harus `blurred` untuk media yang dibagikan |
| `notes` | Catatan |

Mobil video uji **harus berbeda** dari mobil kalibrasi (skenario terkunci).

## `review_subjects.csv` — daftar kendaraan untuk halaman review track

Dua kolom saja: `subject_id` (harus sama dengan `vehicle_id` di
`vehicle_metadata.csv`) dan `label`. Kolom `subject_id` dipertahankan agar
skrip crosswalk penelitian person tetap kompatibel tanpa diubah.

## `take*_events_confirmed.csv` — event crossing terkonfirmasi

Wajib: `event_id` unik, `take_id`, `video_id`, `subject_id` (ID kendaraan;
video uji boleh pseudo-ID seperti `T2_MPV_01`), `marker_id`, `frame_idx`,
`foot_crossing_confirmed`=`TRUE` (legacy column name; untuk mobil artinya bumper
depan pada garis sudah dikonfirmasi), `annotation_source` yang menunjukkan review
manual (validator menolak sumber auto/kandidat), **`body_class_gt`** (label
kelas ground truth — kalibrasi dan oracle memakai kolom ini, bukan output
classifier). Opsional: `gt_distance_m` (harus cocok tabel bila diisi),
`pass_id`, `lane`, `timestamp_s`, `track_id`, `notes`.

Kolom tambahan diabaikan oleh skrip; jangan menghapus/mengganti nama kolom wajib.

## `classifier_video_split.csv` — split berbasis peran video penelitian

Kolom wajib: `source_video_id,split`. Salin template dan isi `split` sebagai:

- `train` — boleh dipakai fine-tuning classifier.
- `val` — boleh dipakai pemilihan model/early stopping, bukan test final.
- `test` — hanya evaluasi final; tidak boleh dipakai training, early stopping,
  pemilihan threshold, atau pengembangan setelah metrik test dilihat.
- `exclude` — tidak dimasukkan ke dataset classifier.

**Jangan menganggap split `train`/`val` bawaan ZIP deteksi sebagai split classifier.**
Pada ZIP saat ini Video 1–3 bertanda train dan Video 4 bertanda val. Bila Video
2–4 adalah video uji utama penelitian, split deteksi tersebut akan membocorkan
Video 2–3 ke fine-tuning. Tetapkan peran berdasarkan protokol eksperimen dan
pastikan semua frame/kendaraan dari video yang sama tetap dalam satu split.
