# Protokol Penelitian — Estimasi Jarak Monokular Mobil (MPV / Sedan / Hatchback)

Dokumen metodologi untuk penelitian rekan. Alur mengikuti pipeline penelitian
person *"Monocular Depth Estimation Using a Quartic Function of Object Pixel
Count"* dengan object diganti mobil, plus penyesuaian yang diwajibkan oleh
perbedaan skenario. Implementasi kode ada di folder `` ini;
dokumen ini menjelaskan **mengapa dan bagaimana**, bukan hanya perintah.

---

## 1. Perbedaan skenario vs penelitian person (dasar semua keputusan)

| Aspek | Penelitian person | Penelitian mobil |
|---|---|---|
| Objek | 4 subjek terdaftar (S01–S04) | Banyak mobil, tidak terdaftar |
| Antar video | Orang yang sama muncul di Video 1–4 | **Mobil berbeda di setiap video** (disjoint) |
| Kuartik "personal" | Per subjek (identitas diketahui via crosswalk manual) | **Mustahil** — mobil uji belum pernah dilihat model |
| Pengganti konsep | — | **Kuartik per kelas body type** (MPV/Sedan/Hatchback) |
| Classifier gender | Label tampilan avatar 3D saja, tidak masuk perhitungan | **Pemilih model kuartik saat inference** — peran metodologis nyata |
| Klaim evaluasi | Konsistensi antar-take pada subjek sama | **Generalisasi ke kendaraan baru sekelas** |

Konsekuensi klaim: hasil Video 2–4 di penelitian mobil mengukur generalisasi
(pernyataan yang lebih kuat), sehingga kalimat batasan di paper harus ditulis
ulang, bukan menyalin kalimat "same subjects" dari paper person.

## 2. Keputusan terkunci dan alasannya

1. **Kuartik per kelas body type** — mobil antar video berbeda; personalisasi
   ke individu digantikan personalisasi ke tipe bentuk kendaraan.
2. **Minimal 2, rekomendasi 3 mobil kalibrasi per kelas.** Kuartik MPV yang
   di-fit dari satu MPV saja sebenarnya kuartik mobil itu: Avanza vs Alphard
   beda ~1 m panjang. Mobil kalibrasi per kelas dipilih agar mencakup rentang
   ukuran dalam kelas. Skrip `03` menolak kelas dengan <2 kendaraan.
3. **Label kelas saat kalibrasi = ground truth manual** (kolom
   `body_class_gt`, diisi dari `vehicle_metadata.csv`). Kesalahan classifier
   yang ikut memcernai fitting akan meracuni model sendiri. Classifier hanya
   dipakai saat inference untuk memilih kuartik mobil tak-dikenal.
4. **Crossing = bumper depan menyentuh garis marker**, mobil **selalu mendekat
   ke kamera** → bumper depan berada di tepi bawah bounding box, sehingga
   proksi bottom-center box dan logika kandidat otomatis identik dengan
   pipeline person.
5. **Fallback ke kuartik global** saat confidence classifier di bawah threshold
   (default 0.6, dibekukan di `fitted_models.json`). Threshold dipilih dari
   data validasi classifier (bukan dari event uji!), lalu dibekukan bersama
   model; evaluator menghitung ulang fallback dari confidence mentah terhadap
   threshold bundle agar tidak tergantung flag CSV classifier.
6. **Video 30 fps** — ketidakpastian frame crossing ±½–1 frame. Pada 20 km/j
   ≈ 0,19 m/frame; 30 km/j ≈ 0,28 m/frame; 40 km/j ≈ 0,37 m/frame (vs 0,05
   m/frame orang berjalan). Mitigasi: aturan review frame (Bagian 5), dan
   ditulis sebagai limitasi paper.
7. **Kamera & skenario fisik**: tripod tetap di titik 0, satu view untuk semua
   take, resolusi sama kalibrasi vs uji (evaluator menolak resolusi berbeda),
   mobil melaju **sepanjang lintasan melewati ketiga garis setiap marker**,
   satu mobil melintas pada satu waktu agar tidak saling menutupi.

## 3. Data dan CSV

Semua template ada di `inputs_templates/` dengan kamus kolom di README-nya.
Ringkasan:

- `marker_ground_truth.csv` — M01–M09 dengan jarak longitudinal terukur
  (pengukuran lapangan; contoh person: sela 6,6 m). **Jangan mencampur** tabel
  GT penelitian person. GT tidak dikoreksi lateral.
- `vehicle_metadata.csv` — kendaraan kalibrasi (`CAL_<KELAS>_<NN>`) dan uji
  (`T2_<KELAS>_<NN>` dst.) dengan `body_class_gt`, dimensi, `plate_action=blurred`.
- `review_subjects.csv` — pasangan `subject_id,label` untuk halaman review
  track Take 1 (kolom `subject_id` dipertahankan demi kompatibilitas skrip
  crosswalk person).
- `takeN_events_confirmed.csv` — event crossing hasil review manual + kolom
  `body_class_gt` (diisi otomatis oleh `05_enrich_events_with_class.py`).

## 4. Protokol Take 1 (kalibrasi)

- Target: **≥ 2 (rekomendasi 3) mobil × 3 kelas × 9 marker = 54–81 event**.
  Setiap kelas wajib mencakup **semua** marker M01–M09 (syarat kuartik orde 4,
  validator menolak marker yang hilang).
- Event hanya dari review manual (validator menolak `annotation_source` auto).
- Fit **satu kuartik monoton global** + **tiga kuartik monoton per kelas**
  (constraint non-increasing `d(D)/d(A) ≤ 0`, verifikasi analitik).
- Metrik Take 1 = **in-sample**, hanya sanity check; angka resmi di Fase uji.
- Bekukan koefisien + normalisasi + threshold fallback dalam
  `fitted_models.json`; selanjutnya **tidak ada refit** di take manapun.

## 5. Protokol review crossing (aturan tegas, karena 30 fps)

1. Definisi event: frame di mana **bumper depan paling menempel garis marker**
   (±½ frame). Aturan ini harus identik antara kalibrasi dan inference karena
   yang masuk kuartik adalah luas box pada frame pilihan.
2. Gunakan strip frame pada halaman review; pilih frame terdekat ke titik
   kontak, jangan "sekitar situ". Jika ragu antara dua frame, pilih yang
   bumper-nya paling bersentuhan dengan garis, dan catat di `notes`.
3. Konsisten untuk semua take; bila operator berganti, sesi review baru wajib
   membaca ulang aturan ini.
4. Mobil yang box-nya terpotong tepi frame: gunakan `--clip-offscreen-bumper`
   saat membuat kandidat dan review lebih hati-hati (luas box terpotong tidak
   valid untuk kuartik — tandain event untuk dieksklusi bila parah).
5. Arah kandidat: mulai `--direction either`, periksa pola, lalu kunci arah
   yang benar (mobil mendekat kamera → bottom-center bergerak turun di gambar)
   di take berikutnya untuk memperkecil kandidat liar.
6. Cek motion blur pada frame crossing mobil terdekat; bila box bergoyang,
   catat sebagai catatan kualitas data.

## 6. Classifier body type (MPV / Sedan / Hatchback)

- Sumber: Kaggle *Indonesian Cars Classification Dataset* (8 kelas, Apache 2.0,
  ~16 ribu gambar ter-crop). Utamakan `scripts/10_prepare_deduplicated_classifier_dataset.py`;
  `00_prepare_classifier_dataset.py` adalah filter lama tanpa exact-hash dedup.
- `10` menghasilkan `train|val|test/{Hatchback,MPV,Sedan}`. Audit dataset ini
  menemukan exact duplicates antar-split dan beberapa gambar identik di lebih
  dari satu label; `10` memindah duplicate satu-label ke split prioritas train>
  val>test dan mengecualikan hash yang label sumbernya konflik. Near-duplicates
  tetap perlu diperiksa terpisah.
- Jika belum ada checkpoint 3-kelas, fine-tune dari pretrained
  `yolov8n-cls.pt` pada dataset bersih. Jika sudah ada checkpoint 3-kelas,
  `scripts/08_finetune_car_classifier.py` dapat melanjutkannya. Catat dataset,
  provenance, epoch, imgsz, batch, learning rate, dan checkpoint hash.
- **Validasi domain nyata**: buat label manual kecil dari crop hasil tracking
  Take 1 dan laporkan Top-1/recall per kelas di sana — angka di dataset web
  tidak mewakili crop perspektif jarak jauh.
- `ultralytics_detection_4videos.zip` hanya memiliki label deteksi `car`; crop
  body type harus diberi label manual. Split train (Video 1–3)/val (Video 4)
  di ZIP dibuat untuk object detection, **bukan** split eksperimen classifier.
  Jika Video 2–4 menjadi video uji estimasi jarak, jangan pakai crop-nya untuk
  fine-tune, early stopping, atau memilih threshold. Gunakan `classifier_video_split.csv`
  untuk menetapkan peran video; bila tidak ada video val non-test, gunakan
  validation split Kaggle/rekaman tambahan. Crop manual blind disiapkan oleh
   `scripts/06_prepare_blind_crop_review.py`.
- Checkpoint lokal yang diperiksa memang mempunyai nama kelas Hatchback/MPV/Sedan
  dan metadata 50 epoch, tetapi path dataset checkpoint bersifat internal. Jangan
  menganggap Kaggle test split untouched untuk checkpoint tersebut sebelum
  provenance training diverifikasi.
- Konsumsi di pipeline: satu crop terbaik per track-segmen (terluas ×
  confidence tertinggi), label dipropagasi ke seluruh segmen agar tidak
  berkedip antar frame; raw softmax disimpan tanpa kalibrasi.
- **Pemilihan threshold fallback (best practice)**: putuskan dari data
  validasi classifier (bukan event uji) pada titik operasi yang diinginkan —
  mis. precision per kelas ≥ 0,8 → confidence minimum di titik itu jadi
  threshold. Simpan nilai akhirnya di `configs/bodytype_config.json` SEBELUM
  kalibrasi Take 1, karena threshold ikut dibekukan ke bundle model. Sensitivitas
  (mis. 0.5/0.6/0.7) boleh dilaporkan sebagai analisis tambahan, tetapi
  kebijakan utama tetap satu nilai beku.

## 7. Evaluasi beku Video 2–4 (tiga varian)

| Varian | Pemilih kuartik | Menjawab |
|---|---|---|
| `end_to_end` | Classifier (fallback → global) | Akurasi sistem nyata yang di-deploy |
| `oracle` | Kelas ground truth manual | Batas atas regresi murni |
| `global` | Kuartik global selalu | Baseline tanpa classifier |

- **Dekomposisi error**: selisih `end_to_end − oracle` = kontribusi classifier;
  `oracle − global` = nilai tambah klasifikasi body type. Keduanya adalah
  analisis utama paper rekan (analog global vs personal di paper person).
- Coverage dilaporkan terpisah: event tanpa deteksi masuk coverage, bukan MAE.
- Peringatan otomatis yang wajib diperiksa: area di luar rentang kalibrasi
  (ekstrapolasi), prediksi nonpositif (tanpa clipping), metadata detektor
  berbeda (hash/imgsz/conf/tracker), take kalibrasi bocor ke uji (ditolak).
- Mobil uji **harus disjoint** dari mobil kalibrasi (skenario terkunci).

## 8. Etika

- **Blur plat nomor** pada semua media yang dibagikan/diunggah (analog blur
  wajah pada penelitian person). Video/foto tetap lokal; hanya CSV parameter,
  metrik, dan grafik yang dibagikan.
- `plate_action=blurred` dicatat per kendaraan di `vehicle_metadata.csv`.

## 9. Limitasi yang dilaporkan di paper

1. GT adalah jarak longitudinal marker; offset lateral kendaraan tidak
   dikoreksi (warisan protokol person).
2. Video 30 fps → ambiguitas frame crossing ±½–1 frame (~0,1–0,2 m pada
   20–30 km/j); kecepatan laju tidak dikendalikan penuh.
3. Variasi ukuran dalam kelas (MPV kecil vs besar) menjadi error model kelas;
   dianalisis lewat residu per kendaraan vs dimensi di `vehicle_metadata.csv`.
4. Label classifier adalah estimasi visual, bukan registrasi manufacturer.
5. Hasil mengukur generalisasi ke kendaraan baru sekelas pada setup kamera
   yang sama; bukan generalisasi lintas lokasi/kamera.

## 10. Arahan visualisasi 3D mobil (berbeda dari scene person)

- Jangan menyalin scene person yang memakai jalan pedesaan/verge hijau dan
  pepohonan. Gunakan **koridor jalan perkotaan** dengan gedung/fasad di kedua
  sisi, trotoar, curb, marka jalan, dan elemen kota seperlunya. Hindari barisan
  pohon sebagai dekorasi utama.
- Gunakan aset lokal `3d model/1990s-hatchback.blend`,
  `1990s-minivan.blend` (representasi visual MPV), dan
  `1990s-sedan.blend`. Aset sudah dibuka, dirender, dan round-trip GLB diuji
  di Blender 5.2.2; skala kira-kira metrik, tekstur terkemas.
- Ketiga model adalah aset statis tanpa kamera/lampu/animasi. Root Empty dapat
  dipakai untuk translasi kendaraan mengikuti hasil jarak; animasi rotasi roda
  bersifat opsional. Scene generation harus membuat kamera, pencahayaan,
  material jalan, gedung-gedung, dan komposisi urban tersendiri.
- Model bertema 1990-an dan bersifat ilustratif, bukan rekonstruksi merek/model
  kendaraan uji atau bukti dimensi fisik. Embedded metadata pada aset mencatat
  Grab3D / Cozy Retro Cars Kit / CC0-1.0; lihat `3d model/README.md`. Ekspor uji
  berhasil round-trip, tetapi visual material GLB tetap harus dicek karena
  beberapa material mempunyai lebih dari satu image node untuk sampler.

## 11. Checklist eksekusi

- [ ] Isi `inputs/` dari template (GT marker, metadata kendaraan, review subjects)
- [ ] Anotasi & profil marker (9/9 terkonfirmasi) + QC overlay tiap take
- [ ] Rekam Take 1 (kalibrasi) & Take 2–4 (uji) — mobil uji berbeda, arah mendekat
- [ ] Tracking `01` per take (konfigurasi identik)
- [ ] Review track→kendaraan Take 1 + crosswalk
- [ ] Kandidat + review crossing + export + enrich `05` (semua take)
- [ ] Fine-tune classifier selesai + validasi domain nyata + threshold dikunci
- [ ] Kalibrasi `03` — periksa monotonicity & warning
- [ ] Classify `02` per take uji
- [ ] Evaluasi `04` — periksa tiga varian + confusion + fallback stats
- [ ] Blur plat nomor di semua media sebelum share
