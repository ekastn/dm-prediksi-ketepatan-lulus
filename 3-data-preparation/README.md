# Data Preparation Report

Fase ETL, feature engineering, dan pembentukan dataset untuk prediksi ketepatan lulus.

## Data Source Architecture

Database SQL Server (`LITIGASI`) dengan 405 tabel. Hanya 4 tabel yang digunakan:

```
tblMHS          — Data master mahasiswa (NIM, nama, angkatan, program, status, demografi)
IPSIPK          — IPS/IPK/SKS per semester (historis akademik)
Qnilai_mhs       — Nilai matakuliah + kehadiran per mahasiswa
Kul_Kehadiran    — Data kehadiran kuliah (supplementary)
```

### Skema Relasi

```
tblMHS (1) ────< (N) IPSIPK        — satu mahasiswa punya banyak record semester
tblMHS (1) ────< (N) Qnilai_mhs     — satu mahasiswa punya banyak record nilai
tblMHS (1) ────< (N) Kul_Kehadiran  — satu mahasiswa punya banyak record kehadiran
```

### Tabel yang Ditolak

| Tabel | Alasan |
|-------|--------|
| `HtblNilai` (39,990 rows) | **0% NIM overlap** — format NIM berbeda (sistem legacy) |
| `Perwalian` (6,222 rows) | `TSKSB` **semua NULL** — tidak bisa untuk SKS |
| `feed_nilai` (1,457 rows) | Cakupan hanya 184 mahasiswa |
| `Luusan` (0 rows) | Tabel kosong — data lulusan di `tblMHS.Status` |
| `tblCuti` (0 rows) | Tabel kosong |

## Extraction Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  SQL Server  │───>│  Bulk Query  │───>│  In-Memory   │───>│  dataset.csv │
│  LITIGASI    │    │  4 tables    │    │  Processing  │    │  608 rows    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

1. **Bulk query** 4 tabel (IPSIPK, Qnilai_mhs, Kul_Kehadiran, tblMHS) — semua data ke memory
2. **Join & aggregate** per mahasiswa di Python (bukan SQL JOIN)
3. **Filter** mahasiswa dengan data cukup & outcome known
4. **Hitung derived features** (ips_trend, avg_ips, sks_completion_ratio, etc)
5. **Tentukan target** (tepat/tidak tepat waktu)
6. **Output**: 3 CSV (full, train, test)

### Filtering Steps

```
1621 mahasiswa di tblMHS (AP + IH)
    │
    ├─ 458 mahasiswa < 3 semester IPSIPK ──> excluded
    │   (angkatan 2024-2025 yang belum mencapai semester 3)
    │
    ├─ 154 mahasiswa dengan 0 IPS valid ──> excluded
    │   (angkatan tua dengan data IPSIPK kosong — efek migrasi vendor)
    │
    ├─ 401 mahasiswa dengan outcome unknown ──> excluded
    │   (status Aktif/Cuti dalam batas durasi normal)
    │
    └─ 608 mahasiswa ──> FINAL DATASET
```

### Running Extraction

```bash
cd 3-data-preparation
../.venv/bin/python3 extract_dataset.py
```

## Preprocessing Pipeline

Setelah ekstraksi, dataset melalui preprocessing pipeline (`03-preprocessing.py` / `03-preprocessing.ipynb`):

```
dataset.csv (608×27, raw, with NULLs)
    │
    ├─ 1. Replace system zeros (IPS=0.0 → NaN)
    │       170 ips_sem1 zeros terkontaminasi
    │
    ├─ 2. Drop 11 kolom (leakage + low-signal + redundant)
    │       student_id, ips_sem4, sks_sem4, ipk_sem4, semester_count,
    │       ips_max, total_sks_lulus_sem4, avg_attendance,
    │       id_agama, jenis_kelamin, has_attendance
    │
    ├─ 3. Median imputation per angkatan
    │       ips_sem1-3 + sks_sem1-3
    │
    ├─ 4. Recompute derived features
    │       ips_trend, avg_ips, ips_std, ips_min, sks_completion_ratio
    │
    ├─ 5. Encode categorical
    │       program: AP→0, IH→1
    │
    └─ Output: dataset_clean.csv (608×17, 0 NULLs, semua numerik)
```

### Preprocessing Decisions (based on EDA)

| Keputusan | Fitur | Alasan |
|-----------|-------|--------|
| Drop | `ips_sem4`, `sks_sem4` | Data leakage (r=+0.877 dengan target) |
| Drop | `ipk_sem4` | Redundant dengan avg_ips |
| Drop | `semester_count` | Circular dengan target |
| Drop | `ips_max` | Tidak mendiskriminasi (r≈0) |
| Drop | `avg_attendance` | 53% missing + r=-0.016 |
| Drop | `id_agama` | r=0.031, hanya 4 Hindu |
| Drop | `jenis_kelamin` | r=0.050, tidak memisahkan kelas |
| Clean | `ips_sem1-4` (IPS=0.0→NaN) | 36% mahasiswa punya 0.0 (placeholder legacy) |
| Impute | `ips_sem1-3`, `sks_sem1-3` | Median per angkatan (robust ke outlier) |

## Final Feature Specification (17 kolom — dataset_clean.csv)

### 1. Identitas & Program

| # | Feature | Type | Range | Description |
|---|---------|------|-------|-------------|
| 1 | `angkatan` | int | 2015–2023 | Tahun masuk |
| 2 | `program` | 0/1 | 0–1 | 0=AP(D3), 1=IH(S1) |

### 2. Performa Akademik Semester 1-3

| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 3-5 | `ips_sem1` – `ips_sem3` | 0.3–4.0 | Indeks Prestasi Semester (imputed) |
| 6-8 | `sks_sem1` – `sks_sem3` | 1–24 | Jumlah SKS diambil per semester (imputed) |

### 3. Performa Matakuliah

| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 9 | `failed_courses` | 0–18 | Jumlah MK gagal (E/D/T) dalam 3 semester pertama |
| 10 | `failed_in_sem1` | 0–7 | Jumlah MK gagal di semester 1 saja |
| 11 | `repeated_courses` | 0–12 | Jumlah MK diulang |

### 4. Derived Features

| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 12 | `ips_trend` | -2.8–2.5 | ips_sem3 - ips_sem1 (positif=membaik) |
| 13 | `avg_ips` | 1.7–3.9 | Rata-rata IPS 3 semester |
| 14 | `ips_std` | 0–1.8 | Volatilitas IPS |
| 15 | `ips_min` | 0.3–3.8 | IPS terendah |
| 16 | `sks_completion_ratio` | 0.1–1.1 | (sks1+sks2+sks3) / 60 |
| 17 | `target` | 0/1 | 1=Tepat Waktu, 0=Tidak Tepat |

## Train/Test Split Design

Split dilakukan berdasarkan **waktu (angkatan)**, bukan random.

| Split | Angkatan | Rows | Tepat | Tidak | % Neg |
|-------|----------|------|-------|-------|-------|
| **Train** | ≤ 2021 | 377 | 363 | 14 | 3.7% |
| **Test** | > 2021 | 231 | 177 | 54 | 23.4% |

### Rasional

1. **Simulasi real deployment** — Model dilatih pada data historis (angkatan yang sudah selesai masa studinya), diuji pada angkatan yang lebih baru.
2. **Menghindari data leakage** — Random split bisa mencampur mahasiswa seangkatan yang punya karakteristik serupa antara train dan test.
3. **Test set lebih challenging** — Train punya 3.7% kelas negatif, Test punya 23.4% — distribusi yang berbeda menguji generalisasi model.

## Final Dataset Statistics

```
Rows:                608
Features:            26 + 1 target (27 columns)
Tepat waktu:         540 (88.8%)
Tidak tepat waktu:   68 (11.2%)

Program breakdown:
  AP (D3):  137 tepat, 10 tidak  (6.8% negative)
  IH (S1):  403 tepat, 58 tidak  (12.6% negative)

Gender:     L=407, P=201
Religion:   1=532 (Islam), 2=72 (Kristen), 4=4 (Hindu)

semester_count distribution:
  3 smt: 191   5 smt: 57    7 smt: 36
  4 smt: 59    6 smt: 252   8-12 smt: 13

Angkatan distribution:
  2015: 116   2018: 46    2021: 46
  2016: 54    2019: 27    2022: 181
  2017: 48    2020: 40    2023: 50
```

## Data Quality Issues

Database hasil migrasi vendor. Dampak pada dataset:

### 1. IPSIPK tidak lengkap untuk angkatan tua
- Mahasiswa angkatan **2014-2015** memiliki record IPSIPK tapi nilai `ips`, `tsks`, `ttsks`, `ipk` semuanya **NULL**.
- Sistem lama tidak menyimpan data per-semester, hanya record placeholder.
- **154 mahasiswa** dikeluarkan karena < 1 semester dengan IPS valid.

### 2. Data kehadiran sangat sparse
- `avg_attendance` **52.8% missing** (321/608).
- Dari 54,587 record Qnilai_mhs: 25% NULL, 56% bernilai 0 (sistem tidak mencatat), 18% memiliki nilai >0.
- Disupplement dengan `Kul_Kehadiran` tapi hanya menambah 146 mahasiswa.
- Kehadiran=0 di Qnilai_mhs diperlakukan sebagai NULL (bukan 0% kehadiran) karena kemungkinan sistem tidak mencatat, bukan berarti mahasiswa tidak hadir.

### 3. Fitur dengan zero variance (dihilangkan)
- `status_masuk`: 100% "Baru"
- `penerima_kps`: 100% "0" (tidak menerima KPS)
- `id_jenis_daftar`: 100% "1"
- `id_jalur_masuk`: 99.4% "12"
- `JalurMasuk`: 100% NULL

### 4. Format NIM tidak seragam antar tabel
- `tblMHS` + `IPSIPK` + `Qnilai_mhs` + `Kul_Kehadiran`: format 9-digit (`207421001`) — kompatibel
- `HtblNilai`: format 10-char (`0301020026    `) atau dotted (`01.01.014.001`) — **0 overlap** dengan tblMHS
- Tabel nilai yang cocok hanya `Qnilai_mhs`

### 5. Periode `K` (Pendek/Singkat)
- Selain Periode 1 (Ganjil) dan 2 (Genap), ada periode `K` (semester pendek/summer).
- Di-map ke nilai 3 untuk pengurutan: 1 < 2 < K

## Output Files

| File | Rows | Description |
|------|------|-------------|
| `dataset.csv` | 608 | Full dataset (raw extraction, 27 cols) |
| `dataset_train.csv` | 377 | Train: angkatan ≤ 2021 |
| `dataset_test.csv` | 231 | Test: angkatan > 2021 |
| `dataset_clean.csv` | 608 | Final dataset pasca-preprocessing (17 cols, 0 NULLs) |
| `02-eda.ipynb` / `.py` | — | EDA notebook + script |
| `02-eda-findings.md` | — | Dokumentasi temuan EDA |
| `03-preprocessing-plan.md` | — | Rencana preprocessing |
| `03-preprocessing.ipynb` / `.py` | — | Notebook + script preprocessing |
| `04-dataset-overview.ipynb` | — | Overview dataset bersih (presentasi) |
| `eda-charts/` | — | 16 visualisasi PNG dari EDA |

## Next Steps (Fase 4: Modeling)

1. **Split dataset bersih** — train (angkatan ≤2021) vs test (angkatan >2021) dari `dataset_clean.csv`
2. **Handle class imbalance** — train hanya 3.7% negatif, pertimbangkan SMOTE atau class_weight
3. **Decision Tree baseline** — latih model awal dengan default hyperparameter
4. **Hyperparameter tuning** — GridSearchCV untuk `max_depth`, `min_samples_split`, `criterion`
5. **Cross-validation** — stratified k-fold pada train set
6. **Model comparison** — bandingkan dengan Random Forest dan Naive Bayes
7. **Feature importance** — analisis fitur mana yang paling berpengaruh
