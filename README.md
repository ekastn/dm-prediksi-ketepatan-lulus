# Prediksi Ketepatan Lulus Mahasiswa — Dataset Preparation

## Project Overview

**Tujuan:** Membangun model Decision Tree untuk memprediksi apakah seorang mahasiswa akan lulus tepat waktu atau tidak, berdasarkan data performa akademik di semester-semester awal.

**Konteks:** Data berasal dari database internal kampus (SQL Server) hasil migrasi dari vendor lama ke vendor baru. Proses migrasi menyebabkan sebagian data historis tidak lengkap atau hilang.

## Data Source Architecture

Database SQL Server (`LITIGASI`) dengan 405 tabel. Hanya tabel berikut yang relevan:

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

## Feature Specification (27 kolom)

### 1. Identitas & Demografi (dari `tblMHS`)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `student_id` | string | NIM |
| 2 | `angkatan` | int | Tahun masuk |
| 3 | `program` | categorical | `AP` = D3, `IH` = S1 |
| 4 | `jenis_kelamin` | categorical | `L` / `P` |
| 5 | `id_agama` | categorical | 1=Islam, 2=Kristen, 3=Katolik, 4=Hindu, etc |

### 2. Performa Akademik Semester 1-4 (dari `IPSIPK`)

| # | Feature | Description |
|---|---------|-------------|
| 6-9 | `ips_sem1` - `ips_sem4` | Indeks Prestasi Semester |
| 10 | `ipk_sem4` | IPK kumulatif sampai semester terakhir yang tersedia (maksimal smt 4) |
| 11-14 | `sks_sem1` - `sks_sem4` | Total SKS yang diambil per semester |
| 15 | `total_sks_lulus_sem4` | Total SKS kumulatif lulus sampai semester terakhir |

Catatan: Mahasiswa dengan hanya 3 semester data akan memiliki `ips_sem4` dan `sks_sem4` = NULL. Model/preprocessing harus handle missing values.

### 3. Performa Matakuliah (dari `Qnilai_mhs`)

| # | Feature | Description |
|---|---------|-------------|
| 16 | `failed_courses` | Jumlah MK dengan nilai E, D, atau T dalam 4 semester pertama |
| 17 | `failed_in_sem1` | Jumlah MK gagal di semester 1 saja (early warning signal) |
| 18 | `repeated_courses` | Jumlah MK yang muncul >1 kali (mengulang) |

### 4. Kehadiran (dari `Qnilai_mhs` + `Kul_Kehadiran`)

| # | Feature | Description |
|---|---------|-------------|
| 19 | `avg_attendance` | Rata-rata persentase kehadiran (0-100). **52.8% missing** |

### 5. Derived Features

| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 20 | `ips_trend` | `ips_last - ips_first` | Tren performa: positif = membaik |
| 21 | `avg_ips` | `mean(ips_sem1..n)` | Rata-rata IPS semester yang tersedia |
| 22 | `ips_std` | `stdev(ips)` | Volatilitas IPS |
| 23 | `ips_max` | `max(ips)` | IPS tertinggi |
| 24 | `ips_min` | `min(ips)` | IPS terendah |
| 25 | `sks_completion_ratio` | `total_sks / 80` | Rasio SKS terhadap ekspektasi (~20/smt) |
| 26 | `semester_count` | `count(IPSIPK)` | Jumlah semester yang sudah ditempuh |

### 6. Target Variable

| # | Feature | Value | Criteria |
|---|---------|-------|----------|
| 27 | `target` | `1` | **Tepat Waktu** — Status `L`/`Lulus` DAN total semester ≤ durasi normal |
| | | `0` | **Tidak Tepat** — Status `Keluar` (dropout), atau lulus > durasi normal, atau Aktif/Cuti melebihi durasi |

**Durasi normal:** IH (S1) = 8 semester, AP (D3) = 6 semester

## Data Quality Issues

Database hasil migrasi vendor. Dampak pada dataset:

### 1. IPSIPK tidak lengkap untuk angkatan tua
- Mahasiswa angkatan **2014-2015** memiliki record IPSIPK tapi nilai `ips`, `tsks`, `ttsks`, `ipk` semuanya **NULL**.
- **154 mahasiswa** dikeluarkan karena < 1 semester dengan IPS valid.

### 2. Data kehadiran sangat sparse
- `avg_attendance` **52.8% missing** (321/608).
- Dari 54,587 record Qnilai_mhs, hanya 18% yang memiliki `Kehadiran > 0`.
- Sudah disupplement dengan `Kul_Kehadiran` tapi hanya menambah 146 mahasiswa.

### 3. Fitur dengan zero variance (dihilangkan)
- `status_masuk`: 100% "Baru"
- `penerima_kps`: 100% "0" (tidak menerima KPS)
- `id_jenis_daftar`: 100% "1"
- `id_jalur_masuk`: 99.4% "12"
- `JalurMasuk`: 100% NULL

### 4. Format NIM tidak seragam antar tabel
- `tblMHS` + `IPSIPK`: format 9-digit (`207421001`)
- `HtblNilai`: format berbeda — **0 overlap** dengan tblMHS
- Tabel nilai yang cocok: `Qnilai_mhs`

## Extraction Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  SQL Server  │───>│  Bulk Query  │───>│  In-Memory   │───>│  dataset.csv │
│  LITIGASI    │    │  4 tables    │    │  Processing  │    │  608 rows    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Filtering Steps

```
1621 mahasiswa di tblMHS (AP + IH)
    │
    ├─ 458 mahasiswa < 3 semester IPSIPK ──> excluded
    │   (angkatan 2024-2025 yang belum mencapai semester 3)
    │
    ├─ 154 mahasiswa dengan 0 IPS valid ──> excluded
    │   (angkatan tua dengan data IPSIPK kosong)
    │
    ├─ 401 mahasiswa dengan outcome unknown ──> excluded
    │   (status Aktif/Cuti dalam batas durasi normal)
    │
    └─ 608 mahasiswa ──> FINAL DATASET
```

### Running the Extraction

```bash
# Setup venv (first time only)
python3 -m venv .venv
.venv/bin/pip install pymssql

# Run extraction
.venv/bin/python3 extract_dataset.py
```

Output:
| File | Rows | Description |
|------|------|-------------|
| `dataset.csv` | 608 | Full dataset |
| `dataset_train.csv` | 377 | Train: angkatan ≤ 2021 |
| `dataset_test.csv` | 231 | Test: angkatan > 2021 (2022-2023) |

### Train/Test Split Design

Split dilakukan berdasarkan **waktu (angkatan)**, bukan random. Alasan:

1. **Simulasi real deployment**: Model dilatih pada data historis (angkatan yang sudah selesai masa studinya), diuji pada angkatan yang lebih baru.
2. ** Menghindari data leakage**: Random split bisa mencampur mahasiswa seangkatan yang punya karakteristik serupa antara train dan test.
3. **Test set lebih challenging**: Train punya 3.7% kelas negatif, Test punya 23.4% — distribusi yang berbeda menguji generalisasi model.

```
Train (≤ 2021):  377 rows  Tepat=363  Tidak=14  (3.7% neg)
Test  (> 2021):  231 rows  Tepat=177  Tidak=54  (23.4% neg)
```

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
```

## Next Steps (untuk fase modeling)

1. **Handle class imbalance** — 11.2% kelas negatif. Opsi:
   - SMOTE / oversampling untuk kelas "tidak tepat waktu"
   - Class weighting di Decision Tree
   - Cost-sensitive learning (FN lebih mahal dari FP)

2. **Handle missing values** — terutama `avg_attendance` (53%), `ips_sem4` (31%), `ips_sem1` (28%). Opsi:
   - Simple imputation (mean/median per program)
   - KNN imputation
   - Gunakan algoritma yang native support missing (C4.5, beberapa implementasi CART)
   - Drop `avg_attendance` jika ternyata tidak signifikan

3. **Feature encoding** — `program`, `jenis_kelamin`, `id_agama` perlu di-encode (one-hot / label / ordinal)

4. **Feature selection** — 26 fitur untuk 68 sampel negatif, perlu seleksi fitur atau regularization untuk hindari overfitting
