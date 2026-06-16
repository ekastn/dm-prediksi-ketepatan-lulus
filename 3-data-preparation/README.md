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

**Catatan:** Mahasiswa dengan hanya 3 semester data akan memiliki `ips_sem4` dan `sks_sem4` = NULL. Model/preprocessing harus handle missing values.

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
| `dataset.csv` | 608 | Full dataset (semua angkatan dengan outcome known) |
| `dataset_train.csv` | 377 | Train: angkatan ≤ 2021 |
| `dataset_test.csv` | 231 | Test: angkatan > 2021 (2022-2023) |

## Next Steps (untuk fase modeling)

1. **Encoding categorical features** — `program` (AP/IH), `jenis_kelamin` (L/P), `id_agama` perlu di-encode (one-hot / label / ordinal)

2. **Handle class imbalance** — 11.2% kelas negatif (train: 3.7%, test: 23.4%). Opsi:
   - SMOTE / oversampling untuk kelas "tidak tepat waktu"
   - Class weighting di Decision Tree
   - Cost-sensitive learning (FN lebih mahal dari FP)

3. **Handle missing values** — terutama `avg_attendance` (53%), `ips_sem4` (31%), `ips_sem1` (28%). Opsi:
   - Simple imputation (mean/median per program)
   - KNN imputation
   - Gunakan algoritma yang native support missing (C4.5, implementasi CART tertentu)
   - Drop `avg_attendance` jika ternyata tidak signifikan

4. **Feature selection** — 26 fitur untuk 68 sampel negatif, perlu seleksi fitur atau regularization untuk hindari overfitting

5. **Hyperparameter tuning** — Grid search untuk `max_depth`, `min_samples_split`, `criterion` (gini/entropy)

6. **Model comparison** — Bandingkan Decision Tree dengan Random Forest dan Naive Bayes sebagai baseline
