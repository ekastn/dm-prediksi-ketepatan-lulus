# Prediksi Ketepatan Lulus Mahasiswa

Project data mining — membangun model Decision Tree untuk memprediksi apakah mahasiswa lulus tepat waktu berdasarkan performa akademik semester awal.

## Metodologi: CRISP-DM

Project ini mengikuti framework **CRISP-DM** (Cross-Industry Standard Process for Data Mining).

```
┌──────────────────────────────────────────────────────────────┐
│  1. Business Understanding                                   │
│     Proposal, rumusan masalah, tujuan                        │
│     📁 1-business-understanding/                             │
├──────────────────────────────────────────────────────────────┤
│  2. Data Understanding                                       │
│     Eksplorasi database, profiling, identifikasi data quality │
│     📁 2-data-understanding/                                 │
├──────────────────────────────────────────────────────────────┤
│  3. Data Preparation                                         │
│     ETL, feature engineering, cleaning, train/test split     │
│     📁 3-data-preparation/                                   │
├──────────────────────────────────────────────────────────────┤
│  4. Modeling                                                 │
│     Decision Tree, hyperparameter tuning, cross-validation   │
│     📁 4-modeling/ (belum ada)                               │
├──────────────────────────────────────────────────────────────┤
│  5. Evaluation                                               │
│     Confusion matrix, accuracy, feature importance, rules    │
│     📁 5-evaluation/ (belum ada)                             │
├──────────────────────────────────────────────────────────────┤
│  6. Deployment / Communication                               │
│     Laporan formal, slide presentasi, artikel IEEE           │
│     📁 6-deployment/ (belum ada)                             │
└──────────────────────────────────────────────────────────────┘
```

## Struktur Direktori

```
prediksi-ketepatan-lulus/
│
├── 1-business-understanding/
│   └── Proposal Data Mining.pdf
│
├── 2-data-understanding/
│   ├── README.md
│   ├── exploration-log.md
│   ├── 06-table-mapping.md
│   ├── 01-data-profiling.ipynb
│   ├── 01-schema-discovery.sql
│   ├── 02-student-profiling.sql
│   ├── 03-academic-profiling.sql
│   ├── 04-grade-profiling.sql
│   ├── 05-data-quality.sql
│   └── ddl.csv
│
├── 3-data-preparation/
│   ├── extract_dataset.py               # ETL + feature engineering
│   ├── dataset.csv / _train.csv / _test.csv  # Raw extracted
│   ├── 02-eda.ipynb / 02-eda.py         # EDA (notebook + script)
│   ├── 02-eda-findings.md               # Hasil analisis EDA
│   ├── eda-charts/                      # 16 PNG visualisasi EDA
│   ├── 03-preprocessing-plan.md         # Rencana preprocessing
│   ├── 03-preprocessing.ipynb / .py     # Pipeline preprocessing
│   ├── 04-dataset-overview.ipynb        # Overview dataset bersih
│   └── dataset_clean.csv                # Dataset siap modeling (608×17)
│
├── 4-modeling/                          # Fase 4 — BELUM ADA
├── 5-evaluation/                        # Fase 5 — BELUM ADA
├── 6-deployment/                        # Fase 6 — BELUM ADA
└── README.md
```

## Dataset Summary

| Metrik | Nilai |
|--------|-------|
| **Total rows** | 608 |
| **Features** | 16 + 1 target (17 kolom) |
| **Target** | `1` = Tepat Waktu, `0` = Tidak Tepat |
| **Class imbalance** | 11.2% negative (train 3.7%, test 23.4%) |
| **NULLs** | 0 (semua numerik) |

### Fitur (16 fitur + 1 target)

| Kategori | Fitur |
|----------|-------|
| **Identitas** | `angkatan` |
| **Program** | `program` (0=AP/D3, 1=IH/S1) |
| **IPS Semester** | `ips_sem1`, `ips_sem2`, `ips_sem3` |
| **SKS Semester** | `sks_sem1`, `sks_sem2`, `sks_sem3` |
| **Nilai MK** | `failed_courses`, `failed_in_sem1`, `repeated_courses` |
| **Derived** | `ips_trend`, `avg_ips`, `ips_std`, `ips_min`, `sks_completion_ratio` |

## Data Quality Notes

- **Database hasil migrasi vendor** — banyak data historis tidak lengkap
- **Angkatan 2014-2015** excluded: IPSIPK seluruhnya NULL
- **TSKS anomaly** — angkatan 2020+ menyimpan SKS kumulatif (bukan per-semester) di `IPSIPK`. Fixed dengan fallback ke `Qnilai_mhs`, di-cap 1-20 MK/semester, outlier → imputasi median.
- **IPS=0.0 dianggap NULL** — 36% mahasiswa punya minimal satu IPS=0.0 (placeholder legacy), dibersihkan saat preprocessing
- **Kehadiran di-drop** — 53% missing + korelasi r≈0 dengan target
- **Fitur di-drop pasca-EDA**: `ips_sem4`, `sks_sem4`, `ipk_sem4` (data leakage), `semester_count`, `ips_max` (tidak mendiskriminasi), `id_agama`, `jenis_kelamin` (r≈0), `avg_attendance` (53% missing)

## Quick Start

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn missingno scikit-learn pymssql

# Extract dataset (requires SQL Server access)
cd 3-data-preparation
python extract_dataset.py

# Run preprocessing (uses extracted dataset_clean.csv)
python 03-preprocessing.py
```

## Status Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | Proposal singkat | ✅ |
| 2 | Dataset + deskripsi variabel | ✅ |
| 3 | Notebook preprocessing | ✅ |
| 4 | Modeling (Decision Tree) | ⬜ Next Phase |
| 5 | Laporan formal Bab I–V | ⬜ |
| 6 | Slide presentasi | ⬜ |
| 7 | Artikel IEEE | ⬜ |

## Tim

- Viola Septianti Elsiana (714230001)
- Muhammad Saladin Eka Septian (714230037)
- Muhammad Hisyam Najwan (714230055)

Program Studi DIV Teknik Informatika — Universitas Logistik dan Bisnis Internasional, 2026
