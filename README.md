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
│   └── Proposal Data Mining.pdf        # Proposal yang disetujui dosen
│
├── 2-data-understanding/
│   ├── README.md                        # Index fase Data Understanding
│   ├── exploration-log.md               # Log kronologis eksplorasi (narasi jujur)
│   ├── 01-data-profiling.ipynb          # Notebook replay profiling interaktif
│   ├── 01-schema-discovery.sql          # SQL: penemuan tabel kunci
│   ├── 02-student-profiling.sql         # SQL: profil mahasiswa & demografi
│   ├── 03-academic-profiling.sql        # SQL: profil IPSIPK & NULL rate
│   ├── 04-grade-profiling.sql           # SQL: profil Qnilai_mhs & kehadiran
│   ├── 05-data-quality.sql              # SQL: deteksi anomali & zero-variance
│   ├── 06-table-mapping.md              # Dokumentasi: relasi tabel & excluded tables
│   └── ddl.csv                          # Referensi DDL (skema database lengkap)
│
├── 3-data-preparation/
│   ├── extract_dataset.py               # Script ETL + feature engineering
│   ├── dataset.csv                      # Full dataset (608 rows × 27 kolom)
│   ├── dataset_train.csv                # Train: angkatan ≤ 2021 (377 rows)
│   └── dataset_test.csv                 # Test:  angkatan > 2021 (231 rows)
│
├── .venv/                               # Python virtual environment (pymssql)
└── README.md                            # File ini
```

## Dataset Summary

| Metrik | Nilai |
|--------|-------|
| **Total rows** | 608 |
| **Features** | 26 + 1 target (27 kolom) |
| **Target** | `1` = Tepat Waktu, `0` = Tidak Tepat |
| **Train** | 377 (angkatan ≤ 2021) |
| **Test** | 231 (angkatan > 2021) |
| **Class imbalance** | 11.2% negative |

| Program | Tepat Waktu | Tidak | % Neg |
|---------|------------|-------|-------|
| AP (D3) | 137 | 10 | 6.8% |
| IH (S1) | 403 | 58 | 12.6% |

### Fitur (26 kolom)

| Kategori | Fitur |
|----------|-------|
| **Identitas** | `student_id`, `angkatan` |
| **Demografi** | `program`, `jenis_kelamin`, `id_agama` |
| **IPS Semester** | `ips_sem1` – `ips_sem4` |
| **SKS & Kumulatif** | `sks_sem1` – `sks_sem4`, `ipk_sem4`, `total_sks_lulus_sem4` |
| **Nilai MK** | `failed_courses`, `failed_in_sem1`, `repeated_courses` |
| **Kehadiran** | `avg_attendance` (53% missing) |
| **Derived** | `ips_trend`, `avg_ips`, `ips_std`, `ips_max`, `ips_min`, `sks_completion_ratio`, `semester_count` |

## Data Quality Notes

- **Database hasil migrasi vendor** — banyak data historis tidak lengkap
- **Angkatan 2014-2015** excluded: IPSIPK seluruhnya NULL (sistem lama tidak mencatat)
- **Kehadiran 53% missing** — fitur `avg_attendance` sparse, di-supplement dengan `Kul_Kehadiran`
- **5 fitur zero-variance dihapus**: `status_masuk`, `penerima_kps`, `id_jenis_daftar`, `id_jalur_masuk`, `JalurMasuk`

## Quick Start

```bash
# Setup virtual environment
python3 -m venv .venv
.venv/bin/pip install pymssql

# Buka notebook profiling
# Buka 2-data-understanding/01-data-profiling.ipynb di VS Code / Jupyter

# Jalankan ulang ekstraksi dataset
cd 3-data-preparation
../.venv/bin/python3 extract_dataset.py
```

## Status Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | Proposal singkat | ✅ `1-business-understanding/` |
| 2 | Dataset + deskripsi variabel | ✅ `3-data-preparation/` |
| 3 | Notebook preprocessing & modeling | ⬜ Fase 4 |
| 4 | Laporan formal Bab I–V | ⬜ |
| 5 | Slide presentasi | ⬜ |
| 6 | Artikel IEEE | ⬜ |

## Tim

- Viola Septianti Elsiana (714230001)
- Muhammad Saladin Eka Septian (714230037)
- Muhammad Hisyam Najwan (714230055)

Program Studi DIV Teknik Informatika — Universitas Logistik dan Bisnis Internasional, 2026
