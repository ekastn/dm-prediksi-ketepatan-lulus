# Preprocessing Plan — Prediksi Ketepatan Lulus

Berdasarkan temuan EDA (numerik + visual).  
Input: `dataset.csv` (608 rows) | Output: `dataset_clean.csv`

---

## Step-by-Step Pipeline

```
dataset.csv (608 rows, 27 kolom, with NULLs + system zeros)
    │
    ├── Step 1: Replace system zeros (IPS=0.0 → NaN)
    │       219 mahasiswa (36%) punya minimal 1 IPS=0.0
    │       79% dari mereka tetap lulus → ini placeholder, bukan nilai riil
    │
    ├── Step 2: Create has_attendance flag & drop avg_attendance
    │       has_attendance = 1 jika ada data kehadiran, 0 jika tidak
    │       r(avg_attendance, target) = -0.016 → sinyal numerik hampir nihil
    │       52.8% missing → terlalu sparse
    │
    ├── Step 3: Drop redundant/leakage features
    │       student_id, ips_sem4, sks_sem4, ipk_sem4,
    │       semester_count, ips_max, total_sks_lulus_sem4
    │
    ├── Step 4: Median imputation per angkatan
    │       ips_sem1, ips_sem2, ips_sem3
    │       sks_sem1, sks_sem2, sks_sem3
    │       imputed based on SAME angkatan median (fitted on full dataset)
    │
    ├── Step 5: Recompute derived features
    │       ips_trend = last_ips - first_ips
    │       avg_ips, ips_std, ips_min
    │       sks_completion_ratio
    │
    ├── Step 6: Encode categorical features
    │       program: AP→0, IH→1
    │       jenis_kelamin: L→0, P→1
    │       id_agama: keep as-is (1/2/4)
    │
    └── Output: dataset_clean.csv (608 rows, ~18 features)
```

---

## Fitur yang Di-drop & Alasan

| Fitur | Alasan | Keputusan |
|-------|--------|-----------|
| `student_id` | Identitas, bukan prediktor | Drop |
| `ips_sem4` | r=+0.877 target, data leakage | Drop |
| `sks_sem4` | r=-0.356, terkait ips_sem4 | Drop |
| `ipk_sem4` | r=+0.633 target + redundant dgn avg_ips (inter-correlation 0.8+) | Drop |
| `semester_count` | Circular — target didefinisikan partial dari durasi studi | Drop |
| `ips_max` | r≈0.0—0.0001, tidak mendiskriminasi kedua kelas | Drop |
| `total_sks_lulus_sem4` | Redundant dgn sks_completion_ratio | Drop |
| `avg_attendance` | 53% missing + r=-0.016 | Ganti `has_attendance` |

---

## Fitur Final (~17 kolom)

| # | Fitur | Tipe | Sumber | Notes |
|---|-------|------|--------|-------|
| 1 | `angkatan` | int | Identitas | |
| 2 | `program` | 0/1 | Demografi | Label-encoded |
| 3 | `jenis_kelamin` | 0/1 | Demografi | Label-encoded |
| 4 | `id_agama` | 1/2/4 | Demografi | Nominal, keep as-is |
| 5 | `ips_sem1` | float | Akademik | 0→NaN → imputed median per angkatan |
| 6 | `ips_sem2` | float | Akademik | Imputed median per angkatan |
| 7 | `ips_sem3` | float | Akademik | Imputed median per angkatan |
| 8 | `sks_sem1` | float | Akademik | 0→NaN → imputed median per angkatan |
| 9 | `sks_sem2` | float | Akademik | Imputed median per angkatan |
| 10 | `sks_sem3` | float | Akademik | Imputed median per angkatan |
| 11 | `failed_courses` | int | Nilai MK | |
| 12 | `failed_in_sem1` | int | Nilai MK | |
| 13 | `repeated_courses` | int | Nilai MK | |
| 14 | `has_attendance` | 0/1 | Kehadiran | 1=jika avg_attendance non-null |
| 15 | `ips_trend` | float | Derived | Recomputed after imputation |
| 16 | `avg_ips` | float | Derived | Recomputed after imputation |
| 17 | `ips_std` | float | Derived | Recomputed after imputation |
| 18 | `ips_min` | float | Derived | Recomputed after imputation |
| 19 | `sks_completion_ratio` | float | Derived | Recomputed |

Plus `target` (0/1).

---

## Kenapa Median Per Angkatan?

1. Missing sistematik by angkatan (2015=1.99 missing avg, 2022=0.03)
2. Median robust ke outlier (IPS ekstrem)
3. Mengikuti praktik standar Decision Tree prep
4. Data cuma 608 rows — KNN/MICE overkill

## Hal yang Belum Diputuskan (Butuh Konfirmasi)

1. **Train/test split ulang?** Dataset sekarang full 608 rows. Preprocessing dilakukan pada full dataset, baru setelah bersih di-split kembali (temporal: ≤2021 train, >2021 test). Atau preprocessing di-split dulu (fit on train, transform on test)?

2. **Binarize grade features?** Designer menyarankan `has_any_failure` (0/1) karena distribusinya extremely zero-inflated (median=0 untuk semua kelas). Ini bisa dicoba nanti di fase modeling.

3. **ips_sem1 tetap dipertahankan?** r=-0.27 (negatif) dengan target. Ini artefak system zero. Tapi setelah dibersihkan (0→NaN→imputed), korelasinya akan berubah. Harus di-evaluasi ulang setelah preprocessing.

---

## File Output Rencana

```
3-data-preparation/
├── dataset.csv                     # Raw (input)
├── 02-eda.py                       # EDA script
├── 02-eda.ipynb                    # EDA notebook
├── 02-eda-findings.md              # Hasil analisis EDA
├── eda-charts/                     # 16 PNG files
├── 03-preprocessing.py             # Preprocessing script (belum dibuat)
└── dataset_clean.csv               # Output bersih (belum dibuat)
```
