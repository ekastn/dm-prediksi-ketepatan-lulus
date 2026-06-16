# Preprocessing Plan — Prediksi Ketepatan Lulus

Berdasarkan temuan EDA (numerik + visual).  
Input: `dataset.csv` (608 rows) | Output: `dataset_clean.csv`

---

## Step-by-Step Pipeline

### Step 0: Fix TSKS Anomaly in extract_dataset.py

**Masalah:** Database `LITIGASI` menyimpan data SKS tidak konsisten di tabel `IPSIPK`:
- Mahasiswa angkatan 2015–2019: `TSKS` = SKS per semester (nilai wajar: 4–24)
- Mahasiswa angkatan 2020+: `TSKS` = total SKS kumulatif (nilai 80–133, salah konteks)
- 310 baris dataset (13%) terpengaruh — nilai SKS sem1-2 mencapai 133/99

**Investigasi:** Query langsung ke database menunjukkan:
- 207422012: sem1 TSKS=131, TTSKS=131 (nilai kumulatif, bukan per-semester)
- 2227421001: sem1 TSKS=4 (normal), sem2 TSKS=97 (kumulatif)
- Tidak bisa di-fix lewat Qnilai_mhs karena data nilai juga terkontaminasi (50+ MK/semester)

**Solusi:** Deteksi otomatis — jika ada TSKS >30 di 4 semester pertama, flag sebagai abnormal. Untuk mahasiswa abnormal, derive SKS dari jumlah distinct Kode_MK di Qnilai_mhs, dengan cap 1–20 MK/semester. Nilai di luar range → NULL → diimputasi median per angkatan di step selanjutnya.

**Hasil:** 0 nilai SKS >30 di dataset final. Imputasi median menangani NULL yang tersisa.

```
dataset.csv (608 rows, 27 kolom, with NULLs + system zeros)
    │
    ├── Step 1: Replace system zeros (IPS=0.0 → NaN)
    │       219 mahasiswa (36%) punya minimal 1 IPS=0.0
    │       79% dari mereka tetap lulus → ini placeholder, bukan nilai riil
    │
    ├── Step 2: Drop redundant/leakage + weak features
    │       student_id, ips_sem4, sks_sem4, ipk_sem4,
    │       semester_count, ips_max, total_sks_lulus_sem4,
    │       avg_attendance, has_attendance, id_agama, jenis_kelamin
    │
    ├── Step 3: Median imputation per angkatan
    │       ips_sem1, ips_sem2, ips_sem3
    │       sks_sem1, sks_sem2, sks_sem3
    │       imputed based on SAME angkatan median (fitted on full dataset)
    │
    ├── Step 4: Recompute derived features
    │       ips_trend = last_ips - first_ips
    │       avg_ips, ips_std, ips_min
    │       sks_completion_ratio
    │
    ├── Step 5: Encode categorical features
    │       program: AP→0, IH→1
    │
    └── Output: dataset_clean.csv (608 rows, ~17 kolom)
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
| `id_agama` | r=0.031, near-zero correlation | Drop |
| `jenis_kelamin` | r=0.050, no class separation | Drop |
| `has_attendance` | Weak signal (4.7% diff), 53% missing | Drop |
| `avg_attendance` | 53% missing + r=-0.016 | Drop |

---

## Fitur Final (~16 fitur)

| # | Fitur | Tipe | Sumber | Notes |
|---|-------|------|--------|-------|
| 1 | `angkatan` | int | Identitas | |
| 2 | `program` | 0/1 | Demografi | Label-encoded |
| 3 | `ips_sem1` | float | Akademik | 0→NaN → imputed median per angkatan |
| 4 | `ips_sem2` | float | Akademik | Imputed median per angkatan |
| 5 | `ips_sem3` | float | Akademik | Imputed median per angkatan |
| 6 | `sks_sem1` | float | Akademik | 0→NaN → imputed median per angkatan |
| 7 | `sks_sem2` | float | Akademik | Imputed median per angkatan |
| 8 | `sks_sem3` | float | Akademik | Imputed median per angkatan |
| 9 | `failed_courses` | int | Nilai MK | |
| 10 | `failed_in_sem1` | int | Nilai MK | |
| 11 | `repeated_courses` | int | Nilai MK | |
| 12 | `ips_trend` | float | Derived | Recomputed after imputation |
| 13 | `avg_ips` | float | Derived | Recomputed after imputation |
| 14 | `ips_std` | float | Derived | Recomputed after imputation |
| 15 | `ips_min` | float | Derived | Recomputed after imputation |
| 16 | `sks_completion_ratio` | float | Derived | Recomputed |

Plus `target` (0/1) → total 17 columns (16 features + 1 target).

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
├── 03-preprocessing.py             # Preprocessing script
└── dataset_clean.csv               # Output bersih
```
