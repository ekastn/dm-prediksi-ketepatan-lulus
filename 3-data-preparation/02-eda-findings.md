# EDA Findings: Prediksi Ketepatan Lulus Mahasiswa

Tanggal: 16 Juni 2026 | Dataset: `dataset.csv` (608 rows x 27 kolom)

---

## A. Missing Values

### A1. Overview

| Fitur | Missing | % |
|-------|---------|---|
| `avg_attendance` | 321 | 52.8% |
| `ips_sem4` / `sks_sem4` | 191 | 31.4% |
| `ips_sem1` / `sks_sem1` | 167 | 27.5% |
| `ips_sem2` / `sks_sem2` | 119 | 19.6% |
| `ips_trend`, `avg_ips`, `ips_std`, `ips_max`, `ips_min` | 118 | 19.4% |
| `ips_sem3` / `sks_sem3` | 18 | 3.0% |

### A2. Pola Missing: Struktural, Bukan Random

Missing by angkatan (rata-rata missing IPS dari 4 per mahasiswa):

| Angkatan | Avg Missing |
|----------|------------|
| 2015 | 1.99 (sangat parah) |
| 2018 | 1.85 |
| 2016 | 0.89 |
| 2017 | 0.88 |
| 2019 | 0.85 |
| 2020 | 0.62 |
| 2021 | 0.50 |
| 2022 | 0.03 |
| 2023 | 0.26 |

Missing by semester_count:

| Semester Count | Avg Missing |
|---------------|-------------|
| 3 | 2.17 |
| 7 | 1.53 |
| 8 | 1.45 |
| 12 | 3.00 |
| 4-6, 9 | ~0 |

**Kesimpulan**: Missing sistematik — angkatan tua (2015, 2018) dan mahasiswa dengan semester ganjil (3, 7, 8) paling parah. Imputasi harus per-angkatan, bukan global.

---

## B. Target Variable

| Program | Tidak Tepat | Tepat Waktu | % Neg |
|---------|-------------|-------------|-------|
| AP (D3) | 10 | 137 | 6.8% |
| IH (S1) | 58 | 403 | 12.6% |
| **Total** | **68** | **540** | **11.2%** |

IH hampir 2x lebih berisiko tidak tepat waktu.

---

## C. Distribusi & Outlier

### C1. IPS Features Summary

| Fitur | Mean | Median | Std | Min | 25% | 75% | Max |
|-------|------|--------|-----|-----|-----|-----|-----|
| ips_sem1 | 1.96 | **3.00** | 1.58 | 0 | **0.00** | 3.25 | 4 |
| ips_sem2 | 3.22 | 3.25 | 0.47 | 0 | 3.13 | 3.38 | 4 |
| ips_sem3 | 3.48 | 3.67 | 0.62 | 0 | 3.15 | 3.88 | 4 |
| ips_sem4 | 2.89 | 3.11 | 1.00 | 0 | 3.00 | 3.43 | 4 |

### C2. CRITICAL: ips_sem1 Terkontaminasi System Zero

**25% dari nilainya = 0.0**. Ini BUKAN nilai IPS riil — ini placeholder dari sistem legacy. Mean (1.96) jauh di bawah median (3.00), menunjukkan distribusi bimodal: mayoritas di sekitar 3.x + spike di 0.

Dampaknya: korelasi `ips_sem1` dengan target = **-0.27** (negatif!) — artinya mahasiswa dengan ips_sem1=0 justru lebih mungkin lulus. Ini artefak data, bukan sinyal.

### C3. Outlier Check

- **219 mahasiswa (36%)** punya minimal satu IPS = 0.0
  - 174 tepat waktu (79%), 45 tidak tepat (21%)
  - Ini konfirmasi: IPS=0 bukan nilai riil (79% tetap lulus)
- **118 mahasiswa (19%)** punya minimal satu IPS = 4.0

---

## D. Pemisahan Kelas (Target=0 vs Target=1)

### D1. Fitur yang Paling Mendiskriminasi

#### ips_trend

| Target | Mean | Median | 25% | 75% |
|--------|------|--------|-----|-----|
| Tidak Tepat | -1.86 | **-3.00** | -3.10 | -0.11 |
| Tepat Waktu | +1.30 | +0.38 | +0.01 | +3.00 |

**Tren negatif = red flag**. Target=0 hampir selalu punya tren turun drastis.

#### avg_ips

| Target | Mean | Median | 25% | 75% |
|--------|------|--------|-----|-----|
| Tidak Tepat | 2.48 | 2.52 | 2.42 | 2.57 |
| Tepat Waktu | 3.01 | 3.06 | 2.53 | 3.40 |

#### ips_min (nilai IPS terendah)

| Target | Mean | Median | 25% | 75% |
|--------|------|--------|-----|-----|
| Tidak Tepat | 0.66 | **0.00** | 0.00 | 0.65 |
| Tepat Waktu | 1.85 | 3.00 | 0.00 | 3.14 |

Target=0 punya median ips_min = 0.00 — semua pernah "nol". Target=1 punya median = 3.00.

#### failed_courses

| Target | Mean | Median | 75% | Max |
|--------|------|--------|-----|-----|
| Tidak Tepat | 3.00 | 0 | 5.25 | 18 |
| Tepat Waktu | 0.09 | 0 | 0 | 14 |

Median sama (0) — kebanyakan mahasiswa tidak gagal MK. Tapi outlier di target=0 sangat ekstrem (mean 3.0 vs 0.09).

#### ips_std (volatilitas)

| Target | Mean | Median |
|--------|------|--------|
| Tidak Tepat | 1.40 | 1.66 |
| Tepat Waktu | 0.83 | 0.32 |

Kinerja tidak stabil = sinyal bahaya.

### D2. Fitur yang GAGAL Mendiskriminasi

- **ips_max**: mean 3.65 untuk BOTH kelas — semua mahasiswa pernah punya semester bagus
- **repeated_courses**: mean lebih TINGGI di target=1 (0.46) vs target=0 (0.12) — counter-intuitive, kemungkinan karena mahasiswa yang bertahan lebih lama punya lebih banyak MK

---

## E. Korelasi dengan Target

### E1. Top Positive

| Fitur | Pearson r |
|-------|-----------|
| ips_sem4 | **+0.877** ⚠️ |
| ipk_sem4 | +0.633 |
| ips_trend | **+0.569** |
| ips_sem2 | +0.421 |
| avg_ips | +0.365 |

### E2. Top Negative

| Fitur | Pearson r |
|-------|-----------|
| failed_courses | -0.452 |
| sks_sem4 | -0.356 |
| angkatan | -0.339 |
| ips_sem1 | **-0.272** ⚠️ |
| failed_in_sem1 | -0.258 |
| ips_std | -0.257 |

### E3. Data Leakage Flag

**ips_sem4 (r=0.877)** terlalu tinggi. Ini SPURIOUS — mahasiswa yang lulus tepat waktu (target=1) otomatis punya ips_sem4 tinggi karena mereka bertahan sampai semester 4. Yang dropout (target=0) punya ips_sem4 rendah atau tidak ada. Model akan belajar tautologi.

`ips_sem4`, `sks_sem4`, `ipk_sem4`, dan `semester_count` semuanya mengandung informasi circular dengan target dan harus di-drop.

### E4. Near-Zero Correlation

- `avg_attendance`: **-0.016** (hampir nol)
- `ips_max`: 0.0001 (hampir nol)
- `id_agama`: 0.031
- `gender_num`: 0.050

---

## F. Attendance Analysis

- avg_attendance: 287 non-null, 321 missing (52.8%)
- Stats non-null: mean=82.35, median=89.80, std=19.31

Target rate by has_attendance:

| | Tidak Tepat | Tepat Waktu |
|---|------------|-------------|
| Missing attendance | 13.4% | 86.6% |
| Has attendance data | 8.7% | 91.3% |

Ada sinyal lemah di flag `has_attendance` (selisih ~4.7%), tapi nilai numerik kehadirannya sendiri hampir tidak membedakan kelas.

---

## G. IPS Inter-Correlations

ips_sem1-4 saling berkorelasi moderat (0.3-0.6) — tidak redundan. Multikolinearitas BUKAN masalah besar antar semester.

ipk_sem4 sangat berkorelasi dengan avg_ips (r=0.83) dan ips_sem4 (r=0.86) — redundant.

---

## H. Implikasi untuk Preprocessing

### Fitur yang Harus Di-drop

| Fitur | Alasan |
|-------|--------|
| `student_id` | Identitas, bukan fitur |
| `ips_sem4`, `sks_sem4` | Data leakage (r=0.877 target) |
| `ipk_sem4` | Redundant dengan avg_ips |
| `semester_count` | Circular — target didefinisikan sebagian darinya |
| `ips_max` | Tidak mendiskriminasi (r≈0) |
| `avg_attendance` | 53% missing + r≈0 |
| `total_sks_lulus_sem4` | Redundant dengan sks_completion_ratio |

### Data Cleaning

1. **Replace IPS = 0.0 dengan NaN** — 36% mahasiswa terpengaruh, ini system zero bukan nilai riil
2. Median imputation per angkatan untuk `ips_sem1-3` dan `sks_sem1-3`
3. Recompute derived features setelah imputasi

### Encoding

- `program`: Label (AP=0, IH=1)
- `jenis_kelamin`: Label (L=0, P=1)
- `id_agama`: Keep as-is (1/2/4)

### Feature yang Dipertahankan (~14 fitur)

1. `angkatan`
2. `program` (encoded)
3. `jenis_kelamin` (encoded)
4. `id_agama`
5. `ips_sem1` (after 0→NaN + imputation)
6. `ips_sem2`
7. `ips_sem3`
8. `sks_sem1`
9. `sks_sem2`
10. `sks_sem3`
11. `failed_courses`
12. `failed_in_sem1`
13. `repeated_courses`
14. `has_attendance` (flag 0/1, ganti avg_attendance)

Derived features (recomputed after imputation):
15. `ips_trend`
16. `avg_ips`
17. `ips_std`
18. `ips_min`
19. `sks_completion_ratio`
