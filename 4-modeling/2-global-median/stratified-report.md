# Stratified Split Baseline Report — Global Median Imputation

**Experiment:** `experiments/global-median/`  
**Tanggal:** 17 Juni 2026  
**Model:** `DecisionTreeClassifier` (default) — 14 fitur (angkatan + program di-drop)  
**Imputation:** Global median (menggantikan median per-angkatan)  

---

## 1. Eksperimen

### 1.1 Tujuan

Menguji apakah **menambah sampel negatif di training** (14 → 54) via stratified random split + **global median imputation** memperbaiki recall kelas 0 yang collapse di temporal baseline (5.6%).

### 1.2 Konfigurasi

| | Temporal (01) | **Stratified (02)** |
|---|---|---|
| Split | angkatan ≤ 2021 / > 2021 | random 80/20, stratify=target |
| Train negatif | 14 (3.7%) | **54 (11.1%)** |
| Test negatif | 54 (23.4%) | 14 (11.5%) |
| Fitur | 16 | **14** (angkatan + program di-drop) |
| Imputation | Global median | Global median |
| Imputed sks_sem2 | 47.5% rows → nilai 18.0 | sama |

### 1.3 Split Detail

```
608 rows → stratified 80/20 (random_state=42)

Train: 486 rows (432 Tepat, 54 Tidak) — 11.1% neg
Test:  122 rows (108 Tepat, 14 Tidak) — 11.5% neg
```

Semua angkatan (2015-2023) tersebar di kedua set. Proporsi kelas seimbang (~11% negatif).

---

## 2. Hasil Evaluasi

### 2.1 Classification Report — Test

```
              precision    recall  f1-score   support

 Tidak Tepat     0.81      0.93      0.87        14
 Tepat Waktu     0.99      0.97      0.98       108

    accuracy                         0.97       122
   macro avg     0.90      0.95      0.92       122
weighted avg     0.97      0.97      0.97       122
```

### 2.2 Key Metrics

| Metrik | Temporal (01) | **Stratified (02)** | Delta |
|--------|:-----------:|:---------------:|--------|
| Accuracy | 0.779 | **0.967** | +0.188 |
| Precision(0) | 1.000 | **0.813** | -0.187 |
| **Recall(0)** | 0.056 | **0.929** | **+0.873** |
| **F1(0)** | 0.105 | **0.867** | **+0.762** |
| **AUC** | 0.528 | **0.950** | **+0.422** |

**Recall 92.9%, precision 81.3%, F1 86.7%** — semua target modeling-plan tercapai di baseline!

### 2.3 Confusion Matrix — Test

```
            Pred 0  Pred 1
Actual 0      13       1     ← hanya 1 false negative
Actual 1       2     106     ← hanya 2 false positive
```

- **13/14** mahasiswa berisiko terdeteksi (recall 92.9%)
- **2/108** false positive (precision 81.3%) — only 2 mahasiswa tepat waktu salah diprediksi
- Trade-off recall vs precision sangat baik untuk baseline unconstrained

### 2.4 Cross-Validation (10-fold, Train only)

| Metrik | Train | CV Test | Gap |
|--------|-------|---------|-----|
| Accuracy | 1.000 | 0.938 | +0.062 |
| Recall | 1.000 | 0.961 | +0.039 |
| F1 | 1.000 | 0.965 | +0.035 |
| ROC-AUC | 1.000 | **0.854** | +0.146 |

**CV ROC-AUC 0.854** — sinyal diskriminasi kuat, masih ada overfitting gap.

---

## 3. Feature Importance

### 3.1 Ranking

| Rank | Fitur | Importance | Kategori |
|------|-------|-----------|----------|
| 1 | `sks_sem2` | **0.432** | SKS |
| 2 | `sks_sem3` | **0.280** | SKS |
| 3 | `ips_sem1` | 0.079 | IPS |
| 4 | `ips_std` | 0.066 | Derived |
| 5 | `ips_sem2` | 0.039 | IPS |
| 6 | `ips_trend` | 0.028 | Derived |
| 7 | `failed_courses` | 0.026 | Nilai MK |
| 8 | `ips_min` | 0.019 | Derived |
| 9 | `ips_sem3` | 0.014 | IPS |
| 10 | `repeated_courses` | 0.010 | Nilai MK |
| 11 | `avg_ips` | 0.005 | Derived |
| 12 | `sks_sem1` | 0.002 | SKS |
| 13 | `failed_in_sem1` | **0.000** | Nilai MK |
| 14 | `sks_completion_ratio` | **0.000** | Derived |

### 3.2 Perbandingan dengan Temporal Baseline

| | Temporal (01) | Stratified (02) |
|---|---|---|
| Top 1 | `ips_min` (0.247) | `sks_sem2` (0.432) |
| Top 2 | `ips_sem1` (0.204) | `sks_sem3` (0.280) |
| Top 3 | `failed_courses` (0.184) | `ips_sem1` (0.079) |
| `sks_sem2` rank | 12 (0.019) | 1 (0.432) |

**Interpretasi:** Dengan hanya 14 sampel negatif, temporal tree split pada fitur IPS (`ips_min` di root). Dengan 54 sampel negatif, stratified tree menemukan pola yang lebih kompleks — SKS semester 2 dan 3 menjadi prediktor utama. Ini menunjukkan **struktur data yang berbeda terungkap** ketika sampel minoritas cukup.

### 3.3 Analisis `sks_sem2` (0.432)

Setelah investigasi (Notebook 03):
- **Bukan artifact** — tidak ada nilai abnormal (>24 atau <2)
- **Separation genuine** — target=0 mean=19.4 SKS vs target=1 mean=13.0
- **Global median (18.0)** tidak menciptakan proxy temporal — semua angkatan dapat nilai imputasi sama
- Dominasi 0.432 (bukan 0.621) masih agak tinggi tapi wajar untuk prediktor kuat

---

## 4. Decision Rules

Tree depth 8, 24 leaves, 47 nodes. Root split:

```
|--- sks_sem2 <= 18.50
|   |--- ips_sem1 <= 2.70
|   |   |--- class: 0          ← IPS sem1 rendah → langsung prediksi telat
|   |--- ips_sem1 > 2.70
|       |--- failed_courses <= 4.00
|           ... (mayoritas Tepat Waktu, campur kompleks)
|       |--- failed_courses > 4.00
|           |--- ips_sem2 <= 3.12 → TEPAT WAKTU (paradoks)
|           |--- ips_sem2 > 3.12  → TIDAK TEPAT
|--- sks_sem2 > 18.50
    |--- sks_sem3 <= 18.50
        |--- ips_std <= 0.27
            |--- ips_sem2 <= 3.46
                ... campuran kompleks
            |--- ips_sem2 > 3.46 → TIDAK TEPAT
        |--- ips_std > 0.27 → TIDAK TEPAT
    |--- sks_sem3 > 18.50
        |--- ips_sem1 <= 2.93 → TIDAK TEPAT
        |--- ips_sem1 > 3.26 → TEPAT WAKTU
        |--- antara 2.93-3.26: campuran dengan ips_min, ips_sem3, ips_std, ips_trend
```

### Aturan yang masuk akal secara bisnis:

1. **`sks_sem2 <= 18.5 AND ips_sem1 <= 2.70 → TIDAK TEPAT`** — SKS normal + IPS rendah semester 1 = red flag klasik.

2. **`sks_sem2 > 18.5 AND sks_sem3 > 18.5 AND ips_sem1 <= 2.93 → TIDAK TEPAT`** — Mahasiswa overload SKS di sem 2-3 + IPS sem 1 rendah = risiko tinggi.

3. **`sks_sem2 > 18.5 AND ips_std > 0.27 → TIDAK TEPAT`** — SKS tinggi + volatilitas IPS tinggi = performa tidak stabil = risiko.

### Aturan yang perlu diwaspadai:

- **`failed_courses > 4 AND ips_sem2 <= 3.12 → TEPAT WAKTU`** — kontra-intuitif, mungkin artefak sampel spesifik
- Split sangat granular pada `ips_std <= 0.25`, `ips_trend <= 0.18`, dsb. — indikasi overfitting

---

## 5. Perbandingan: Per-Angkatan vs Global Median Imputation

| | Per-Angkatan | **Global Median** | Change |
|---|---|---|---|
| **Performance** | | | |
| Recall(0) | 0.929 | **0.929** | = |
| Precision(0) | 0.650 | **0.813** | +0.163 ↑ |
| **F1(0)** | 0.765 | **0.867** | **+0.102 ↑** |
| AUC | 0.932 | **0.950** | +0.018 ↑ |
| **Feature Importance** | | | |
| `sks_sem2` | **0.621** | **0.432** | -0.189 ↓ |
| `sks_sem3` | 0.095 | **0.280** | +0.185 ↑ |
| Top-2 combined | 0.717 | **0.712** | = |
| Top-3 spread | mono (0.62) | dual (0.43+0.28) | lebih seimbang |

**Global median lebih baik di semua aspek:**
1. Feature importance lebih balance (0.43 vs 0.62)
2. Performance justru meningkat (F1 0.87 vs 0.77)
3. Tidak ada proxy temporal — imputasi bersih
4. `sks_sem3` naik prominen — sinyal genuine yang sebelumnya tertutup dominance `sks_sem2`

---

## 6. Perbandingan Semua Eksperimen

| | 01 Temporal | 02 Stratified (per-angkatan) | **02 Stratified (global median)** |
|---|---|---|---|
| Train negatif | 14 | 54 | 54 |
| Imputation | Per-angkatan | Per-angkatan | **Global median** |
| Fitur | 16 | 14 | 14 |
| Recall(0) | 0.037 | 0.929 | **0.929** |
| Precision(0) | 1.000 | 0.650 | **0.813** |
| **F1(0)** | 0.071 | 0.765 | **0.867** |
| AUC | 0.519 | 0.932 | **0.950** |
| Top feature | ips_min (0.25) | sks_sem2 (0.62) | **sks_sem2 (0.43)** |
| Masalah | underfit | proxy temporal | sks_sem2 masih dominan |

**Stratified + Global Median adalah baseline terbaik sejauh ini.**

---

## 7. Rekomendasi untuk Step Berikutnya

### 7.1 Adopsi Global Median

Global median imputation sudah terbukti lebih baik — adopsi ke pipeline utama.

### 7.2 Handling Overfitting

Tree masih unconstrained (depth 8, train perfect 1.0):
- Constraint `max_depth=4-6`
- `min_samples_leaf=5-10`
- `ccp_alpha` pruning path

### 7.3 Class Balancing (Opsional)

Stratified split sudah beri 54 sampel negatif — cukup untuk baseline. Tapi SMOTE atau `class_weight='balanced'` masih bisa dicoba untuk boost precision(0) dari 0.81 ke ≥0.90.

### 7.4 Feature Engineering

- Binarize `sks_sem2`: `sks_sem2_high` = sks_sem2 > 18 (flag overload)
- Ini bisa membuat tree lebih sederhana tapi sama informatifnya

### 7.5 Validasi Temporal

Setelah model final ditemukan, uji ulang dengan temporal split untuk konfirmasi generalisasi ke skenario nyata.

---

## 8. Deliverables

| File | Deskripsi |
|------|-----------|
| `experiments/global-median/preprocessing.py` | Script preprocessing global median |
| `experiments/global-median/dataset_clean.csv` | Dataset hasil |
| `experiments/global-median/02-stratified-baseline.ipynb` | Notebook stratified baseline |
| `experiments/global-median/02-stratified-results.md` | Output + 3 chart |
| `experiments/global-median/02-stratified-results_files/` | PNG (confusion, importance, tree) |
| `experiments/global-median/rules_stratified.txt` | 71 baris decision rules |
| `experiments/global-median/README.md` | Report eksperimen (high-level) |
| `experiments/global-median/stratified-report.md` | **Dokumen ini** |
