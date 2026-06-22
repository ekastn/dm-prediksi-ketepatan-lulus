# Stratified Split Baseline Report

**Fase 4 — Modeling | Variasi: Stratified Split + Drop Angkatan**  
Tanggal: 17 Juni 2026  
Model: `DecisionTreeClassifier` (default) — 14 fitur (angkatan + program di-drop)

---

## 1. Eksperimen

### 1.1 Tujuan

Menguji apakah **menambah sampel negatif di training** (14 → 54) via stratified random split memperbaiki recall kelas 0 yang collapse di temporal baseline (3.7%).

### 1.2 Perubahan dari Notebook 01 (Temporal)

| | 01 (Temporal) | 02 (Stratified) |
|---|---|---|
| Split | angkatan ≤ 2021 / > 2021 | random 80/20, stratify=target |
| Train negatif | 14 (3.7%) | **54 (11.1%)** |
| Test negatif | 54 (23.4%) | 14 (11.5%) |
| Fitur | 16 (termasuk angkatan) | **14 (angkatan + program di-drop)** |
| Alasan drop | — | angkatan = data leakage (2023 = semua target=0) |

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

 Tidak Tepat     0.65      0.93      0.76        14
 Tepat Waktu     0.99      0.94      0.96       108

    accuracy                         0.93       122
   macro avg     0.82      0.93      0.86       122
weighted avg     0.95      0.93      0.94       122
```

### 2.2 Key Metrics

| Metrik | Temporal (01) | **Stratified (02)** | Delta |
|--------|:-----------:|:---------------:|--------|
| Accuracy | 0.775 | **0.934** | +0.159 |
| Precision(0) | 1.000 | **0.650** | -0.350 |
| **Recall(0)** | 0.037 | **0.929** | **+0.892** |
| **F1(0)** | 0.071 | **0.765** | **+0.694** |
| **AUC** | 0.519 | **0.932** | **+0.413** |

### 2.3 Confusion Matrix — Test

```
            Pred 0  Pred 1
Actual 0      13       1     ← hanya 1 false negative
Actual 1       7     101     ← 7 false positive
```

- **13/14** mahasiswa berisiko terdeteksi (recall 92.9%)
- **7** false positive — mahasiswa tepat diprediksi telat (precision 65%)
- Model jauh lebih berguna: trade-off recall vs precision yang wajar

### 2.4 Cross-Validation (10-fold, Train only)

| Metrik | Train | CV Test | Gap |
|--------|-------|---------|-----|
| Accuracy | 1.000 | 0.959 | +0.041 |
| Recall | 1.000 | 0.975 | +0.026 |
| F1 | 1.000 | 0.977 | +0.023 |
| ROC-AUC | 1.000 | **0.901** | +0.099 |

**CV ROC-AUC 0.90** — sinyal diskriminasi yang jauh lebih kuat dibanding temporal baseline (0.69).

---

## 3. Feature Importance

### 3.1 Ranking

| Rank | Fitur | Importance | Kategori |
|------|-------|-----------|----------|
| 1 | `sks_sem2` | **0.621** | SKS |
| 2 | `sks_sem3` | 0.095 | SKS |
| 3 | `ips_std` | 0.068 | Derived |
| 4 | `ips_sem3` | 0.061 | IPS |
| 5 | `ips_sem1` | 0.060 | IPS |
| 6 | `failed_courses` | 0.030 | Nilai MK |
| 7 | `ips_min` | 0.024 | Derived |
| 8 | `avg_ips` | 0.017 | Derived |
| 9 | `repeated_courses` | 0.010 | Nilai MK |
| 10 | `ips_trend` | 0.010 | Derived |
| 11 | `ips_sem2` | 0.002 | IPS |
| 12 | `sks_sem1` | 0.002 | SKS |
| 13 | `failed_in_sem1` | **0.000** | Nilai MK |
| 14 | `sks_completion_ratio` | **0.000** | Derived |

### 3.2 Analisis

**PERINGATAN: `sks_sem2` dominance (0.621).** Root split ada di `sks_sem2 <= 18.50`. Ini perlu investigasi:

1. **Kemungkinan artifact TSKS bug:** Data quality issue yang sudah di-fix di `extract_dataset.py` mungkin belum tuntas untuk semester 2. Beberapa mahasiswa mungkin masih punya nilai SKS abnormal yang berkorelasi dengan target.

2. **Atau genuine signal:** Mahasiswa yang mengambil sedikit SKS di semester 2 (dropout efektif, cuti tidak resmi) memang cenderung telat. Tapi 62% importance untuk satu fitur mencurigakan.

**Zero importance:** `failed_in_sem1` dan `sks_completion_ratio` tidak digunakan tree.

**Perbandingan dengan temporal baseline:** Di temporal, top features adalah `ips_min` (0.25), `failed_courses` (0.18), `ips_sem1` (0.16). Di stratified, fitur-fitur IPS didorong ke bawah oleh dominasi SKS. Ini menunjukkan **struktur data yang berbeda terungkap** tergantung komposisi train set.

---

## 4. Decision Rules

Tree depth 8, 23 leaves. Root split:

```
|--- sks_sem2 <= 18.50
|   |--- ips_sem1 <= 2.70
|   |   |--- class: 0          ← IPS sem1 rendah → langsung prediksi telat
|   |--- ips_sem1 > 2.70
|       |--- failed_courses <= 4.00
|           ... (mayoritas Tepat Waktu)
|       |--- failed_courses > 4.00
|           |--- ips_std <= 0.55 → TIDAK TEPAT
|           |--- ips_std > 0.55  → TEPAT WAKTU
|--- sks_sem2 > 18.50
    |--- [cabang kompleks: sks_sem3, ips_std, avg_ips, ips_min]
        ... campuran Tepat dan Tidak Tepat
```

### Aturan yang masuk akal secara bisnis:

1. **`ips_sem1 <= 2.70 → TIDAK TEPAT`** — Mahasiswa dengan IPS semester 1 di bawah 2.70 berisiko. Masuk akal.

2. **`failed_courses > 4 AND ips_std <= 0.55 → TIDAK TEPAT`** — Banyak gagal MK + volatilitas rendah = kemungkinan konsisten buruk (bukan cuma satu semester jelek). Masuk akal.

3. **Cabang `sks_sem2 > 18.50`** — Mahasiswa dengan SKS tinggi di semester 2 dianalisis lebih dalam dengan `sks_sem3`, `ips_std`, `avg_ips`. Ini mencakup mahasiswa yang "jalan normal" tapi tetap perlu diskrining.

### Aturan yang mencurigakan:

- Beberapa split pada `ips_std <= 0.10`, `ips_sem1 <= 3.12`, dsb. sangat granular — kemungkinan overfit ke sampel spesifik.

---

## 5. Perbandingan Tiga Eksperimen

| | 01—Temporal | 02—Stratified (dengan angkatan) | **02—Stratified (no angkatan)** |
|---|---|---|---|
| Train negatif | 14 | 54 | 54 |
| Fitur | 16 | 16 | **14** |
| Recall(0) | 0.037 | 0.929 | **0.929** |
| F1(0) | 0.071 | 0.813 | **0.765** |
| AUC | 0.519 | 0.941 | **0.932** |
| Top feature | ips_min | **angkatan (0.68)** | **sks_sem2 (0.62)** |
| Masalah | underfit (min sampel) | data leakage (angkatan) | sks_sem2 dominance? |

**Kesimpulan:** 02—Stratified (no angkatan) adalah **baseline paling jujur** sejauh ini. Recall 92.9% dengan precision 65% adalah starting point yang layak untuk tuning. Namun dominasi `sks_sem2` perlu diinvestigasi.

---

## 6. Analisis `sks_sem2` Dominance

Perlu dicek: apakah SKS semester 2 benar-benar prediktor kuat, atau ini artifact data?

**Hipotesis artifact:**
- Database menyimpan SKS tidak konsisten (TSKS kumulatif vs per-semester)
- Fix di `extract_dataset.py` menggunakan cap MK 1-20 + fallback imputasi
- Jika imputasi median per angkatan tidak sempurna, residual artifact bisa muncul

**Hipotesis genuine:**
- SKS semester 2 rendah → mahasiswa mengurangi beban (atau cuti tidak resmi) → indikator awal masalah
- Ini align dengan EDA finding bahwa `sks_sem2` berkorelasi r=-0.35 dengan target

**Rekomendasi:** Perlu dicek distribusi `sks_sem2` per target class. Jika ada pemisahan jelas (misal: target=0 mostly <10 SKS), ini genuine. Jika ada cluster aneh (nilai >24 atau 0), ini artifact.

---

## 7. Rekomendasi untuk Step Berikutnya

### 7.1 Investigasi `sks_sem2`

- Plot distribusi `sks_sem2` per target class
- Cek apakah ada nilai abnormal (>24 atau <1)
- Jika artifact: fix di preprocessing, re-run
- Jika genuine: keep, dan ini interesting finding

### 7.2 Handling Class Imbalance (lanjut)

- **Stratified split sudah membantu signifikan** (54 negatif di train vs 14)
- Tapi masih bisa ditingkatkan dengan SMOTE atau class_weight
- Target: precision(0) naik dari 0.65 ke ≥0.75

### 7.3 Hyperparameter Tuning

- Constraint tree complexity: `max_depth=3-6`, `min_samples_leaf=5-10`
- Scoring untuk GridSearchCV: `'recall'` atau `'f1'` dengan `pos_label=0`
- Pruning dengan `ccp_alpha`

### 7.4 Validasi dengan Temporal Split

- Setelah model optimal ditemukan di stratified split
- Uji ulang dengan temporal split sebagai sanity check
- Jika recall(0) temporal tetap rendah → model overfit ke stratified distribution

---

## 8. Deliverables

| File | Deskripsi |
|------|-----------|
| `02-stratified-baseline.ipynb` | Notebook stratified split + drop angkatan |
| `02-stratified-results.md` | Output nbconvert — full results + 3 charts |
| `02-stratified-results_files/` | 3 PNG (confusion matrix, feature importance, decision tree) |
| `rules_stratified.txt` | 68 baris decision rules |
| `stratified-report.md` | Dokumen ini |
