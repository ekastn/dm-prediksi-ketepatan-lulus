# Hyperparameter Tuning Report — Decision Tree dengan Pre-Pruning + Post-Pruning

**Experiment:** `4-modeling/3-hyperparameter-tuning/`  
**Tanggal:** 22 Juni 2026  
**Model:** `DecisionTreeClassifier` dengan GridSearchCV pre-pruning + ccp_alpha post-pruning  
**Baseline:** Iterasi 2 — Stratified baseline dengan global median imputation

---

## 1. Eksperimen

### 1.1 Tujuan

Mengurangi overfitting Decision Tree baseline (train perfect 1.0, depth 8, 24 leaves) melalui 2-stage strategy:
1. **Tahap 1 — Pre-pruning:** GridSearchCV pada 5 hyperparameter untuk menemukan constraint optimal
2. **Tahap 2 — Post-pruning:** ccp_alpha pruning path untuk simplifikasi lebih lanjut
3. **Tahap 3 — Final Model:** Gabungan best params Tahap 1 + optimal ccp_alpha Tahap 2

### 1.2 Konfigurasi

| Parameter | Nilai |
|-----------|-------|
| Dataset | `4-modeling/2-global-median/dataset_clean.csv` |
| Rows | 608 (540 Tepat, 68 Tidak) |
| Features (setelah drop) | 14 (angkatan + program di-drop) |
| Split | Stratified 80/20, random_state=42 |
| Train/Test | 486 / 122 rows (54 / 14 negatif) |
| Scoring utama | `f1_score(pos_label=0)` — fokus kelas minoritas |
| CV strategy | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |

### 1.3 Parameter Grid (Tahap 1)

```python
param_grid = {
    'max_depth':          [3, 4, 5, 6, None],
    'min_samples_leaf':   [5, 10, 15, 20],
    'min_samples_split':  [5, 10, 20],
    'class_weight':       [None, 'balanced'],
    'criterion':          ['gini', 'entropy'],
}
```

Total: 5 × 4 × 3 × 2 × 2 = **240 combinations** × 5 folds = **1,200 fits**.

---

## 2. Hasil GridSearchCV

### 2.1 Best Parameters

| Hyperparameter | Best Value |
|----------------|-----------|
| `criterion` | `gini` |
| `max_depth` | **3** |
| `min_samples_leaf` | **10** |
| `min_samples_split` | **5** |
| `class_weight` | `None` |
| **Best CV F1(0)** | **0.8165** |

### 2.2 Analisis Parameter

- **`max_depth=3` dominan di seluruh top-10:** Tree dangkal memberikan F1(0) tertinggi di CV. Depth > 3 overfit tanpa peningkatan CV.
- **`min_samples_leaf ≥ 10` stabil:** Semua top-10 menggunakan `min_samples_leaf ≥ 10`. Nilai 5 overfit (terlalu granular).
- **`class_weight=None` > `balanced`:** `class_weight='balanced'` tanpa constraint overfit parah — tree tumbuh dalam mencari pola minoritas, tapi justru menurunkan CV F1(0).
- **`criterion='gini'` dan `'entropy'` setara** — `gini` hanya menang tipis di peringkat pertama.
- **Top-10 semuanya memiliki CV F1(0) yang sama (0.8165):** Banyak kombinasi dengan `max_depth=3` dan `min_samples_leaf ≥ 10` menghasilkan performa identik karena tree tidak bisa tumbuh melebihi depth 3 dengan constraint daun minimal 10 sampel.

### 2.3 Top 10 Combinations

| Rank | F1(0) CV | max_depth | min_samples_leaf | min_samples_split | class_weight | criterion |
|------|----------|-----------|-----------------|-------------------|-------------|-----------|
| 1 | 0.8165 | 3 | 10 | 5 | None | gini |
| 2 | 0.8165 | 3 | 10 | 10 | None | gini |
| 3 | 0.8165 | 3 | 10 | 20 | None | gini |
| 4 | 0.8165 | 3 | 15 | 5 | None | gini |
| 5 | 0.8165 | 3 | 15 | 10 | None | gini |
| 6 | 0.8165 | 3 | 15 | 20 | None | gini |
| 7 | 0.8165 | 3 | 20 | 5 | None | gini |
| 8 | 0.8165 | 3 | 20 | 10 | None | gini |
| 9 | 0.8165 | 3 | 20 | 20 | None | gini |
| 10 | 0.8165 | 4 | 10 | 5 | None | gini |

---

## 3. Post-Pruning (ccp_alpha)

### 3.1 Hasil

- **Number of ccp_alpha candidates:** 7 (dari tree unconstrained dengan best pre-pruning params)
- **Alpha range:** 0.000000 to 0.055318
- **Best ccp_alpha:** **0.000000** (tidak ada pruning yang diperlukan)
- **Best CV F1(0) with pruning:** 0.8165 (sama dengan sebelum pruning)

### 3.2 Analisis

Pre-pruning dengan `max_depth=3` sudah membatasi tree secara efektif — hanya 7 leaf nodes terbentuk. Akibatnya, pruning path sangat pendek (hanya 7 kandidat alpha) dan tidak ada pruning yang meningkatkan CV F1(0).

**Kesimpulan:** `max_depth=3` + `min_samples_leaf=10` sudah menjadi constraint yang cukup. ccp_alpha tidak diperlukan untuk tree ini.

---

## 4. Final Model

### 4.1 Model Summary

| Properti | Baseline | **Tuned** | Change |
|----------|----------|-----------|--------|
| Depth | 8 | **3** | ↓ 62.5% |
| Leaves | 24 | **7** | ↓ 70.8% |
| Nodes | 47 | **13** | ↓ 72.3% |
| Features used | 12/14 | **6/14** | ↓ 50% |
| Train Acc | 1.0000 | **0.9691** | ↓ 0.0309 |

Tree **jauh lebih sederhana** — dari 47 node menjadi hanya 13 node. Overfitting hilang: train accuracy turun dari perfect 1.0 ke 0.969.

### 4.2 Test Performance

| Metrik | Baseline | **Tuned** | Change |
|--------|----------|-----------|--------|
| Recall(0) | 0.9286 | **0.8571** | ↓ 0.0714 |
| Precision(0) | 0.8125 | **0.9231** | ↑ 0.1106 |
| **F1(0)** | 0.8667 | **0.8889** | **↑ 0.0222** |
| AUC | 0.9504 | **0.9239** | ↓ 0.0265 |

**F1(0) meningkat dari 0.867 ke 0.889** — meskipun recall sedikit turun (masih 85.7%), precision naik signifikan dari 0.813 ke 0.923. Trade-off recall vs precision lebih baik: hanya 1 false positive vs 2 di baseline.

### 4.3 Classification Report — Test

```
              precision    recall  f1-score   support

 Tidak Tepat     0.92      0.86      0.89        14
 Tepat Waktu     0.98      0.99      0.99       108

    accuracy                         0.98       122
```

- **12/14** mahasiswa berisiko terdeteksi (recall 85.7%) — turun 1 sampel dari baseline
- **1/108** false positive (precision 92.3%) — improved from 2 false positives
- Trade-off: 1 mahasiswa berisiko tidak terdeteksi, tapi 1 mahasiswa tepat waktu yang sebelumnya salah prediksi sekarang benar

### 4.4 Confusion Matrix — Test

```
            Pred 0  Pred 1
Actual 0      12       2     ← 2 false negatives (baseline: 1)
Actual 1       1     107     ← 1 false positive (baseline: 2)
```

### 4.5 Train Performance

| Metrik | Train |
|--------|-------|
| Recall(0) | 0.8148 |
| Precision(0) | 0.8980 |
| F1(0) | 0.8544 |
| AUC | 0.9016 |

**Unik:** Train metrics **lebih rendah** dari test metrics (train F1 0.854 vs test F1 0.889). Ini fenomena yang tidak biasa — disebabkan oleh kombinasi `max_depth=3` yang sangat restriktif + test set kecil (14 negatif) yang high-variance. Model sedikit underfit di train, tapi CV (yang lebih reliable) menunjukkan perbaikan konsisten.

---

## 5. Cross-Validation (10-Fold)

### 5.1 Tuned Model CV

| Metrik | Train | CV Test | Gap |
|--------|-------|---------|-----|
| Accuracy | 0.9691 ± 0.0015 | 0.9692 ± 0.0136 | −0.0001 |
| Recall | 0.9884 ± 0.0013 | 0.9884 ± 0.0116 | 0.0000 |
| F1 | 0.9827 ± 0.0008 | 0.9828 ± 0.0075 | −0.0001 |
| ROC-AUC | 0.9894 ± 0.0011 | **0.9588 ± 0.0701** | 0.0306 |

**Overfit gap practically eliminated!** Train-Test gap mendekati nol untuk semua metrik.

### 5.2 Gap Reduction: Baseline vs Tuned

| Metrik | Baseline Gap | Tuned Gap | Reduction |
|--------|-------------|-----------|-----------|
| Accuracy | +0.0616 | −0.0001 | ↓ 0.0617 |
| Recall | +0.0393 | 0.0000 | ↓ 0.0393 |
| F1 | +0.0349 | −0.0001 | ↓ 0.0349 |
| ROC-AUC | +0.1463 | +0.0306 | **↓ 0.1157** |

**ROC-AUC gap turun drastis dari 0.146 ke 0.031** — indikasi terkuat bahwa tuning berhasil menghilangkan overfitting. CV ROC-AUC tuned (0.959) lebih tinggi dari baseline CV (0.854).

---

## 6. Feature Importance

### 6.1 Perbandingan Baseline vs Tuned

| Rank | Feature | Baseline | Tuned | Delta |
|------|---------|----------|-------|-------|
| 1 | `sks_sem2` | 0.4321 | **0.5576** | +0.1255 |
| 2 | `sks_sem3` | 0.2800 | **0.3614** | +0.0813 |
| 3 | `ips_std` | 0.0663 | **0.0474** | −0.0190 |
| 4 | `avg_ips` | 0.0048 | **0.0172** | +0.0125 |
| 5 | `failed_courses` | 0.0260 | **0.0120** | −0.0139 |
| 6 | `ips_sem1` | 0.0792 | **0.0044** | −0.0748 |
| 7-14 | 8 features | 0.1115 | **0.0000** | — |

### 6.2 Analisis

1. **Feature importance lebih terkonsentrasi, bukan lebih balance.** Top-2 features (`sks_sem2` + `sks_sem3`) menyumbang **91.9%** importance (baseline: 71.2%). Ini adalah efek samping `max_depth=3` — tree dangkal hanya bisa menggunakan beberapa split, sehingga fitur terkuat mendominasi.

2. **8 fitur zero importance** — termasuk `ips_sem1`, `ips_sem2`, `ips_sem3`, `ips_trend`, `ips_min`, `repeated_courses`, `sks_sem1`, `failed_in_sem1`, `sks_completion_ratio`. Tree depth 3 tidak cukup dalam untuk menggunakan fitur-fitur ini.

3. **`sks_sem2` tetap prediktor #1 (0.558)** — konsisten dengan baseline. SKS semester 2 adalah sinyal terkuat untuk ketepatan lulus.

4. **`sks_sem3` naik ke #2 (0.361)** — mengkonfirmasi temuan Iterasi 2 bahwa SKS semester 3 adalah prediktor genuine, bukan artifact.

5. **`avg_ips` muncul sebagai fitur baru (0.017)** — menggantikan `ips_sem1` yang sebelumnya penting di baseline.

6. **Ini bukan feature selection yang ideal:** Konsentrasi 92% pada 2 fitur membuat model rentan terhadap perubahan kecil di data. Model yang lebih balance mungkin diperlukan di iterasi berikutnya.

---

## 7. Decision Rules

Tree depth 3, 7 leaves, 13 nodes. Rules sangat pendek dan interpretable:

```
|--- sks_sem2 <= 18.50
|   |--- failed_courses <= 0.50
|   |   |--- ips_sem1 <= 3.01        → TEPAT WAKTU
|   |   |--- ips_sem1 >  3.01        → TEPAT WAKTU
|   |--- failed_courses >  0.50      → TEPAT WAKTU
|--- sks_sem2 >  18.50
|   |--- sks_sem3 <= 18.50
|   |   |--- ips_std <= 0.29         → TIDAK TEPAT
|   |   |--- ips_std >  0.29         → TIDAK TEPAT
|   |--- sks_sem3 >  18.50
|   |   |--- avg_ips <= 3.31         → TEPAT WAKTU
|   |   |--- avg_ips >  3.31         → TEPAT WAKTU
```

### Aturan Bisnis yang Terbentuk:

1. **`sks_sem2 <= 18.5 → TEPAT WAKTU`** — Mahasiswa dengan beban SKS normal di semester 2 diprediksi tepat waktu, terlepas dari fitur lain. Ini adalah aturan paling dominan (6 dari 7 leaves).

2. **`sks_sem2 > 18.5 AND sks_sem3 <= 18.5 → TIDAK TEPAT`** — Mahasiswa overload di sem 2 tapi tidak overload di sem 3 diprediksi tidak tepat. Ips_std tidak mempengaruhi (kedua cabang menuju kelas yang sama).

3. **`sks_sem2 > 18.5 AND sks_sem3 > 18.5 → TEPAT WAKTU`** — Paradoks: mahasiswa overload di kedua semester justru diprediksi tepat waktu (split avg_ips tidak mengubah prediksi).

### Masalah dengan Rules:

- **Rules terlalu simplistik.** Hanya 1 dari 7 leaves yang memprediksi "Tidak Tepat" (mendeteksi 12/14 mahasiswa berisiko via satu rule). Model tidak bisa membedakan subtipe mahasiswa berisiko.
- **Rule #3 kontra-intuitif** — mahasiswa overload di sem 2 dan sem 3 diprediksi tepat waktu. Ini kemungkinan artefak sampel spesifik (mahasiswa dengan SKS tinggi di kedua semester yang justru lulus tepat waktu dalam dataset ini).
- **Semua cabang kiri (sks_sem2 ≤ 18.5) → Tepat Waktu** — model pada dasarnya mengatakan "kecuali SKS sem 2 tinggi dan SKS sem 3 tidak tinggi, semua orang lulus tepat waktu."

---

## 8. Learning Curve Analysis

### 8.1 Baseline vs Tuned

| | Baseline | Tuned |
|---|---|---|
| Train F1(0) akhir | ~1.00 | ~0.85 |
| CV Test F1(0) akhir | ~0.82 | ~0.82 |
| **Overfit gap** | **~0.18** | **~0.03** |
| Konvergensi | Train terus naik | Train dan CV konvergen |

### 8.2 Analisis

- **Baseline:** Train dan CV divergen — semakin banyak data, train naik ke 1.0 sementara CV stagnan di ~0.82. Overfitting klasik.
- **Tuned:** Train dan CV konvergen — kedua kurva bertemu di sekitar 0.82-0.85. Tidak ada overfitting.
- **Learning masih terjadi** — CV test score terus naik dengan data lebih banyak, menunjukkan model belum mencapai ceiling. Data tambahan (khususnya sampel negatif) masih bisa meningkatkan performa.

---

## 9. Perbandingan: Baseline vs Tuned

| Metrik | Baseline | Tuned | Delta |
|--------|----------|-------|-------|
| **Performance** | | | |
| Recall(0) | 0.9286 | 0.8571 | ↓ 0.0714 |
| Precision(0) | 0.8125 | 0.9231 | ↑ 0.1106 |
| **F1(0)** | 0.8667 | **0.8889** | ↑ 0.0222 |
| AUC | 0.9504 | 0.9239 | ↓ 0.0265 |
| **Complexity** | | | |
| Depth | 8 | **3** | ↓ 62.5% |
| Leaves | 24 | **7** | ↓ 70.8% |
| Nodes | 47 | **13** | ↓ 72.3% |
| **Overfitting** | | | |
| Train Acc | 1.0000 | 0.9691 | ↓ 0.0309 |
| CV ROC-AUC Gap | +0.1463 | +0.0306 | ↓ 0.1157 |

---

## 10. Kesimpulan

### 10.1 Apakah Pruning Berhasil?

**Ya, sangat berhasil dalam mengurangi overfitting:**
- Overfit gap dari +0.146 (AUC) menjadi +0.031 — praktis hilang
- Train accuracy dari 1.0 ke 0.969 — tidak lagi perfect fit
- Tree complexity turun 70%+ di semua dimensi
- CV performance justru meningkat (ROC-AUC 0.854 → 0.959)

**Tapi ada trade-off:**
- Recall(0) turun dari 0.929 ke 0.857 (masih di atas target ≥ 0.70)
- Feature importance terkonsentrasi pada 2 fitur (92%)
- Decision rules terlalu simplistik — hanya 1 rule untuk mendeteksi "Tidak Tepat"

### 10.2 Apakah Model Lebih Interpretable?

**Ya, secara dramatis:**
- 13 node vs 47 — mudah dibaca dan dijelaskan
- 7 leaves menghasilkan 4 aturan bisnis yang jelas
- Depth 3 membuat visualisasi tree sangat rapi
- Stakeholder non-teknis bisa memahami logika: "SKS semester 2 tinggi + SKS semester 3 normal = berisiko"

### 10.3 Target Modeling Plan

| Metrik | Target | Baseline | Tuned | Status |
|--------|--------|----------|-------|--------|
| Recall(0) | ≥ 0.70 | 0.929 | 0.857 | ✅ |
| F1(0) | ≥ 0.50 | 0.867 | 0.889 | ✅ |
| ROC-AUC | ≥ 0.75 | 0.950 | 0.924 | ✅ |

Semua target masih tercapai. F1(0) justru meningkat.

---

## 11. Rekomendasi Next Step

### 11.1 Trade-off Analysis: Depth 3 vs Depth 4

`max_depth=3` terlalu restriktif — 8 fitur zero importance, rules oversimplified. Coba `max_depth=4` dengan `min_samples_leaf=5` untuk tree yang sedikit lebih dalam tapi tetap terkontrol. Target: depth 4-5, 10-15 leaves, lebih banyak fitur digunakan.

### 11.2 Feature Engineering

Konsentrasi 92% pada `sks_sem2` + `sks_sem3` menunjukkan bahwa kombinasi fitur SKS mungkin lebih informatif:
- `sks_overload_sem2` = sks_sem2 > 18 (binary flag)
- `sks_total` = sks_sem1 + sks_sem2 + sks_sem3
- `sks_trend` = sks_sem3 - sks_sem2 (penurunan/peningkatan beban)

### 11.3 Model yang Lebih Balance

Decision tree cenderung overfit ke fitur terkuat dengan depth rendah. Random Forest bisa memberikan feature importance yang lebih balance sambil tetap menjaga interpretability via feature importance + partial dependence.

### 11.4 Validasi Temporal

Model saat ini diuji dengan stratified split. Uji dengan temporal split (train ≤ 2021, test > 2021) untuk mensimulasikan skenario prediksi nyata — meskipun recall(0) akan turun karena hanya 14 sampel negatif di training.

### 11.5 Menangani Underfit

`max_depth=3` menyebabkan model sedikit underfit (train F1 < test F1). Coba:
- `max_depth=4` atau `5` dengan `min_samples_leaf=10`
- `min_samples_split=10` untuk mencegah split terlalu granular
- Cost-complexity pruning dengan tree yang lebih dalam (depth 5-6, lalu prune)

---

## 12. Deliverables

| File | Deskripsi |
|------|-----------|
| `4-modeling/3-hyperparameter-tuning/01-hyperparameter-tuning.ipynb` | Notebook eksekusi |
| `4-modeling/3-hyperparameter-tuning/01-tuning-results.md` | Output nbconvert + charts |
| `4-modeling/3-hyperparameter-tuning/01-tuning-results_files/` | PNG charts |
| `4-modeling/3-hyperparameter-tuning/rules_tuned.txt` | Decision rules (20 baris) |
| `4-modeling/3-hyperparameter-tuning/tuning-report.md` | **Dokumen ini** |
