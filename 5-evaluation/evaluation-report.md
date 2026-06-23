# Evaluation Report — Decision Tree untuk Prediksi Ketepatan Lulus

**CRISP-DM Phase 5: Validasi model final dengan temporal split, repeated CV, error analysis, dan permutation importance.**
**Tanggal: 23 Juni 2026**

---

## 1. Overview

### 1.1 Objective

Model 01-Tuned (`max_depth=3, min_samples_leaf=10`) mencapai F1(0)=0.889 pada stratified split, tapi **belum pernah diuji pada temporal split** — skenario yang mensimulasikan prediksi nyata (train pada data historis, prediksi pada angkatan baru). Fase ini menjawab pertanyaan: **apakah model generalizable ke skenario deployment?**

### 1.2 Model Under Evaluation

```python
DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
```

| Properti | Nilai |
|----------|-------|
| Depth | 3 |
| Leaves | 7 (2 "Tidak Tepat", 5 "Tepat Waktu") |
| Features used | 6/14 |
| Top features (MDI) | `sks_sem2` (0.558), `sks_sem3` (0.361) |
| Known F1(0) | 0.889 (stratified single split) |

### 1.3 Dataset

| | Temporal Split | Stratified Split |
|---|---|---|
| **Train** | 377 (≤2021), 14 neg (3.7%) | 486 (80%), 54 neg (11.1%) |
| **Test** | 231 (>2021), 54 neg (23.4%) | 122 (20%), 14 neg (11.5%) |
| **Source** | `5-evaluation/dataset_clean.csv` (global median imputation) |

### 1.4 Evaluation Scope

| # | Phase | Method |
|---|-------|--------|
| 1 | Temporal Validation | Train best model on temporal split, compare vs stratified |
| 2 | Deep Error Analysis | Individual FN/FP profiling, rule coverage, program/angkatan breakdown |
| 3 | Repeated CV | 10×10-fold RepeatedStratifiedKFold (100 evaluations) |
| 4 | Permutation Importance | 30 repeats on stratified test, scoring='f1' |
| 5 | Rule Stability | 10-fold CV on full dataset, extract rules per fold |
| 6 | Summary Dashboard | Consolidated findings + deployment recommendation |

---

## 2. Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | `dataset_clean.csv` (608 rows, 17 cols) |
| Preprocessing | Global median imputation, drop `angkatan` + `program` |
| Features | 14 (6 IPS+SKS + 3 nilai MK + 5 derived) |
| Target | 1=Tepat Waktu (540), 0=Tidak Tepat (68) |
| Best model params | `max_depth=3, min_samples_leaf=10` |
| Scoring focus | `f1_score(pos_label=0)` |
| Random state | `42` (all operations) |
| Temporal train | `angkatan <= 2021` |
| Temporal test | `angkatan > 2021` |

---

## 3. Phase 1: Temporal Validation — THE CRITICAL FINDING

### 3.1 Summary

Model dilatih ulang dengan hyperparameter yang sama pada temporal split. Hasilnya: **model gagal total.**

| Metric | Stratified | Temporal | Delta |
|--------|-----------|----------|-------|
| Accuracy | **0.9754** | 0.7662 | −0.2092 |
| Precision(0) | **0.9231** | 0.0000 | −0.9231 |
| **Recall(0)** | **0.8571** | **0.0000** | −0.8571 |
| **F1(0)** | **0.8889** | **0.0000** | −0.8889 |
| AUC | **0.9550** | 0.7098 | −0.2452 |

### 3.2 Classification Report — Temporal Test

```
              precision    recall  f1-score   support

 Tidak Tepat       0.00      0.00      0.00        54
 Tepat Waktu       0.77      1.00      0.87       177

    accuracy                           0.77       231
```

**Model memprediksi SEMUA 231 mahasiswa sebagai "Tepat Waktu".** Tidak ada satupun mahasiswa berisiko yang terdeteksi. Recall(0) = 0/54 = 0.0000.

### 3.3 Confusion Matrix — Temporal Test

```
            Pred 0  Pred 1
Actual 0       0       54     ← semua false negative — 54/54 terlewat
Actual 1       0      177     ← semua true positive
```

### 3.4 Root Cause Analysis

Model temporal memiliki **struktur tree yang sepenuhnya berbeda** dari model stratified:

**Stratified model (54 train neg):**
```
|--- sks_sem2 <= 18.50 → ... → TEPAT WAKTU
|--- sks_sem2 >  18.50
|   |--- sks_sem3 <= 18.50 → ... → TIDAK TEPAT  ← 2 leaves predict NEGATIVE
|   |--- sks_sem3 >  18.50 → ... → TEPAT WAKTU
```

**Temporal model (14 train neg):**
```
|--- failed_courses <= 1.50 → ... → TEPAT WAKTU    ← ALL leaves predict POSITIVE
|--- failed_courses >  1.50 → TEPAT WAKTU           ← NO negative-predicting leaf
```

Dengan hanya **14 sampel negatif di training** (3.7%) dan `min_samples_leaf=10`, tree temporal **tidak bisa membentuk leaf yang memprediksi kelas minoritas**. Semua 7 leaf default ke kelas mayoritas (Tepat Waktu). Root split berubah dari `sks_sem2` menjadi `failed_courses`.

### 3.5 Stratified Reference (Validation)

Stratified model tetap berfungsi dengan baik (sebagai kontrol):

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Tidak Tepat (0) | 0.92 | 0.86 | 0.89 | 14 |
| Tepat Waktu (1) | 0.98 | 0.99 | 0.99 | 108 |

Confusion matrix:
```
            Pred 0  Pred 1
Actual 0      12       2     ← 2 FN (dari 14)
Actual 1       1     107     ← 1 FP (dari 108)
```

---

## 4. Phase 2: Deep Error Analysis

### 4.1 False Negative Profiling

Karena model temporal memprediksi **semua sebagai Tepat Waktu**, **semua 54 mahasiswa berisiko menjadi false negative** (100% miss rate). Tidak ada True Negative — model tidak mendeteksi siapapun.

**FN Aggregate Profile:**

| Karakteristik | FN (n=54) |
|---------------|-----------|
| Program IH (S1) | 52 (96.3%) |
| Program AP (D3) | 2 (3.7%) |
| Angkatan 2022 | 5 (9.3%) |
| Angkatan 2023 | 49 (90.7%) |
| Mean sks_sem2 | 19.67 |
| Mean sks_sem3 | 18.15 |
| Mean avg_ips | 3.08 |
| Mean failed_courses | 3.09 |
| Mean ips_std | 0.91 |

**Kenapa model melewatkan mereka?**

Semua FN melewati path:
- `failed_courses <= 1.50 → TEPAT WAKTU` (mayoritas dari 2022)  
- `failed_courses > 1.50 → TEPAT WAKTU` (angkatan 2023 dengan banyak MK gagal)

Dengan struktur tree temporal yang tidak memiliki leaf negatif, **tidak ada mekanisme untuk mendeteksi mahasiswa berisiko**. Faktor risiko seperti SKS tinggi, IPS rendah, dan banyak MK gagal tidak tertangkap karena depth=3 terlalu restriktif dengan hanya 14 sampel negatif.

### 4.2 False Positive Profiling

**Temporal test: 0 False Positives.** Karena model memprediksi semua sebagai "Tepat Waktu", tidak ada mahasiswa yang salah di-flag.

### 4.3 Rule Coverage Analysis

Menggunakan struktur rule dari model stratified (sebagai referensi) dan mengaplikasikannya ke temporal test:

| Rule | Coverage | Actual Neg | Predicted Neg (temporal model) |
|------|----------|-----------|-------------------------------|
| Rule 1: sks_sem2 ≤ 18.5 → Tepat | 177 | 0 | 0 |
| Rule 2: sks_sem2 > 18.5, sks_sem3 ≤ 18.5 → **Tidak Tepat** | 52 | **52** | **0** |
| Rule 3: sks_sem2 > 18.5, sks_sem3 > 18.5 → Tepat | 2 | 2 | 0 |

**Temuan kritis Rule 2:** Dari 52 mahasiswa yang masuk rule "overload lalu collapse", **100% (52/52) adalah actual negatif** — precision rule ini sempurna jika model bisa memprediksi negatif. Tapi karena model temporal tidak punya leaf negatif, **Recall Rule 2 = 0.000** — tidak ada yang terdeteksi.

### 4.4 Program-Level Breakdown

| Program | N | True Neg | False Neg | Recall(0) |
|---------|---|----------|-----------|-----------|
| AP (D3) | 2 | 0 | 2 | 0.0 |
| IH (S1) | 229 | 0 | 52 | 0.0 |

Kegagalan terjadi di kedua program. IH mendominasi FN karena memiliki lebih banyak mahasiswa berisiko.

**Stratified reference:**
| Program | N | Recall(0) | Precision(0) | F1(0) |
|---------|---|-----------|--------------|-------|
| AP (D3) | 33 | 0.0 | 0.000 | 0.00 |
| IH (S1) | 89 | 1.0 | 0.923 | 0.96 |

### 4.5 Angkatan-Level Breakdown

| Angkatan | N | Neg | FN | Recall(0) |
|----------|---|-----|-----|-----------|
| 2022 | 181 | 5 | 5 | 0.0 |
| 2023 | 50 | 49 | 49 | 0.0 |

**Angkatan 2023 sangat bermasalah:** 49/50 mahasiswa (98%) adalah tidak tepat waktu. Ini adalah data leakage yang sudah diidentifikasi di fase modeling — hampir seluruh angkatan 2023 belum lulus (target=0). Model temporal memprediksi semuanya sebagai "Tepat Waktu" — complete failure pada angkatan ini.

### 4.6 Distribution Shift Analysis

| Angkatan | sks_sem2 mean | sks_sem3 mean | avg_ips mean | Neg/Total |
|----------|---------------|---------------|-------------|-----------|
| 2020 | 10.47 | 8.25 | 3.18 | 4/40 |
| 2021 | 11.37 | 9.43 | 3.11 | 3/46 |
| **2022** | **17.64** | **9.10** | **2.71** | **5/181** |
| **2023** | **19.20** | **17.78** | **2.93** | **49/50** |

Distribusi fitur berubah signifikan antara angkatan training (2015–2021) dan test (2022–2023):
- `sks_sem2` naik dari ~11 (2021) ke ~19 (2023)
- Pola SKS angkatan baru berbeda dari pola historis
- Model yang dilatih pada data historis tidak bisa menggeneralisasi ke distribusi baru

---

## 5. Phase 3: Repeated Stratified 10×10-Fold CV

100 evaluasi (10 folds × 10 repeats) memberikan confidence intervals yang lebih reliable:

| Metric | Mean | Std | 95% CI Low | 95% CI High |
|--------|------|-----|-----------|------------|
| F1(0) | **0.8327** | 0.0955 | 0.8140 | 0.8514 |
| Recall(0) | 0.7833 | 0.1353 | 0.7568 | 0.8099 |
| Precision(0) | 0.9105 | 0.1028 | 0.8903 | 0.9306 |
| AUC | 0.9700 | 0.0400 | 0.9622 | 0.9779 |
| Accuracy | 0.9660 | 0.0181 | 0.9624 | 0.9695 |

### 5.1 Single Split vs CV

| Metric | Single Split | CV Mean | Diff | Status |
|--------|-------------|---------|------|--------|
| Accuracy | 0.9754 | 0.9660 | +0.0095 | Aligned |
| Precision(0) | 0.9231 | 0.9105 | +0.0126 | Aligned |
| Recall(0) | 0.8571 | 0.7833 | **+0.0738** | **Optimistic** |
| F1(0) | 0.8889 | 0.8327 | **+0.0562** | **Optimistic** |
| AUC | 0.9550 | 0.9700 | −0.0150 | Aligned |

**Single split recall(0) = 0.857 overestimates sebesar +7.4%** dibandingkan CV mean (0.783). Single split F1(0) juga overestimates +5.6%. Ini mengkonfirmasi bahwa test set dengan hanya 14 sampel negatif menghasilkan metrik high-variance yang unreliable.

CV mean F1(0) = 0.83 memberikan estimasi yang lebih realistis untuk performa model pada stratified scenario — tapi ini **tidak berlaku untuk temporal scenario** (di mana CV tidak dilakukan karena ketidakseimbangan temporal).

---

## 6. Phase 4: Permutation Importance vs MDI

### 6.1 Comparison

| Rank | Feature | MDI | Perm Mean | Perm Std | dRank |
|------|---------|-----|-----------|----------|-------|
| 1 | `sks_sem2` | 0.558 | 0.128 | 0.019 | +0 |
| 2 | `sks_sem3` | 0.361 | 0.066 | 0.012 | +0 |
| 3 | `ips_std` | 0.047 | 0.000 | 0.000 | −5 |
| 4 | `avg_ips` | 0.017 | 0.000 | 0.000 | −4 |
| 5 | `failed_courses` | 0.012 | 0.000 | 0.000 | −3 |
| 6 | `ips_sem1` | 0.004 | 0.000 | 0.000 | −2 |
| 7–14 | 8 features | 0.000 | 0.000 | 0.000 | — |

**Spearman rank correlation: 0.6749 (p=0.0081)** — moderate agreement.

### 6.2 Key Findings

1. **`sks_sem2` dan `sks_sem3` tetap dominan di kedua metode** — mengkonfirmasi bahwa SKS adalah prediktor genuine, bukan artifact MDI.
2. **Permutation importance absolute values jauh lebih kecil** — 0.128 vs 0.558 untuk `sks_sem2`. Permutation mengukur actual impact pada F1 score, bukan impurity reduction.
3. **10 features zero importance di permutation** — depth=3 terlalu dangkal; hanya 2 fitur yang memberikan kontribusi terukur.
4. **Semua fitur zero-MDI juga zero-permutation** — tidak ada hidden signal yang terlewat MDI.

---

## 7. Phase 5: Decision Rule Stability

### 7.1 Stability Across 10 Stratified CV Folds

| Metrik | Hasil |
|--------|-------|
| Root split | **`sks_sem2` in 10/10 folds (100%)** |
| Mean leaves | 7.0 (std=0.0, range 7–7) |
| Mean neg leaves | 1.5 (range 1–2) |
| Mean features used | 5.2 |
| Always used | `ips_sem1`, `sks_sem2`, `sks_sem3`, `failed_courses`, `ips_std` |
| Sometimes used | `sks_completion_ratio`, `ips_min` |
| Never used | `ips_sem2`, `ips_sem3`, `sks_sem1`, `failed_in_sem1`, `repeated_courses`, `ips_trend`, `avg_ips` |

### 7.2 Verdict

- **Root split STABLE** — `sks_sem2` selalu menjadi split pertama di semua 10 folds. Ini adalah sinyal kuat bahwa `sks_sem2` adalah prediktor paling robust.
- **Leaf count STABLE** — tree selalu 7 leaves (terkendala `max_depth=3` + `min_samples_leaf=10`).
- **Feature set mostly stable** — 5 fitur selalu digunakan, 2 kadang-kadang, 7 tidak pernah.
- **TAPI: rules ini hanya berlaku untuk stratified split.** Temporal model memiliki struktur tree yang sepenuhnya berbeda (root: `failed_courses`).

---

## 8. Summary Dashboard

### 8.1 Performance Summary

| Scenario | F1(0) | Recall(0) | Precision(0) | AUC |
|----------|-------|-----------|--------------|-----|
| **Stratified (modeling)** | **0.8889** | 0.8571 | 0.9231 | 0.9550 |
| Repeated CV (100 evals) | 0.8327 | 0.7833 | 0.9105 | 0.9700 |
| **Temporal (deployment)** | **0.0000** | **0.0000** | **0.0000** | **0.7098** |

### 8.2 Error Profile (Temporal Test)

| | Count |
|---|---|
| Total test | 231 |
| True Neg (detected) | 0 |
| False Neg (missed) | 54 |
| False Pos (flagged) | 0 |

**100% miss rate** — semua mahasiswa berisiko tidak terdeteksi.

### 8.3 Feature Importance Summary

| Method | Top Feature | Second | Third |
|--------|------------|--------|-------|
| MDI (Gini) | sks_sem2 (0.558) | sks_sem3 (0.361) | ips_std (0.047) |
| Permutation | sks_sem2 (0.128) | sks_sem3 (0.066) | ips_sem2 (0.000) |

### 8.4 Rule Coverage

| Rule | Test Coverage | Actual Neg | Impact |
|------|-------------|------------|--------|
| `sks_sem2 ≤ 18.5 → Tepat` | 177 (76.6%) | 0 | Correctly predicts safe students |
| `sks_sem2 > 18.5, sks_sem3 ≤ 18.5 → Tidak Tepat` | 52 (22.5%) | **52** | Rule covers 96.3% of at-risk but model doesn't fire it |
| `sks_sem2 > 18.5, sks_sem3 > 18.5 → Tepat` | 2 (0.9%) | 2 | Misses 2 at-risk |

---

## 9. Discussion

### 9.1 The Stratified Split Was Misleading

Stratified split (80/20 random, stratify=target) memberikan 54 sampel negatif di training — cukup untuk tree depth=3 membentuk leaf negatif dan mencapai F1(0)=0.889. Tapi ini adalah **skenario artificial**: dalam deployment nyata, model dilatih pada data historis di mana proporsi mahasiswa berisiko jauh lebih rendah (3.7% di train temporal vs 11.1% di train stratified).

### 9.2 Why Temporal Fails

Tiga faktor berkontribusi pada kegagalan temporal:

1. **Training negatif terlalu sedikit (14).** Dengan `min_samples_leaf=10`, tree tidak bisa membentuk leaf minoritas yang valid. Tree default ke kelas mayoritas untuk semua leaf.
2. **Distribution shift.** Pola SKS angkatan 2022–2023 berbeda signifikan dari 2015–2021. `sks_sem2` mean naik dari ~13 ke ~18, `sks_sem3` dari ~11 ke ~14.
3. **max_depth=3 terlalu restriktif** untuk dataset dengan 14 sampel negatif. Tree tidak cukup dalam untuk menemukan pola minoritas yang kompleks.

### 9.3 What the Stratified Model Actually Learned

Stratified model mengandalkan rule sederhana:
- `sks_sem2 > 18.5 AND sks_sem3 <= 18.5 → TIDAK TEPAT`

Rule ini bekerja pada stratified test karena distribusi SKS di test stratified mirip dengan training stratified. Tapi pada temporal data (>2021), pola "overload lalu collapse" tidak terdeteksi karena:
- Rule 2 mencakup 52 mahasiswa dengan 52 actual negatif di temporal test — **precision rule sempurna**
- Tapi model temporal **tidak pernah memprediksi negatif** karena tree-nya tidak punya leaf negatif
- Ini adalah **paradoks**: rule yang benar ada di data, tapi model tidak bisa mengaksesnya karena keterbatasan training

### 9.4 Limitations

1. **Model tidak generalizable ke temporal split** — stratified performance tidak mencerminkan performa deployment.
2. **Data angkatan 2023 contaminated** — 49/50 mahasiswa adalah target=0 karena belum lulus, menciptakan spike negatif artifisial.
3. **Permutation importance terbatas** — depth=3 hanya menggunakan 2 fitur secara efektif; 10 fitur zero importance.
4. **Rule extraction misleading** — rules dari stratified model tidak berlaku di temporal model.
5. **Temporal split juga bermasalah** — train hanya 14 negatif (3.7%), test 54 negatif (23.4%). Distribusi yang sangat berbeda membuat generalisasi hampir mustahil untuk single Decision Tree.

---

## 10. Recommendations

### 10.1 Deployment Verdict

**[WARN] MODEL BELUM SIAP UNTUK DEPLOYMENT.**

Model stratified mencapai target performa (F1(0)=0.889, Recall(0)=0.857) tapi **gagal total pada temporal validation** — skenario yang mensimulasikan deployment nyata. Model tidak bisa mendeteksi mahasiswa berisiko pada angkatan baru (2022–2023).

### 10.2 Short-Term Improvements

| # | Improvement | Expected Impact |
|---|-------------|-----------------|
| 1 | **Tambah data training** — kumpulkan lebih banyak sampel mahasiswa tidak tepat waktu dari angkatan 2015–2021 | Training neg dari 14 → 30+ |
| 2 | **Handle class imbalance** — SMOTE atau class_weight pada temporal split | Memberi bobot lebih pada kelas minoritas |
| 3 | **Relax constraint** — coba `max_depth=4-5` untuk temporal split | Tree lebih dalam bisa menemukan pola minoritas |
| 4 | **Feature engineering** — `sks_ratio` = sks_sem3/sks_sem2, flag `sks_sem2_high` | Fitur yang lebih robust terhadap distribution shift |
| 5 | **Separate models per program** — AP (D3) vs IH (S1) memiliki karakteristik berbeda | Recall lebih baik untuk masing-masing program |

### 10.3 Medium-Term Improvements

| # | Improvement | Rationale |
|---|-------------|-----------|
| 1 | **Random Forest** — ensemble untuk feature importance lebih balance dan stabilitas lebih baik | Single DT oversimplified + overfit ke top-2 features |
| 2 | **Retrain berkala** — update model setiap tahun dengan data angkatan baru | Mengatasi distribution shift |
| 3 | **Rule-based fallback** — gunakan rule `sks_sem2 > 18.5 AND sks_sem3 ≤ 18.5` sebagai alert sistem, BUKAN sebagai model prediksi tunggal | Rule memiliki precision sempurna di temporal test |
| 4 | **Add more data sources** — data demografi, jalur masuk, beasiswa | Fitur tambahan bisa membantu diskriminasi |

### 10.4 Long-Term Strategy

1. **Kumpulkan data longitudinal** — track mahasiswa dari semester 1 sampai lulus/DO untuk meningkatkan sampel negatif.
2. **Implementasi dual-model:** model stratified untuk eksplorasi + model temporal untuk deployment.
3. **Monitor distribution shift** — alert jika distribusi fitur angkatan baru berbeda signifikan dari training.

---

## 11. Comparison to Modeling Phase

| Aspect | Modeling Phase | Evaluation Phase |
|--------|---------------|------------------|
| Split | Stratified 80/20 | Temporal (≤2021/>2021) + Stratified |
| Train neg | 54 (11.1%) | 14 (3.7%) temporal, 54 stratified |
| Test neg | 14 (11.5%) | 54 (23.4%) temporal, 14 stratified |
| F1(0) | 0.889 | 0.000 temporal, 0.889 stratified |
| Overfit check | Train-CV gap ≈ 0 | NOT applicable (temporal fails) |
| Verdict | **Model excellent** | **Model not generalizable** |

**Kontradiksi utama:** Model yang "sempurna" di stratified (F1=0.889, no overfit, depth=3) ternyata **completely useless** di temporal (F1=0.000). Ini membuktikan bahwa stratified split memberikan ilusi performa yang baik — model belajar dari distribusi kelas yang artificial (54 neg di train), bukan dari skenario deployment sesungguhnya (14 neg di train).

---

## 12. Deliverables

| File | Description |
|------|-------------|
| `5-evaluation/01-final-evaluation.ipynb` | Notebook evaluasi (25 cells) |
| `5-evaluation/01-evaluation-results.md` | Output nbconvert (2,289 lines, 10 charts) |
| `5-evaluation/01-evaluation-results_files/` | 10 chart PNGs |
| `5-evaluation/evaluation_metrics.json` | Structured metrics export |
| `5-evaluation/rules_temporal.txt` | Decision rules from temporal model |
| `5-evaluation/rules_stratified.txt` | Decision rules from stratified model |
| `5-evaluation/evaluation-report.md` | **This document** |

---

## 13. References

- Phase 4 Modeling: `4-modeling/README.md`
- Tuning Report: `4-modeling/3-hyperparameter-tuning/tuning-report.md`
- Baseline Report: `4-modeling/2-global-median/stratified-report.md`
- EDA Findings: `3-data-preparation/02-eda-findings.md`
- Evaluation Plan: `5-evaluation/README.md`
