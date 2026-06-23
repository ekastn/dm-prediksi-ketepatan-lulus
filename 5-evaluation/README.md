# Fase 5 — Evaluation

**CRISP-DM Phase 5: Validasi model final dengan temporal split, repeated CV, error analysis, dan permutation importance.**

---

## Objective

Model 01-Tuned (`max_depth=3, min_samples_leaf=10`) telah mencapai F1(0)=0.889 pada stratified split, tapi **belum diuji pada temporal split** — skenario yang mensimulasikan prediksi nyata (train pada data historis, prediksi pada angkatan baru). Fase ini menjawab pertanyaan: **apakah model generalizable ke skenario deployment?**

---

## Dataset

| File | Source | Description |
|------|--------|-------------|
| `dataset_clean.csv` | Copy dari `4-modeling/2-global-median/` | Global median imputation, 608 rows × 17 cols |

Akan di-split menjadi:
- **Temporal train:** `angkatan ≤ 2021` (~377 rows, ~14 negatif)
- **Temporal test:** `angkatan > 2021` (~231 rows, ~54 negatif)
- **Stratified split:** 80/20 random, stratify=target (sama seperti di modeling — untuk perbandingan)

---

## Experiment Plan

### Phase 1: Temporal Validation — The True Test

```python
# Split berdasarkan angkatan
train_mask = df['angkatan'] <= 2021
X_train_t = df[train_mask].drop(columns=['target', 'angkatan', 'program'])
X_test_t  = df[~train_mask].drop(columns=['target', 'angkatan', 'program'])
y_train_t = df[train_mask]['target']
y_test_t  = df[~train_mask]['target']
```

Latih **exact same best model** (`max_depth=3, min_samples_leaf=10, random_state=42`) pada temporal split, evaluasi.

**Expected:**
- Recall(0) akan turun karena train hanya ~14 sampel negatif (vs 54 di stratified)
- Precision(0) mungkin naik (model lebih konservatif)
- F1(0) likely lebih rendah dari 0.889 — pertanyaannya: seberapa rendah?
- Confusion matrix akan punya lebih banyak false negatives (model "terlalu optimis")

**Deliverables:**
- Classification report untuk temporal train + test
- Confusion matrix side-by-side: stratified vs temporal test
- Tabel perbandingan: stratified performance vs temporal performance

### Phase 2: Deep Error Analysis (Temporal Test)

Temporal test set punya ~54 sampel negatif — 4x lebih banyak dari stratified test (14). Ini memberi kesempatan untuk **error profiling** yang lebih bermakna.

**2a. False Negative Profiling (at-risk students missed)**

Untuk setiap mahasiswa target=0 yang diprediksi target=1 (false negative):
- Berapa `sks_sem2`, `sks_sem3`, `ips_sem1`, `failed_courses`?
- Apakah mereka memenuhi rule "overload lalu collapse" (`sks_sem2 > 18.5 AND sks_sem3 ≤ 18.5`)?
- Apakah ada pola: angkatan tertentu, program tertentu?
- Kenapa model "melewatkan" mereka — fitur apa yang membuat mereka terlihat seperti "Tepat Waktu"?

**2b. False Positive Profiling (safe students falsely flagged)**

Untuk setiap mahasiswa target=1 yang diprediksi target=0 (false positive):
- Profil akademik mereka dibandingkan dengan true positives
- Apakah mereka "borderline" — dekat dengan decision boundary?
- Apakah ada fitur yang tidak proporsional (misal: ips_sem1 rendah karena system zero yang tidak sempurna diimputasi)?

**2c. Rule Coverage Analysis**

Ambil decision rules dari model terbaik:

```
|--- sks_sem2 <= 18.50 → ... → TEPAT WAKTU (rule 1)
|--- sks_sem2 > 18.50
|   |--- sks_sem3 <= 18.50 → ... → TIDAK TEPAT (rule 2 — satu-satunya rule "berisiko")
|   |--- sks_sem3 > 18.50 → ... → TEPAT WAKTU (rule 3)
```

Untuk temporal test set:
- Berapa banyak mahasiswa yang masuk ke rule 2 (diprediksi "Tidak Tepat")?
- Dari yang masuk rule 2, berapa yang benar-benar "Tidak Tepat" (precision rule 2)?
- Dari total "Tidak Tepat" di test, berapa yang terdeteksi rule 2 (recall rule 2)?
- Apakah rule 3 (`sks_sem2 > 18.5 AND sks_sem3 > 18.5 → TEPAT WAKTU`) menghasilkan false negatives?

**2d. Program-Level Breakdown**

| Program | N | True Neg | False Neg | True Pos | False Pos | Recall(0) | Precision(0) |
|---------|---|----------|-----------|----------|-----------|-----------|--------------|
| AP (D3) | ? | ? | ? | ? | ? | ? | ? |
| IH (S1) | ? | ? | ? | ? | ? | ? | ? |

Apakah model lebih akurat untuk salah satu program?

**2e. Angkatan-Level Breakdown**

Apakah error terkonsentrasi di angkatan tertentu? Misal: angkatan 2022-2023 (paling baru) mungkin punya karakteristik berbeda dari 2016-2019.

### Phase 3: Repeated Stratified K-Fold

Single stratified split menghasilkan metrik dengan varians tinggi (test hanya 14 negatif). Repeated CV memberikan confidence intervals.

```python
from sklearn.model_selection import RepeatedStratifiedKFold

rcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
# 10 folds × 10 repeats = 100 evaluations
```

**Deliverables:**
- Mean ± 95% CI untuk accuracy, recall(0), precision(0), F1(0), AUC
- Perbandingan: single-split metrics vs CV metrics (apakah single split optimis/pesimis?)
- Boxplot distribusi metrik across 100 evaluations

### Phase 4: Permutation Importance

MDI (Mean Decrease in Impurity) yang digunakan sejauh ini **biased** terhadap continuous features dengan banyak unique values — menjelaskan kenapa `sks_sem2` dan `sks_sem3` selalu mendominasi. Permutation importance lebih robust.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    final_model, X_test, y_test,
    n_repeats=30, random_state=42, scoring='f1'
)
```

**Deliverables:**
- Tabel perbandingan: MDI ranking vs Permutation ranking
- Apakah top features berubah?
- Apakah fitur yang sebelumnya zero importance (failed_in_sem1, sks_completion_ratio) muncul di permutation?
- Bar chart side-by-side: MDI vs Permutation

### Phase 5: Decision Rule Stability

Apakah rules dari model konsisten across CV folds? Atau rules berubah tergantung subset training?

```python
# Untuk setiap fold di 10-fold CV:
# - Train model pada fold
# - Extract decision rules (export_text)
# - Hitung rule similarity (Jaccard pada fitur yang digunakan di root split)
```

**Deliverables:**
- Root split stability: fitur apa yang muncul di root paling sering?
- Rule count distribution: berapa leaves per fold?
- Fitur yang selalu muncul vs yang kadang muncul
- Jika rules tidak stabil → model tidak reliable untuk deployment

### Phase 6: Summary Dashboard

Satu sel final yang merangkum semua temuan:

| Section | Content |
|---------|---------|
| Model Configuration | Hyperparameters, training details |
| Stratified Performance | Single-split metrics (known from Phase 4) |
| Temporal Performance | **Key result** — apakah model generalize? |
| Repeated CV | Metrics with 95% CI |
| Error Profile | Karakteristik FN dan FP |
| Feature Ranking | MDI vs Permutation comparison |
| Rule Coverage | Coverage + precision per rule |
| Limitations | Apa yang model tidak bisa lakukan |
| Recommendation | Model ready for deployment? Atau butuh improvement? |

---

## Notebook Structure

`5-evaluation/01-final-evaluation.ipynb`

| Cell | Phase | Content |
|------|-------|---------|
| 1 | Setup | Imports, load dataset, define splits (temporal + stratified) |
| 2 | Phase 1 | Train best model on temporal split, evaluate, compare vs stratified |
| 3 | Phase 2a | FN profiling — karakteristik mahasiswa berisiko yang terlewat |
| 4 | Phase 2b | FP profiling — karakteristik mahasiswa aman yang salah flag |
| 5 | Phase 2c | Rule coverage analysis pada temporal test set |
| 6 | Phase 2d | Program-level breakdown |
| 7 | Phase 2e | Angkatan-level breakdown |
| 8 | Phase 3 | Repeated stratified 10×10-fold CV dengan confidence intervals |
| 9 | Phase 4 | Permutation importance vs MDI comparison |
| 10 | Phase 5 | Decision rule stability across CV folds |
| 11 | Phase 6 | Summary dashboard + final recommendation |
| 12 | Export | Save charts, metrics, and rules |

---

## Expected Findings

| Scenario | Likely Outcome | Implication |
|----------|---------------|-------------|
| Temporal recall(0) ≈ 0.3–0.5 | Model struggles with only 14 train negatives | Need SMOTE or different class balancing |
| Temporal recall(0) ≈ 0.6–0.8 | Model generalizes reasonably | Acceptable for deployment with caveats |
| Temporal recall(0) ≈ 0.8+ | Model generalizes well | Ready for deployment |
| Permutation importance ≠ MDI | sks_sem2 dominance is partly MDI artifact | Revise feature importance narrative |
| Rules stable across CV folds | Model structure is reliable | Rules can be used for policy |
| Rules unstable across CV folds | Model structure is data-dependent | Need ensemble or more data |
| FN concentrated in IH (S1) | Program-specific risk factors | Separate models per program? |
| FN concentrated in angkatan baru | Distribution shift over time | Model needs retraining with new data |

---

## Deliverables

```
5-evaluation/
├── README.md                        ← Dokumen ini
├── evaluation-plan.md               ← Rencana evaluasi detail
├── dataset_clean.csv                ← Global median dataset (copy)
├── 01-final-evaluation.ipynb        ← Notebook evaluasi
├── 01-evaluation-results.md         ← Output nbconvert
└── 01-evaluation-results_files/     ← Charts
```
