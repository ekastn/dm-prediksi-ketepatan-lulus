# Combined Tuning Report — `max_features` + `min_impurity_decrease`

**Experiment:** `4-modeling/3-hyperparameter-tuning/03-combined-tuning.ipynb`  
**Tanggal:** 22 Juni 2026  
**Model:** `DecisionTreeClassifier(max_features=5, min_impurity_decrease=0.001, max_depth=7, min_samples_leaf=10)`  
**Predecessors:** Baseline → 01-Tuned (depth=3) → ~~02-Engineered~~ (failed, deleted)

---

## 1. Rationale

01-Tuned dan 02-Engineered menghasilkan model identik (F1=0.889, top-2=90-92%, 2 leaves "Tidak Tepat").  
Tiga masalah struktural persisten:
1. `max_depth` adalah hard cap — menciptakan cosmetic splits di boundary
2. Fitur terkuat (`sks_sem2`) memenangkan setiap split — konsentrasi 90%
3. Tree depth 3-4 tidak cukup dalam untuk menggunakan >6 fitur

Dua parameter baru untuk mengatasi:
- `max_features` — paksa tree memilih dari subset random per split → distribusikan importance
- `min_impurity_decrease` — block split yang tidak meaningful → hilangkan cosmetics

---

## 2. Phase 0: Impurity Landscape Analysis

Sebelum tuning, kita latih tree unconstrained untuk memahami distribusi impurity decrease.

### 2.1 Hasil

| Statistik | Nilai |
|-----------|-------|
| Tree depth | 7 |
| Internal nodes | 12 |
| Leaves | 13 |
| **Cosmetic splits** (both children same class) | **8/12 (67%)** |

| Impurity Decrease | Value |
|-------------------|-------|
| Min | 0.000000 |
| Q25 | 0.000663 |
| Median | 0.001944 |
| Q75 | 0.004853 |
| Max | 0.085354 |

### 2.2 Survival by Threshold

| Threshold | Splits Surviving | Cosmetic Killed |
|-----------|-----------------|-----------------|
| 0.000 | 12 | 0 |
| **0.001** | **7** | **5** |
| 0.002 | 6 | 6 |
| 0.005 | 3 | 8 |
| 0.010 | 2 | 8 |
| 0.020 | 2 | 8 |

### 2.3 Insight

- **67% splits are cosmetic** — tree mostly makes garbage splits at deeper levels. Ini menjelaskan kenapa depth-3/4 sudah optimal: split lebih dalam tidak menghasilkan pemisahan kelas yang berbeda.
- **Threshold 0.001 kills 5 cosmetic splits** — hanya 7 splits bertahan. Threshold 0.005 terlalu agresif (3 splits only).
- **Ada split dengan impurity decrease 0.000000** — literally zero value. Tree membuat split hanya untuk memenuhi `max_depth`.

---

## 3. Phase 1: GridSearchCV

### 3.1 Configuration

| Parameter | Values |
|-----------|--------|
| `max_features` | [None, 'sqrt', 4, 5] |
| `min_impurity_decrease` | [0.0, 0.001, 0.002, 0.005, 0.008, 0.01, 0.02] |
| Fixed | max_depth=7, min_samples_leaf=10, criterion='gini' |
| Total | 4 × 7 = 28 combinations × 5 folds = 140 fits |

### 3.2 Best Parameters

| Parameter | Best Value |
|-----------|-----------|
| `max_features` | **5** |
| `min_impurity_decrease` | **0.001** |
| **Best CV F1(0)** | **0.8371** |

### 3.3 Heatmap: CV F1(0)

| min_impurity_decrease | max_features=None | max_features=4 | **max_features=5** | max_features=sqrt |
|-----------------------|-------------------|----------------|---------------------|-------------------|
| 0.000 | 0.8165 | 0.7022 | 0.8347 | 0.7459 |
| **0.001** | 0.8165 | 0.7234 | **0.8371** | 0.7724 |
| 0.002 | 0.8165 | 0.7234 | **0.8371** | 0.7864 |
| 0.005 | 0.8165 | 0.7234 | **0.8371** | 0.7864 |
| 0.008 | 0.8165 | 0.7234 | **0.8371** | 0.7864 |
| 0.010 | 0.8165 | 0.7234 | **0.8371** | 0.7864 |
| 0.020 | 0.8165 | 0.7958 | **0.8371** | 0.7864 |

### 3.4 Key Findings

1. **`max_features=5` consistently gives F1=0.8371**, vs 0.8165 for `max_features=None`. Ini adalah peningkatan CV F1(0) sebesar **+0.0206** — pertama kalinya tuning menghasilkan perbaikan CV.

2. **`max_features=5` robust across all thresholds** — F1(0) stabil di 0.8371 dari 0.001 sampai 0.02. Tidak ada trade-off antara impurity threshold dan performa.

3. **`max_features=None` (all features) stuck at 0.8165** — ini adalah ceiling untuk tree yang melihat semua fitur di setiap split. 01-Tuned dan 02-Engineered menggunakan ini.

4. **`max_features=sqrt` (~4) memberi performa terburuk** — 0.7459 pada threshold 0.0, naik ke 0.7864 pada threshold tinggi. Terlalu sedikit fitur per split untuk dataset kecil.

5. **`max_features=4` intermediate** — 0.7234 pada threshold optimal. Kurang stabil dari 5.

6. **`min_impurity_decrease` hanya berdampak pada edges** — dari 0.0 ke 0.001 ada lompatan kecil, lalu flat. Threshold > 0.001 tidak mengubah CV F1(0) karena split yang dieliminasi sudah cosmetic.

---

## 4. Phase 2: `min_impurity_decrease` Sweep

Dengan fixed `max_features=5`, sweep 31 thresholds dari 0 ke 0.03.

### Temuan

- **CV F1(0) flat** dari 0.001 ke ~0.02, lalu turun drastis (underfit).
- **Cosmetic splits turun** dari 4 (threshold=0) ke 2 (threshold=0.02).
- **Tidak_leaves tetap 2** di semua thresholds — tree tidak bisa menghasilkan lebih dari 2 aturan untuk kelas minoritas.
- **Leaves turun** dari 6 (threshold=0) ke 2 (threshold=0.02) — tree menjadi stump.

---

## 5. Phase 3: `max_features` Comparison

Fixed `min_impurity_decrease=0.001`, bandingkan 4 nilai `max_features`.

| max_features | Depth | Leaves | Feats Used | Top-2% |
|-------------|-------|--------|-----------|--------|
| None (14) | 4 | 7 | 6/14 | 91.2% |
| sqrt (~4) | 4 | 6 | 3/14 | 97.6% |
| 4 | 4 | 6 | 5/14 | 87.0% |
| **5** | **4** | **6** | **5/14** | **93.6%** |

**Analisis:**
- `max_features` tidak mengurangi konsentrasi — top-2 tetap 87-98%.
- `max_features=sqrt` justru memperparah (97.6%) — dengan hanya 4 fitur per split, `sks_sem2` hampir selalu tersedia.
- `max_features=4` sedikit menurunkan konsentrasi (87%) tapi dengan CV F1 yang jauh lebih rendah (0.7234 vs 0.8371).
- **Trade-off:** CV F1 yang lebih tinggi (max_features=5) = konsentrasi tetap tinggi (93.6%). Tidak ada cara untuk mendapatkan keduanya dengan single tree.

---

## 6. Final Model

| Properti | 01-Tuned | 03-Combined | Change |
|----------|----------|-------------|--------|
| Depth | 3 | **4** | +1 |
| Leaves | 7 | **6** | −1 |
| Nodes | 13 | **11** | −2 |
| Cosmetic splits | 2 | **4** | +2 (worse!) |
| Feats used | 6/14 | **5/14** | −1 |
| Top-2% | 92.0% | **93.6%** | +1.6% (worse!) |
| Tidak leaves | 2 | **2** | = |
| Train Acc | 0.9691 | **0.9712** | +0.0021 |

### 6.1 Test Performance — Identical (Again)

| Metrik | 01-Tuned | 03-Combined | Change |
|--------|----------|-------------|--------|
| Recall(0) | 0.8571 | **0.8571** | = |
| Precision(0) | 0.9231 | **0.9231** | = |
| **F1(0)** | 0.8889 | **0.8889** | = |
| AUC | 0.9239 | **0.9239** | = |
| Train F1(0) | 0.8544 | **0.8627** | +0.0083 |

**Test F1(0) tetap 0.8889 dalam 3 eksperimen berturut-turut.** CV F1(0) naik (0.8165 → 0.8371), tapi test set terlalu kecil (14 negatif) untuk merefleksikan perbaikan ini.

---

## 7. Decision Rules — Lebih Buruk

```
|--- sks_sem2 <= 18.50
|   |--- failed_courses <= 0.50
|   |   |--- ips_sem1 <= 3.01  → TEPAT WAKTU
|   |   |--- ips_sem1 > 3.01   → TEPAT WAKTU  ← COSMETIC
|   |--- failed_courses > 0.50 → TEPAT WAKTU
|--- sks_sem2 > 18.50
|   |--- sks_sem3 <= 18.50
|   |   |--- ips_std <= 0.29   → TIDAK TEPAT
|   |   |--- ips_std > 0.29    → TIDAK TEPAT  ← COSMETIC
|   |--- sks_sem3 > 18.50
|   |   |--- ips_sem1 <= 3.26  → TEPAT WAKTU
|   |   |--- ips_sem1 > 3.26   → TEPAT WAKTU  ← COSMETIC
```

**4/8 internal nodes are cosmetic** — 50% splits tidak mengubah prediksi. Bandingkan dengan 2/13 (15%) di 01-Tuned. Tree 03 lebih "wasteful" meskipun lebih kecil.

---

## 8. 4-Way Comparison

| Metrik | Baseline | 01-Tuned | 02-Eng | **03-Combined** | Best |
|--------|----------|----------|--------|-----------------|------|
| **Performance** |
| Recall(0) | **0.9286** | 0.8571 | 0.8571 | 0.8571 | Baseline |
| Precision(0) | 0.8125 | **0.9231** | **0.9231** | **0.9231** | 01/02/03 |
| **F1(0)** | 0.8667 | **0.8889** | **0.8889** | **0.8889** | 01/02/03 |
| AUC | **0.9504** | 0.9239 | 0.9239 | 0.9239 | Baseline |
| **Complexity** |
| Depth | 8 | **3** | 4 | 4 | 01 |
| Leaves | 24 | 7 | 9 | **6** | 03 |
| Nodes | 47 | 13 | 17 | **11** | 03 |
| **Feature Balance** |
| Feats Used | **12** | 6 | 6 | 5 | Baseline |
| Top-2% | **71.2%** | 92.0% | 90.3% | 93.6% | Baseline |
| **Overfitting** |
| Train Acc | 1.0000 | **0.9691** | **0.9691** | 0.9712 | 01/02 |
| **Win Count** | 3 | 4 | 0 | 3 | — |

**03-Combined tidak mengungguli 01-Tuned dalam metrik apa pun yang penting.** CV F1(0) memang lebih tinggi, tapi ini tidak terlihat di test set. 01-Tuned tetap menjadi model terbaik karena memiliki complexity terendah (depth=3) dengan performa terbaik (F1=0.889).

---

## 9. Kesimpulan

### 9.1 Apakah `max_features` + `min_impurity_decrease` Berhasil?

**Partial success:**
- CV F1(0) meningkat dari 0.8165 → 0.8371 — perbaikan genuine di generalisasi
- `min_impurity_decrease=0.001` mengeliminasi 5/12 cosmetic splits di landscape

**Tapi gagal di objective utama:**
- Test F1(0) tetap 0.8889 — tidak ada perbaikan di holdout
- Top-2 still 93.6% — konsentrasi fitur tidak teratasi
- Hanya 2 leaves untuk "Tidak Tepat" — tidak bisa membedakan subtipe mahasiswa berisiko
- Cosmetic splits justru meningkat (2 → 4)

### 9.2 Mengapa `max_features` Tidak Memperbaiki Feature Concentration?

**Jawaban matematis:** Probabilitas `sks_sem2` terpilih dalam random subset:
- `max_features=5` dari 14 fitur: P(sks_sem2 terpilih) = 5/14 ≈ 36%
- Ketika terpilih: tree memilih `sks_sem2` (impurity decrease 0.085)
- Ketika tidak terpilih: tree memilih `sks_sem3` (impurity decrease ~0.02)
- Fitur lain hanya punya chance ketika KEDUA fitur SKS tidak terpilih: P = (9/14) × (8/13) ≈ 40%

Jadi di 60% split, tree masih menggunakan `sks_sem2` atau `sks_sem3`. Feature importance tetap terkonsentrasi.

**Untuk benar-benar mendistribusikan importance**, diperlukan probabilitas yang jauh lebih rendah (max_features=2) atau mekanisme yang berbeda (ensemble, yaitu Random Forest).

### 9.3 Ceiling Single Decision Tree Tercapai

Tiga eksperimen terpisah (01, 02, 03) semuanya konvergen ke **F1(0) = 0.8889**. Ini adalah ceiling single Decision Tree untuk dataset ini dengan constraint yang reasonable. Tidak ada parameter yang bisa menembus batas ini.

---

## 10. Rekomendasi

### 10.1 Stop Single Decision Tree Tuning

3 eksperimen = 1 hasil. Tidak ada parameter yang mengubah test F1(0). Single DT sudah mencapai batasnya.

### 10.2 01-Tuned Adalah Model Single-Tree Terbaik

| Alasan | Detail |
|--------|--------|
| Complexity terendah | Depth=3, 7 leaves, 13 nodes |
| Performa terbaik | F1(0)=0.889, precision=0.923 |
| Interpretable | Rules pendek, mudah dibaca |
| CV gap minimal | Near-zero overfitting |
| Tidak ada parameter "tambahan" | Tanpa max_features atau min_impurity_decrease — lebih reproducible |

### 10.3 Next: Ensemble Methods (Iterasi 4)

Random Forest secara inheren menyelesaikan masalah yang tidak bisa diselesaikan single DT:
- **Feature concentration**: setiap tree dibangun dari bootstrap sample + random feature subset yang berbeda → importance otomatis terdistribusi
- **Subtype detection**: multiple trees membangun aturan berbeda → lebih banyak leaves untuk "Tidak Tepat"
- **Stability**: voting/averaging → tidak bergantung pada satu split pertama

Target: F1(0) ≥ 0.90, top-2 < 60%, > 3 leaves "Tidak Tepat".

---

## 11. Deliverables

| File | Deskripsi |
|------|-----------|
| `4-modeling/3-hyperparameter-tuning/03-combined-tuning.ipynb` | Notebook eksekusi |
| `4-modeling/3-hyperparameter-tuning/03-combined-results.md` | Output nbconvert + charts |
| `4-modeling/3-hyperparameter-tuning/03-combined-results_files/` | PNG charts |
| `4-modeling/3-hyperparameter-tuning/rules_combined.txt` | Decision rules |
| `4-modeling/3-hyperparameter-tuning/combined-tuning-report.md` | **Dokumen ini** |
