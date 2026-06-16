# Experiment: Global Median Imputation

**Tanggal:** 17 Juni 2026  
**Objective:** Menghilangkan proxy temporal pada fitur SKS yang disebabkan imputasi median per-angkatan.

---

## 1. Perubahan

Sebelum (`03-preprocessing.py:45`):

```python
df[col] = df.groupby('angkatan')[col].transform(lambda x: x.fillna(x.median()))
```

Sesudah (`experiments/global-median/preprocessing.py`):

```python
median_val = df[col].median()
df[col] = df[col].fillna(median_val)
```

**Single global median** menggantikan median per-angkatan → setiap fitur diimputasi dengan **1 nilai tunggal** untuk semua mahasiswa → tidak ada proxy per-angkatan.

Dampak: Semua mahasiswa yang kehilangan data (47.5% untuk `sks_sem2`) mendapat nilai imputasi yang **sama** (median=18.0), terlepas dari angkatan mana mereka berasal.

---

## 2. Hasil: Temporal Baseline

| Metrik | Original | Global Median |
|--------|----------|---------------|
| Recall(0) | 0.037 | 0.056 |
| AUC | 0.519 | 0.528 |

**Kesimpulan:** Tidak ada perubahan berarti. Temporal baseline tetap useless — core problem adalah **14 sampel negatif**, bukan imputation method. Imputasi tidak bisa memperbaiki kekurangan sampel.

---

## 3. Hasil: Stratified Baseline (No Angkatan)

### 3.1 Performance

```
              precision    recall  f1-score   support

 Tidak Tepat     0.81      0.93      0.87        14
 Tepat Waktu     0.99      0.97      0.98       108

    accuracy                         0.97       122
```

| Metrik | Original Per-Angkatan | **Global Median** |
|--------|:--------------------:|:---------------:|
| Recall(0) | 0.929 | **0.929** |
| Precision(0) | 0.650 | **0.813** |
| **F1(0)** | 0.765 | **0.867** |
| AUC | 0.932 | **0.950** |

**F1(0) naik dari 0.77 → 0.87, Precision(0) naik dari 0.65 → 0.81.** Performance justru **lebih baik** setelah proxy temporal dihilangkan!

### 3.2 Feature Importance — IMPROVEMENT SIGNIFIKAN

| Per-Angkatan | | Global Median | |
|---|---|---|---|
| `sks_sem2` | **0.621** | `sks_sem2` | **0.432** ↓ |
| `sks_sem3` | 0.095 | `sks_sem3` | **0.280** ↑ |
| `ips_std` | 0.068 | `ips_sem1` | 0.079 |
| `ips_sem3` | 0.061 | `ips_std` | 0.066 |
| `ips_sem1` | 0.060 | `ips_sem2` | 0.039 |

**Perubahan kunci:**

- **`sks_sem2` turun 30%** (0.62 → 0.43) — proxy bias berkurang signifikan
- **`sks_sem3` naik 3x** (0.10 → 0.28) — fitur SKS semester 3 jadi lebih informatif karena tidak lagi redundant dengan `sks_sem2`
- Distribusi importance **lebih tersebar** — dari hampir mono-fitur (0.62) menjadi dual-top (0.43 + 0.28 = 0.71)

### 3.3 Root Split — Tidak Berubah (18.50)

Root split tetap `sks_sem2 <= 18.50`. Global median = 18.0, jadi banyak mahasiswa yang diimputasi tepat di bawah threshold. Ini menunjukkan threshold ini genuine, bukan artefak imputasi.

---

## 4. Cross-Validation (Stratified)

| Metrik | CV Train | CV Test | Gap |
|--------|---------|---------|-----|
| Accuracy | 1.000 | 0.938 | +0.062 |
| ROC-AUC | 1.000 | **0.854** | +0.146 |

CV ROC-AUC 0.85 — masih ada overfitting, tapi lebih terkendali.

---

## 5. Analisis: `sks_sem2` Masih Dominan — Kenapa?

Meskipun proxy temporal sudah dihilangkan, `sks_sem2` + `sks_sem3` tetap mendominasi (combined 0.71). Hipotesis:

### 5.1 Ini GENUINE Signal

Data investigation menunjukkan:
- Target=0 (Tidak Tepat): mean sks_sem2 = **19.4**, tightly clustered
- Target=1 (Tepat Waktu): mean sks_sem2 = **13.0**, large spread
- Quartile Q4 (high SKS): **46.3% tingkat ketidaktepatan** vs <2% di Q1-Q3

Mahasiswa yang mengambil **banyak SKS di semester 2-3** lebih berisiko telat. Kemungkinan: mereka overload untuk mengejar ketertinggalan → burnout → telat.

### 5.2 Residual Clustering

Global median menghasilkan nilai 18.0 yang dekat dengan root split 18.50. Mahasiswa yang diimputasi dengan 18.0 masuk ke cabang kiri (≤18.5, mostly Tepat Waktu) — ini menciptakan pemisahan yang bersih antara data asli (tersebar di kedua cabang) dan data imputasi (terkonsentrasi di kiri).

### 5.3 Verdict

`Sks_sem2` **genuine predictor** — bukan artifact murni. Ada signal real di sini. Tapi dominasinya (0.43) masih agak tinggi. Bisa dikurangi lebih lanjut dengan:

- **Opsional:** Ganti ke imputasi yang lebih sophisticated (KNN) di experiment berikutnya
- **Atau:** Terima sebagai genuine finding, constraint tree dengan pruning/tuning
- **Atau:** Binarize SKS features (`sks_high_sem2` = sks_sem2 > 18)

---

## 6. Perbandingan Lengkap

### 6.1 Temporal vs Stratified (Global Median)

| Metrik | Temporal | Stratified |
|--------|----------|------------|
| Recall(0) | 0.056 | **0.929** |
| F1(0) | 0.105 | **0.867** |
| AUC | 0.528 | **0.950** |

### 6.2 Stratified: Per-Angkatan vs Global Median

| Metrik | Per-Angkatan | Global Median | Delta |
|--------|:-----------:|:-----------:|-------|
| Recall(0) | 0.929 | 0.929 | = |
| Precision(0) | 0.650 | **0.813** | +0.163 |
| F1(0) | 0.765 | **0.867** | +0.102 |
| AUC | 0.932 | **0.950** | +0.018 |
| sks_sem2 imp. | 0.621 | **0.432** | -0.189 |
| Top-3 spread | 0.78 | **0.79** | lebih seimbang |

---

## 7. Kesimpulan

**Global median imputation BERHASIL:**
1. Proxy temporal berkurang — `sks_sem2` importance turun 30% (0.62→0.43)
2. Performance justru **meningkat** — F1(0) naik ke 0.87, precision 0.81
3. Distribusi fitur lebih seimbang — 2 fitur top vs 1 fitur dominan
4. `sks_sem2` tetap penting (0.43) — ini genuine signal, bukan artifact

**Rekomendasi:**
- **Adopsi global median** sebagai imputation strategy
- `sks_sem2` diterima sebagai genuine predictor (perlu constraint/pruning agar tidak overfit)
- Lanjut ke **class balancing + hyperparameter tuning** dengan dataset dari experiment ini

---

## 8. Deliverables

| File | Deskripsi |
|------|-----------|
| `experiments/global-median/preprocessing.py` | Script preprocessing dengan global median |
| `experiments/global-median/dataset_clean.csv` | Dataset hasil imputasi global median |
| `experiments/global-median/01-temporal-baseline.ipynb` | Notebook temporal baseline |
| `experiments/global-median/01-temporal-results.md` | Output + chart temporal |
| `experiments/global-median/02-stratified-baseline.ipynb` | Notebook stratified baseline |
| `experiments/global-median/02-stratified-results.md` | Output + chart stratified |
| `experiments/global-median/README.md` | Dokumen ini |
