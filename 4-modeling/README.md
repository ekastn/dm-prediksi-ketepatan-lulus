# Fase 4 — Modeling

**CRISP-DM Phase 4: Decision Tree untuk klasifikasi ketepatan lulus mahasiswa.**

---

## Struktur

```
4-modeling/
├── README.md                    ← Dokumen ini
├── 1-baseline/                  ← Iterasi 1: Baseline Decision Tree
├── 2-global-median/             ← Iterasi 2: Global Median Imputation
└── ...                          ← Future iterations
```

---

## Iterasi 1: Baseline (`1-baseline/`)

**Tujuan:** Membangun Decision Tree baseline dengan hyperparameter default.

### Eksperimen

| Notebook | Deskripsi | Hasil Kunci |
|----------|-----------|-------------|
| `01-temporal-baseline` | Split temporal (≤2021 train, >2021 test), DT default | Recall(0)=0.04 — gagal total |
| `02-stratified-baseline` | Split stratified 80/20, drop angkatan+program | Recall(0)=0.93 — bagus, tapi sks_sem2 dominance |
| `03-sks-investigation` | Investigasi penyebab sks_sem2 importance 0.62 | Imputasi per-angkatan menciptakan proxy temporal |

### Temuan Utama

1. **Temporal split (14 sampel negatif) → model underfit parah.** Decision tree hanya bisa belajar dari 14 sampel minoritas, menghasilkan recall 3.7% di test.

2. **Stratified split (54 sampel negatif) → perbaikan dramatis.** Recall 0.93, F1 0.77. Tapi feature importance didominasi `sks_sem2` (0.62) karena imputasi median per-angkatan menciptakan nilai unik per angkatan → proxy untuk angkatan 2023 (semua target=0).

3. **`angkatan` sebagai fitur = data leakage.** Angkatan 2023 = semua target=0 karena mahasiswa belum lulus. Tree langsung belajar shortcut ini.

4. **5 fitur zero importance:** `program`, `sks_sem1`, `angkatan`, `failed_in_sem1`, `ips_std` (di temporal baseline).

### Laporan
- `baseline-report.md` — Laporan lengkap temporal baseline
- `stratified-report.md` — Laporan lengkap stratified baseline

---

## Iterasi 2: Global Median Imputation (`2-global-median/`)

**Tujuan:** Menghilangkan proxy temporal yang disebabkan imputasi median per-angkatan.

### Perubahan

```python
# Sebelum (per-angkatan)
df[col] = df.groupby('angkatan')[col].transform(lambda x: x.fillna(x.median()))

# Sesudah (global)
df[col] = df[col].fillna(df[col].median())
```

### Eksperimen

| Notebook | Deskripsi | Hasil Kunci |
|----------|-----------|-------------|
| `01-temporal-baseline` | Temporal split dengan data global median | Recall(0)=0.06 — tetap useless (14 neg) |
| `02-stratified-baseline` | Stratified split, drop angkatan+program | **Recall(0)=0.93, F1(0)=0.87, AUC=0.95** |

### Temuan Utama

1. **Feature importance lebih balance:** `sks_sem2` turun dari 0.62 → 0.43, `sks_sem3` naik dari 0.10 → 0.28.

2. **Performance justru meningkat:** F1(0) naik dari 0.77 → 0.87, precision(0) naik dari 0.65 → 0.81.

3. **`sks_sem2` tetap prediktor kuat (0.43):** Investigasi menunjukkan ini genuine signal — mahasiswa yang overload SKS di semester 2 (mean 19.4) cenderung telat vs yang beban normal (mean 13.0). Bukan artifact.

4. **Semua target modeling-plan sudah tercapai di baseline:**
   - Recall(0) ≥ 0.70 → **0.93** ✅
   - F1(0) ≥ 0.50 → **0.87** ✅
   - AUC ≥ 0.75 → **0.95** ✅

5. **Masih ada overfitting:** Train perfect 1.0, depth 8, 24 leaves. Tree unconstrained.

### Laporan
- `README.md` — High-level experiment summary + comparison
- `stratified-report.md` — Laporan lengkap stratified baseline (global median)

---

## Perbandingan Antar Iterasi

| | 1—Temporal | 1—Stratified | **2—Stratified (global)** |
|---|---|---|---|
| Train negatif | 14 | 54 | 54 |
| Imputation | Per-angkatan | Per-angkatan | **Global median** |
| Feature count | 16 | 14 | 14 |
| Recall(0) | 0.037 | 0.929 | **0.929** |
| Precision(0) | 1.000 | 0.650 | **0.813** |
| **F1(0)** | 0.071 | 0.765 | **0.867** |
| AUC | 0.519 | 0.932 | **0.950** |
| Top feature | ips_min (0.25) | sks_sem2 (0.62) | **sks_sem2 (0.43)** |
| Key problem | Underfit | Proxy temporal | Overfit (unconstrained) |

---

## Rencana Selanjutnya

### Iterasi 3: Hyperparameter Tuning

- GridSearchCV dengan scoring F1(0)
- Parameter grid: `max_depth`, `min_samples_leaf`, `min_samples_split`, `class_weight`, `criterion`
- Tujuan: kurangi overfitting, tingkatkan precision(0), tree lebih interpretable

### Iterasi 4: Model Comparison (optional)

- Random Forest
- Gaussian Naive Bayes

### Iterasi 5: Final Evaluation

- Repeated stratified k-fold
- Temporal split sanity check
- Rule extraction + domain interpretation
- Final model export

---

## Dataset

Semua eksperimen menggunakan dataset dari `3-data-preparation/`:
- **Input raw:** `dataset.csv` (608 × 27, hasil ekstraksi SQL Server)
- **Clean (per-angkatan):** `3-data-preparation/dataset_clean.csv`
- **Clean (global median):** `2-global-median/dataset_clean.csv`

### Fitur (setelah preprocessing)

| # | Fitur | Kategori |
|---|-------|----------|
| 1 | `angkatan` | Identitas |
| 2 | `program` | Demografi |
| 3-5 | `ips_sem1`, `ips_sem2`, `ips_sem3` | IPS |
| 6-8 | `sks_sem1`, `sks_sem2`, `sks_sem3` | SKS |
| 9-11 | `failed_courses`, `failed_in_sem1`, `repeated_courses` | Nilai MK |
| 12-16 | `ips_trend`, `avg_ips`, `ips_std`, `ips_min`, `sks_completion_ratio` | Derived |

**Catatan:** `angkatan` dan `program` di-drop di stratified split (data leakage).

---

## Key Decisions Made

1. **Stratified split > Temporal split** — untuk eksplorasi modeling dengan sampel minoritas cukup
2. **Global median > Per-angkatan median** — untuk menghilangkan proxy temporal
3. **Drop `angkatan` + `program`** — untuk mencegah data leakage
4. **SMOTE ditunda** — 54 sampel negatif sudah cukup untuk baseline
5. **F1(0) sebagai scoring utama** — lebih informatif dari accuracy untuk imbalanced data

---

## Metrik Target (dari `modeling-plan.md`)

| Metrik | Target | Status (Best) |
|--------|--------|---------------|
| Recall(0) | ≥ 0.70 | ✅ 0.929 |
| F1(0) | ≥ 0.50 | ✅ 0.867 |
| ROC-AUC | ≥ 0.75 | ✅ 0.950 |
