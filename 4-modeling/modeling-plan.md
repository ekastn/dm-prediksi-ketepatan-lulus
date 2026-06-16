# Modeling Plan — Prediksi Ketepatan Lulus Mahasiswa

Fase 4 CRISP-DM: Decision Tree untuk klasifikasi ketepatan lulus.

---

## 1. Kondisi Saat Ini & Input

### Dataset Final

| Metrik | Nilai |
|--------|-------|
| **File** | `3-data-preparation/dataset_clean.csv` |
| **Rows** | 608 mahasiswa |
| **Features** | 16 numerik + 1 target |
| **Target** | `1` = Tepat Waktu, `0` = Tidak Tepat |
| **NULLs** | 0 |

### 16 Fitur

| # | Fitur | Tipe | Kategori |
|---|-------|------|----------|
| 1 | `angkatan` | int (2015–2023) | Identitas |
| 2 | `program` | 0/1 (AP=0, IH=1) | Demografi |
| 3 | `ips_sem1` | float (1.3–4.0) | IPS Semester |
| 4 | `ips_sem2` | float (0.57–4.0) | IPS Semester |
| 5 | `ips_sem3` | float (0.3–4.0) | IPS Semester |
| 6 | `sks_sem1` | float (2–24) | SKS Semester |
| 7 | `sks_sem2` | float (2–22) | SKS Semester |
| 8 | `sks_sem3` | float (1–22) | SKS Semester |
| 9 | `failed_courses` | int (0–18) | Nilai MK |
| 10 | `failed_in_sem1` | int (0–7) | Nilai MK |
| 11 | `repeated_courses` | int (0–12) | Nilai MK |
| 12 | `ips_trend` | float (-2.79–2.5) | Derived |
| 13 | `avg_ips` | float (1.68–3.9) | Derived |
| 14 | `ips_std` | float (0–1.8) | Derived |
| 15 | `ips_min` | float (0.3–3.8) | Derived |
| 16 | `sks_completion_ratio` | float (0.12–1.07) | Derived |

### Train/Test Split

**Strategi: Temporal split** dari `dataset_clean.csv`:

```
Train: angkatan ≤ 2021  →  377 rows
Test:  angkatan > 2021   →  231 rows
```

| Set | Tepat (1) | Tidak (0) | % Neg |
|-----|----------|-----------|-------|
| Train | 363 | **14** | **3.7%** |
| Test | 177 | 54 | 23.4% |

Split temporal dipakai karena mensimulasikan skenario nyata: model dilatih pada data historis (outcome sudah diketahui), diuji pada angkatan baru (outcome belum diketahui sepenuhnya).

### Problem Kritis: Class Imbalance

Training set hanya punya **14 sampel negatif** dari 377 (3.7%). Tanpa handling imbalance, model akan cenderung memprediksi "Tepat Waktu" untuk semua input — useless untuk deteksi mahasiswa berisiko.

**Strategi handling:**
1. `class_weight='balanced'`
2. SMOTE (Synthetic Minority Oversampling)
3. Kombinasi keduanya

---

## 2. Workflow Modeling

```
dataset_clean.csv (608 rows)
    │
    ├── Step 1: Re-split temporal (angkatan ≤ 2021 train, > 2021 test)
    │
    ├── Step 2: Baseline DecisionTreeClassifier (default params)
    │        Evaluasi: accuracy, precision, recall, F1, confusion matrix
    │        Baseline untuk perbandingan improvement
    │
    ├── Step 3: Handling class imbalance
    │        ├── 3a: class_weight='balanced'
    │        ├── 3b: SMOTE oversampling
    │        └── 3c: SMOTE + class_weight combined
    │        Pilih strategi terbaik berdasarkan recall kelas 0
    │
    ├── Step 4: Hyperparameter tuning
    │        GridSearchCV dengan stratified 5-fold CV:
    │          - max_depth: [3, 5, 7, 10, 15, None]
    │          - min_samples_split: [2, 5, 10, 20]
    │          - min_samples_leaf: [1, 2, 5, 10]
    │          - criterion: [gini, entropy]
    │          - ccp_alpha: pruning path (cost-complexity)
    │
    ├── Step 5: Feature selection & pruning
    │        - Feature importance dari best model
    │        - Drop fitur dengan importance < threshold (opsional)
    │        - Post-pruning dengan optimal ccp_alpha
    │        - Evaluasi ulang model pruned
    │
    ├── Step 6: Decision tree rule extraction
    │        - Ekstrak aturan dari tree (sklearn.tree.export_text)
    │        - Format sebagai if-then rules
    │        - Translate ke bahasa domain:
    │          "Jika IPS Trend < -0.5 DAN Failed Courses > 2 → Berisiko Telat"
    │
    ├── Step 7: Comparison models
    │        ├── Random Forest (ensemble, lebih robust)
    │        ├── Gaussian Naive Bayes (baseline probabilistik)
    │        └── Bandingkan: F1, ROC-AUC, Precision-Recall curve
    │
    └── Step 8: Final evaluation & deliverables
             - 10-fold stratified CV pada train
             - Confusion matrix + classification report
             - ROC curve + AUC
             - Precision-Recall curve (lebih informatif untuk imbalanced)
             - Feature importance bar chart
             - Decision tree visualization (export_graphviz)
             - Learning curve
             - Save model dengan joblib
```

---

## 3. Metrik Evaluasi

### Prioritas (karena fokus pada deteksi mahasiswa berisiko)

| Metrik | Prioritas | Alasan |
|--------|-----------|--------|
| **Recall kelas 0** | ⭐⭐⭐ | Menangkap sebanyak mungkin mahasiswa yang akan telat. False negative (gagal deteksi) lebih mahal dari false positive |
| **F1-score kelas 0** | ⭐⭐⭐ | Balance antara precision dan recall untuk kelas minoritas |
| **Precision kelas 0** | ⭐⭐ | Jangan terlalu banyak false alarm (memprediksi telat padahal tepat) |
| **ROC-AUC** | ⭐⭐ | Overall separability kedua kelas |
| Accuracy | ⭐ | Menyesatkan karena imbalance — model yang selalu prediksi "tepat" dapat accuracy 88% |

### Target Minimum

| Metrik | Target |
|--------|--------|
| Recall kelas 0 | ≥ 0.70 |
| F1-score kelas 0 | ≥ 0.50 |
| ROC-AUC | ≥ 0.75 |

---

## 4. Teknis Implementasi

### Library

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold,
    cross_val_score, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import joblib
```

### File Output

```
4-modeling/
├── modeling-plan.md              # Dokumen ini
├── 01-modeling.ipynb            # Notebook modeling utama
├── 01-modeling.py               # Script modeling (opsional)
├── charts/                      # Visualisasi hasil
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall.png
│   ├── feature_importance.png
│   ├── decision_tree.png
│   └── learning_curve.png
├── rules.txt                    # Aturan decision tree (teks)
└── model_dt.pkl                 # Model tersimpan (joblib)
```

---

## 5. Risiko & Mitigasi

| Risiko | Severity | Mitigasi |
|--------|----------|----------|
| Overfitting karena sampel negatif terlalu sedikit (14) | High | Pruning ketat, max_depth kecil, stratified CV |
| SMOTE menciptakan synthetic samples yang tidak realistis | Medium | Bandingkan dengan class_weight='balanced' saja |
| Distribution shift train→test (3.7% vs 23.4% neg) | Medium | Tidak bisa dihindari — dokumentasikan sebagai batasan temporal |
| Decision Tree terlalu sederhana, underfit | Low | Bandingkan dengan Random Forest |
| Fitur `angkatan` jadi proxy untuk target di test set | Medium | Coba train tanpa `angkatan`, bandingkan hasil |

---

## 6. Timeline Perkiraan

| Step | Estimasi |
|------|----------|
| Setup + Re-split | 15 menit |
| Baseline DT | 15 menit |
| Imbalance handling (3 strategi) | 30 menit |
| Hyperparameter tuning | 30 menit |
| Feature selection & pruning | 20 menit |
| Rule extraction | 15 menit |
| Comparison models (RF, NB) | 30 menit |
| Final evaluation & deliverables | 30 menit |
| **Total** | ~3 jam |

---

## 7. Open Questions

1. ~~Train/test split ulang dari `dataset_clean.csv`?~~ → **Ya**, re-split temporal dari clean dataset
2. Fitur `angkatan`: dipertahankan atau didrop? → **Dipertahankan dulu**, evaluasi dari feature importance. Bisa jadi proxy yang valid (angkatan tua cenderung sudah lulus).
3. `has_any_failure` sebagai fitur biner? → **Coba di Step 5** (feature engineering opsional), karena `failed_courses` extremely zero-inflated.
4. Stratified k-fold: 5-fold atau 10-fold? → **5-fold** untuk tuning (cepat), **10-fold** untuk evaluasi final.
