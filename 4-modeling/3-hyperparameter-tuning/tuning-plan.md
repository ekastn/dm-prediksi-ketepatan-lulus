# Iterasi 3 — Hyperparameter Tuning + Pruning

**Fase 4 CRISP-DM | Dataset: Global median imputation (`2-global-median/dataset_clean.csv`)**

---

## 1. Objective

Tune Decision Tree hyperparameters untuk mengurangi overfitting (train 1.0 → lebih rendah),
meningkatkan precision(0), dan menghasilkan tree yang lebih kecil + interpretable.

## 2. Baseline (dari Iterasi 2)

| Metrik | Nilai | Notes |
|--------|-------|-------|
| Depth | 8 | Unconstrained |
| Leaves | 24 | |
| Nodes | 47 | |
| Recall(0) | 0.929 | 13/14 terdeteksi |
| Precision(0) | 0.813 | 2 false positive |
| F1(0) | 0.867 | |
| AUC | 0.950 | |
| Train | Perfect 1.0 | **Overfit** |

## 3. Strategy: Pre-pruning + Post-pruning

### Tahap 1: Pre-pruning (GridSearchCV)

```python
param_grid = {
    'max_depth':          [3, 4, 5, 6, None],
    'min_samples_leaf':   [5, 10, 15, 20],
    'min_samples_split':  [5, 10, 20],
    'class_weight':       [None, 'balanced'],
    'criterion':          ['gini', 'entropy'],
}
```

Scoring: `make_scorer(f1_score, pos_label=0)`  
CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

### Tahap 2: Post-pruning (ccp_alpha)

Dari best model Tahap 1 → compute `cost_complexity_pruning_path` → pilih optimal `ccp_alpha` yang memaksimalkan F1(0) di CV.

### Tahap 3: Final Model

Train dengan best params Tahap 1 + optimal ccp_alpha Tahap 2.

## 4. Evaluation Plan

- Classification report + confusion matrix (test set)
- 10-fold stratified CV pada final model
- Baseline vs Tuned comparison chart (4 metrik + tree size)
- Feature importance (bandingkan dengan baseline)
- Decision tree visualization (tree kecil, readable!)
- Decision rules (export_text — pendek)
- Learning curve
- Summary table

## 5. Expected Outcome

| | Baseline | Expected After Tuning |
|---|---|---|
| Depth | 8 | 3-5 |
| Leaves | 24 | 8-15 |
| Recall(0) | 0.93 | 0.85-0.93 |
| Precision(0) | 0.81 | 0.85-0.95 |
| F1(0) | 0.87 | 0.85-0.92 |
| Overfit gap | +0.10 | <0.05 |
| Tree readability | Complex | Interpretable |

## 6. Deliverables

```
4-modeling/3-hyperparameter-tuning/
├── tuning-plan.md                   ← Dokumen ini
├── dataset_clean.csv                ← Copy dari 2-global-median/
├── 01-hyperparameter-tuning.ipynb   ← Notebook
├── 01-tuning-results.md + _files/   ← Output nbconvert
└── tuning-report.md                 ← Report
```
