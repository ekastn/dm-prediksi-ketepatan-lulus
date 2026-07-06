# AGENTS.md — Project Context for AI Agents

This file is the single source of truth for any AI agent working on this repository. Read it fully before doing anything.

---

## Project Identity

| Field | Value |
|-------|-------|
| **Title** | Prediksi Ketepatan Lulus Mahasiswa Menggunakan Decision Tree |
| **Course** | Data Mining, Semester 06 |
| **Institution** | DIV Teknik Informatika, Universitas Logistik dan Bisnis Internasional (ULBI) |
| **Year** | 2026 |
| **Framework** | CRISP-DM |
| **Team** | Viola Septianti Elsiana (714230001), Muhammad Saladin Eka Septian (714230037), Muhammad Hisyam Najwan (714230055) |
| **Primary model** | `sklearn.tree.DecisionTreeClassifier` |
| **Language** | Python 3, Jupyter notebooks |
| **Database** | SQL Server 2022 (Docker), database `LITIGASI` |
| **VCS** | Git, last commit `de63ec5` — "Phase 5 evaluation: temporal validation reveals model not generalizable" |

---

## Problem Statement

Predict whether students in programs **AP (D3 Administrasi Peradilan)** and **IH (S1 Ilmu Hukum)** will graduate on time (`Tepat Waktu` / `Tidak Tepat`), using only academic performance data from their first 3 semesters.

**Targets (set in Business Understanding phase):**

| Metric | Target |
|--------|--------|
| Recall(0) — detect at-risk students | ≥ 0.70 |
| F1(0) — balanced minority-class score | ≥ 0.50 |
| AUC-ROC | ≥ 0.75 |
| Tree depth | ≤ 5 |
| Train-CV gap | < 0.05 |
| Leaves | ≤ 15 |

---

## Data Pipeline (602 students, 14 features, 1 target)

### Source
- **SQL Server `LITIGASI`** running in Docker at `localhost:1433`, user `sa`, password `MSSQL01#`
- 405 tables → only 4 used: `tblMHS` (1,621 students), `IPSIPK` (6,228 semester records), `Qnilai_mhs` (54,587 grade records), `Kul_Kehadiran` (attendance)
- Tried and rejected: `HtblNilai` (0% NIM overlap — incompatible legacy format), `Perwalian` (all NULL), `feed_nilai` (184 students only), `Luusan` (empty), `tblCuti` (empty)

### Filtering (1,621 → 608)
1. Remove angkatan 2024–2025 (< 3 semesters of IPSIPK): −458
2. Remove angkatan 2014 and others with zero valid IPS (legacy migration gaps): −154
3. Remove students with unknown outcome (Aktif/Cuti): −401
4. **Final: 608 students**

### Class Balance
- Tepat Waktu: 540 (88.8%)
- Tidak Tepat: 68 (11.2%) — highly imbalanced
- Program AP (D3): 147 students (137 tepat, 10 tidak)
- Program IH (S1): 461 students (403 tepat, 58 tidak)

### Features (14 predictors)
| Category | Features |
|----------|----------|
| IPS (semester GPA) | `ips_sem1`, `ips_sem2`, `ips_sem3` |
| SKS (credit load) | `sks_sem1`, `sks_sem2`, `sks_sem3` |
| Failed courses | `failed_courses` (total), `failed_in_sem1` (sem1 only), `repeated_courses` |
| Derived from IPS | `ips_trend` (sem3−sem1), `avg_ips` (mean of 3), `ips_std` (std dev), `ips_min` |
| Derived from SKS | `sks_completion_ratio` (completed/taken) |

**Dropped from model (not predictive features):** `angkatan` (would leak — all 2023 students are target=0), `program` (near-zero importance), `ips_sem4` (leakage, r=0.877 with target), `semester_count`, `avg_attendance` (53% missing), `id_agama`, `jenis_kelamin`.

### Target Definition
- `1` = Tepat Waktu (graduated within normal duration: D3=6 sem, S1=8 sem)
- `0` = Tidak Tepat (exceeded normal duration, or DO/mengundurkan diri)

### Critical Data Quality Issue
**SKS migration bug:** Database was migrated by a vendor. For angkatan 2020+, SKS stored in `IPSIPK.TSKS` is cumulative (80–133 for semesters 1–2), not per-semester. Fixed in `extract_dataset.py` by falling back to counting distinct `Kode_MK` from `Qnilai_mhs`, capped at 1–20 courses/semester, outliers → NULL → median imputation.

**Zero-IPS placeholder:** 219 students (36%) have `IPS=0.0` as a system placeholder (not actual zero). Preprocessing step 1 replaces `0.0 → NaN` before imputation.

### Key Datasets
| File | Description |
|------|-------------|
| `3-data-preparation/dataset.csv` | Raw extraction, 608×27 |
| `3-data-preparation/dataset_train.csv` | Temporal train (angkatan ≤2021, 377 rows) |
| `3-data-preparation/dataset_test.csv` | Temporal test (angkatan >2021, 231 rows) |
| `3-data-preparation/dataset_clean.csv` | Clean, per-angkatan median imputation |
| `4-modeling/2-global-median/dataset_clean.csv` | Clean, global median imputation (better) |
| `5-evaluation/dataset_clean.csv` | Copy of global median dataset |


## Complete Experiment Log

### Phase 4 — Modeling

#### Iteration 1: Baseline (`4-modeling/1-baseline/`)
| Exp | Split | Key Result | Verdict |
|-----|-------|-----------|---------|
| 01 — Temporal baseline | Train ≤2021, Test >2021, DT default | Recall(0)=0.056, F1(0)=0.105 | ❌ Failed — only 14 neg in train |
| 02 — Stratified baseline | 80/20 random, stratify=target, DT default | Recall(0)=0.929, F1(0)=0.867 | ⚠️ Overfit (train=1.0, depth=8) |
| 03 — SKS investigation | Diagnostic | Discovered per-angkatan imputation creates temporal proxy via `sks_sem2` | Root cause found |

**Decision:** Adopt stratified split for modeling exploration (54 neg in train vs 14 in temporal).

#### Iteration 2: Global Median Imputation (`4-modeling/2-global-median/`)
| Exp | Split | Key Change | F1(0) | Verdict |
|-----|-------|-----------|-------|---------|
| 01 | Temporal | Global median instead of per-angkatan | 0.060 | ❌ Still fails (14 neg) |
| 02 | Stratified | Global median instead of per-angkatan | **0.867** | ✅ Major improvement (+0.102 over per-angkatan) |

**Key insight:** Per-angkatan median imputation leaked cohort info — each angkatan got different imputed values, making `sks_sem2` a temporal proxy. Global median fixed this.

#### Iteration 3: Hyperparameter Tuning (`4-modeling/3-hyperparameter-tuning/`)
| Exp | Method | Result |
|-----|--------|--------|
| 01 — Pre-Pruning | GridSearchCV: 240 combos × 5-fold (max_depth, min_samples_leaf, criterion, splitter) | **Best: max_depth=3, min_samples_leaf=10, criterion='gini'** → F1(0)=0.889 |
| 02 — Feature Engineering | 4 binary flags + aggregate features | ❌ All engineered features importance=0.0000; deleted |
| 03 — Combined tuning | 28 combos (max_features + min_impurity_decrease) | CV improved (0.837), test same (0.889) — ceiling reached |

**Ceiling:** Single Decision Tree F1(0) = **0.889**. Three different tuning approaches converge to the same value. Further improvements require ensemble methods or different data splits.

### Best Model (01-Tuned)
```python
DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
```
- Depth: 3, Leaves: 7 (2 "Tidak Tepat", 5 "Tepat Waktu"), Nodes: 13
- Uses 6/14 features
- **Stratified test:** F1(0)=0.889, Recall(0)=0.857, Precision(0)=0.923, AUC=0.955, Acc=0.98

### Decision Rules
```
|--- sks_sem2 <= 18.50
|   |--- failed_courses <= 0.50 → TEPAT WAKTU (both ips_sem1 branches)
|   |--- failed_courses >  0.50 → TEPAT WAKTU
|--- sks_sem2 >  18.50
|   |--- sks_sem3 <= 18.50 → TIDAK TEPAT (both ips_std branches)
|   |--- sks_sem3 >  18.50 → TEPAT WAKTU (both avg_ips branches)
```

**Business rule:** Students with SKS sem2 > 18.5 AND SKS sem3 ≤ 18.5 are at risk — the "overload then collapse" pattern.

### Feature Importance (MDI)
| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `sks_sem2` | 0.558 |
| 2 | `sks_sem3` | 0.361 |
| 3 | `ips_std` | 0.047 |
| 4 | `avg_ips` | 0.017 |
| 5 | `failed_courses` | 0.012 |
| 6 | `ips_sem1` | 0.004 |

SKS dominates IPS — 91.9% combined. Study load patterns > academic grades for predicting graduation timeliness.

---

### Phase 5 — Evaluation (`5-evaluation/`)
6 evaluation phases were run. The central notebook is `01-final-evaluation.ipynb`.

#### Phase 1: Temporal Validation — CRITICAL FINDING
Same best model hyperparameters, trained on temporal split (train ≤2021, test >2021):

| Metric | Stratified Split | Temporal Split | Delta |
|--------|-----------------|----------------|-------|
| F1(0) | **0.889** | **0.000** | −0.889 |
| Recall(0) | **0.857** | **0.000** | −0.857 |
| Precision(0) | 0.923 | 0.000 | −0.923 |
| AUC | 0.955 | **0.710** | −0.245 |

**Model predicted ALL 231 temporal test students as "Tepat Waktu."** 0/54 at-risk students detected.

**Root cause:** Temporal train has only **14 negative samples (3.7%)** vs 54 (11.1%) in stratified train. With `min_samples_leaf=10`, the tree cannot form any leaf predicting the minority class. All 7 leaves default to majority. Root split changes from `sks_sem2` to `failed_courses`. The tree structure completely changes.

#### Phase 2: Deep Error Analysis
- 54/54 at-risk students in temporal test are false negatives
- 96.3% of FNs from program IH, 90.7% from angkatan 2023
- The "overload-collapse" rule (`sks_sem2 > 18.5 AND sks_sem3 ≤ 18.5`) covers 52 students in temporal test — and **52 actually are negative** (perfect rule precision), but the temporal model never applies this rule because it never learns it from training

#### Phase 3: Repeated CV
10×10-fold RepeatedStratifiedKFold (100 evaluations):
- F1(0) mean = **0.833** [95% CI: 0.814–0.851]
- Recall(0) mean = 0.783 [95% CI: 0.757–0.810]
- Single stratified split overestimates recall(0) by +7.4% vs CV mean

#### Phase 4: Permutation Importance
- `sks_sem2` and `sks_sem3` remain dominant — no hidden signals missed by MDI
- Rank correlation between MDI and permutation importance: 0.675 (p=0.008) — good agreement

#### Phase 5: Rule Stability
- Root split `sks_sem2` stable in 10/10 CV folds on stratified full-dataset training
- Leaves mean=7.2, std=0.42

#### Phase 6: Distribution Shift
- `sks_sem2` mean jumps from ~11 (angkatan 2015–2021) to ~19 (angkatan 2022–2023)
- Historical models cannot generalize to newer angkatan with different SKS distributions


## Current Status: CRISP-DM Phases

| Phase | Status | Key Output |
|-------|--------|-----------|
| 1. Business Understanding | ✅ Complete | `Proposal Data Mining.pdf` |
| 2. Data Understanding | ✅ Complete | 5 SQL profiling scripts, `exploration-log.md`, `01-data-profiling.ipynb` |
| 3. Data Preparation | ✅ Complete | `extract_dataset.py`, EDA, preprocessing, `dataset_clean.csv` (608×17) |
| 4. Modeling | ✅ Complete | 3 iterations, best model selected |
| 5. Evaluation | ✅ Complete | 6-phase evaluation, `evaluation-report.md`, `evaluation_metrics.json` |
| 6. Deployment | ❌ Not started | Model flagged as NOT deployable |

## Outstanding Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | Proposal singkat | ✅ |
| 2 | Dataset + deskripsi variabel | ✅ |
| 3 | Notebook preprocessing | ✅ |
| 4 | Modeling (3 iterasi) | ✅ |
| 5 | Hyperparameter tuning | ✅ |
| 6 | Evaluation (6 phases) | ✅ |
| 7 | Laporan formal Bab I–V | ⬜ NOT DONE |
| 8 | Slide presentasi (.pptx) | 🔶 Partially done — `make_pptx.py`, `ppt.html`, `template_ppt.html`, two `.pptx` files exist |
| 9 | Artikel IEEE | ⬜ NOT DONE |

---

## Key Files Quick Reference

### Documentation
| File | Content |
|------|---------|
| `README.md` | Master technical report (463 lines) — the definitive project summary |
| `reference.md` | Reference paper outline (rice price prediction) — used as structural template for Bab I–V |
| `AGENTS.md` | This file — agent context |

### Phase 2 — Data Understanding
| File | Content |
|------|---------|
| `2-data-understanding/README.md` | Phase overview + table mapping decisions |
| `2-data-understanding/exploration-log.md` | Chronological narrative (385 lines) — how we explored 405 tables |
| `2-data-understanding/01-schema-discovery.sql` | Found 4 key tables |
| `2-data-understanding/02-student-profiling.sql` | Student demographics |
| `2-data-understanding/03-academic-profiling.sql` | IPS/IPK/SKS analysis |
| `2-data-understanding/04-grade-profiling.sql` | Grade distributions |
| `2-data-understanding/05-data-quality.sql` | Zero-variance, NIM mismatch |
| `2-data-understanding/06-table-mapping.md` | Table relationships + rejected tables |
| `2-data-understanding/ddl.csv` | DDL reference (2,880 lines) |

### Phase 3 — Data Preparation
| File | Content |
|------|---------|
| `3-data-preparation/README.md` | ETL pipeline docs (244 lines) |
| `3-data-preparation/extract_dataset.py` | Main pipeline: SQL Server → in-memory processing → CSV (424 lines) |
| `3-data-preparation/02-eda.ipynb` | Exploratory data analysis |
| `3-data-preparation/02-eda-findings.md` | EDA findings |
| `3-data-preparation/03-preprocessing.ipynb` | Preprocessing: zero→NaN, drop, impute, encode |
| `3-data-preparation/03-preprocessing-plan.md` | Preprocessing plan |

### Phase 4 — Modeling
| File | Content |
|------|---------|
| `4-modeling/README.md` | Modeling overview (319 lines) |
| `4-modeling/1-baseline/modeling-plan.md` | Iteration 1 plan |
| `4-modeling/1-baseline/01-modeling.ipynb` | Temporal baseline |
| `4-modeling/1-baseline/02-stratified-baseline.ipynb` | Stratified baseline |
| `4-modeling/1-baseline/03-sks-investigation.ipynb` | SKS temporal proxy investigation |
| `4-modeling/2-global-median/preprocessing.py` | Global median preprocessing script |
| `4-modeling/2-global-median/01-temporal-baseline.ipynb` | Temporal + global median |
| `4-modeling/2-global-median/02-stratified-baseline.ipynb` | Stratified + global median |
| `4-modeling/3-hyperparameter-tuning/tuning-plan.md` | Tuning experiment plan |
| `4-modeling/3-hyperparameter-tuning/tuning-report.md` | Exp 01 results |
| `4-modeling/3-hyperparameter-tuning/combined-tuning-report.md` | Exp 03 results |
| `4-modeling/3-hyperparameter-tuning/01-hyperparameter-tuning.ipynb` | Exp 01 + Exp 02 |
| `4-modeling/3-hyperparameter-tuning/03-combined-tuning.ipynb` | Exp 03 |
| `4-modeling/3-hyperparameter-tuning/rules_tuned.txt` | Best model rules |
| `4-modeling/3-hyperparameter-tuning/rules_combined.txt` | Combined model rules |

### Phase 5 — Evaluation
| File | Content |
|------|---------|
| `5-evaluation/README.md` | Evaluation plan (219 lines) |
| `5-evaluation/evaluation-report.md` | Comprehensive report (440 lines) |
| `5-evaluation/01-final-evaluation.ipynb` | Main notebook — all 6 phases |
| `5-evaluation/01-evaluation-results.md` | nbconvert output (2,289 lines) |
| `5-evaluation/01-evaluation-results_files/` | 10 chart PNGs |
| `5-evaluation/evaluation_metrics.json` | Structured metrics export |
| `5-evaluation/rules_stratified.txt` | Rules from stratified model |
| `5-evaluation/rules_temporal.txt` | Rules from temporal model |

### Presentation
| File | Content |
|------|---------|
| `make_pptx.py` | HTML → PPTX converter (416 lines) |
| `ppt.html` | Presentation HTML template |
| `template_ppt.html` | Alternative template |
| `prediksi_ketepatan_lulus.pptx` | Generated presentation v1 |
| `prediksi_ketepatan_lulus2.pptx` | Generated presentation v2 |

---

## Critical Architectural Knowledge

### Why the model works on stratified split but fails on temporal

1. **Sample imbalance in temporal training:** Only 14 negative samples (3.7%) in temporal train (angkatan ≤2021) vs 54 (11.1%) in stratified. With `min_samples_leaf=10`, the tree mathematically cannot split to find the minority class.

2. **Distribution shift:** SKS patterns changed over time. Mean `sks_sem2` rose from ~11 (pre-2020) to ~19 (2022–2023). Students in newer cohorts take more credits per semester. The model trained on older data doesn't learn the "overload-collapse" pattern that became more common in newer cohorts.

3. **Stratified sampling is misleading for temporal prediction:** Stratified split distributes negative samples evenly across train/test, ensuring the model "sees" enough minority class during training. But in real deployment, you don't have future data to stratify by — you only have historical data.

4. **The "overload-collapse" rule is real but untrained:** The rule `sks_sem2 > 18.5 AND sks_sem3 ≤ 18.5` perfectly identifies at-risk students (52/52 in temporal test), but the temporal model never learns it because it never sees enough examples of this pattern in historical training data.

### Do NOT: Common Mistakes When Working on This Project

- **Don't use `angkatan` as a feature** — it would leak (all 2023 students are target=0) and was already dropped
- **Don't use `ips_sem4` as a predictor** — it has r=0.877 with target (leakage from later semesters)
- **Don't use per-angkatan median imputation** — it creates a temporal proxy that inflates model performance artificially. Always use global median
- **Don't treat stratified performance as deployment performance** — the model's true capability is F1(0)=0.000, not 0.889
- **Don't retrain on the full dataset and report performance** — this project has already established that stratified metrics are illusory
- **The SKS data for angkatan 2020+ from `IPSIPK.TSKS` is cumulative, not per-semester** — use the fixed pipeline in `extract_dataset.py`, not raw TSKS values

### Accepted Project Conventions

- Random state: always `42`
- Scoring focus: `f1_score(pos_label=0)` — minority class (Tidak Tepat) is the metric that matters
- Temporal split boundary: `angkatan <= 2021` = train, `> 2021` = test
- All preprocessing must happen AFTER splitting to avoid leakage
- `target` column: 0 = Tidak Tepat (minority), 1 = Tepat Waktu (majority)
- Notebook outputs (`*-results.md`) are nbconvert artifacts — don't edit them directly

---

## Recommendations for Next Steps (from the project team)

### If continuing modeling work:
1. **SMOTE or class_weight='balanced'** on temporal split to handle 14-neg-sample problem
2. **Ensemble methods** — Random Forest or Gradient Boosting for better stability with imbalanced data
3. **Collect more data** — specifically more negative samples (not on-time graduates)
4. **Separate models per program** — AP and IH have different credit structures (D3 vs S1)
5. **Rule-based hybrid system** — combine the perfect-precision rule (`sks_sem2 > 18.5 AND sks_sem3 ≤ 18.5`) with ML for coverage
6. **Periodic retraining** — update model yearly with new cohort data to handle distribution shift

### If working on documentation/report:
- `reference.md` is an unrelated paper (rice price prediction) used as a structural template for the Bab I–V formal report format
- The formal report (laporan Bab I–V) is the highest-priority remaining deliverable
- Use `README.md` as the technical source — it contains all data, results, and figures needed for the report
- Polish the `.pptx` slides using the existing `make_pptx.py` pipeline

### If working on deployment:
- The model is explicitly flagged as NOT ready for deployment
- Any deployment work should first resolve the temporal generalization failure
- The simplest deployable system would be the rule-based alert: flag any student with `sks_sem2 > 18.5 AND sks_sem3 ≤ 18.5`
