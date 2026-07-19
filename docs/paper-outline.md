# Title

**Balancing Performance and Interpretability Through Hyperparameter Tuning of Decision Trees for Student Graduation Prediction**

**Pengaruh Hyperparameter Tuning terhadap Kinerja dan Interpretabilitas Model Decision Tree untuk Prediksi Ketepatan Kelulusan Mahasiswa**

---

# Research Questions

This study aims to answer the following research questions:

**RQ1.** How does hyperparameter tuning affect the predictive performance of Decision Tree models for student graduation prediction?

**RQ2.** How does hyperparameter tuning influence the interpretability and complexity of Decision Tree models?

**RQ3.** To what extent does the tuned Decision Tree model generalize under different validation strategies?

---

# Paper Outline

## Abstract *(To be written last)*

* Background
* Problem Statement
* Proposed Method
* Experimental Results
* Conclusion

---

# I. Introduction

### A. Background

* Importance of predicting student graduation outcomes.
* Educational Data Mining and early warning systems.
* Decision Tree as an interpretable classification algorithm.

### B. Problem Statement

* Decision Trees are prone to overfitting.
* Academic datasets often contain missing values and severe class imbalance.
* Improving predictive performance should not sacrifice model interpretability.

### C. Research Gap

* Existing studies primarily focus on improving predictive performance while providing limited discussion on the trade-off between model performance and interpretability after hyperparameter tuning.

### D. Research Questions

* Present RQ1–RQ3.

---

# II. Related Work

### A. Student Graduation Prediction

* Previous studies on graduation prediction.
* Educational Data Mining applications.

### B. Decision Tree in Educational Data Mining

* Advantages of Decision Trees.
* Interpretability in educational decision support systems.

### C. Hyperparameter Tuning

* Hyperparameter optimization techniques.
* GridSearchCV for Decision Tree optimization.
* Summary of existing limitations.

---

# III. Methodology

### A. Dataset

* Data source.
* Student population.
* Features.
* Target variable.

### B. Data Preprocessing

* Data filtering.
* Missing value handling.
* Global median imputation.
* Feature engineering.
* Dataset splitting.

### C. Decision Tree Model

* CART algorithm.
* Gini impurity.
* Selected hyperparameters.

### D. Hyperparameter Tuning

* GridSearchCV configuration.
* Hyperparameter search space.
* Cross-validation strategy.
* Model selection criteria.

### E. Experimental Setup

* Baseline model.
* Tuned model.
* Evaluation metrics.
* Validation strategies.

---

# IV. Results and Discussion

### A. Baseline Model Performance

* Baseline evaluation.
* Initial observations.
* Overfitting analysis.

### B. Impact of Hyperparameter Tuning

* Performance comparison.
* Evaluation metrics.
* Best hyperparameter configuration.

### C. Model Interpretability Analysis

* Tree depth.
* Number of leaves.
* Decision rules.
* Feature importance.
* Model complexity comparison.

### D. Generalization Analysis

* Stratified validation.
* Temporal validation.
* Discussion of generalization capability.

---

# V. Conclusion

### A. Summary of Findings

* Summary of the experimental results.
* Answers to the research questions.

### B. Limitations

* Dataset limitations.
* Generalization limitations.

### C. Future Work

* Ensemble learning methods.
* Class imbalance handling.
* Temporal-aware modeling.

---

# Planned Figures

| Figure | Description                                                       |
| ------ | ----------------------------------------------------------------- |
| Fig. 1 | Overall research workflow based on CRISP-DM                       |
| Fig. 2 | Best-performing Decision Tree visualization                       |
| Fig. 3 | Feature importance ranking                                        |
| Fig. 4 | Performance comparison between stratified and temporal validation |

---

# Planned Tables

| Table     | Description                                |
| --------- | ------------------------------------------ |
| Table I   | Dataset characteristics                    |
| Table II  | Hyperparameter search space                |
| Table III | Baseline vs. tuned model performance       |
| Table IV  | Model complexity comparison                |
| Table V   | Stratified vs. temporal validation results |

---

## Writing Order

To keep the writing process efficient, we'll write the paper in the following order:

1. Methodology
2. Results and Discussion
3. Related Work
4. Introduction
5. Conclusion
6. Abstract

This order ensures that the Introduction and Abstract accurately reflect the completed experiments and findings, reducing the need for major revisions later.
