# Breast Cancer Classification — Model Comparison Study

## Overview

This project builds and compares three machine learning models to classify tumors as malignant or benign using the Breast Cancer Wisconsin dataset.

The objective was not only to achieve high accuracy, but to evaluate models through multiple performance lenses, particularly recall, which is critical in medical diagnosis.

## Dataset

- 569 tumor biopsy samples
- 30 numerical features
- Target:
  - 0 = Malignant
  - 1 = Benign
- Mild class imbalance (~63% benign, ~37% malignant)

Loaded directly from `sklearn.datasets`.

---

## Models Compared

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

---

## Evaluation Techniques

- Train/Test Split
- Confusion Matrix
- Precision, Recall, F1 Score
- ROC Curve & AUC
- Threshold Tuning
- 5-Fold Stratified Cross-Validation

---

## Key Findings

- Logistic Regression consistently outperformed the other models.
- Highest cross-validated recall (~0.992).
- Lowest false negatives (critical for cancer detection).
- Most stable performance across folds.
- Increasing model complexity did not improve performance on this dataset.

---

## Visualizations

- Confusion Matrix Comparison
- Feature Importance (Random Forest)
- ROC Curve Comparison

---

## Why Logistic Regression Was Selected

Although ensemble methods are often more powerful, this dataset appears largely linearly separable. Logistic Regression provided:

- Highest recall
- Strong precision
- Stable cross-validation performance
- Simpler and more interpretable decision boundary

In medical contexts, minimizing false negatives is critical. Logistic Regression achieved this most consistently.

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
python ML_Breast_Cancer.py
```
