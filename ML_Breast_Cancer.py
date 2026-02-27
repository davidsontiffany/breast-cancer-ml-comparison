from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# Load dataset
cancer = load_breast_cancer()

# Convert to dataframe
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

# Explore
print("Target names:", cancer.target_names)
print("Shape:", df.shape)
print("\nFirst rows:")
print(df.head())

print("\nSummary stats:")
print(df.describe())

# -----------------------------
# STEP 1 OBSERVATIONS
# -----------------------------

# Observation 1:
# The dataset contains 569 rows and 30 numerical features plus 1 target column.
# This confirms the problem is a supervised classification task where the model
# learns patterns from numeric predictors to classify tumors as malignant or benign.

# Observation 2:
# The target variable is binary (0 = malignant, 1 = benign), indicating a classification
# problem rather than regression. This means evaluation metrics like precision, recall,
# and F1 score will be more important than just accuracy.

# Observation 3:
# All features are continuous numeric values and appear to have very different ranges
# (e.g., mean area vs fractal dimension). This suggests feature scaling will be important
# for models sensitive to magnitude such as Logistic Regression.

# Observation 4:
# There are no missing values shown in the summary statistics, which means data cleaning
# will be minimal and we can move directly into modeling.

# -----------------------------
# STEP 2 — Check class balance
# -----------------------------

print("\nClass counts:")
print(df['target'].value_counts())

print("\nClass distribution (%):")
print(df['target'].value_counts(normalize=True).round(3))

# -----------------------------
# STEP 2 OBSERVATIONS — CLASS BALANCE
# -----------------------------

# The dataset is mildly imbalanced, with ~63% benign cases and ~37% malignant cases.
# This means accuracy alone could be misleading. A naive model that predicts "benign"
# for every case would already achieve ~63% accuracy without actually learning anything.

# Because this is a medical prediction problem, recall becomes especially important.
# Missing a malignant tumor (false negative) is far more dangerous than incorrectly
# flagging a benign tumor as malignant (false positive). Therefore, recall and F1 score
# will be key evaluation metrics when comparing models.

# -----------------------------
# STEP 3 — Train/test split and scaling
# -----------------------------


X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# STEP 3 OBSERVATIONS — SPLIT & SCALING
# -----------------------------

# The parameter stratify=y ensures that the train and test sets maintain the
# same class distribution (benign vs malignant) as the original dataset.
# Since this dataset is mildly imbalanced (~63% benign, ~37% malignant),
# stratification prevents one set from accidentally containing too many
# examples of one class. This helps ensure fair and reliable model evaluation.

# StandardScaler was applied ONLY to the training data using fit_transform(),
# and then transform() was applied to the test data. This prevents data leakage,
# which would occur if information from the test set influenced the scaling.
# Scaling is especially important for Logistic Regression because it relies
# on gradient-based optimization and is sensitive to feature magnitude.

# -----------------------------
# STEP 4 — Logistic Regression
# -----------------------------


lr_model = LogisticRegression(max_iter=10000, random_state=42)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

# Explanation:
# max_iter controls how many iterations the algorithm is allowed to use
# to converge. If set too low, the model may fail to converge and produce
# unstable or inaccurate results. We used scaled features because Logistic
# Regression is sensitive to feature magnitude.

# -----------------------------
# STEP 5 — Random Forest
# -----------------------------


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# Explanation:
# Random Forest builds many decision trees and combines their predictions.
# Unlike Logistic Regression, it does not require scaled features because
# tree-based models split data based on thresholds rather than feature magnitude.

# -----------------------------
# STEP 6 — Evaluate models
# -----------------------------


print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))
print("F1 Score:", f1_score(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# -----------------------------
# STEP 6 ANALYSIS
# -----------------------------

# Logistic Regression achieved higher accuracy, precision, recall, and F1 score
# compared to Random Forest.

# Most importantly, Logistic Regression produced only 1 false negative,
# while Random Forest produced 2 false negatives. Since false negatives
# represent missed malignant tumors, minimizing them is critical in a
# medical diagnosis context.

# Therefore, Logistic Regression appears to perform better on this dataset,
# particularly in terms of recall, which is the most important metric for
# cancer detection.

# -----------------------------
# STEP 7 — Plot Confusion Matrices
# -----------------------------


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

labels = cancer.target_names

ConfusionMatrixDisplay.from_predictions(
    y_test, lr_pred, display_labels=labels,
    ax=axes[0], colorbar=False
)
axes[0].set_title('Logistic Regression')

ConfusionMatrixDisplay.from_predictions(
    y_test, rf_pred, display_labels=labels,
    ax=axes[1], colorbar=False
)
axes[1].set_title('Random Forest')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()

# -----------------------------
# STEP 7 ANALYSIS — CONFUSION MATRICES
# -----------------------------

# The most important difference between the two models is the number of
# false negatives (missed malignant tumors).

# Logistic Regression produced 1 false negative, while Random Forest produced 2.
# In a medical context, false negatives are the most dangerous type of error
# because they represent missed cancer diagnoses.

# Although both models perform well overall, Logistic Regression is safer
# because it minimizes missed malignant cases. This makes it more suitable
# for deployment in a healthcare screening context where recall is critical.

# -----------------------------
# STEP 8 — Feature Importance (Random Forest)
# -----------------------------

importances = rf_model.feature_importances_
feature_names = cancer.feature_names

feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feat_df.head(10))

# Plot top 10
plt.figure(figsize=(10, 6))
plt.barh(feat_df['Feature'][:10][::-1],
         feat_df['Importance'][:10][::-1])

plt.xlabel('Importance Score')
plt.title('Top 10 Most Important Features (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# -----------------------------
# STEP 8 ANALYSIS — FEATURE IMPORTANCE
# -----------------------------

# The most important features are dominated by the "worst" measurements
# (e.g., worst area, worst concave points, worst radius, worst perimeter).
# This indicates that the most extreme tumor characteristics are the strongest
# predictors of malignancy.

# Mean-based features also appear in the top rankings, while standard error (SE)
# features are largely absent. This suggests that overall tumor size and shape
# matter more than variability in measurements.

# A possible explanation is that malignant tumors exhibit more extreme structural
# characteristics, making "worst-case" measurements more informative for prediction.
# This aligns with clinical intuition, where aggressive tumors show more abnormal
# morphology.

# The model’s reliance on these features appears reasonable and trustworthy,
# as they correspond to biologically meaningful indicators of tumor severity.


"""
-----------------------------
STEP 9 — FINAL MODEL RECOMMENDATION
-----------------------------

Based on the evaluation results, Logistic Regression would be the preferred model
for deployment in this breast cancer classification task. While both models performed
well, Logistic Regression achieved higher accuracy, precision, recall, and F1 score
compared to Random Forest.

The most important metric driving this decision is recall. In a medical context,
false negatives represent missed malignant tumors, which can have severe consequences
for patient outcomes. Logistic Regression produced fewer false negatives than Random
Forest, making it the safer option for real-world screening scenarios where identifying
cancer cases is critical.

Although Random Forest is often considered a more powerful model, in this dataset it
produced slightly more false negatives and false positives. These errors could lead to
missed diagnoses or unnecessary follow-up procedures, both of which carry real-world
costs in healthcare environments.

To improve the weaker model (Random Forest), one potential step would be hyperparameter
tuning, such as adjusting the number of trees, tree depth, or class weighting to better
handle the class imbalance. Additionally, cross-validation could provide a more reliable
performance estimate and help refine the model further.

Overall, Logistic Regression is recommended because it balances strong performance,
interpretability, and a lower risk of missing malignant cases, which is the most
critical objective in this classification problem.
"""

# ROC Curve: Plot the ROC curve for both models and compute the AUC. What does a
# larger AUC mean, and how does it relate to the metrics you already computed?


# Get probability scores
lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

# Calculate AUC
lr_auc = roc_auc_score(y_test, lr_probs)
rf_auc = roc_auc_score(y_test, rf_probs)

print("\nROC AUC Scores:")
print("Logistic Regression AUC:", lr_auc)
print("Random Forest AUC:", rf_auc)

plt.figure(figsize=(8, 6))

plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.3f})')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})')

# Random guessing line
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()

# -----------------------------
# STRETCH GOAL ANALYSIS — ROC & AUC
# -----------------------------

# Both models achieved extremely high AUC scores, indicating excellent ability
# to distinguish between malignant and benign tumors across all classification
# thresholds.

# Logistic Regression achieved an AUC of ~0.995, while Random Forest achieved
# ~0.994. This means Logistic Regression slightly outperforms Random Forest in
# ranking malignant cases higher than benign ones.

# A larger AUC indicates better model performance because it reflects stronger
# separation between classes regardless of the chosen probability threshold.
# Unlike accuracy, precision, and recall, which depend on a single decision
# cutoff (e.g., 0.50), AUC evaluates performance across all thresholds.

# The high AUC values align with the previously observed metrics, including
# strong recall and precision. This confirms that both models are highly
# effective classifiers, with Logistic Regression showing a slight advantage
# in overall discrimination ability.


# Threshold tuning: Logistic Regression outputs probabilities, not just class labels. Figure
# out how to access those probabilities and experiment with changing the decision
# threshold. Document the trade-off you observe.

# -----------------------------
# STRETCH GOAL — THRESHOLD TUNING
# -----------------------------


# Function to apply threshold

def predict_with_threshold(probs, threshold):
    return (probs >= threshold).astype(int)


# Try different thresholds
thresholds = [0.3, 0.5, 0.7]

for t in thresholds:
    preds = predict_with_threshold(lr_probs, t)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\nThreshold: {t}")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)

    # -----------------------------
# STRETCH GOAL ANALYSIS — THRESHOLD TUNING
# -----------------------------

# Adjusting the decision threshold significantly impacts the trade-off between
# precision and recall.

# At a threshold of 0.3, recall increased to 1.0, meaning the model detected
# all malignant tumors. However, precision decreased slightly, indicating
# more false positives. This setting prioritizes sensitivity and would be
# appropriate in medical screening scenarios where missing cancer is unacceptable.

# At the default threshold of 0.5, precision and recall were balanced,
# providing strong overall performance.

# At a higher threshold of 0.7, precision remained high, but recall dropped,
# meaning more malignant cases were missed. This conservative setting reduces
# false alarms but increases the risk of missed diagnoses.

# This experiment demonstrates how threshold tuning allows practitioners
# to align model behavior with real-world risk tolerance. In healthcare,
# lowering the threshold may be preferable to maximize recall and minimize
# false negatives.


# Cross-validation: A single train/test split can be lucky or unlucky. Research a technique
# that gives you a more reliable performance estimate and apply it to both models.

# ==========================================================
# STRETCH GOAL — CROSS VALIDATION
# ==========================================================

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression with scaling pipeline
lr_cv_model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=10000, random_state=42))
])

# Random Forest
rf_cv_model = RandomForestClassifier(n_estimators=100, random_state=42)

scoring = ['accuracy', 'precision', 'recall', 'f1']

lr_cv_results = cross_validate(lr_cv_model, X, y, cv=cv, scoring=scoring)
rf_cv_results = cross_validate(rf_cv_model, X, y, cv=cv, scoring=scoring)

def summarize_cv(name, results):
    print(f"\n=== {name} (5-Fold Cross Validation) ===")
    for metric in scoring:
        scores = results[f'test_{metric}']
        print(f"{metric.title():<10} Mean={scores.mean():.4f}  Std={scores.std():.4f}")

summarize_cv("Logistic Regression", lr_cv_results)
summarize_cv("Random Forest", rf_cv_results)

# -----------------------------
# STRETCH GOAL ANALYSIS — CROSS-VALIDATION
# -----------------------------

# Cross-validation was used to provide a more reliable performance estimate
# by evaluating both models across multiple train/test splits rather than
# relying on a single split.

# Logistic Regression consistently outperformed Random Forest across all
# evaluation metrics, particularly recall and F1 score. The mean recall for
# Logistic Regression (~0.992) was higher than Random Forest (~0.967),
# indicating that Logistic Regression is more effective at detecting
# malignant tumors across different subsets of the data.

# Standard deviation values were relatively low for both models, suggesting
# stable performance. However, Logistic Regression demonstrated slightly
# more consistent recall across folds, reinforcing its reliability.

# These results confirm earlier findings from the confusion matrix, ROC/AUC,
# and threshold tuning analyses: Logistic Regression is the stronger and
# more dependable model for this classification task, particularly in a
# medical context where minimizing false negatives is critical.

# Cross-validation strengthens confidence that the observed performance is
# not due to a lucky train/test split but reflects the model’s true ability
# to generalize to new data.


#Third model: Add one more classifier of your choice to the comparison. Justify why you
#picked it and whether the result surprised you


# -----------------------------
# STRETCH GOAL — THIRD MODEL (Gradient Boosting)
# -----------------------------

#Train the Model

gb_model = GradientBoostingClassifier(random_state=42)

gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)

#Evaluate It

print("\n=== Gradient Boosting ===")
print("Accuracy:", accuracy_score(y_test, gb_pred))
print("Precision:", precision_score(y_test, gb_pred))
print("Recall:", recall_score(y_test, gb_pred))
print("F1 Score:", f1_score(y_test, gb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, gb_pred))

#Add Cross-Validation for it

gb_cv_model = GradientBoostingClassifier(random_state=42)

gb_cv_results = cross_validate(gb_cv_model, X, y, cv=cv, scoring=scoring)

summarize_cv("Gradient Boosting", gb_cv_results)

# -----------------------------
# STRETCH GOAL — THIRD MODEL ANALYSIS
# -----------------------------

# Gradient Boosting was selected as a third model because it represents
# a boosting-based ensemble method, which differs from Logistic Regression
# (linear model) and Random Forest (bagging-based ensemble). Gradient
# Boosting sequentially builds trees that correct the errors of prior trees,
# and it is often highly effective on structured tabular data.

# On the test set, Gradient Boosting achieved strong recall (0.986),
# matching Logistic Regression in minimizing false negatives.
# However, it produced more false positives than Logistic Regression,
# resulting in slightly lower precision and accuracy overall.

# Cross-validation results further confirmed that Logistic Regression
# maintained higher average recall and overall stability across folds.
# Although Gradient Boosting performed well, it did not outperform
# Logistic Regression on this dataset.

# This outcome was somewhat surprising, as boosting methods frequently
# outperform linear models. However, the dataset appears to be well-separated
# and largely linear, meaning Logistic Regression is sufficient and does not
# require additional model complexity.

# Therefore, even after introducing a third model, Logistic Regression
# remains the most reliable and interpretable choice for deployment,
# particularly in a medical context where minimizing false negatives
# and maintaining model stability are critical.