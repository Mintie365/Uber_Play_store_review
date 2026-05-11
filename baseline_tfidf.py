# ===== Results: 3-Class TF-IDF + Logistic Regression (LLM Augmented) =====
# Accuracy              : 0.8064
# Precision (macro)     : 0.5619
# Recall (macro)        : 0.5919
# F1-score (macro)      : 0.5671

# ===== Classification Report =====
#               precision    recall  f1-score   support

#          low     0.6715    0.9267    0.7787       150
#          mid     0.0714    0.0417    0.0526        24
#         high     0.9429    0.8073    0.8699       327

#     accuracy                         0.8064       501
#    macro avg     0.5619    0.5919    0.5671       501
# weighted avg     0.8199    0.8064    0.8034       501


# ===== Confusion Matrix =====
#            Pred_low  Pred_mid  Pred_high
# True_low        139         3          8
# True_mid         15         1          8
# True_high        53        10        264


import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================
# 1. Load Ready-to-use Data
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "data_train_ready.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "data_test_ready.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


#
train_df["review_description"] = train_df["review_description"].fillna("").astype(str)
test_df["review_description"] = test_df["review_description"].fillna("").astype(str)
#

X_train_resampled = train_df["review_description"]
y_train_resampled = train_df["label_id"]

X_test = test_df["review_description"]
y_test = test_df["label_id"]

label_names = ["low", "mid", "high"]

print(f"Loaded balanced Train set from: {TRAIN_PATH} ({len(X_train_resampled)} rows)")
print(f"Loaded Test set from: {TEST_PATH} ({len(X_test)} rows)")

# =========================
# 2. TF-IDF feature extraction
# =========================
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000
)

X_train_tfidf = tfidf.fit_transform(X_train_resampled)
X_test_tfidf = tfidf.transform(X_test)

# =========================
# 3. Logistic Regression
# =========================
clf = LogisticRegression(

    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

clf.fit(X_train_tfidf, y_train_resampled)

# =========================
# 4. Prediction & Evaluation
# =========================
y_pred = clf.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("\n===== Results: 3-Class TF-IDF + Logistic Regression (LLM Augmented) =====")
print(f"Accuracy              : {accuracy:.4f}")
print(f"Precision (macro)     : {precision_macro:.4f}")
print(f"Recall (macro)        : {recall_macro:.4f}")
print(f"F1-score (macro)      : {f1_macro:.4f}")

print("\n===== Classification Report =====")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=label_names,
    digits=4,
    zero_division=0
))

print("\n===== Confusion Matrix =====")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
cm_df = pd.DataFrame(
    cm,
    index=[f"True_{name}" for name in label_names],
    columns=[f"Pred_{name}" for name in label_names]
)
print(cm_df)