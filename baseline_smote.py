# ===== Results: 3-Class TF-IDF + Logistic Regression + SMOTE =====
# Accuracy              : 0.7685
# Precision (macro)     : 0.5758
# Recall (macro)        : 0.5826
# F1-score (macro)      : 0.5700
# Precision (weighted)  : 0.8404
# Recall (weighted)     : 0.7685
# F1-score (weighted)   : 0.7971

# ===== Classification Report =====
#               precision    recall  f1-score   support

#          low     0.7241    0.8400    0.7778       150
#          mid     0.0517    0.1250    0.0732        24
#         high     0.9517    0.7829    0.8591       327

#     accuracy                         0.7685       501
#    macro avg     0.5758    0.5826    0.5700       501
# weighted avg     0.8404    0.7685    0.7971       501


# ===== Confusion Matrix =====
#            Pred_low  Pred_mid  Pred_high
# True_low        126        17          7
# True_mid         15         3          6
# True_high        33        38        256
# (4205) PS D:\year4sem2\dsai4205\group> 

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
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
# 1. Load data
# =========================
file_path = "./data_cleaned/cleaned_reviews_strict_en.csv"
df = pd.read_csv(file_path)

# Keep only needed columns
df = df[["review_description", "rating"]].copy()

# Drop missing / blank text
df["review_description"] = df["review_description"].astype(str).str.strip()
df = df[df["review_description"].notna()]
df = df[df["review_description"] != ""]
df = df[df["review_description"].str.lower() != "nan"]

# Keep only valid ratings
df = df[df["rating"].isin([1, 2, 3, 4, 5])].copy()

# =========================
# 2. Convert 5-class to 3-class
#    1,2 -> low(0)
#    3   -> mid(1)
#    4,5 -> high(2)
# =========================
rating_map = {
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 2
}
label_map = {
    0: "low",
    1: "mid",
    2: "high"
}
label_names = ["low", "mid", "high"]

df["rating_3class"] = df["rating"].map(rating_map)

print("Dataset shape after cleaning:", df.shape)
print("\n3-class distribution before split:")
print(df["rating_3class"].map(label_map).value_counts())

# =========================
# 3. Train-test split
# =========================
X = df["review_description"]
y = df["rating_3class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining set distribution before SMOTE:")
print(pd.Series(y_train).map(label_map).value_counts())

print("\nTest set distribution:")
print(pd.Series(y_test).map(label_map).value_counts())

# =========================
# 4. TF-IDF feature extraction
#    fit only on training set
# =========================
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# =========================
# 5. Handle imbalance on TRAIN only with SMOTE
# =========================
class_counts = pd.Series(y_train).value_counts()
minority_count = class_counts.min()

# k_neighbors must be < minority class sample count
k_neighbors = min(5, minority_count - 1)
if k_neighbors < 1:
    k_neighbors = 1

smote = SMOTE(
    random_state=42,
    k_neighbors=k_neighbors
)

X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

print("\nTraining set distribution after SMOTE:")
print(pd.Series(y_train_smote).map(label_map).value_counts())

# =========================
# 6. Logistic Regression baseline
# =========================
clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=2000,
    random_state=42
)

clf.fit(X_train_smote, y_train_smote)

# =========================
# 7. Prediction
# =========================
y_pred = clf.predict(X_test_tfidf)

# =========================
# 8. Evaluation
# =========================
accuracy = accuracy_score(y_test, y_pred)

precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n===== Results: 3-Class TF-IDF + Logistic Regression + SMOTE =====")
print(f"Accuracy              : {accuracy:.4f}")
print(f"Precision (macro)     : {precision_macro:.4f}")
print(f"Recall (macro)        : {recall_macro:.4f}")
print(f"F1-score (macro)      : {f1_macro:.4f}")
print(f"Precision (weighted)  : {precision_weighted:.4f}")
print(f"Recall (weighted)     : {recall_weighted:.4f}")
print(f"F1-score (weighted)   : {f1_weighted:.4f}")

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