# ===== Results: 3-Class TF-IDF + Logistic Regression =====
# Accuracy              : 0.8144
# Precision (macro)     : 0.5642
# Recall (macro)        : 0.5912
# F1-score (macro)      : 0.5714
# Precision (weighted)  : 0.8225
# Recall (weighted)     : 0.8144
# F1-score (weighted)   : 0.8122

# ===== Classification Report =====
#               precision    recall  f1-score   support

#          low     0.6923    0.9000    0.7826       150
#          mid     0.0625    0.0417    0.0500        24
#         high     0.9379    0.8318    0.8817       327

#     accuracy                         0.8144       501
#    macro avg     0.5642    0.5912    0.5714       501
# weighted avg     0.8225    0.8144    0.8122       501


# ===== Confusion Matrix =====
#            Pred_low  Pred_mid  Pred_high
# True_low        135         5         10
# True_mid         15         1          8
# True_high        45        10        272


import pandas as pd
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
from sklearn.utils import resample

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

train_df = pd.DataFrame({
    "review_description": X_train,
    "rating": y_train
})

print("\nTraining set distribution before resampling:")
print(train_df["rating"].map(label_map).value_counts())

# =========================
# 4. Handle imbalance on TRAIN only
#    Oversample minority classes to the largest class size
# =========================
class_counts = train_df["rating"].value_counts()
target_count = class_counts.max()

resampled_parts = []

for rating_value, group in train_df.groupby("rating"):
    current_count = len(group)

    if current_count < target_count:
        upsampled = resample(
            group,
            replace=True,
            n_samples=target_count,
            random_state=42
        )
        resampled_parts.append(upsampled)
    else:
        resampled_parts.append(group)

train_resampled = pd.concat(resampled_parts).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nTraining set distribution after resampling:")
print(train_resampled["rating"].map(label_map).value_counts())

X_train_resampled = train_resampled["review_description"]
y_train_resampled = train_resampled["rating"]

# =========================
# 5. TF-IDF feature extraction
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
# 6. Logistic Regression baseline
# =========================
clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

clf.fit(X_train_tfidf, y_train_resampled)

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

print("\n===== Results: 3-Class TF-IDF + Logistic Regression =====")
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