import os
import re
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from imblearn.over_sampling import BorderlineSMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# =========================================================
# 0. Config
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = r"./data_cleaned/cleaned_reviews_strict_en.csv"
OUTPUT_DIR = "word2vec_borderline_smote_svm_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Word2Vec 参数
W2V_DIM = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_SG = 1          # 1 = skip-gram, 0 = CBOW
W2V_EPOCHS = 20

# 三分类映射
# 1,2 -> low
# 3   -> mid
# 4,5 -> high
LABEL_NAMES = ["low", "mid", "high"]
LABEL_TO_ID = {"low": 0, "mid": 1, "high": 2}
ID_TO_LABEL = {0: "low", 1: "mid", 2: "high"}

# 是否把元信息拼进文本
USE_META_FIELDS = True

# PCA
USE_PCA = True
PCA_DIM = 256

# Borderline-SMOTE
SMOTE_KIND = "borderline-1"

# Linear SVM 参数
C_LIST = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0]
MAX_ITER = 5000


# =========================================================
# 1. Utility
# =========================================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_meta_text(text, default_value):
    text = clean_text(text)
    return text if text else default_value


def normalize_thumbs_up(x):
    if pd.isna(x):
        return 0
    try:
        return int(float(x))
    except Exception:
        return 0


def normalize_review_month(x):
    x = clean_meta_text(x, "unknown_date")
    if x == "unknown_date":
        return x

    m = re.search(r"(\d{4}-\d{2})", x)
    if m:
        return m.group(1)

    if len(x) >= 7:
        return x[:7]

    return x


def map_rating_to_3class(rating):
    if rating in [1, 2]:
        return "low"
    elif rating == 3:
        return "mid"
    elif rating in [4, 5]:
        return "high"
    return None


def build_input_text(row, available_cols):
    desc = clean_text(row["review_description"])
    if not desc:
        desc = "no_review"

    if not USE_META_FIELDS:
        return desc

    parts = []

    if "appVersion" in available_cols:
        version = clean_meta_text(row["appVersion"], "unknown_version")
        parts.append(f"AppVersion: {version}.")
    if "thumbs_up" in available_cols:
        thumbs = normalize_thumbs_up(row["thumbs_up"])
        parts.append(f"ThumbsUp: {thumbs}.")
    if "review_date_parsed" in available_cols:
        review_month = normalize_review_month(row["review_date_parsed"])
        parts.append(f"ReviewDate: {review_month}.")

    parts.append(f"Review: {desc}")
    return " ".join(parts)


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return text.split()


def get_sentence_vector(tokens, w2v_model, embed_dim):
    vectors = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv.key_to_index]
    if not vectors:
        return np.zeros(embed_dim, dtype=np.float32)
    return np.mean(vectors, axis=0)


def transform_to_vectors(token_lists, w2v_model, embed_dim):
    return np.vstack([get_sentence_vector(tokens, w2v_model, embed_dim) for tokens in token_lists])


def compute_metrics(y_true, y_pred, label_names):
    acc = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=label_names,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "report": report,
        "confusion_matrix": cm
    }


def save_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, fontsize=10)
    plt.yticks(tick_marks, labels, fontsize=10)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 2. Load data
# =========================================================
df = pd.read_csv(DATA_PATH)

required_cols = ["review_description", "rating"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

possible_meta_cols = ["appVersion", "thumbs_up", "review_date_parsed"]
available_cols = [c for c in possible_meta_cols if c in df.columns]

keep_cols = required_cols + available_cols
df = df[keep_cols].copy()

df["review_description"] = df["review_description"].apply(clean_text)
df = df[df["review_description"].notna()]
df = df[df["review_description"].str.strip() != ""]
df = df[df["review_description"].str.lower() != "nan"]

df = df[df["rating"].isin([1, 2, 3, 4, 5])].copy()

df["label_name"] = df["rating"].apply(map_rating_to_3class)
df = df[df["label_name"].notna()].copy()
df["label_id"] = df["label_name"].map(LABEL_TO_ID)

df["input_text"] = df.apply(lambda row: build_input_text(row, available_cols), axis=1)

print("Dataset shape after cleaning:", df.shape)
print("\nOriginal rating distribution:")
print(df["rating"].value_counts().sort_index())
print("\n3-class distribution:")
print(df["label_name"].value_counts())
print("\n3-class distribution by rating mapping check:")
print(df.groupby(["rating", "label_name"]).size())


# =========================================================
# 3. Split: 70 / 15 / 15
# =========================================================
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["label_id"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label_id"]
)

print("\nTrain distribution:")
print(train_df["label_name"].value_counts())
print("\nValidation distribution:")
print(val_df["label_name"].value_counts())
print("\nTest distribution:")
print(test_df["label_name"].value_counts())


# =========================================================
# 4. Word2Vec embedding
# =========================================================
train_df["tokens"] = train_df["input_text"].apply(tokenize)
val_df["tokens"] = val_df["input_text"].apply(tokenize)
test_df["tokens"] = test_df["input_text"].apply(tokenize)

print("\nTraining Word2Vec model ...")
w2v_model = Word2Vec(
    sentences=train_df["tokens"].tolist(),
    vector_size=W2V_DIM,
    window=W2V_WINDOW,
    min_count=W2V_MIN_COUNT,
    sg=W2V_SG,
    seed=SEED,
    workers=1,
    epochs=W2V_EPOCHS
)

X_train = transform_to_vectors(train_df["tokens"].tolist(), w2v_model, W2V_DIM)
X_val = transform_to_vectors(val_df["tokens"].tolist(), w2v_model, W2V_DIM)
X_test = transform_to_vectors(test_df["tokens"].tolist(), w2v_model, W2V_DIM)

y_train = train_df["label_id"].values
y_val = val_df["label_id"].values
y_test = test_df["label_id"].values

print("\nEmbedding shape:", X_train.shape)


# =========================================================
# 5. Scale + PCA
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

if USE_PCA:
    pca_dim = min(PCA_DIM, X_train_scaled.shape[1], len(X_train_scaled) - 1)
    pca = PCA(n_components=pca_dim, random_state=SEED)

    X_train_feat = pca.fit_transform(X_train_scaled)
    X_val_feat = pca.transform(X_val_scaled)
    X_test_feat = pca.transform(X_test_scaled)

    print(f"\nPCA enabled. Reduced dimension: {pca_dim}")
    print(f"Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")
else:
    X_train_feat = X_train_scaled
    X_val_feat = X_val_scaled
    X_test_feat = X_test_scaled


# =========================================================
# 6. Borderline-SMOTE on TRAIN only
# =========================================================
train_counts = Counter(y_train)
print("\nTrain class counts BEFORE Borderline-SMOTE:")
for i in range(3):
    print(f"{ID_TO_LABEL[i]}: {train_counts[i]}")

minority_count = min(train_counts.values())
k_neighbors = min(5, minority_count - 1)
if k_neighbors < 1:
    k_neighbors = 1

smote = BorderlineSMOTE(
    kind=SMOTE_KIND,
    random_state=SEED,
    k_neighbors=k_neighbors
)

X_train_res, y_train_res = smote.fit_resample(X_train_feat, y_train)

train_counts_after = Counter(y_train_res)
print("\nTrain class counts AFTER Borderline-SMOTE:")
for i in range(3):
    print(f"{ID_TO_LABEL[i]}: {train_counts_after[i]}")


# =========================================================
# 7. Model selection on validation set
# =========================================================
best_val_f1 = -1.0
best_C = None
best_model = None
history = []

for c in C_LIST:
    clf = LinearSVC(
        C=c,
        max_iter=MAX_ITER,
        random_state=SEED
    )
    clf.fit(X_train_res, y_train_res)

    val_pred = clf.predict(X_val_feat)
    val_metrics = compute_metrics(y_val, val_pred, LABEL_NAMES)

    history.append({
        "C": c,
        "val_accuracy": val_metrics["accuracy"],
        "val_precision_macro": val_metrics["precision_macro"],
        "val_recall_macro": val_metrics["recall_macro"],
        "val_f1_macro": val_metrics["f1_macro"],
        "val_f1_weighted": val_metrics["f1_weighted"]
    })

    print(
        f"\nC={c:.2f} | "
        f"Val Acc={val_metrics['accuracy']:.4f} | "
        f"Val Macro-F1={val_metrics['f1_macro']:.4f} | "
        f"Val Weighted-F1={val_metrics['f1_weighted']:.4f}"
    )

    if val_metrics["f1_macro"] > best_val_f1:
        best_val_f1 = val_metrics["f1_macro"]
        best_C = c
        best_model = clf

print(f"\nBest C selected by validation macro-F1: {best_C}")


# =========================================================
# 8. Test
# =========================================================
test_pred = best_model.predict(X_test_feat)
test_metrics = compute_metrics(y_test, test_pred, LABEL_NAMES)

print("\n================ FINAL TEST RESULTS ================")
print("Embedding Model       : Word2Vec")
print(f"Embedding Dimension   : {W2V_DIM}")
print("Classifier            : LinearSVC")
print("Resampling            : Borderline-SMOTE")
print(f"Best C                : {best_C}")
print(f"Accuracy              : {test_metrics['accuracy']:.4f}")
print(f"Precision (macro)     : {test_metrics['precision_macro']:.4f}")
print(f"Recall (macro)        : {test_metrics['recall_macro']:.4f}")
print(f"F1-score (macro)      : {test_metrics['f1_macro']:.4f}")
print(f"Precision (weighted)  : {test_metrics['precision_weighted']:.4f}")
print(f"Recall (weighted)     : {test_metrics['recall_weighted']:.4f}")
print(f"F1-score (weighted)   : {test_metrics['f1_weighted']:.4f}")

print("\n===== Classification Report =====")
print(test_metrics["report"])

print("\n===== Confusion Matrix =====")
cm_df = pd.DataFrame(
    test_metrics["confusion_matrix"],
    index=[f"True_{x}" for x in LABEL_NAMES],
    columns=[f"Pred_{x}" for x in LABEL_NAMES]
)
print(cm_df)


# =========================================================
# 9. Save outputs
# =========================================================
history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(OUTPUT_DIR, "validation_history.csv"), index=False)

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
save_confusion_matrix(test_metrics["confusion_matrix"], LABEL_NAMES, cm_path)

with open(os.path.join(OUTPUT_DIR, "test_results.txt"), "w", encoding="utf-8") as f:
    f.write("================ FINAL TEST RESULTS ================\n")
    f.write("Embedding Model       : Word2Vec\n")
    f.write(f"Embedding Dimension   : {W2V_DIM}\n")
    f.write("Classifier            : LinearSVC\n")
    f.write("Resampling            : Borderline-SMOTE\n")
    f.write(f"Best C                : {best_C}\n")
    f.write(f"Accuracy              : {test_metrics['accuracy']:.4f}\n")
    f.write(f"Precision (macro)     : {test_metrics['precision_macro']:.4f}\n")
    f.write(f"Recall (macro)        : {test_metrics['recall_macro']:.4f}\n")
    f.write(f"F1-score (macro)      : {test_metrics['f1_macro']:.4f}\n")
    f.write(f"Precision (weighted)  : {test_metrics['precision_weighted']:.4f}\n")
    f.write(f"Recall (weighted)     : {test_metrics['recall_weighted']:.4f}\n")
    f.write(f"F1-score (weighted)   : {test_metrics['f1_weighted']:.4f}\n\n")

    f.write("===== Classification Report =====\n")
    f.write(test_metrics["report"])
    f.write("\n\n===== Confusion Matrix =====\n")
    f.write(cm_df.to_string())

print(f"\nAll outputs saved to: {OUTPUT_DIR}")