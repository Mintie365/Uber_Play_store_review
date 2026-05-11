# ================ FINAL TEST RESULTS ================
# Best Epoch          : 2
# Test Loss           : 0.3352
# Accuracy            : 0.8404
# Precision (macro)   : 0.6140
# Recall (macro)      : 0.6117
# F1-score (macro)    : 0.6117
# Precision (weighted): 0.8578
# Recall (weighted)   : 0.8404
# F1-score (weighted) : 0.8488

# ===== Classification Report (low / mid / high) =====
#               precision    recall  f1-score   support

#          low     0.8440    0.8214    0.8326       112
#          mid     0.0769    0.1111    0.0909        18
#         high     0.9212    0.9024    0.9117       246

#     accuracy                         0.8404       376
#    macro avg     0.6140    0.6117    0.6117       376
# weighted avg     0.8578    0.8404    0.8488       376


# ===== Confusion Matrix (low / mid / high) =====
# [[ 92   9  11]
#  [  8   2   8]
#  [  9  15 222]]

# All outputs saved to: survey_roberta_output_3class


import os
import re
import random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

import matplotlib.pyplot as plt


# =========================================================
# 0. Config
# =========================================================
SEED = 42
DATA_PATH = r"data_cleaned\cleaned_reviews_strict_en.csv"

DESC_COL = "review_description"
VERSION_COL = "appVersion"
THUMBS_COL = "thumbs_up"
DATE_COL = "review_date_parsed"
LABEL_COL = "rating"

MODEL_NAME = "roberta-base"
MAX_LEN = 192
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 2

USE_MINORITY_AUGMENTATION = True
FOCAL_GAMMA = 2.0

OUTPUT_DIR = "survey_roberta_output_3class"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# 三分类映射
# low  = 1,2
# mid  = 3
# high = 4,5
THREE_CLASS_NAMES = ["low", "mid", "high"]
THREE_CLASS_TO_ID = {"low": 0, "mid": 1, "high": 2}
ID_TO_THREE_CLASS = {0: "low", 1: "mid", 2: "high"}


# =========================================================
# 1. Reproducibility
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================================================
# 2. Device info
# =========================================================
print("Using device:", DEVICE)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
else:
    print("WARNING: CUDA not available, using CPU!")


# =========================================================
# 3. Text / meta cleaning
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


# =========================================================
# 4. Build multi-field input text
# =========================================================
def build_input_text(row):
    desc = clean_text(row[DESC_COL])
    version = clean_meta_text(row[VERSION_COL], "unknown_version")
    thumbs = normalize_thumbs_up(row[THUMBS_COL])
    review_month = normalize_review_month(row[DATE_COL])

    if not desc:
        desc = "no_review"

    combined = (
        f"AppVersion: {version}. "
        f"ThumbsUp: {thumbs}. "
        f"ReviewDate: {review_month}. "
        f"Review: {desc}"
    )
    return combined


# =========================================================
# 5. Light augmentation for minority classes
#    只增强 review_description
# =========================================================
def random_deletion(words, p=0.1):
    if len(words) <= 4:
        return words
    new_words = [w for w in words if random.random() > p]
    if len(new_words) == 0:
        return words
    return new_words


def random_swap(words, n=1):
    if len(words) <= 3:
        return words[:]
    words = words[:]
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return words


def random_duplicate(words, p=0.08):
    if len(words) <= 3:
        return words
    new_words = []
    for w in words:
        new_words.append(w)
        if random.random() < p:
            new_words.append(w)
    return new_words


def augment_one_text(text):
    text = clean_text(text)
    words = text.split()
    if len(words) < 4:
        return text

    op_choice = random.choice(["delete", "swap", "mix", "duplicate"])

    if op_choice == "delete":
        words = random_deletion(words, p=0.1)
    elif op_choice == "swap":
        words = random_swap(words, n=1)
    elif op_choice == "duplicate":
        words = random_duplicate(words, p=0.08)
    else:
        words = random_deletion(words, p=0.08)
        words = random_swap(words, n=1)

    aug_text = " ".join(words).strip()
    return aug_text if len(aug_text) >= 3 else text


def augment_row_text(row):
    row = row.copy()
    row[DESC_COL] = augment_one_text(row[DESC_COL])
    return row


# =========================================================
# 6. Dataset
# =========================================================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# =========================================================
# 7. Focal Loss
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = (1 - target_probs) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = -alpha_t * focal_factor * target_log_probs
        else:
            loss = -focal_factor * target_log_probs

        return loss.mean()


# =========================================================
# 8. Metrics
# =========================================================
def compute_metrics(y_true, y_pred, labels_for_report, target_names=None):
    acc = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true, y_pred,
        labels=labels_for_report,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels_for_report)

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


# =========================================================
# 9. Confusion matrix plot
# =========================================================
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
# 10. Train / Eval
# =========================================================
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            loss = criterion(logits, labels)

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, all_labels, all_preds


# =========================================================
# 11. Load and prepare data
# =========================================================
def map_rating_to_3class(rating):
    if rating in [1, 2]:
        return "low"
    elif rating == 3:
        return "mid"
    elif rating in [4, 5]:
        return "high"
    return None


keep_cols = [DESC_COL, VERSION_COL, THUMBS_COL, DATE_COL, LABEL_COL]

df = pd.read_csv(DATA_PATH)
df = df[keep_cols].copy()

df[DESC_COL] = df[DESC_COL].apply(clean_text)
df[VERSION_COL] = df[VERSION_COL].apply(lambda x: clean_meta_text(x, "unknown_version"))
df[DATE_COL] = df[DATE_COL].apply(lambda x: clean_meta_text(x, "unknown_date"))
df[THUMBS_COL] = df[THUMBS_COL].apply(normalize_thumbs_up)

# 评分只保留 1~5
df = df[df[LABEL_COL].isin([1, 2, 3, 4, 5])].copy()

# 正文不能为空
df = df[df[DESC_COL].notna()]
df = df[df[DESC_COL].str.strip() != ""]
df = df[df[DESC_COL].str.lower() != "nan"]

# 构建三分类标签
df["label_name"] = df[LABEL_COL].apply(map_rating_to_3class)
df = df[df["label_name"].notna()].copy()
df["label_id"] = df["label_name"].map(THREE_CLASS_TO_ID)

# 构建最终输入文本
df["input_text"] = df.apply(build_input_text, axis=1)

print("Dataset shape after cleaning:", df.shape)
print("\nFull dataset original rating distribution:")
print(df[LABEL_COL].value_counts().sort_index())

print("\nFull dataset 3-class distribution:")
print(df["label_name"].value_counts())


# 70 / 15 / 15
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

print("\nTrain distribution BEFORE augmentation:")
print(train_df["label_name"].value_counts())

print("\nValidation distribution:")
print(val_df["label_name"].value_counts())

print("\nTest distribution:")
print(test_df["label_name"].value_counts())


# =========================================================
# 12. Minority augmentation on TRAIN only
# =========================================================
if USE_MINORITY_AUGMENTATION:
    train_class_counts = train_df["label_name"].value_counts().to_dict()
    target_count = max(train_class_counts.values())
    augmented_rows = []

    for cls_name in THREE_CLASS_NAMES:
        cls_df = train_df[train_df["label_name"] == cls_name].copy()
        current_count = len(cls_df)

        if current_count < target_count and current_count > 0:
            need = target_count - current_count
            sampled_idx = np.random.choice(cls_df.index, size=need, replace=True)

            for idx in sampled_idx:
                row = cls_df.loc[idx].copy()
                row = augment_row_text(row)
                row["input_text"] = build_input_text(row)
                augmented_rows.append(row)

    if len(augmented_rows) > 0:
        aug_df = pd.DataFrame(augmented_rows)
        train_df = pd.concat([train_df, aug_df], ignore_index=True)

train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

print("\nTrain distribution AFTER minority augmentation:")
print(train_df["label_name"].value_counts())


# =========================================================
# 13. Tokenizer / Dataset / DataLoader
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = ReviewDataset(
    train_df["input_text"].tolist(),
    train_df["label_id"].tolist(),
    tokenizer,
    MAX_LEN
)

val_dataset = ReviewDataset(
    val_df["input_text"].tolist(),
    val_df["label_id"].tolist(),
    tokenizer,
    MAX_LEN
)

test_dataset = ReviewDataset(
    test_df["input_text"].tolist(),
    test_df["label_id"].tolist(),
    tokenizer,
    MAX_LEN
)

# WeightedRandomSampler
train_label_counts = Counter(train_df["label_id"].tolist())
sample_weights = [1.0 / train_label_counts[label] for label in train_df["label_id"].tolist()]
sample_weights = torch.DoubleTensor(sample_weights)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# =========================================================
# 14. Class weights for Focal Loss
# =========================================================
num_classes = 3
train_counts_for_weights = np.array([train_label_counts[i] for i in range(num_classes)], dtype=np.float32)
total_train = train_counts_for_weights.sum()

class_weights = total_train / (num_classes * train_counts_for_weights)
class_weights = class_weights / class_weights.mean()
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

print("\nClass weights used in Focal Loss:")
for i in range(num_classes):
    print(f"{ID_TO_THREE_CLASS[i]} -> weight {class_weights[i]:.4f}")


# =========================================================
# 15. Model / Optimizer / Scheduler / Loss
# =========================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

criterion = FocalLoss(alpha=class_weights_tensor, gamma=FOCAL_GAMMA)
scaler = GradScaler("cuda", enabled=torch.cuda.is_available())


# =========================================================
# 16. Training loop
# =========================================================
best_val_f1 = -1.0
best_epoch = -1
patience_counter = 0
best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")

history = []

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, criterion, DEVICE, scaler
    )

    val_loss, val_true, val_pred = evaluate(
        model, val_loader, criterion, DEVICE
    )

    val_metrics = compute_metrics(
        y_true=val_true,
        y_pred=val_pred,
        labels_for_report=[0, 1, 2],
        target_names=THREE_CLASS_NAMES
    )

    val_macro_f1 = val_metrics["f1_macro"]

    print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")
    print(f"Train Loss        : {train_loss:.4f}")
    print(f"Val Loss          : {val_loss:.4f}")
    print(f"Val Accuracy      : {val_metrics['accuracy']:.4f}")
    print(f"Val PrecisionMacro: {val_metrics['precision_macro']:.4f}")
    print(f"Val Recall Macro  : {val_metrics['recall_macro']:.4f}")
    print(f"Val F1 Macro      : {val_metrics['f1_macro']:.4f}")
    print(f"Val F1 Weighted   : {val_metrics['f1_weighted']:.4f}")

    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_metrics["accuracy"],
        "val_precision_macro": val_metrics["precision_macro"],
        "val_recall_macro": val_metrics["recall_macro"],
        "val_f1_macro": val_metrics["f1_macro"],
        "val_f1_weighted": val_metrics["f1_weighted"]
    })

    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        best_epoch = epoch
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model updated at epoch {epoch}.")
    else:
        patience_counter += 1
        print(f"No improvement. Patience {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break


# =========================================================
# 17. Load best model and test
# =========================================================
print(f"\nLoading best model from epoch {best_epoch} ...")
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))

test_loss, test_true, test_pred = evaluate(
    model, test_loader, criterion, DEVICE
)

test_metrics = compute_metrics(
    y_true=test_true,
    y_pred=test_pred,
    labels_for_report=[0, 1, 2],
    target_names=THREE_CLASS_NAMES
)

test_true_names = [ID_TO_THREE_CLASS[x] for x in test_true]
test_pred_names = [ID_TO_THREE_CLASS[x] for x in test_pred]
cm_original = confusion_matrix(test_true_names, test_pred_names, labels=THREE_CLASS_NAMES)

print("\n================ FINAL TEST RESULTS ================")
print(f"Best Epoch          : {best_epoch}")
print(f"Test Loss           : {test_loss:.4f}")
print(f"Accuracy            : {test_metrics['accuracy']:.4f}")
print(f"Precision (macro)   : {test_metrics['precision_macro']:.4f}")
print(f"Recall (macro)      : {test_metrics['recall_macro']:.4f}")
print(f"F1-score (macro)    : {test_metrics['f1_macro']:.4f}")
print(f"Precision (weighted): {test_metrics['precision_weighted']:.4f}")
print(f"Recall (weighted)   : {test_metrics['recall_weighted']:.4f}")
print(f"F1-score (weighted) : {test_metrics['f1_weighted']:.4f}")

print("\n===== Classification Report (low / mid / high) =====")
print(test_metrics["report"])

print("\n===== Confusion Matrix (low / mid / high) =====")
print(cm_original)


# =========================================================
# 18. Save outputs
# =========================================================
history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "test_results.txt"), "w", encoding="utf-8") as f:
    f.write("================ FINAL TEST RESULTS ================\n")
    f.write(f"Best Epoch          : {best_epoch}\n")
    f.write(f"Test Loss           : {test_loss:.4f}\n")
    f.write(f"Accuracy            : {test_metrics['accuracy']:.4f}\n")
    f.write(f"Precision (macro)   : {test_metrics['precision_macro']:.4f}\n")
    f.write(f"Recall (macro)      : {test_metrics['recall_macro']:.4f}\n")
    f.write(f"F1-score (macro)    : {test_metrics['f1_macro']:.4f}\n")
    f.write(f"Precision (weighted): {test_metrics['precision_weighted']:.4f}\n")
    f.write(f"Recall (weighted)   : {test_metrics['recall_weighted']:.4f}\n")
    f.write(f"F1-score (weighted) : {test_metrics['f1_weighted']:.4f}\n\n")
    f.write("===== Classification Report (low / mid / high) =====\n")
    f.write(test_metrics["report"])
    f.write("\n\n===== Confusion Matrix (low / mid / high) =====\n")
    f.write(str(cm_original))

save_confusion_matrix(
    cm_original,
    labels=THREE_CLASS_NAMES,
    save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")