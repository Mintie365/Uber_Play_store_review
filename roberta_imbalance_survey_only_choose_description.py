# ================ FINAL TEST RESULTS ================
# Best Epoch          : 4
# Test Loss           : 0.5643
# Accuracy            : 0.6223
# Precision (macro)   : 0.4081
# Recall (macro)      : 0.4010
# F1-score (macro)    : 0.3864
# Precision (weighted): 0.7367
# Recall (weighted)   : 0.6223
# F1-score (weighted) : 0.6624

# ===== Classification Report (label ids 0~4) =====
#               precision    recall  f1-score   support

#            0     0.7636    0.8660    0.8116        97
#            1     0.1667    0.0667    0.0952        15
#            2     0.0545    0.1667    0.0822        18
#            3     0.1538    0.2667    0.1951        30
#            4     0.9020    0.6389    0.7480       216

#     accuracy                         0.6223       376
#    macro avg     0.4081    0.4010    0.3864       376
# weighted avg     0.7367    0.6223    0.6624       376


# ===== Confusion Matrix (ratings 1~5) =====
# [[ 84   3   4   2   4]
#  [ 11   1   1   2   0]
#  [  7   1   3   7   0]
#  [  2   1   8   8  11]
#  [  6   0  39  33 138]]


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

TITLE_COL = "review_title"
DESC_COL = "review_description"
VERSION_COL = "appVersion"
COUNTRY_COL = "country_code"
LABEL_COL = "rating"

MODEL_NAME = "roberta-base"   # 显存不够可改成 "distilroberta-base" 或 "bert-base-uncased"
MAX_LEN = 192
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 2

USE_MINORITY_AUGMENTATION = True
MINORITY_CLASSES = [2, 3, 4]
AUG_TARGET_REFERENCE_CLASS = 1
FOCAL_GAMMA = 2.0

OUTPUT_DIR = "survey_roberta_4fields_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


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
# 3. Text cleaning
# =========================================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_meta_text(text, unknown_token="unknown"):
    text = clean_text(text)
    return text if text else unknown_token


# =========================================================
# 4. Multi-field text builder
# =========================================================
def build_input_text(title, desc, version, country):
    title = clean_text(title)
    desc = clean_text(desc)
    version = clean_meta_text(version, "unknown_version")
    country = clean_meta_text(country, "unknown_country")

    if not title:
        title = "no_title"
    if not desc:
        desc = "no_review"

    combined = (
        f"Title: {title}. "
        f"AppVersion: {version}. "
        f"Country: {country}. "
        f"Review: {desc}"
    )
    return combined


# =========================================================
# 5. Light augmentation for minority classes
#    只增强标题和正文，不增强版本号/国家码
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
    if random.random() < 0.7:
        row[TITLE_COL] = augment_one_text(row[TITLE_COL])
    if random.random() < 0.9:
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
def compute_metrics(y_true, y_pred, labels_for_report):
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
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

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
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

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
df = pd.read_csv(DATA_PATH)

keep_cols = [TITLE_COL, DESC_COL, VERSION_COL, COUNTRY_COL, LABEL_COL]
df = df[keep_cols].copy()

for col in [TITLE_COL, DESC_COL, VERSION_COL, COUNTRY_COL]:
    df[col] = df[col].apply(clean_text)

# 评分只保留 1~5
df = df[df[LABEL_COL].isin([1, 2, 3, 4, 5])].copy()

# 至少正文不能空；标题可以空
df = df[df[DESC_COL].notna()]
df = df[df[DESC_COL].str.strip() != ""]
df = df[df[DESC_COL].str.lower() != "nan"]

# 构建最终输入文本
df["input_text"] = df.apply(
    lambda r: build_input_text(
        r[TITLE_COL], r[DESC_COL], r[VERSION_COL], r[COUNTRY_COL]
    ),
    axis=1
)

print("Dataset shape after cleaning:", df.shape)
print("\nFull dataset class distribution:")
print(df[LABEL_COL].value_counts().sort_index())

# 标签映射：1~5 -> 0~4
label2id = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
id2label = {v: k for k, v in label2id.items()}
df["label_id"] = df[LABEL_COL].map(label2id)

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
print(train_df[LABEL_COL].value_counts().sort_index())
print("\nValidation distribution:")
print(val_df[LABEL_COL].value_counts().sort_index())
print("\nTest distribution:")
print(test_df[LABEL_COL].value_counts().sort_index())


# =========================================================
# 12. Minority augmentation on TRAIN only
# =========================================================
if USE_MINORITY_AUGMENTATION:
    target_count = train_df[train_df[LABEL_COL] == AUG_TARGET_REFERENCE_CLASS].shape[0]
    augmented_rows = []

    for cls in MINORITY_CLASSES:
        cls_df = train_df[train_df[LABEL_COL] == cls].copy()
        current_count = len(cls_df)

        if current_count < target_count:
            need = target_count - current_count
            sampled_idx = np.random.choice(cls_df.index, size=need, replace=True)

            for idx in sampled_idx:
                row = cls_df.loc[idx].copy()
                row = augment_row_text(row)
                row["input_text"] = build_input_text(
                    row[TITLE_COL], row[DESC_COL], row[VERSION_COL], row[COUNTRY_COL]
                )
                augmented_rows.append(row)

    if len(augmented_rows) > 0:
        aug_df = pd.DataFrame(augmented_rows)
        train_df = pd.concat([train_df, aug_df], ignore_index=True)

train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

print("\nTrain distribution AFTER minority augmentation:")
print(train_df[LABEL_COL].value_counts().sort_index())


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
num_classes = 5
train_counts_for_weights = np.array([train_label_counts[i] for i in range(num_classes)], dtype=np.float32)
total_train = train_counts_for_weights.sum()

class_weights = total_train / (num_classes * train_counts_for_weights)
class_weights = class_weights / class_weights.mean()
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

print("\nClass weights used in Focal Loss:")
for i in range(num_classes):
    print(f"Rating {id2label[i]} -> weight {class_weights[i]:.4f}")


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
        labels_for_report=[0, 1, 2, 3, 4]
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
    labels_for_report=[0, 1, 2, 3, 4]
)

# 转回原始评分 1~5
test_true_original = [id2label[x] for x in test_true]
test_pred_original = [id2label[x] for x in test_pred]
cm_original = confusion_matrix(test_true_original, test_pred_original, labels=[1, 2, 3, 4, 5])

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

print("\n===== Classification Report (label ids 0~4) =====")
print(test_metrics["report"])

print("\n===== Confusion Matrix (ratings 1~5) =====")
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
    f.write("===== Classification Report =====\n")
    f.write(test_metrics["report"])
    f.write("\n\n===== Confusion Matrix (ratings 1~5) =====\n")
    f.write(str(cm_original))

save_confusion_matrix(
    cm_original,
    labels=[1, 2, 3, 4, 5],
    save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")


