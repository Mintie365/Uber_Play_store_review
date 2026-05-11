'''
==================================================
  FINAL TEST RESULTS (GloVe + BiLSTM)
==================================================
Accuracy    : 0.8244
Macro F1    : 0.5927
Weighted F1 : 0.8205

Classification Report:
              precision    recall  f1-score   support

         low     0.7330    0.8600    0.7914       150
         mid     0.1250    0.0833    0.1000        24
        high     0.9126    0.8624    0.8868       327

    accuracy                         0.8244       501
   macro avg     0.5902    0.6019    0.5927       501
weighted avg     0.8211    0.8244    0.8205       501


Confusion Matrix:
           Pred_low  Pred_mid  Pred_high
True_low        129         5         16
True_mid         11         2         11
True_high        36         9        282
'''

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score,
)
from sklearn.model_selection import train_test_split

# =========================================================
# 1. 基础配置
# =========================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "data_train_ready.csv")
TEST_PATH  = os.path.join(BASE_DIR, "data", "data_test_ready.csv")
GLOVE_PATH = os.path.join(BASE_DIR, "data", "glove.6B.100d.txt")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "best_bilstm.pt")
CURVE_PATH = os.path.join(BASE_DIR, "bilstm_training_curve.png")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM  = 100       # GloVe维度
HIDDEN_DIM = 128       # BiLSTM单向隐层维度，双向后=256
NUM_LAYERS = 2         # LSTM层数
DROPOUT    = 0.4
MAX_LEN    = 64        # 句子最大词数
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 1e-3
NUM_CLASS  = 3

print(f"使用设备: {DEVICE}")

# =========================================================
# 2. 前端：GloVe 预训练词向量加载（Embedding侧）
# =========================================================
print("\n正在加载 GloVe 预训练词向量...")

def load_glove(glove_path):
    word2vec = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            word2vec[word] = vec
    print(f"GloVe 加载完成，共 {len(word2vec)} 个词向量")
    return word2vec

glove = load_glove(GLOVE_PATH)

# =========================================================
# 3. 构建词表 & Embedding矩阵
# =========================================================
def tokenize(text):
    text = str(text).lower()
    return re.sub(r"[^a-z0-9\s]", " ", text).split()

print("\n正在加载数据...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

train_sub, val_sub = train_test_split(
    train_df, test_size=0.2, random_state=SEED, stratify=train_df["label_id"]
)

# 词表仅用 train_sub，避免验证集文本参与建表
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
word2idx  = {PAD_TOKEN: 0, UNK_TOKEN: 1}

for text in train_sub["review_description"]:
    for word in tokenize(text):
        if word not in word2idx:
            word2idx[word] = len(word2idx)

VOCAB_SIZE = len(word2idx)
print(f"词表大小（train_sub）: {VOCAB_SIZE}")

# 构建Embedding矩阵（GloVe权重）；未命中向量用固定种子，便于复现
embed_rng = np.random.default_rng(SEED)
embed_matrix = np.zeros((VOCAB_SIZE, EMBED_DIM), dtype=np.float32)
hit, miss = 0, 0
for word, idx in word2idx.items():
    if word in glove:
        embed_matrix[idx] = glove[word]
        hit += 1
    else:
        embed_matrix[idx] = embed_rng.normal(0, 0.1, EMBED_DIM).astype(np.float32)
        miss += 1

print(f"GloVe命中: {hit} | 未命中(随机初始化): {miss}")

# =========================================================
# 4. Dataset & DataLoader
# =========================================================
def encode(text, word2idx, max_len):
    tokens = tokenize(text)[:max_len]
    ids    = [word2idx.get(t, word2idx[UNK_TOKEN]) for t in tokens]
    # Padding
    ids   += [word2idx[PAD_TOKEN]] * (max_len - len(ids))
    return ids

class ReviewDataset(Dataset):
    def __init__(self, df, word2idx, max_len):
        self.X = [encode(t, word2idx, max_len)
                  for t in df["review_description"]]
        self.y = df["label_id"].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long)
        )

# 损失函数类别权重：仅按 train_sub 统计；label_id 约定 0=low, 1=mid, 2=high
_sub_y = train_sub["label_id"].to_numpy(dtype=np.int64)
counts = np.bincount(_sub_y, minlength=NUM_CLASS)
max_c = float(counts.max()) if counts.size else 1.0
mid_count = int(counts[1])
mid_w = (max_c / mid_count) if mid_count > 0 else 1.0
class_weights = torch.tensor(
    [1.0, float(mid_w), 1.0], dtype=torch.float32
).to(DEVICE)
print(
    f"\n训练子集各类样本数 low/mid/high: {counts[0]}/{counts[1]}/{counts[2]} | "
    f"Mid 类损失权重: {class_weights[1]:.3f}x"
)

train_loader = DataLoader(
    ReviewDataset(train_sub, word2idx, MAX_LEN),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    ReviewDataset(val_sub, word2idx, MAX_LEN),
    batch_size=BATCH_SIZE
)
test_loader = DataLoader(
    ReviewDataset(test_df, word2idx, MAX_LEN),
    batch_size=BATCH_SIZE
)

# =========================================================
# 5. 后端：BiLSTM 模型定义（分析模型侧）
# =========================================================
class GloveBiLSTM(nn.Module):
    def __init__(self, embed_matrix, hidden_dim, num_layers, dropout, num_class):
        super().__init__()

        vocab_size, embed_dim = embed_matrix.shape

        # Embedding层：加载GloVe权重，冻结不更新
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embed_matrix), requires_grad=False  # 冻结GloVe
        )

        # BiLSTM层：双向，两层
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        # 全连接分类层：双向所以 hidden_dim*2
        self.fc = nn.Linear(hidden_dim * 2, num_class)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)

        output, (hidden, _) = self.bilstm(embedded)
        # 取最后一层正向和反向的hidden state拼接
        # hidden: (num_layers*2, batch, hidden_dim)
        forward_h  = hidden[-2]               # 正向最后层
        backward_h = hidden[-1]               # 反向最后层
        concat     = torch.cat([forward_h, backward_h], dim=1)  # (batch, hidden*2)

        out = self.dropout(concat)
        out = self.fc(out)                    # (batch, num_class)
        return out

model = GloveBiLSTM(
    embed_matrix=embed_matrix,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    num_class=NUM_CLASS
).to(DEVICE)

print(f"\n模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# =========================================================
# 6. 训练
# =========================================================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=3, factor=0.5
)

def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

print("\n" + "="*50)
print("  开始训练 BiLSTM...")
print("="*50)

best_val_f1   = 0
best_epoch    = 0
train_f1_hist = []
val_f1_hist   = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    # 验证
    train_preds, train_labels = evaluate(train_loader)
    val_preds,   val_labels   = evaluate(val_loader)

    train_f1 = f1_score(train_labels, train_preds, average="macro")
    val_f1   = f1_score(val_labels,   val_preds,   average="macro")

    train_f1_hist.append(train_f1)
    val_f1_hist.append(val_f1)

    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Loss: {total_loss/len(train_loader):.4f} | "
          f"Train Macro F1: {train_f1:.4f} | "
          f"Val Macro F1: {val_f1:.4f}"
          + (" ← best" if epoch == best_epoch else ""))

# =========================================================
# 7. 测试集评估
# =========================================================
print(f"\n最优模型来自 Epoch {best_epoch}，验证集 Macro F1: {best_val_f1:.4f}")
model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
)

test_preds, test_labels = evaluate(test_loader)

print("\n" + "="*50)
print("  FINAL TEST RESULTS (GloVe + BiLSTM)")
print("="*50)
print(f"Accuracy    : {accuracy_score(test_labels, test_preds):.4f}")
print(f"Macro F1    : {f1_score(test_labels, test_preds, average='macro'):.4f}")
print(f"Weighted F1 : {f1_score(test_labels, test_preds, average='weighted'):.4f}")

print("\nClassification Report:")
print(classification_report(
    test_labels, test_preds,
    target_names=["low", "mid", "high"],
    digits=4
))

print("\nConfusion Matrix:")
cm = confusion_matrix(test_labels, test_preds)
print(pd.DataFrame(
    cm,
    index=["True_low", "True_mid", "True_high"],
    columns=["Pred_low", "Pred_mid", "Pred_high"]
))

# =========================================================
# 8. 可视化：训练曲线
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), train_f1_hist, label="Train Macro F1", marker="o")
plt.plot(range(1, EPOCHS+1), val_f1_hist,   label="Val Macro F1",   marker="s")
plt.axvline(x=best_epoch, color="red", linestyle="--", label=f"Best Epoch ({best_epoch})")
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("GloVe + BiLSTM Training Curve")
plt.legend()
plt.tight_layout()
plt.savefig(CURVE_PATH, dpi=150)
plt.show()
print(f"\n训练曲线已保存: {CURVE_PATH}")

print("\n全部完成！")