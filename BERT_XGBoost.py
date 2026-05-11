'''
==================================================
  FINAL TEST RESULTS (BERT Feature + XGBoost Optuna)
==================================================
Optuna 验证目标得分   : 0.9097
Accuracy             : 0.8363
Macro F1             : 0.6205
Weighted F1          : 0.8307

Classification Report:
              precision    recall  f1-score   support

         low     0.7283    0.8933    0.8024       150
         mid     0.2500    0.1250    0.1667        24
        high     0.9246    0.8624    0.8924       327

    accuracy                         0.8363       501
   macro avg     0.6343    0.6269    0.6205       501
weighted avg     0.8335    0.8363    0.8307       501


Confusion Matrix:
           Pred_low  Pred_mid  Pred_high
True_low        134         1         15
True_mid         13         3          8
True_high        37         8        282

'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)

# =========================================================
# 1. 基础配置与路径
# =========================================================
SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "data_train_ready.csv")
TEST_PATH  = os.path.join(BASE_DIR, "data", "data_test_ready.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_MODEL_NAME = "bert-base-uncased"
print(f"使用设备: {DEVICE}")

# =========================================================
# 2. 前端：BERT 特征提取器（解耦层）
# =========================================================
print(f"\n正在启动前端：载入 BERT 特征提取器 (Device: {DEVICE})...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
bert_model.eval()

def extract_bert_embeddings(text_list, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size)):
            batch = text_list[i: i + batch_size]
            batch = [str(t) if pd.notna(t) else "" for t in batch]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(DEVICE)
            outputs = bert_model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    return np.vstack(embeddings)

# =========================================================
# 3. 加载数据并提取 BERT 向量
# =========================================================
print("\n正在加载数据...")
train_full_df = pd.read_csv(TRAIN_PATH)
test_df       = pd.read_csv(TEST_PATH)

print("\n正在通过 BERT 提取训练集特征（这可能需要几分钟）...")
X_train_full = extract_bert_embeddings(train_full_df["review_description"].tolist())
y_train_full = train_full_df["label_id"].values

print("\n正在通过 BERT 提取测试集特征...")
X_test = extract_bert_embeddings(test_df["review_description"].tolist())
y_test = test_df["label_id"].values

# 分出验证集供 Optuna 使用
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=SEED,
    stratify=y_train_full
)

print(f"\nEmbedding 提取完成！向量维度: {X_train_full.shape}")
print(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")

# =========================================================
# 4. 后端：Optuna 自动调参（XGBoost）
# =========================================================
# 验证集优化目标：宏 F1 + high 类 F1 加权，避免纯 macro 时 mid_weight 过大、大量 high→mid
VAL_OBJECTIVE_MACRO_WEIGHT = 0.55
VAL_OBJECTIVE_HIGH_F1_WEIGHT = 0.45


def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 150, 600),
        "max_depth":        trial.suggest_int("max_depth", 4, 12),
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "subsample":        trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 3.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "max_delta_step":   trial.suggest_int("max_delta_step", 0, 5),
    }

    # 1 = 不对 mid 额外加权；上限略降，减轻过判 mid
    mid_weight = trial.suggest_int("mid_weight", 1, 18)
    sample_weights = np.where(y_train == 1, mid_weight, 1).astype(float)

    model = XGBClassifier(
        **params,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    preds = model.predict(X_val)

    f1_macro = f1_score(y_val, preds, average="macro")
    f1_per = f1_score(y_val, preds, average=None, labels=[0, 1, 2], zero_division=0)
    f1_high = float(f1_per[2])

    return (
        VAL_OBJECTIVE_MACRO_WEIGHT * f1_macro
        + VAL_OBJECTIVE_HIGH_F1_WEIGHT * f1_high
    )

print("\n" + "="*50)
print("  正在运行 Optuna 寻找后端模型最优参数...")
print("="*50)
print(
    f"  验证目标: {VAL_OBJECTIVE_MACRO_WEIGHT:.0%}·macro_F1 + "
    f"{VAL_OBJECTIVE_HIGH_F1_WEIGHT:.0%}·F1_high"
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"\n调优完成！最优验证目标得分: {study.best_value:.4f}")
print(f"最优参数: {study.best_params}")

# 用最优参数在 train 上重训一次，汇报验证集 macro F1 / F1_high（便于和 RF 等指标对齐）
_report_params = study.best_params.copy()
_mid_w = _report_params.pop("mid_weight")
_sw = np.where(y_train == 1, _mid_w, 1).astype(float)
_val_report = XGBClassifier(
    **_report_params,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
)
_val_report.fit(X_train, y_train, sample_weight=_sw, verbose=False)
_val_pred = _val_report.predict(X_val)
_f1m = f1_score(y_val, _val_pred, average="macro")
_f1h = f1_score(y_val, _val_pred, average=None, labels=[0, 1, 2], zero_division=0)[2]
print(f"（同参数）验证集 Macro F1: {_f1m:.4f}  |  F1_high: {float(_f1h):.4f}")

# =========================================================
# 5. 最终训练与评估
# =========================================================
# 把 mid_weight 从 best_params 里单独取出来
best_params = study.best_params.copy()
mid_weight_best = best_params.pop("mid_weight")
sample_weights_full = np.where(y_train_full == 1, mid_weight_best, 1).astype(float)

best_xgb = XGBClassifier(
    **best_params,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
)
best_xgb.fit(X_train_full, y_train_full, sample_weight=sample_weights_full)

y_pred = best_xgb.predict(X_test)

print("\n" + "="*50)
print("  FINAL TEST RESULTS (BERT Feature + XGBoost Optuna)")
print("="*50)
print(f"Optuna 验证目标得分   : {study.best_value:.4f}")
print(f"Accuracy             : {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro F1             : {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Weighted F1          : {f1_score(y_test, y_pred, average='weighted'):.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["low", "mid", "high"],
    digits=4
))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(
    cm,
    index=["True_low", "True_mid", "True_high"],
    columns=["Pred_low", "Pred_mid", "Pred_high"]
))

# =========================================================
# 6. 可解释性：Feature Importance
# =========================================================
importances = best_xgb.feature_importances_
indices = np.argsort(importances)[-15:]

plt.figure(figsize=(10, 6))
plt.title("Analytic Back-end: Top 15 BERT Semantic Dimensions (XGBoost)")
plt.barh(range(len(indices)), importances[indices], color="steelblue", align="center")
plt.yticks(range(len(indices)), [f"Dim_{i}" for i in indices])
plt.xlabel("XGBoost Feature Importance Score")
plt.tight_layout()
plt.show()

print("\n全部完成！")