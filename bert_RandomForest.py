import os
import torch
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# ==================================================
#   FINAL TEST RESULTS (BERT Feature + RF Optuna)
# ==================================================
# Optuna Best Macro F1: 0.9190
# Accuracy: 0.8283

# Classification Report:
#               precision    recall  f1-score   support

#          low     0.7158    0.8733    0.7868       150
#          mid     0.1250    0.0417    0.0625        24
#         high     0.9129    0.8654    0.8885       327

#     accuracy                         0.8283       501
#    macro avg     0.5846    0.5935    0.5793       501
# weighted avg     0.8162    0.8283    0.8185       501


# Confusion Matrix:
#            Pred_low  Pred_mid  Pred_high
# True_low        131         0         19
# True_mid         15         1          8
# True_high        37         7        283

# =========================================================
# 1. 基础配置与路径
# =========================================================
SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "data_train_ready.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "data_test_ready.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_MODEL_NAME = 'bert-base-uncased'

# =========================================================
# 2. 前端：BERT 特征提取器 (解耦层)
# =========================================================
print(f"正在启动前端：载入 BERT 特征提取器 (Device: {DEVICE})...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
bert_model.eval()

def extract_bert_embeddings(text_list, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size)):
            batch = text_list[i : i + batch_size]
            # 处理 NaN 或非字符串
            batch = [str(t) if pd.notna(t) else "" for t in batch]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
            outputs = bert_model(**inputs)
            # 提取 [CLS] 向量作为句向量 (768维)
            batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_emb)
    return np.vstack(embeddings)

# =========================================================
# 3. 加载并转换数据
# =========================================================
train_full_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\n正在通过 BERT 提取训练集特征 (这可能需要几分钟)...")
X_train_full = extract_bert_embeddings(train_full_df["review_description"].tolist())
y_train_full = train_full_df["label_id"].values

print("\n正在通过 BERT 提取测试集特征...")
X_test = extract_bert_embeddings(test_df["review_description"].tolist())
y_test = test_df["label_id"].values

# 分出验证集供 Optuna 使用
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=SEED, stratify=y_train_full
)

# =========================================================
# 4. 后端：Optuna 自动调参 (Random Forest 优化)
# =========================================================
def objective(trial):
    # 定义搜索空间
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        # 强制包含手动加权选项，救回 Mid 类 (1)
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
    }
    
    # 针对 Mid 类 (1) 注入手动加权奖励 (暴力加权 1:15:1)
    # 你可以修改这里，如果 Optuna 选 None，我们就强行给它加权
    cw = params['class_weight']
    if cw is None:
        cw = {0: 1, 1: 50, 2: 1} # 暴力救回 Mid
    
    rf = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        class_weight=cw,
        random_state=SEED,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    val_preds = rf.predict(X_val)
    
    # 我们优化 Macro F1，这样模型必须重视 Mid 才能拿高分
    score = f1_score(y_val, val_preds, average='macro')
    return score

print("\n" + "="*50)
print(" 正在运行 Optuna 寻找后端模型最优参数...")
print("="*50)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30) # 建议跑 30-50 次

print(f"\n✅ 调优完成！最优参数: {study.best_params}")

# =========================================================
# 5. 最终训练与输出 (对齐你的格式)
# =========================================================
# 使用最优参数重新训练 (如果 best 没选加权，我们依然手动补上加权)
final_params = study.best_params
if final_params['class_weight'] is None:
    final_params['class_weight'] = {0: 1, 1: 15, 2: 1}

best_rf = RandomForestClassifier(**final_params, random_state=SEED, n_jobs=-1)
best_rf.fit(X_train_full, y_train_full)

y_pred = best_rf.predict(X_test)

print("\n" + "==================================================")
print("  FINAL TEST RESULTS (BERT Feature + RF Optuna)")
print("==================================================")
print(f"Optuna Best Macro F1: {study.best_value:.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["low", "mid", "high"], digits=4))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["True_low", "True_mid", "True_high"], 
                   columns=["Pred_low", "Pred_mid", "Pred_high"])
print(cm_df)

# =========================
# 6. 可解释性：特征重要性图
# =========================
importances = best_rf.feature_importances_
indices = np.argsort(importances)[-15:]
plt.figure(figsize=(10, 6))
plt.title("Analytic Back-end: Top 15 BERT Semantic Dimensions")
plt.barh(range(len(indices)), importances[indices], color='lightgreen', align="center")
plt.yticks(range(len(indices)), [f"Dim_{i}" for i in indices])
plt.xlabel("Gini Importance")
plt.tight_layout()
plt.show()