# fasttext + RandomForestClassifier 的完整实现，包含 Optuna 调参和最终评估结果展示
# 最优参数获取完成:
# {'n_estimators': 208, 'max_depth': 42, 'min_samples_split': 8, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'bootstrap': False, 'class_weight': None}
# ==================================================
#   FINAL RESULTS WITH OPTUNA OPTIMIZATION
# ==================================================
# Test Accuracy: 0.8383

# Classification Report:
#               precision    recall  f1-score   support

#          low       0.72      0.88      0.79       150
#          mid       0.25      0.08      0.12        24
#         high       0.93      0.87      0.90       327

#     accuracy                           0.84       501
#    macro avg       0.63      0.61      0.60       501
# weighted avg       0.83      0.84      0.83       501


# Confusion Matrix:
#            Pred_low  Pred_mid  Pred_high
# True_low        132         0         18
# True_mid         17         2          5
# True_high        35         6        286



import os
import re
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import FastText
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

# =========================================================
# 0. 配置与路径
# =========================================================
SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "data_train_ready.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "data_test_ready.csv")
EMBED_DIM = 100

def tokenize(text):
    text = str(text).lower()
    return re.sub(r"[^a-z0-9\s]", " ", text).split()

def get_sentence_vector(tokens, ft_model):
    vectors = [ft_model.wv[t] for t in tokens if t in ft_model.wv.key_to_index]
    if not vectors: return np.zeros(EMBED_DIM)
    return np.mean(vectors, axis=0)

# =========================================================
# 1. 加载数据与特征提取 (Embedding 阶段)
# =========================================================
print("正在载入增强数据并训练 Embedding Model...")
train_full_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 分出验证集供 Optuna 使用
train_df, val_df = train_test_split(
    train_full_df, test_size=0.2, random_state=SEED, stratify=train_full_df["label_id"]
)

train_df["tokens"] = train_df["review_description"].apply(tokenize)
val_df["tokens"] = val_df["review_description"].apply(tokenize)
test_df["tokens"] = test_df["review_description"].apply(tokenize)

# 训练 FastText (作为独立的 Embedding Model)
ft_model = FastText(vector_size=EMBED_DIM, window=5, min_count=2, sg=1, seed=SEED)
ft_model.build_vocab(corpus_iterable=train_df["tokens"].tolist())
ft_model.train(corpus_iterable=train_df["tokens"].tolist(), total_examples=len(train_df), epochs=20)

def transform_to_vectors(df):
    X = np.vstack([get_sentence_vector(t, ft_model) for t in df["tokens"]])
    return X

X_train = transform_to_vectors(train_df)
X_val = transform_to_vectors(val_df)
X_test = transform_to_vectors(test_df)
y_train, y_val, y_test = train_df["label_id"], val_df["label_id"], test_df["label_id"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 2. Optuna 调参 (Analytic Model 阶段)
# =========================================================
def objective(trial):
    # 定义随机森林的搜索空间
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
    }
    
    model = RandomForestClassifier(**params, random_state=SEED, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_val_scaled)
    # 核心：优化 Macro F1 以照顾少数类
    score = f1_score(y_val, preds, average='macro')
    return score

print("\n开始 Optuna 贝叶斯超参数优化...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50) # 跑50次实验

print("\n最优参数获取完成:")
print(study.best_params)

# =========================================================
# 3. 最终模型训练与评估
# =========================================================
best_rf = RandomForestClassifier(**study.best_params, random_state=SEED, n_jobs=-1)
best_rf.fit(X_train_scaled, y_train)

y_pred = best_rf.predict(X_test_scaled)

print("\n" + "="*50)
print("  FINAL RESULTS WITH OPTUNA OPTIMIZATION")
print("="*50)
print(f"Test Accuracy: {best_rf.score(X_test_scaled, y_test):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["low", "mid", "high"]))

# 可解释性图表
importances = best_rf.feature_importances_
indices = np.argsort(importances)[-15:]
plt.figure(figsize=(10, 6))
plt.title("Optimized Analytic Model: Top 15 Feature Importance")
plt.barh(range(len(indices)), importances[indices], color='lightgreen', align="center")
plt.yticks(range(len(indices)), [f"Dim_{i}" for i in indices])
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(pd.DataFrame(cm, index=["True_low", "True_mid", "True_high"], 
                   columns=["Pred_low", "Pred_mid", "Pred_high"]))