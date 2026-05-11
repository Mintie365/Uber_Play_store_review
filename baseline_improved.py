import os
import pandas as pd
import numpy as np
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import warnings
warnings.filterwarnings("ignore")

# =========================
# 1. 配置与路径加载
# =========================
SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 请确保这里的路径正确
TRAIN_PATH = os.path.join(BASE_DIR, "data", "data_train_ready.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "data_test_ready.csv")

print("正在加载数据集...")
train_full_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_full_df["review_description"] = train_full_df["review_description"].fillna("").astype(str)
test_df["review_description"] = test_df["review_description"].fillna("").astype(str)

y_train_full = train_full_df["label_id"].values
y_test = test_df["label_id"].values

# =========================
# 2. 改进的 TF-IDF 特征提取
# =========================
print("正在构建改进版 TF-IDF (1-2 grams + sublinear_tf) 特征矩阵...")
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2), 
    min_df=2,
    max_features=20000,
    sublinear_tf=True  # 【核心改进】：对数缩放词频，大幅缓解长短文本差异带来的方差问题！
)

X_train_tfidf = tfidf.fit_transform(train_full_df["review_description"])
X_test_tfidf = tfidf.transform(test_df["review_description"])

# =========================
# 3. 交叉验证 Optuna 调参 (拒绝过拟合)
# =========================
def objective(trial):
    # 只搜索最核心的正则化参数 C
    c_val = trial.suggest_float("C", 0.1, 10.0, log=True)
    
    clf = LogisticRegression(
        C=c_val,
        class_weight="balanced", # 保持严谨的数学比例平衡
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED
    )
    
    # 【核心改进】：使用 5 折交叉验证算 Macro F1，防止模型在单一验证集上作弊过拟合
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X_train_tfidf, y_train_full, cv=cv, scoring="f1_macro", n_jobs=-1)
    
    return scores.mean()

print("\n" + "="*50)
print(" 开始运行严格的 5-Fold CV Optuna 优化...")
print("="*50)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")

# 强制 Optuna 第一个尝试默认参数 C=1.0，确保它至少能找到不差于 Baseline 的结果
study.enqueue_trial({"C": 1.0})
study.optimize(objective, n_trials=30)

print(f"\n✅ 调优完成！5-Fold CV 平均最高 Macro F1: {study.best_value:.4f}")
print(f"最优参数: {study.best_params}")

# =========================
# 4. 最终测试集评估
# =========================
best_c = study.best_params["C"]

best_clf = LogisticRegression(
    C=best_c,
    class_weight="balanced",
    solver="lbfgs",
    max_iter=2000,
    random_state=SEED
)

best_clf.fit(X_train_tfidf, y_train_full)
y_pred = best_clf.predict(X_test_tfidf)

print("\n" + "="*50)
print("  FINAL TEST RESULTS (Robust Optimized Baseline)")
print("="*50)
print(f"Accuracy              : {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro F1              : {f1_score(y_test, y_pred, average='macro'):.4f}")

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred, target_names=["low", "mid", "high"], digits=4))

print("\n===== Confusion Matrix =====")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm, 
    index=["True_low", "True_mid", "True_high"], 
    columns=["Pred_low", "Pred_mid", "Pred_high"]
)
print(cm_df)