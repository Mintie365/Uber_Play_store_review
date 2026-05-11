# pip install gensim


# ==================================================
#   FINAL TEST RESULTS (LLM Augmented + FastText)
# ==================================================
# 正在寻找最佳 SVM 参数 C (当前手动权重: {0: 1, 1: 1, 2: 1})...
# ✅ 调优完成！选择最佳参数 C=0.1，对应的验证集 Macro-F1=0.7938
# Best SVM C: 0.1
# Accuracy: 0.7385

# Classification Report:
#               precision    recall  f1-score   support

#          low       0.68      0.77      0.72       150
#          mid       0.09      0.25      0.14        24
#         high       0.93      0.76      0.84       327

#     accuracy                           0.74       501
#    macro avg       0.57      0.59      0.57       501
# weighted avg       0.82      0.74      0.77       501


# Confusion Matrix:
#            Pred_low  Pred_mid  Pred_high
# True_low        116        23         11
# True_mid         11         6          7
# True_high        44        35        248

# Confusion Matrix:
#            Pred_low  Pred_mid  Pred_high
# True_low        121        17         12
# True_mid         13         3          8
# True_high        46        31        250



# ==================================================
#   FINAL TEST RESULTS (LLM Augmented + FastText)
# ==================================================
# Best SVM C: 0.01
# Accuracy: 0.7166
# 正在寻找最佳 SVM 参数 C (当前手动权重: {0: 1, 1: 1.5, 2: 1})...
# ✅ 调优完成！选择最佳参数 C=0.01，对应的验证集 Macro-F1=0.7867

# Classification Report:
#               precision    recall  f1-score   support

#          low       0.65      0.73      0.69       150
#          mid       0.07      0.21      0.11        24
#         high       0.93      0.75      0.83       327

#     accuracy                           0.72       501
#    macro avg       0.55      0.56      0.54       501
# weighted avg       0.80      0.72      0.75       501


# Confusion Matrix:
#            Pred_low  Pred_mid  Pred_high
# True_low        109        30         11
# True_mid         11         5          8
# True_high        48        34        245
import os
import re
import warnings
import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =========================================================
# 1. 路径设置 (指向你的 data 文件夹)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "data_train_ready.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "data_test_ready.csv")

SEED = 42
EMBED_DIM = 100

# =========================================================
# 2. 核心工具 (仅保留分词和向量化，这是模型必须的步骤)
# =========================================================
def tokenize(text):
    """FastText 需要列表格式的输入"""
    return str(text).lower().split()

def get_sentence_vector(tokens, ft_model):
    """将分词列表转为 100 维向量"""
    vectors = [ft_model.wv[t] for t in tokens if t in ft_model.wv.key_to_index]
    if not vectors:
        return np.zeros(EMBED_DIM)
    return np.mean(vectors, axis=0)

# =========================================================
# 3. 加载已经处理好的数据
# =========================================================
print(f"正在加载处理好的数据...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 将训练集切出一部分做验证集，用来选最好的 C 参数
train_sub, val_sub = train_test_split(
    train_df, test_size=0.2, random_state=SEED, stratify=train_df["label_id"]
)

# 分词
train_sub["tokens"] = train_sub["review_description"].apply(tokenize)
val_sub["tokens"] = val_sub["review_description"].apply(tokenize)
test_df["tokens"] = test_df["review_description"].apply(tokenize)

# =========================================================
# 4. 训练 FastText (学习词向量)
# =========================================================
print("正在训练 FastText 嵌入层...")
ft_model = FastText(vector_size=EMBED_DIM, window=5, min_count=2, sg=1, seed=SEED)
ft_model.build_vocab(corpus_iterable=train_sub["tokens"].tolist())
ft_model.train(corpus_iterable=train_sub["tokens"].tolist(), total_examples=len(train_sub), epochs=20)

# =========================================================
# 5. 向量化
# =========================================================
X_train = np.vstack([get_sentence_vector(t, ft_model) for t in train_sub["tokens"]])
X_val = np.vstack([get_sentence_vector(t, ft_model) for t in val_sub["tokens"]])
X_test = np.vstack([get_sentence_vector(t, ft_model) for t in test_df["tokens"]])

y_train, y_val, y_test = train_sub["label_id"], val_sub["label_id"], test_df["label_id"]

# SVM 必须做标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# =========================================================
# 6. 训练 LinearSVC (手动加权版)
# =========================
# 在这里定义你的“暴力权重”：{类别ID: 权重倍数}
# 0: low, 1: mid (我们需要救回来的类), 2: high
# 建议尝试：5, 10, 15，甚至 20
# 如果设为 15，意味着模型分错一个 mid 的代价是分错一个 high 的 15 倍！
custom_weights = {0: 1, 1: 1, 2: 1} 

print(f"正在寻找最佳 SVM 参数 C (当前手动权重: {custom_weights})...")
best_f1, best_c, best_clf = -1, 0, None

# 依然保留 C 的循环，因为权重变化后，正则化强度也需要重新匹配
for c in [0.01, 0.1, 1, 5]:
    # 将原来的 class_weight="balanced" 替换为我们的字典
    clf = LinearSVC(
        C=c, 
        class_weight=custom_weights, 
        random_state=SEED, 
        max_iter=10000 # 增加权重后模型更难收敛，建议调大迭代次数
    )
    clf.fit(X_train, y_train)
    
    # 在验证集上评估
    val_pred = clf.predict(X_val)
    # 这里我们优化 Macro F1，它能迫使模型平衡三个类的表现
    report = classification_report(y_val, val_pred, output_dict=True, zero_division=0)
    f1 = report['macro avg']['f1-score']
    
    # 打印每个 C 下的 mid 类表现，方便你观察
    mid_recall = report['1']['recall'] if '1' in report else 0
    print(f"Testing C={c:4}: Macro-F1={f1:.4f}, Mid-Recall={mid_recall:.4f}")
    
    if f1 > best_f1:
        best_f1, best_c, best_clf = f1, c, clf

print(f"\n✅ 调优完成！选择最佳参数 C={best_c}，对应的验证集 Macro-F1={best_f1:.4f}")
# =========================================================
# 7. 最终测试结果
# =========================================================
y_pred = best_clf.predict(X_test)

print("\n" + "="*50)
print("  FINAL TEST RESULTS (LLM Augmented + FastText)")
print("="*50)
print(f"Best SVM C: {best_c}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["low", "mid", "high"]))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=["True_low", "True_mid", "True_high"], 
                   columns=["Pred_low", "Pred_mid", "Pred_high"]))