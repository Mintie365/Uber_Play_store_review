import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# =========================
# 1. 路径配置 (子文件夹 data)
# =========================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = DATA_DIR / "UberCustomerReviews.csv"

# 清洗结果与报告
STRICT_OUTPUT = DATA_DIR / "cleaned_reviews_strict_en.csv"
SUMMARY_OUTPUT = DATA_DIR / "cleaning_summary.txt"

# 最终给模型使用的 Ready 文件
TRAIN_OUTPUT = DATA_DIR / "data_train_ready.csv"
TEST_OUTPUT = DATA_DIR / "data_test_ready.csv"


# =========================
# 2. 基础函数 (严格保留，修复了 Bug)
# =========================
def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

def normalize_text(text: str) -> str:
    text = safe_str(text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def word_count(text: str) -> int:
    text = safe_str(text).strip()
    if not text:
        return 0
    return len(text.split())

def char_count(text: str) -> int:
    return len(safe_str(text).strip())

def is_blank_text(text: str) -> bool:
    return len(safe_str(text).strip()) == 0

def is_only_punct_or_space(text: str) -> bool:
    text = safe_str(text).strip()
    if not text:
        return True
    return re.fullmatch(r"[\W_]+", text) is not None

def has_repeated_char_pattern(text: str) -> bool:
    text = safe_str(text).strip()
    if len(text) < 4:
        return False
    return re.fullmatch(r"(.)\1{3,}", text) is not None

def ascii_ratio(text: str) -> float:
    text = safe_str(text)
    if len(text) == 0:
        return 1.0
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    return ascii_count / len(text)

def contains_suspicious_encoding(text: str) -> bool:
    text = safe_str(text)
    # 【Bug 已修复】去掉了空字符串，加入了有效的乱码特征字符
    suspicious_patterns = ["Ã", "â", "ð", "Ø", "Ù", ""]
    # 加了 if p 防止再次出现空字符串匹配全宇宙的情况
    return any(p in text for p in suspicious_patterns if p)

def looks_non_english_or_corrupted(text: str) -> bool:
    text = safe_str(text).strip()
    if not text:
        return True
    ratio = ascii_ratio(text)
    if ratio < 0.60:
        return True
    if contains_suspicious_encoding(text):
        return True
    return False


# =========================
# 3. 读取数据
# =========================
if not INPUT_PATH.exists():
    raise FileNotFoundError(f"找不到原始文件，请确保 {INPUT_PATH} 存在。")

df = pd.read_csv(INPUT_PATH)
raw_rows, raw_cols = df.shape

# 统一列名
df.columns = [c.strip() for c in df.columns]

required_cols = ["review_description", "rating"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"缺少必要字段: {col}")


# =========================
# 4. 删除明显低价值字段
# =========================
drop_cols = [
    "review_title", "review_id", "user_name", "developer_response", 
    "developer_response_date", "laguage_code", "country_code", "source"
]
existing_drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=existing_drop_cols).copy()


# =========================
# 5. 核心文本清洗与特征生成
# =========================
df["review_description_raw"] = df["review_description"].copy()
df["review_description"] = df["review_description"].apply(normalize_text)

df["text_char_count"] = df["review_description"].apply(char_count)
df["text_word_count"] = df["review_description"].apply(word_count)

# 噪声标记
df["is_blank_text"] = df["review_description"].apply(is_blank_text)
df["is_only_punct_or_space"] = df["review_description"].apply(is_only_punct_or_space)
df["has_repeated_char_pattern"] = df["review_description"].apply(has_repeated_char_pattern)
df["ascii_ratio"] = df["review_description"].apply(ascii_ratio)
df["has_suspicious_encoding"] = df["review_description_raw"].apply(contains_suspicious_encoding)
df["non_english_or_corrupted_risk"] = df["review_description_raw"].apply(looks_non_english_or_corrupted)

if "thumbs_up" in df.columns:
    df["thumbs_up"] = df["thumbs_up"].fillna(0)
if "appVersion" in df.columns:
    df["appVersion_missing"] = df["appVersion"].isna().astype(int)
    df["appVersion"] = df["appVersion"].fillna("unknown")
if "review_date" in df.columns:
    df["review_date_parsed"] = pd.to_datetime(df["review_date"], errors="coerce")

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df[df["rating"].isin([1, 2, 3, 4, 5])].copy()
df["rating"] = df["rating"].astype(int)


# =========================
# 6. 删除脏样本 (仅保留 Strict 逻辑)
# =========================
before_filter = len(df)

# 执行过滤
df = df[~df["is_blank_text"]].copy()
df = df[~df["is_only_punct_or_space"]].copy()
df = df[~df["has_repeated_char_pattern"]].copy()
df = df[~df["non_english_or_corrupted_risk"]].copy()

after_filter = len(df)

before_dedup = len(df)
df = df.drop_duplicates().copy()
after_dedup = len(df)


# =========================
# 7. 输出清洗报告与数据
# =========================
df.to_csv(STRICT_OUTPUT, index=False, encoding="utf-8-sig")

summary_lines = [
    "STRICT CLEANING SUMMARY",
    "=" * 60,
    f"Input file: {INPUT_PATH}",
    f"Raw shape: {raw_rows} rows × {raw_cols} cols\n",
    "[1] Dropped columns",
    f"Dropped columns: {existing_drop_cols}\n",
    "[2] Strict cleaning result",
    f"Rows before filter: {before_filter}",
    f"Rows after filter:  {after_filter}",
    f"Rows before dedup:  {before_dedup}",
    f"Rows after dedup:   {after_dedup}",
    f"Cleaned shape:      {df.shape[0]} rows × {df.shape[1]} cols\n",
    "[3] Rating distribution (strict cleaned)",
    df["rating"].value_counts().sort_index().to_string() + "\n",
    "[4] Output files",
    str(STRICT_OUTPUT),
    str(SUMMARY_OUTPUT)
]

with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("清洗完成。")
print(f"数据量从 {raw_rows} 行减少至 {df.shape[0]} 行。")
print(f"清洗报告已生成: {SUMMARY_OUTPUT}")


# =========================================================
# 8. 切分数据集与 NLLB 回译增强
# =========================================================
print("\n" + "="*60)
print("开始执行 3分类映射、划分集合与 NLLB 数据增强...")

# 映射为 3 分类
RATING_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
df["label_id"] = df["rating"].map(RATING_MAP)
df = df.dropna(subset=["label_id", "review_description"])

if len(df) < 10:
    raise ValueError(f"严重错误：数据清洗后只剩下 {len(df)} 行，不足以进行模型训练！")

# 先划分训练集和测试集 (Train 80% / Test 20%)
train_df, test_df = train_test_split(
    df, 
    test_size=0.20, 
    random_state=42, 
    stratify=df["label_id"]
)

print(f"划分完成。训练集分布:\n{train_df['label_id'].value_counts().sort_index().to_string()}")

# 准备 NLLB-200 翻译模型
MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n正在载入 NLLB-200 模型至 {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

def back_translate(text):
    """英 -> 中 -> 英 回译"""
    try:
        inputs_cn = tokenizer(text, return_tensors="pt", truncation=True, max_length=150).to(DEVICE)
        result_cn = model.generate(**inputs_cn, forced_bos_token_id=tokenizer.lang_code_to_id["zho_Hans"], max_length=150)
        cn_text = tokenizer.batch_decode(result_cn, skip_special_tokens=True)[0]

        inputs_en = tokenizer(cn_text, return_tensors="pt", truncation=True, max_length=150).to(DEVICE)
        result_en = model.generate(**inputs_en, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=150)
        return tokenizer.batch_decode(result_en, skip_special_tokens=True)[0]
    except Exception:
        return text

# 只对训练集进行少数类增强
counts = train_df["label_id"].value_counts()
target_count = counts.max() 
final_train_list = [train_df] 

for lid in [0, 1]:
    if lid not in counts: continue
    num_to_augment = target_count - counts[lid]
    if num_to_augment > 0:
        print(f"\n正在增强训练集类别 {lid} (需生成 {num_to_augment} 条新数据)...")
        samples = train_df[train_df["label_id"] == lid].sample(n=num_to_augment, replace=True, random_state=42)
        
        aug_data = []
        for _, row in tqdm(samples.iterrows(), total=num_to_augment):
            new_text = back_translate(row["review_description"])
            new_row = row.copy() # 拷贝原来的元信息
            new_row["review_description"] = new_text
            aug_data.append(new_row)
            
        final_train_list.append(pd.DataFrame(aug_data))

# 合并增强后的训练集
train_df_final = pd.concat(final_train_list).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n[任务完成] 数据增强结束。")
print(f"增强后训练集分布:\n{train_df_final['label_id'].value_counts().sort_index().to_string()}")

# 保存 Ready 文件
train_df_final.to_csv(TRAIN_OUTPUT, index=False, encoding="utf-8-sig")
test_df.to_csv(TEST_OUTPUT, index=False, encoding="utf-8-sig")

print(f"\n最终文件已保存：")
print(f"训练集(已增强) : {TRAIN_OUTPUT}")
print(f"测试集(纯原始) : {TEST_OUTPUT}")