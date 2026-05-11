import pandas as pd
import numpy as np
import re
from pathlib import Path

# =========================
# 1. 路径配置
# =========================
INPUT_PATH = r"D:\year4sem2\dsai4205\group\data\UberCustomerReviews.csv"
OUTPUT_DIR = Path(r"D:\year4sem2\dsai4205\group\data_cleaned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOFT_OUTPUT = OUTPUT_DIR / "cleaned_reviews_soft.csv"
STRICT_OUTPUT = OUTPUT_DIR / "cleaned_reviews_strict_en.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "cleaning_summary.txt"


# =========================
# 2. 基础函数
# =========================
def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


def normalize_text(text: str) -> str:
    """
    温和文本清洗：
    1) 转字符串
    2) 去除首尾空格
    3) 小写
    4) 替换换行/制表
    5) 压缩多空格
    6) 保留标点，不激进去掉
    """
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
    """
    简单检测类似 aaaaa / !!!!! / 111111 这种明显噪声
    """
    text = safe_str(text).strip()
    if len(text) < 4:
        return False
    return re.fullmatch(r"(.)\1{3,}", text) is not None


def ascii_ratio(text: str) -> float:
    """
    英文倾向粗判断：
    返回 ASCII 字符占比
    """
    text = safe_str(text)
    if len(text) == 0:
        return 1.0
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    return ascii_count / len(text)


def contains_suspicious_encoding(text: str) -> bool:
    """
    检测常见乱码特征
    """
    text = safe_str(text)
    suspicious_patterns = [
        "Ã", "â", "ð", "Ø", "Ù", "�"
    ]
    return any(p in text for p in suspicious_patterns)


def looks_non_english_or_corrupted(text: str) -> bool:
    """
    严格版过滤条件：
    - ASCII比例过低
    - 存在常见乱码模式
    """
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
df = pd.read_csv(INPUT_PATH)

raw_rows, raw_cols = df.shape

# 统一列名，避免后续出问题
df.columns = [c.strip() for c in df.columns]

required_cols = ["review_description", "rating"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"缺少必要字段: {col}")


# =========================
# 4. 删除明显低价值字段
# =========================
drop_cols = [
    "review_title",              # 缺失率极高
    "review_id",                 # 唯一ID，无建模意义
    "user_name",                 # 噪声较大
    "developer_response",        # 不应与原始评论直接混合
    "developer_response_date",   # 同上
    "laguage_code",              # 常量列
    "country_code",              # 常量列
    "source"                     # 信息量很低，可先删
]

existing_drop_cols = [c for c in drop_cols if c in df.columns]
df_soft = df.drop(columns=existing_drop_cols).copy()


# =========================
# 5. 核心文本清洗
# =========================
df_soft["review_description_raw"] = df_soft["review_description"].copy()
df_soft["review_description"] = df_soft["review_description"].apply(normalize_text)

# 文本长度特征
df_soft["text_char_count"] = df_soft["review_description"].apply(char_count)
df_soft["text_word_count"] = df_soft["review_description"].apply(word_count)

# 噪声标记
df_soft["is_blank_text"] = df_soft["review_description"].apply(is_blank_text)
df_soft["is_only_punct_or_space"] = df_soft["review_description"].apply(is_only_punct_or_space)
df_soft["has_repeated_char_pattern"] = df_soft["review_description"].apply(has_repeated_char_pattern)
df_soft["ascii_ratio"] = df_soft["review_description"].apply(ascii_ratio)
df_soft["has_suspicious_encoding"] = df_soft["review_description_raw"].apply(contains_suspicious_encoding)

# 严格英文风险标记
df_soft["non_english_or_corrupted_risk"] = df_soft["review_description_raw"].apply(looks_non_english_or_corrupted)

# thumbs_up 缺失少，可填0
if "thumbs_up" in df_soft.columns:
    df_soft["thumbs_up"] = df_soft["thumbs_up"].fillna(0)

# appVersion 缺失较多：保留，但增加缺失标记并填 unknown
if "appVersion" in df_soft.columns:
    df_soft["appVersion_missing"] = df_soft["appVersion"].isna().astype(int)
    df_soft["appVersion"] = df_soft["appVersion"].fillna("unknown")

# 日期字段可保留原样，也可尝试转datetime
if "review_date" in df_soft.columns:
    df_soft["review_date_parsed"] = pd.to_datetime(df_soft["review_date"], errors="coerce")

# rating 转整数并校验
df_soft["rating"] = pd.to_numeric(df_soft["rating"], errors="coerce")
df_soft = df_soft[df_soft["rating"].isin([1, 2, 3, 4, 5])].copy()
df_soft["rating"] = df_soft["rating"].astype(int)


# =========================
# 6. 删除真正应删的脏样本（温和版）
# =========================
# 只删除：
# 1) 空文本
# 2) 纯标点/空格
# 3) 极明显重复字符噪声（如 aaaa, !!!!!）
before_soft_filter = len(df_soft)

df_soft = df_soft[~df_soft["is_blank_text"]].copy()
df_soft = df_soft[~df_soft["is_only_punct_or_space"]].copy()
df_soft = df_soft[~df_soft["has_repeated_char_pattern"]].copy()

after_soft_filter = len(df_soft)

# 去掉完全重复行（保险起见）
before_soft_dedup = len(df_soft)
df_soft = df_soft.drop_duplicates().copy()
after_soft_dedup = len(df_soft)

# 注意：这里不删除“相同 review_description”
# 因为评论平台里很多人写相同短句是正常现象


# =========================
# 7. 构建严格英文版
# =========================
df_strict = df_soft[~df_soft["non_english_or_corrupted_risk"]].copy()


# =========================
# 8. 输出结果
# =========================
df_soft.to_csv(SOFT_OUTPUT, index=False, encoding="utf-8-sig")
df_strict.to_csv(STRICT_OUTPUT, index=False, encoding="utf-8-sig")


# =========================
# 9. 生成摘要
# =========================
summary_lines = []

summary_lines.append("CLEANING SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Input file: {INPUT_PATH}")
summary_lines.append(f"Raw shape: {raw_rows} rows × {raw_cols} cols")
summary_lines.append("")

summary_lines.append("[1] Dropped columns")
summary_lines.append(f"Dropped columns: {existing_drop_cols}")
summary_lines.append("")

summary_lines.append("[2] Soft cleaning result")
summary_lines.append(f"Rows before soft filter: {before_soft_filter}")
summary_lines.append(f"Rows after soft filter:  {after_soft_filter}")
summary_lines.append(f"Rows before dedup:       {before_soft_dedup}")
summary_lines.append(f"Rows after dedup:        {after_soft_dedup}")
summary_lines.append(f"Soft cleaned shape:      {df_soft.shape[0]} rows × {df_soft.shape[1]} cols")
summary_lines.append("")

summary_lines.append("[3] Strict English-like result")
summary_lines.append(f"Strict cleaned shape:    {df_strict.shape[0]} rows × {df_strict.shape[1]} cols")
summary_lines.append(f"Rows removed by strict English/corruption filter: {df_soft.shape[0] - df_strict.shape[0]}")
summary_lines.append("")

summary_lines.append("[4] Noise statistics in soft cleaned data")
summary_lines.append(f"non_english_or_corrupted_risk = 1: {int(df_soft['non_english_or_corrupted_risk'].sum())}")
summary_lines.append(f"text_word_count <= 1:             {int((df_soft['text_word_count'] <= 1).sum())}")
summary_lines.append(f"text_word_count <= 3:             {int((df_soft['text_word_count'] <= 3).sum())}")
summary_lines.append("")

summary_lines.append("[5] Rating distribution (soft cleaned)")
summary_lines.append(df_soft["rating"].value_counts().sort_index().to_string())
summary_lines.append("")

summary_lines.append("[6] Rating distribution (strict cleaned)")
summary_lines.append(df_strict["rating"].value_counts().sort_index().to_string())
summary_lines.append("")

summary_lines.append("[7] Output files")
summary_lines.append(str(SOFT_OUTPUT))
summary_lines.append(str(STRICT_OUTPUT))
summary_lines.append(str(SUMMARY_OUTPUT))

with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("Cleaning finished.")
print(f"Soft cleaned file   : {SOFT_OUTPUT}")
print(f"Strict cleaned file : {STRICT_OUTPUT}")
print(f"Summary report      : {SUMMARY_OUTPUT}")
print()
print("Soft cleaned shape  :", df_soft.shape)
print("Strict cleaned shape:", df_strict.shape)