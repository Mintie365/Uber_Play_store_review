import pandas as pd
import numpy as np
import re
from collections import Counter

# Optional language detection
USE_LANGDETECT = True
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False
    USE_LANGDETECT = False

INPUT_PATH = r"D:\year4sem2\dsai4205\group\data\UberCustomerReviews.csv"
OUTPUT_REPORT_PATH = r"D:\year4sem2\dsai4205\group\data\dataset_issue_audit_report.txt"


def safe_detect_lang(text):
    if not USE_LANGDETECT or not LANGDETECT_AVAILABLE:
        return "unknown"
    try:
        text = str(text).strip()
        if len(text) < 3:
            return "too_short"
        return detect(text)
    except Exception:
        return "error"


def is_blank_text(text):
    return str(text).strip() == ""


def basic_text_stats(text):
    text = str(text)
    words = text.split()
    return {
        "char_len": len(text),
        "word_len": len(words),
        "has_url": int(bool(re.search(r"http\S+|www\.\S+", text))),
        "has_email": int(bool(re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", text))),
        "has_html": int(bool(re.search(r"<.*?>", text))),
        "has_non_ascii": int(any(ord(c) > 127 for c in text)),
        "has_digit": int(bool(re.search(r"\d", text))),
        "has_only_punct_or_space": int(bool(re.fullmatch(r"[\W_]+", text.strip()))) if text.strip() else 0,
        "has_repeated_char_pattern": int(bool(re.search(r"(.)\1{4,}", text.lower()))),  # e.g. gooooood
    }


def looks_like_noise(text):
    text = str(text).strip()
    if text == "":
        return True
    # only punctuation/symbols
    if re.fullmatch(r"[\W_]+", text):
        return True
    # too short and not alphabetic
    if len(text) <= 2 and not re.search(r"[a-zA-Z]", text):
        return True
    return False


def ascii_english_like_ratio(text):
    """
    Rough heuristic: proportion of ASCII letters among non-space chars.
    Not a real language detector, but useful as a quick signal.
    """
    text = str(text)
    non_space = [c for c in text if not c.isspace()]
    if len(non_space) == 0:
        return np.nan
    ascii_letters = sum(1 for c in non_space if ('a' <= c.lower() <= 'z'))
    return ascii_letters / len(non_space)


def main():
    df = pd.read_csv(INPUT_PATH)

    report = []
    report.append("DATASET ISSUE AUDIT REPORT")
    report.append("=" * 70)
    report.append(f"Input file: {INPUT_PATH}")
    report.append("This script only audits the dataset and does NOT clean or modify it.")
    report.append("")

    # --------------------------------------------------
    # 1. Basic structure
    # --------------------------------------------------
    report.append("[1] BASIC STRUCTURE")
    report.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    report.append("Columns:")
    for col in df.columns:
        report.append(f"  - {col} ({df[col].dtype})")
    report.append("")

    # --------------------------------------------------
    # 2. Missing values
    # --------------------------------------------------
    report.append("[2] MISSING VALUE ANALYSIS")
    missing_counts = df.isnull().sum().sort_values(ascending=False)
    total_missing = int(missing_counts.sum())
    report.append(f"Total missing cells: {total_missing}")
    for col, cnt in missing_counts.items():
        pct = cnt / len(df) * 100
        report.append(f"  - {col}: {cnt} ({pct:.2f}%)")
    report.append("")

    # --------------------------------------------------
    # 3. Duplicate analysis
    # --------------------------------------------------
    report.append("[3] DUPLICATE ANALYSIS")
    full_dups = int(df.duplicated().sum())
    report.append(f"Fully duplicated rows: {full_dups}")

    if "review_id" in df.columns:
        dup_review_id = int(df.duplicated(subset=["review_id"]).sum())
        report.append(f"Duplicated review_id rows: {dup_review_id}")

    if "review_description" in df.columns:
        dup_review_text = int(df.duplicated(subset=["review_description"]).sum())
        report.append(f"Duplicated review_description rows: {dup_review_text}")

    combo_cols = [c for c in ["user_name", "review_description", "review_date"] if c in df.columns]
    if len(combo_cols) >= 2:
        dup_combo = int(df.duplicated(subset=combo_cols).sum())
        report.append(f"Duplicated rows by {combo_cols}: {dup_combo}")
    report.append("")

    # --------------------------------------------------
    # 4. Rating / target-label issues
    # --------------------------------------------------
    report.append("[4] TARGET LABEL (RATING) ANALYSIS")
    if "rating" in df.columns:
        rating_num = pd.to_numeric(df["rating"], errors="coerce")
        invalid_rating = int(((~rating_num.between(1, 5)) | rating_num.isna()).sum())
        report.append(f"Invalid or missing rating rows: {invalid_rating}")

        rating_dist = rating_num.value_counts(dropna=False).sort_index()
        report.append("Rating distribution:")
        for rating, cnt in rating_dist.items():
            pct = cnt / len(df) * 100
            report.append(f"  - {rating}: {cnt} ({pct:.2f}%)")

        valid_rating = rating_num.dropna()
        if len(valid_rating) > 0:
            vc = valid_rating.value_counts()
            imbalance_ratio = vc.max() / vc.min() if vc.min() > 0 else np.inf
            report.append(f"Imbalance ratio (largest class / smallest class): {imbalance_ratio:.2f}")
            report.append("Interpretation: higher ratio means stronger class imbalance.")
    else:
        report.append("Column 'rating' not found.")
    report.append("")

    # --------------------------------------------------
    # 5. Constant / low-information columns
    # --------------------------------------------------
    report.append("[5] CONSTANT OR LOW-INFORMATION COLUMNS")
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        top_values = df[col].value_counts(dropna=False).head(3).to_dict()
        report.append(f"  - {col}: unique_non_null={nunique}, top_values={top_values}")
    report.append("")

    # --------------------------------------------------
    # 6. Text field diagnostics
    # --------------------------------------------------
    report.append("[6] TEXT FIELD DIAGNOSTICS")
    text_col = None
    for candidate in ["review_description", "review_text", "content", "review"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        report.append("No obvious text column found.")
        report.append("")
    else:
        report.append(f"Detected text column: {text_col}")

        txt = df[text_col].fillna("").astype(str)

        blank_count = int(txt.apply(is_blank_text).sum())
        noise_count = int(txt.apply(looks_like_noise).sum())

        text_stats_df = txt.apply(basic_text_stats).apply(pd.Series)
        text_stats_df["ascii_english_like_ratio"] = txt.apply(ascii_english_like_ratio)

        char_len = text_stats_df["char_len"]
        word_len = text_stats_df["word_len"]

        report.append(f"Blank text rows: {blank_count}")
        report.append(f"Rows that look like pure noise: {noise_count}")
        report.append(f"Average char length: {char_len.mean():.2f}")
        report.append(f"Median char length: {char_len.median():.2f}")
        report.append(f"Average word length: {word_len.mean():.2f}")
        report.append(f"Median word length: {word_len.median():.2f}")

        # Very short / very long
        very_short = int((word_len <= 1).sum())
        q1 = word_len.quantile(0.25)
        q3 = word_len.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        long_outliers = int((word_len > upper_bound).sum())

        report.append(f"Very short reviews (<=1 word): {very_short}")
        report.append(f"Long-text outliers by IQR: {long_outliers} (threshold > {upper_bound:.2f} words)")

        # Text pattern issues
        for col in ["has_url", "has_email", "has_html", "has_non_ascii", "has_digit",
                    "has_only_punct_or_space", "has_repeated_char_pattern"]:
            count = int(text_stats_df[col].sum())
            pct = count / len(df) * 100
            report.append(f"{col}: {count} ({pct:.2f}%)")

        # English / non-English risk
        report.append("")
        report.append("Language risk checks:")
        non_ascii_count = int(text_stats_df["has_non_ascii"].sum())
        report.append(f"Rows containing non-ASCII characters: {non_ascii_count}")

        # Heuristic
        low_ascii_ratio = int((text_stats_df["ascii_english_like_ratio"] < 0.5).sum())
        report.append(f"Rows with low ASCII-English-like ratio (<0.5): {low_ascii_ratio}")

        # langdetect sample-based full audit
        if USE_LANGDETECT and LANGDETECT_AVAILABLE:
            report.append("Running language detection on all rows. This may take some time...")
            df["_detected_lang"] = txt.apply(safe_detect_lang)
            lang_counts = df["_detected_lang"].value_counts(dropna=False)
            report.append("Detected language distribution:")
            for lang, cnt in lang_counts.items():
                pct = cnt / len(df) * 100
                report.append(f"  - {lang}: {cnt} ({pct:.2f}%)")
        else:
            report.append("langdetect is not installed. Language detection skipped.")
            report.append("Install with: pip install langdetect")

        # Top suspicious examples
        report.append("")
        report.append("Examples of suspicious / potentially non-English / noisy reviews:")
        suspicious_mask = (
            (text_stats_df["has_non_ascii"] == 1) |
            (text_stats_df["has_url"] == 1) |
            (text_stats_df["has_only_punct_or_space"] == 1) |
            (text_stats_df["ascii_english_like_ratio"] < 0.5)
        )
        suspicious_examples = txt[suspicious_mask].head(10).tolist()
        if suspicious_examples:
            for i, ex in enumerate(suspicious_examples, 1):
                report.append(f"  {i}. {ex[:200]}")
        else:
            report.append("  None found in first scan.")
        report.append("")

    # --------------------------------------------------
    # 7. Date / version / metadata issues
    # --------------------------------------------------
    report.append("[7] METADATA FIELD DIAGNOSTICS")

    if "review_date" in df.columns:
        review_date_parsed = pd.to_datetime(df["review_date"], errors="coerce")
        invalid_dates = int(review_date_parsed.isna().sum())
        report.append(f"Unparseable review_date rows: {invalid_dates}")

    if "developer_response_date" in df.columns:
        dev_date_parsed = pd.to_datetime(df["developer_response_date"], errors="coerce")
        invalid_dev_dates = int(dev_date_parsed.isna().sum())
        report.append(f"Unparseable developer_response_date rows (including missing/non-date text): {invalid_dev_dates}")

    if "appVersion" in df.columns:
        missing_app_version = int(df["appVersion"].isna().sum())
        unique_versions = int(df["appVersion"].nunique(dropna=True))
        report.append(f"Missing appVersion rows: {missing_app_version}")
        report.append(f"Unique appVersion values: {unique_versions}")
        report.append(f"Top 10 appVersion values: {df['appVersion'].value_counts(dropna=False).head(10).to_dict()}")
    report.append("")

    # --------------------------------------------------
    # 8. Potential leakage / modelling concern
    # --------------------------------------------------
    report.append("[8] MODELLING RISK CHECKS")
    report.append("Potential modelling concerns to consider:")
    report.append("  - Class imbalance in rating")
    report.append("  - Non-English or mixed-language reviews")
    report.append("  - Very short reviews that may be hard to classify")
    report.append("  - Long complaint-style reviews that may dominate vocabulary")
    report.append("  - developer_response may be useful as a feature, but should not be merged into the user review text blindly")
    report.append("  - review_id and user_name should not be used as text features")
    report.append("")

    # --------------------------------------------------
    # 9. Recommended next actions
    # --------------------------------------------------
    report.append("[9] RECOMMENDED NEXT ACTIONS (DO NOT CLEAN YET)")
    report.append("Recommended workflow:")
    report.append("  1. Confirm whether non-English reviews are frequent enough to require filtering or special handling.")
    report.append("  2. Check whether very short reviews are truly informative before dropping anything.")
    report.append("  3. Decide whether review_title should be dropped based on missingness and usefulness.")
    report.append("  4. Keep class imbalance as a modelling issue, not a cleaning issue.")
    report.append("  5. After diagnosis, design a task-aware cleaning plan instead of applying a blind standard pipeline.")
    report.append("")

    # Save report
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        for line in report:
            f.write(line + "\n")

    print("Audit completed.")
    print("Report saved to:")
    print(OUTPUT_REPORT_PATH)


if __name__ == "__main__":
    main()