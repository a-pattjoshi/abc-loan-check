"""Structural cleaning of the raw loan file.

Imputation and scaling are NOT done here. They live inside the model pipeline
(Phase 3) so that no information leaks from the hold-out set.
Decisions: see reports/decision_log.md (D3 to D6).
"""
import re

import numpy as np
import pandas as pd

TARGET = "Decision"
MOJIBAKE = "â€“"  # en dash saved as UTF-8 and read as cp1252


def _norm_text(s: pd.Series) -> pd.Series:
    """Fix mojibake, unify dashes, spacing and case, and glue digits to CC."""
    s = s.astype("string").str.replace(MOJIBAKE, "-", regex=False)
    s = s.str.replace(r"\s*[-–]\s*", " - ", regex=True)
    s = s.str.upper().str.replace(r"(\d+)\s*CC\b", r"\1CC", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def clean_raw(raw: pd.DataFrame):
    """Return (clean_df, log_df). The log lists every fix and its row count."""
    log = []

    def note(issue, rows, action):
        log.append({"issue": issue, "rows_affected": int(rows), "action": action})

    df = raw.copy()
    obj_cols = df.select_dtypes(exclude="number").columns
    for c in obj_cols:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)

    # 1. fully empty row (its Age holds a space, so it hid from dropna)
    empty = df.isna().all(axis=1)
    df = df[~empty].copy()
    note("1 empty row", empty.sum(), "dropped")

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Pincode"] = df["Pincode"].astype("Int64")

    # 2. rows with 1 to 4 blank fields: keep, add flag (imputed later in pipeline)
    feat_cols = [c for c in df.columns if c not in ("ID", TARGET)]
    df["missing_any"] = df[feat_cols].isna().any(axis=1).astype(int)
    note("2 blank fields", df["missing_any"].sum(), "kept, missing_any flag added, impute in pipeline")

    # 3. Age over 60: keep in training, policy gate runs in the chatbot
    note("3 Age > 60", (df["Age"] > 60).sum(), "kept (D4), policy gate in chatbot")

    # 4. LTV = 0 is impossible
    ltv0 = df["LTV"] == 0
    df.loc[ltv0, "LTV"] = np.nan
    df["ltv_was_zero"] = ltv0.astype(int)
    note("4 LTV = 0", ltv0.sum(), "set to missing, ltv_was_zero flag, impute in pipeline")

    # 5. zero salary is valid for students and non-earning members
    df["no_income"] = (df["Net_salary"] == 0).astype(int)
    note("5 Net_salary = 0", df["no_income"].sum(), "kept, no_income flag added")

    # 6. salary outliers: capped later (features step), only counted here
    p995 = df["Net_salary"].quantile(0.995)
    note("6 salary outliers", (df["Net_salary"] > p995).sum(), f"above p99.5 ({p995:,.0f}); cap + log in Phase 2")

    # 7 and 8. mojibake and spelling variants
    moji = df["Model_Description"].str.contains(MOJIBAKE, regex=False, na=False)
    note("7 mojibake", moji.sum(), "fixed to '-' (D5)")
    before = df["Model_Variant"].nunique(), df["Model_Description"].nunique()
    for c in ("Model_Description", "Model_Variant"):
        df[c] = _norm_text(df[c])
    note("8 variant spelling", 0, f"normalised; Model_Variant {before[0]} -> {df['Model_Variant'].nunique()}, "
                                   f"Model_Description {before[1]} -> {df['Model_Description'].nunique()} unique")

    # 9. duplicates ignoring ID (D6): keep first; drop both rows of conflicting pairs
    key = [c for c in df.columns if c not in ("ID", "missing_any", "ltv_was_zero", "no_income")]
    key_nolabel = [c for c in key if c != TARGET]
    grp_n = df.groupby(key_nolabel, dropna=False)[TARGET].transform("nunique")
    dup_all = df.duplicated(subset=key_nolabel, keep=False)
    conflict = dup_all & (grp_n > 1) & (df["missing_any"] == 0)  # D7: rows with blanks stay
    plain_dup = df.duplicated(subset=key, keep="first") & ~conflict
    conflicts = df[conflict].copy()
    df = df[~conflict & ~plain_dup].copy()
    note("9 duplicates", plain_dup.sum(), "dropped (kept first)")
    note("9 conflicting duplicates", conflict.sum(), "dropped all rows of the pair (D6), only rows without blanks (D7), listed in log")

    # 12. pincode prefixes for later target encoding
    pin = df["Pincode"].astype("string")
    df["pin2"] = pin.str[:2]
    df["pin3"] = pin.str[:3]

    df = df.reset_index(drop=True)
    return df, pd.DataFrame(log), conflicts


def to_num(col: pd.Series) -> pd.Series:
    """Tolerant number reader for files typed by people: "25,000", "Rs. 25000", "₹25k" is NOT read (k is ambiguous),
    "30 years", "80%". Anything else that is not a number becomes NaN. Negative and infinite values become NaN."""
    t = col.astype("string").str.strip()
    t = t.str.replace(r"(?i)\b(rs\.?|inr|rupees?|yrs?|years?|%)", "", regex=True).str.replace(r"[₹,%\s]", "", regex=True)
    out = pd.to_numeric(t, errors="coerce").astype(float)
    out[~np.isfinite(out) | (out < 0)] = np.nan
    return out


def prepare_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Row-preserving version of clean_raw for scoring a new file: nothing is dropped, row order is kept.

    Same fixes as training (blank handling, LTV = 0, no-income flag, text normalising, pincode prefixes).
    Imputation, encoding and unseen categories are handled inside the saved model pipeline.
    """
    df = raw.copy()
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)
    for c in ("Age", "Pincode", "Loan_Amount", "LTV", "Net_salary"):
        df[c] = to_num(df[c])  # plain float64 so NA compares as False (text files give nullable dtypes)
    df["Pincode"] = df["Pincode"].round().astype("Int64")
    feat_cols = [c for c in df.columns if c not in ("ID", TARGET)]
    df["missing_any"] = df[feat_cols].isna().any(axis=1).astype(int)
    ltv0 = df["LTV"] == 0
    df.loc[ltv0, "LTV"] = np.nan
    df["ltv_was_zero"] = ltv0.astype(int)
    df["no_income"] = (df["Net_salary"] == 0).astype(int)
    for c in ("Model_Description", "Model_Variant"):
        df[c] = _norm_text(df[c])
    pin = df["Pincode"].astype("string")
    df["pin2"], df["pin3"] = pin.str[:2], pin.str[:3]
    return df
