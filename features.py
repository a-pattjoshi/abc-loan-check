"""Derived features and the preprocessing step.

add_features() is stateless. Everything that learns from data (median fill,
one-hot, smoothed target encoding of the pincode prefix) sits in the
ColumnTransformer, so it is fitted inside each CV fold and cannot leak.
Unseen categories are safe: one-hot ignores them, the target encoder falls
back to the global rate.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, TargetEncoder

from config import ALWAYS_NUM, N_FOLDS, RANDOM_STATE, SALARY_CAP

CAT_COLS = ["Final_Tier", "Make_Code", "Product_Code", "Model_Variant", "Employment_Type",
            "PAST_LOANS_ACTIVE", "Gender", "Qualifications", "Resident_Type", "pin2", "pin3"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in CAT_COLS:
        if c in d:
            d[c] = d[c].astype(object).where(d[c].notna(), "MISSING").astype(str)
    sal = d["Net_salary"].astype(float).clip(upper=SALARY_CAP)
    ltv = d["LTV"].astype(float)
    loan = d["Loan_Amount"].astype(float)
    d["log_salary"] = np.log1p(sal)
    d["vehicle_value"] = loan / (ltv / 100.0)
    d["down_payment"] = d["vehicle_value"] - loan
    d["LTI"] = loan / sal.where(sal > 0)
    for c in ("Age", "LTV", "Loan_Amount"):
        d[c] = d[c].astype(float)
    return d


def build_preprocessor(groups: dict, names=None, scale=None) -> ColumnTransformer:
    """Preprocessor for the chosen field groups (dict name -> columns by type)."""
    use = {k: v for k, v in groups.items() if names is None or k in names}
    num = list(ALWAYS_NUM)
    onehot, te = [], []
    for g in use.values():
        num += g.get("num", [])
        onehot += g.get("onehot", [])
        te += g.get("te", [])
    num_tf = SimpleImputer(strategy="median")
    if scale == "quantile":  # for linear models: robust to the salary and LTI tails
        num_tf = Pipeline([("imp", num_tf), ("q", QuantileTransformer(
            n_quantiles=200, output_distribution="normal", random_state=RANDOM_STATE))])
    parts = [("num", num_tf, num)]
    if onehot:
        parts.append(("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                          sparse_output=False), onehot))
    if te:
        parts.append(("te", TargetEncoder(target_type="binary", smooth="auto",
                                          cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)), te))
    return ColumnTransformer(parts, remainder="drop")


def make_model(groups: dict, names=None, **hgb_kw) -> Pipeline:
    from sklearn.ensemble import HistGradientBoostingClassifier
    params = dict(max_iter=150, learning_rate=0.1, random_state=RANDOM_STATE)
    params.update(hgb_kw)
    return Pipeline([("prep", build_preprocessor(groups, names)),
                     ("clf", HistGradientBoostingClassifier(**params))])
