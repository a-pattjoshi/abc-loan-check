"""Score one application. The decision comes ONLY from the model probability vs the threshold.

The age 18 to 60 check is a policy rule that runs before the model.
"""
import json
import time
from functools import lru_cache

import joblib
import pandas as pd

import config as C
import features as F
import lookups

MODEL_VERSION = None


@lru_cache(maxsize=1)
def _load():
    model = joblib.load(C.MODEL_PATH)
    meta = json.loads((C.MODEL_PATH.parent / "metadata.json").read_text())
    return model, meta, lookups.load()


def explain_factors(a: dict, top: int = 3, min_delta: float = 0.01) -> list:
    """Which answers pushed the risk score up compared with a typical applicant. Explanation only, never the decision.

    For each answer we swap in a neutral value (loan at 60% of the price, an unseen area, an unlisted vehicle, income 25,000,
    age 35, an active loan) and see how much the score falls. Returns [(label, drop_in_risk), ...] largest first.
    """
    base = score(a)["probability"]
    tests = [("Your loan is a high share of the vehicle price", {"loan": a["price"] * 0.6}),
             ("Your area (pincode)", {"pincode": "999999"}),
             ("The vehicle you chose", {"make": "OTHER", "product": "OTHER", "variant": "OTHER"}),
             ("Your age", {"age": 35})]
    if a["employment"] not in C.NO_INCOME_TYPES:
        tests.append(("Your income", {"income": 25000.0}))
    if not a["active_loan"]:
        tests.append(("Having no loan history on file", {"active_loan": True}))
    out = []
    for label, change in tests:
        drop = base - score({**a, **change})["probability"]
        if drop >= min_delta:
            out.append((label, drop))
    return sorted(out, key=lambda t: -t[1])[:top]


def policy_gate(age: int):
    """Return None if allowed, else the reason text."""
    lo, hi = C.POLICY_AGE
    if not lo <= age <= hi:
        return f"Applicants must be between {lo} and {hi} years old."
    return None


def derive_tier(pincode: str):
    """(tier, note). Unknown pincodes fall back to the 3-digit area, then to MISSING."""
    lk = _load()[2]
    if pincode in lk["pin_tier"]:
        return lk["pin_tier"][pincode], None
    if pincode[:3] in lk["pin3_tier"]:
        return lk["pin3_tier"][pincode[:3]], "area estimate"
    return "MISSING", "unknown area"


def build_row(a: dict) -> pd.DataFrame:
    """a: pincode, make, product, variant, price, loan, employment, income, age, active_loan (bool)."""
    tier, _ = derive_tier(a["pincode"])
    income = 0.0 if a["employment"] in C.NO_INCOME_TYPES else float(a["income"])
    row = {
        "Pincode": int(a["pincode"]), "Final_Tier": tier, "pin2": a["pincode"][:2], "pin3": a["pincode"][:3],
        "Make_Code": a["make"], "Product_Code": a["product"], "Model_Variant": a["variant"],
        "LTV": 100.0 * a["loan"] / a["price"], "ltv_was_zero": 0, "Loan_Amount": float(a["loan"]),
        "Employment_Type": a["employment"], "Net_salary": income, "no_income": int(income == 0),
        "Age": float(a["age"]),
        "PAST_LOANS_ACTIVE": "PAST_LOANS_ACTIVE" if a["active_loan"] else "NO_PAST_LOANS",
        "missing_any": 0, "Gender": None, "Qualifications": None, "Resident_Type": None,
    }
    return F.add_features(pd.DataFrame([row]))


def score(a: dict) -> dict:
    model, meta, _ = _load()
    t0 = time.perf_counter()
    X = build_row(a)
    p = float(model.predict_proba(X)[:, 1][0])
    thr = float(meta["threshold"])
    return {"probability": p, "threshold": thr, "decision": "DECLINED" if p >= thr else "APPROVED",
            "features": X.iloc[0].to_dict(), "model": meta["model"], "saved_at": meta["saved_at"],
            "latency_ms": (time.perf_counter() - t0) * 1000}


INCOME_GRID = (1000.0, 8000.0, 15000.0, 25000.0, 40000.0, 70000.0, 150000.0, 1000000.0)


def decision_bounds(a: dict) -> dict:
    """Lowest and highest risk score over every plausible answer to the questions that are not asked yet.

    `a` holds the answers so far: age, price, loan and pincode are required; make/product/variant, employment, income and
    active_loan are optional. Missing answers are replaced by every vehicle in the menu, every employment type, a grid of
    incomes and both active-loan answers. If even the highest score stays below the cut-off the decision is certainly
    APPROVED; if even the lowest is at or above it the decision is certainly DECLINED. The decision still comes only from
    the model: asking more questions could not change it.
    """
    model, meta, lk = _load()
    thr = float(meta["threshold"])
    tier, _ = derive_tier(a["pincode"])
    vehicles = ([(a["make"], a["product"], a["variant"])] if "make" in a else
                sorted({(v["make"], v["product"], v["variant"]) for v in lk["vehicles"]} | {("OTHER", "OTHER", "OTHER")}))
    emps = [a["employment"]] if "employment" in a else list(C.EMPLOYMENT_LABELS)
    rows = []
    for make, product, variant in vehicles:
        for emp in emps:
            if emp in C.NO_INCOME_TYPES:
                incomes = [0.0]
            else:
                incomes = [float(a["income"])] if "income" in a else list(INCOME_GRID)
            for inc in incomes:
                for act in ([bool(a["active_loan"])] if "active_loan" in a else [False, True]):
                    rows.append({
                        "Pincode": int(a["pincode"]), "Final_Tier": tier, "pin2": a["pincode"][:2], "pin3": a["pincode"][:3],
                        "Make_Code": make, "Product_Code": product, "Model_Variant": variant,
                        "LTV": 100.0 * a["loan"] / a["price"], "ltv_was_zero": 0, "Loan_Amount": float(a["loan"]),
                        "Employment_Type": emp, "Net_salary": inc, "no_income": int(inc == 0), "Age": float(a["age"]),
                        "PAST_LOANS_ACTIVE": "PAST_LOANS_ACTIVE" if act else "NO_PAST_LOANS",
                        "missing_any": 0, "Gender": None, "Qualifications": None, "Resident_Type": None})
    X = F.add_features(pd.DataFrame(rows))
    p = model.predict_proba(X)[:, 1]
    mid_i = int(abs(p - float(pd.Series(p).median())).argmin())
    lo, hi = float(p.min()), float(p.max())
    certain = "DECLINED" if lo >= thr else "APPROVED" if hi < thr else None
    return {"lo": lo, "hi": hi, "mid": float(p[mid_i]), "threshold": thr, "certain": certain, "combinations": len(rows),
            "features": X.iloc[mid_i].to_dict(), "model": meta["model"], "saved_at": meta["saved_at"]}


def suggest_lower_loan_partial(a: dict, step=0.05, floor=0.5):
    """Like suggest_lower_loan, for an early stop: the largest loan after which the decision is certainly APPROVED."""
    loan = a["loan"]
    k = 1
    while loan * (1 - step * k) >= max(C.MIN_LOAN, loan * floor):
        trial = {**a, "loan": round(loan * (1 - step * k), -2)}
        if decision_bounds(trial)["certain"] == "APPROVED":
            return trial["loan"]
        k += 1
    return None


def suggest_lower_loan(a: dict, step=0.05, floor=0.5):
    """Largest loan (in 5% steps down from the request, not below 50% of it) that the model approves."""
    loan = a["loan"]
    k = 1
    while loan * (1 - step * k) >= max(C.MIN_LOAN, loan * floor):
        trial = {**a, "loan": round(loan * (1 - step * k), -2)}
        if score(trial)["decision"] == "APPROVED":
            return trial["loan"]
        k += 1
    return None


# ---------------------------------------------------------------- batch scoring of a file
RAW_COLS = ["ID", "Qualifications", "Employment_Type", "Age", "Pincode", "Gender", "Product_Code", "Loan_Amount", "LTV",
            "Resident_Type", "Net_salary", "Final_Tier", "Make_Code", "Model_Description", "Model_Variant",
            "PAST_LOANS_ACTIVE", "Decision"]
MODEL_COLS = ["Pincode", "Make_Code", "Product_Code", "Model_Variant", "LTV", "Loan_Amount", "Employment_Type",
              "Net_salary", "Age", "PAST_LOANS_ACTIVE"]  # columns the model reads; missing ones are imputed (and warned)
UPPER_COLS = ["Employment_Type", "Make_Code", "Product_Code", "PAST_LOANS_ACTIVE"]


class FileFormatError(ValueError):
    """The uploaded file cannot be read as a table. The message is safe to show to a user."""


def read_any_csv(src) -> pd.DataFrame:
    """Read a CSV from a path or bytes. Tolerates BOM, cp1252/latin-1, comma/semicolon/tab/pipe, header case and spaces.

    Everything is read as text; numbers are converted (and bad ones blanked) later in clean.prepare_rows.
    """
    import csv
    import io
    from pathlib import Path
    raw = src if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
    text = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None or not text.strip():
        raise FileFormatError("The file is empty.")
    try:
        sep = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, sep=sep, engine="python", on_bad_lines="skip", skip_blank_lines=False)
    except Exception as e:  # pandas raises several types for unreadable text
        raise FileFormatError(f"The file could not be read as a table ({type(e).__name__}).") from None
    canon = {c.lower(): c for c in RAW_COLS}
    names = [canon.get(str(c).strip().lower().replace(" ", "_"), str(c).strip()) for c in df.columns]
    df.columns = names
    return df.loc[:, ~pd.Index(names).duplicated()]


def score_frame(raw: pd.DataFrame, threshold=None):
    """Score a table in the training-file layout. Returns (result_df, info).

    One output row per input row, same order. info has rows, threshold, declined_share, warnings, missing_columns
    and, when a Decision column is present with both classes, roc_auc / pr_auc / recall_declined.
    """
    import clean
    model, meta, lk = _load()
    thr = float(meta["threshold"] if threshold is None else threshold)
    warnings, n = [], len(raw)
    info = {"rows": n, "threshold": thr, "warnings": warnings}
    if n == 0:
        raise FileFormatError("The file has a header but no rows.")
    raw = raw.copy()
    missing = [c for c in MODEL_COLS if c not in raw.columns]
    info["missing_columns"] = missing
    if missing:
        warnings.append("Columns not in the file, filled with typical values: " + ", ".join(missing))
    extra = [c for c in raw.columns if c not in RAW_COLS]
    info["ignored_columns"] = extra
    if extra:
        warnings.append("Columns the model was not trained on were ignored (for example a credit score or tax return field): "
                        + ", ".join(str(c) for c in extra[:12]) + (" ..." if len(extra) > 12 else "")
                        + ". They do not change any prediction.")
    for c in RAW_COLS:
        if c not in raw.columns:
            raw[c] = pd.NA
    for c in UPPER_COLS:  # categories are upper case in training; "sal" must mean "SAL"
        raw[c] = raw[c].astype("string").str.strip().str.upper().replace("", pd.NA)
    yes = {"YES", "Y", "TRUE", "1", "ACTIVE", "PAST_LOANS_ACTIVE"}
    no = {"NO", "N", "FALSE", "0", "NONE", "NO_PAST_LOANS", "NIL"}
    raw["PAST_LOANS_ACTIVE"] = raw["PAST_LOANS_ACTIVE"].map(
        lambda v: pd.NA if pd.isna(v) else ("PAST_LOANS_ACTIVE" if v in yes else "NO_PAST_LOANS" if v in no else v)).astype("string")
    by_label = {lab.upper(): code for code, lab in C.EMPLOYMENT_LABELS.items()}
    raw["Employment_Type"] = raw["Employment_Type"].map(lambda v: pd.NA if pd.isna(v) else by_label.get(v, v)).astype("string")
    for col, ok, nice in (("Employment_Type", set(C.EMPLOYMENT_LABELS), "employment type"),
                          ("PAST_LOANS_ACTIVE", {"PAST_LOANS_ACTIVE", "NO_PAST_LOANS"}, "active-loan value")):
        odd = int((raw[col].notna() & ~raw[col].isin(ok)).sum())
        if odd:
            warnings.append(f"{odd} rows have an {nice} not seen in training ({col}). They are scored with the typical profile.")
    tiers = {v.upper(): v for v in set(lk["pin_tier"].values())}
    raw["Final_Tier"] = raw["Final_Tier"].astype("string").str.strip().str.upper().map(tiers).astype("string")
    prep = clean.prepare_rows(raw)
    blank = prep["Final_Tier"].isna() & prep["Pincode"].notna()
    prep.loc[blank, "Final_Tier"] = prep.loc[blank, "Pincode"].astype(str).map(
        lambda p: lk["pin_tier"].get(p) or lk["pin3_tier"].get(p[:3]))
    ltv = prep["LTV"].dropna()
    if len(ltv) and ltv.max() <= 1.5:
        warnings.append("LTV values are all 1.5 or below. The model expects a percentage (for example 80, not 0.8). "
                        "Scores are not reliable until this is fixed.")
    bad = {c: int((raw[c].notna() & clean.to_num(raw[c]).isna()).sum()) for c in ("Age", "Loan_Amount", "LTV", "Net_salary")}
    bad = {c: k for c, k in bad.items() if k > 0}
    if bad:
        warnings.append("Values that are not valid positive numbers were treated as blank: " + ", ".join(f"{c} ({k})" for c, k in bad.items()))
    known_makes = {v["make"] for v in lk["vehicles"]}
    unseen = int((prep["Make_Code"].notna() & ~prep["Make_Code"].isin(known_makes)).sum())
    if unseen:
        warnings.append(f"{unseen} rows have a make not seen in training. They are scored with the typical profile.")
    X = F.add_features(prep)
    p = model.predict_proba(X)[:, 1]
    ids = raw["ID"] if raw["ID"].notna().any() else pd.Series(range(1, n + 1))
    ids_num = pd.to_numeric(ids, errors="coerce")
    if ids_num.notna().all() and (ids_num % 1 == 0).all():
        ids = ids_num.astype("Int64")
    res = pd.DataFrame({"ID": ids.to_numpy()})
    res["Decision"] = pd.Series(p >= thr).map({True: C.POSITIVE, False: "APPROVED"}).to_numpy()
    res["P_DECLINED"] = p.round(6)
    age = prep["Age"]
    res["policy_age_ok"] = (age.between(*C.POLICY_AGE) | age.isna()).to_numpy()
    info["declined_share"] = float((p >= thr).mean())
    info["policy_age_flagged"] = int((~res["policy_age_ok"]).sum())
    if C.TARGET in raw.columns:
        lab = raw[C.TARGET].astype("string").str.strip().str.upper()
        known = lab.isin([C.POSITIVE, "APPROVED"]).fillna(False).to_numpy(dtype=bool)
        y = (lab == C.POSITIVE).fillna(False).astype(int).to_numpy()
        if known.sum() and len(set(y[known])) == 2:
            from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
            info.update(roc_auc=round(roc_auc_score(y[known], p[known]), 4), pr_auc=round(average_precision_score(y[known], p[known]), 4),
                        recall_declined=round(recall_score(y[known], (p >= thr)[known]), 4), labelled_rows=int(known.sum()))
    return res, info


def predict_file(inp, out, fmt="both", threshold=None):
    """Score a CSV file and write the result. fmt: 'label', 'prob' or 'both' (ID is always first).

    The decision is the model probability against the saved threshold. The age 18 to 60 rule is reported in
    the policy_age_ok column but does not change the label (D18).
    """
    res, info = score_frame(read_any_csv(inp), threshold)
    cols = ["ID"] + (["Decision"] if fmt in ("label", "both") else []) + (["P_DECLINED"] if fmt in ("prob", "both") else []) + ["policy_age_ok"]
    res[cols].to_csv(out, index=False)
    return info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Score a CSV file with the saved model.")
    ap.add_argument("input")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--format", choices=["label", "prob", "both"], default="both")
    ap.add_argument("--threshold", type=float, default=None, help="override the saved threshold")
    args = ap.parse_args()
    try:
        print(predict_file(args.input, args.out, args.format, args.threshold))
    except FileFormatError as e:
        raise SystemExit(f"ERROR: {e}")
