"""Shared constants. Field groups are what the ablation drops one at a time."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # flat layout: code and model files sit in one folder
RAW_CSV = ROOT.parent / "training_data.csv"  # read-only
CLEAN_PARQUET = ROOT / "data" / "clean.parquet"
SPLIT_CSV = ROOT / "data" / "holdout_ids.csv"
MODEL_PATH = ROOT / "model.joblib"

RANDOM_STATE = 42
HOLDOUT_SIZE = 0.20
N_FOLDS = 5
N_JOBS = 2
TARGET = "Decision"
POSITIVE = "DECLINED"

SALARY_CAP = 100_000  # p99.5 of Net_salary on the full file (issue 6)
POLICY_AGE = (18, 60)
POLICY_MAX_LTV = 100.0  # hard rule: a loan of 100% or more of the vehicle price is rejected up front (no applicant in the data had LTV above 99.6)

# Always-on flags from cleaning (not part of the ablation)
ALWAYS_NUM = ["missing_any"]

# The 7 chatbot questions map to these groups (CLAUDE.md design table)
GROUPS = {
    "pincode": {"onehot": ["Final_Tier"], "te": ["pin2", "pin3"]},
    "vehicle": {"onehot": ["Make_Code", "Product_Code", "Model_Variant"]},
    "ltv": {"num": ["LTV", "ltv_was_zero"]},
    "loan": {"num": ["Loan_Amount", "vehicle_value", "down_payment"]},
    "employment": {"onehot": ["Employment_Type"]},
    "income": {"num": ["log_salary", "no_income", "LTI"]},
    "age": {"num": ["Age"]},
    "past_loans": {"onehot": ["PAST_LOANS_ACTIVE"]},
}
# Nearly flat fields: tested by adding each to the full set
EXTRA_GROUPS = {
    "gender": {"onehot": ["Gender"]},
    "qualifications": {"onehot": ["Qualifications"]},
    "resident_type": {"onehot": ["Resident_Type"]},
}

# ---- Chatbot (Phase 4) ----
LOOKUPS_PATH = ROOT / "lookups.json"
PRICE_RANGE = (30_000, 500_000)      # on-road price, rupees (training data: 50k to 363k)
PRICE_ACCEPT = (10_000, 2_000_000)   # chat accepts this wide range; outside the training range it adds a caution
TRAIN_PRICE = (50_000, 363_000)      # price range the model saw in training
INCOME_MAX = 1_000_000               # monthly, rupees
MIN_LOAN = 10_000
# Optional wording layer (Groq). Off by default. Key goes in loan-chatbot/.env as GROQ_API_KEY (AP adds it).
USE_LLM = False
GROQ_MODEL = "llama-3.3-70b-versatile"  # AP can change; model name lives only here

EMPLOYMENT_LABELS = {  # code -> label shown to the user (meanings from the project brief, section 5)
    "SAL": "Salaried", "SEP": "Self-employed", "NPP": "Non-permanent probationary",
    "AGR": "Agriculture / farming", "NREGI": "Non-regular income", "PEN": "Pensioner",
    "STU": "Student", "NONEARNMEM": "Non-earning family member",
}
NO_INCOME_TYPES = ("STU", "NONEARNMEM")  # skip the income question, income = 0
MAKE_LABELS = {"TVS": "TVS iQube (electric)", "ZEST": "Scooty Zest", "SCOOTY": "Scooty",
               "MOPEDS": "Mopeds (XL series)", "SPORT": "Sport", "CITY PLUS": "Star City Plus"}
