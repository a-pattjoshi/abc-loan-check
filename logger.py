"""Session logging to SQLite with PII masking.

What is stored: every question and masked answer, the model's input features (without the full pincode),
the model score, threshold, decision, model version and latency. The chatbot does not ask for a name or
phone number. hash_pii() exists for any field that would carry one (salted SHA-256, never stored raw).
Invalid inputs are never stored as typed (a user may type a phone number into the pincode box).
"""
import hashlib
import json
import os
import re
import secrets
import sqlite3
import sys
import uuid
from datetime import datetime, timezone

import pandas as pd

import amounts
import config as C

DB_PATH = C.ROOT / "logs" / "sessions.db"
SALT_PATH = C.ROOT / "logs" / ".salt"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT, status TEXT,
  model_version TEXT, threshold REAL, probability REAL, decision TEXT,
  latency_ms REAL, features_json TEXT);
CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts TEXT,
  step TEXT, role TEXT, text TEXT, valid INTEGER);
CREATE TABLE IF NOT EXISTS applicants (
  session_id TEXT PRIMARY KEY, consent INTEGER, consent_ts TEXT, name_hash TEXT, phone_hash TEXT);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _db_path():
    return os.environ.get("LOAN_LOG_DB") or str(DB_PATH)  # tests point this at another file


def _conn():
    path = _db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.executescript(SCHEMA)
    return con


# ------------------------------------------------------------------ masking
def _salt() -> str:
    if os.environ.get("LOG_SALT"):
        return os.environ["LOG_SALT"]
    if not SALT_PATH.exists():
        SALT_PATH.parent.mkdir(exist_ok=True)
        SALT_PATH.write_text(secrets.token_hex(16))
        SALT_PATH.chmod(0o600)
    return SALT_PATH.read_text().strip()


def hash_pii(value: str) -> str:
    """Salted SHA-256 for names or phone numbers (first 16 hex chars)."""
    return hashlib.sha256((_salt() + str(value).strip().lower()).encode()).hexdigest()[:16]


def income_band(x) -> str:
    x = float(x)
    for hi, name in [(10_000, "under 10k"), (20_000, "10k-20k"), (30_000, "20k-30k"),
                     (50_000, "30k-50k"), (100_000, "50k-100k")]:
        if x < hi:
            return name
    return "100k+"


def mask_answer(step: str, raw: str, valid: bool) -> str:
    """What gets stored for a user's typed answer."""
    if not valid:
        return f"[invalid input, {len(raw)} chars]"
    if step in ("name", "phone"):
        return f"[{step} provided]"
    if step == "pin":
        return re.sub(r"\s", "", raw)[:3] + "***"
    if step == "inc":
        num = amounts.parse_amount(raw)
        return "[income]" if num is None else f"income band {income_band(num)}"
    return raw.strip()


def mask_text(text: str) -> str:
    """Strip long digit runs and emails before any text goes to an LLM."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]", text)
    return re.sub(r"\d[\d\s-]{5,}\d", "[number]", text)


# ------------------------------------------------------------------ writing
def start_session(model_version: str = "") -> str:
    sid = uuid.uuid4().hex
    try:
        with _conn() as con:
            con.execute("INSERT INTO sessions(session_id, started_at, status, model_version) VALUES (?,?,?,?)",
                        (sid, _now(), "in_progress", model_version))
    except Exception as e:  # logging must never break the chat
        print("logger.start_session failed:", e, file=sys.stderr)
    return sid


def log_turn(sid: str, step: str, role: str, text: str, valid: bool = True):
    try:
        with _conn() as con:
            con.execute("INSERT INTO turns(session_id, ts, step, role, text, valid) VALUES (?,?,?,?,?,?)",
                        (sid, _now(), step, role, text, int(valid)))
    except Exception as e:
        print("logger.log_turn failed:", e, file=sys.stderr)


def log_consent(sid: str, agreed: bool):
    try:
        with _conn() as con:
            con.execute("""INSERT INTO applicants(session_id, consent, consent_ts) VALUES (?,?,?)
                           ON CONFLICT(session_id) DO UPDATE SET consent=excluded.consent, consent_ts=excluded.consent_ts""",
                        (sid, int(agreed), _now()))
    except Exception as e:
        print("logger.log_consent failed:", e, file=sys.stderr)


def save_applicant(sid: str, name: str | None = None, phone: str | None = None):
    """Store only salted hashes of name and phone, never the values."""
    nh = hash_pii(name) if name else None
    ph = hash_pii(phone) if phone else None
    try:
        with _conn() as con:
            con.execute("""INSERT INTO applicants(session_id, name_hash, phone_hash) VALUES (?,?,?)
                           ON CONFLICT(session_id) DO UPDATE SET name_hash=COALESCE(excluded.name_hash, name_hash),
                                                                  phone_hash=COALESCE(excluded.phone_hash, phone_hash)""",
                        (sid, nh, ph))
    except Exception as e:
        print("logger.save_applicant failed:", e, file=sys.stderr)


def _clean_features(f: dict) -> dict:
    out = {}
    for k, v in f.items():
        if k == "Pincode":  # the full pincode is not stored; pin2, pin3 and the tier are enough to replay the model
            continue
        out[k] = None if (v is None or (isinstance(v, float) and v != v)) else (v.item() if hasattr(v, "item") else v)
    return out


def finish_session(sid: str, res: dict, status: str = "finished"):
    try:
        feats = json.dumps(_clean_features(res["features"])) if "features" in res else None
        with _conn() as con:
            con.execute("""UPDATE sessions SET finished_at=?, status=?, model_version=COALESCE(NULLIF(?, ''), model_version),
                           threshold=?, probability=?, decision=?, latency_ms=?, features_json=? WHERE session_id=?""",
                        (_now(), status, f"{res.get('model', '')} @ {res.get('saved_at', '')}".strip(" @"),
                         res.get("threshold"), res.get("probability"), res["decision"], res.get("latency_ms"), feats, sid))
    except Exception as e:
        print("logger.finish_session failed:", e, file=sys.stderr)


# ------------------------------------------------------------------ reading
def masked_view(limit: int = 50, db: str | None = None) -> pd.DataFrame:
    """Safe table for a UI or a report: pincode prefix and income band only."""
    path = db or _db_path()
    if not os.path.exists(path):
        return pd.DataFrame()
    con = sqlite3.connect(path)
    rows = con.execute("""SELECT session_id, started_at, status, decision, probability, threshold, latency_ms, features_json
                          FROM sessions ORDER BY started_at DESC LIMIT ?""", (limit,)).fetchall()
    out = []
    for sid, st, status, dec, p, thr, lat, fj in rows:
        f = json.loads(fj) if fj else {}
        out.append({"session": sid[:8], "started_at": st, "status": status, "decision": dec,
                    "risk": None if p is None else round(p, 4), "threshold": thr,
                    "pincode_prefix": f.get("pin3"), "income_band": income_band(f["Net_salary"]) if f.get("Net_salary") is not None else None,
                    "LTV": None if f.get("LTV") is None else round(f["LTV"], 1), "latency_ms": None if lat is None else round(lat, 1)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(masked_view(int(sys.argv[1]) if len(sys.argv) > 1 else 20).to_string(index=False))
