"""Read typed rupee amounts. Numbers only: 95000, 95,000, 9,50,000, 95000.50, optionally with a leading Rs, INR or the rupee sign.

Commas and spaces are removed. Letters ("80k", "1.2 lakh", "abc"), signs, exponents ("1e5") and anything else are refused.
"""
import math
import re

_PLAIN = re.compile(r"[0-9][0-9,\s]*(\.[0-9]{1,2})?")


def parse_amount(text: str):
    """Return a float, or None if the text is not a plain number."""
    t = str(text).strip()
    if not t or len(t) > 20:
        return None
    t = re.sub(r"(?i)^(₹|rs\.?|inr)\s*", "", t)
    if not _PLAIN.fullmatch(t):
        return None
    v = float(re.sub(r"[,\s]", "", t))
    return v if math.isfinite(v) else None
