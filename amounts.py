"""Parse typed rupee amounts: 95000, 95,000, Rs 95000, 80k, 1.2 lakh, 2 lac, 0.5 crore."""
import math
import re

_MULT = {"k": 1e3, "l": 1e5, "lac": 1e5, "lacs": 1e5, "lakh": 1e5, "lakhs": 1e5, "cr": 1e7, "crore": 1e7, "crores": 1e7}
_SUFFIX = re.compile(r"(\d+(?:\.\d+)?)(k|lakhs?|lacs?|l|crores?|cr)")


def parse_amount(text: str):
    """Return a float, or None if the text is not an amount."""
    t = re.sub(r"[₹,\s]|rs\.?|inr", "", str(text).lower())
    m = _SUFFIX.fullmatch(t)
    if m:
        return float(m.group(1)) * _MULT[m.group(2)]
    try:
        v = float(t)
    except ValueError:
        return None
    return v if math.isfinite(v) else None  # "nan" and "inf" are not amounts
