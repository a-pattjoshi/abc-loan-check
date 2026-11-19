"""Indian pincode checks. India Post pincodes have 6 digits; the first two digits name a postal circle.

Valid circles: 11-19, 20-28, 30-34, 36-49, 50-53, 56-64, 67-79, 80-85 (90-99 are army post offices, not places).
"""
import re

_CIRCLES = [
    (11, 11, "Delhi"), (12, 13, "Haryana"), (14, 16, "Punjab"), (17, 17, "Himachal Pradesh"), (18, 19, "Jammu and Kashmir"),
    (20, 28, "Uttar Pradesh or Uttarakhand"), (30, 34, "Rajasthan"), (36, 39, "Gujarat"), (40, 44, "Maharashtra"),
    (45, 48, "Madhya Pradesh"), (49, 49, "Chhattisgarh"), (50, 53, "Telangana or Andhra Pradesh"), (56, 59, "Karnataka"),
    (60, 64, "Tamil Nadu"), (67, 69, "Kerala"), (70, 74, "West Bengal"), (75, 77, "Odisha"), (78, 78, "Assam"),
    (79, 79, "the North East"), (80, 85, "Bihar or Jharkhand"),
]
_CITIES = {"110": "Delhi", "400": "Mumbai", "411": "Pune", "380": "Ahmedabad", "500": "Hyderabad",
           "560": "Bengaluru", "600": "Chennai", "700": "Kolkata", "302": "Jaipur", "226": "Lucknow"}


def is_indian_pincode(pin: str) -> bool:
    return bool(re.fullmatch(r"[0-9]{6}", pin)) and any(lo <= int(pin[:2]) <= hi for lo, hi, _ in _CIRCLES)


def place(pin: str) -> str:
    """City if we know it, else the state or region for the pincode's first two digits."""
    if pin[:3] in _CITIES:
        return _CITIES[pin[:3]]
    p = int(pin[:2])
    return next((n for lo, hi, n in _CIRCLES if lo <= p <= hi), "India")


def extract(raw: str):
    """Pull a 6-digit pincode out of what a person typed: "560001", "560 001", "Delhi, 110095". None if there is not exactly one."""
    digits = re.sub(r"[^0-9]", "", raw)
    if len(digits) == 6:
        return digits
    found = re.findall(r"(?<![0-9])[0-9]{6}(?![0-9])", raw)
    return found[0] if len(found) == 1 else None
