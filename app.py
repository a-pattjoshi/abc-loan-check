"""ABC Credit two-wheeler loan check: a few questions, one decision.

Run:  .venv/bin/streamlit run src/app.py
The decision comes only from the trained model (predict.score). The age rule is a policy gate that
runs first. The optional wording layer (wording.py) only rephrases the explanation text.
"""
import math
import re
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st  # noqa: E402

import amounts  # noqa: E402
import explain as explain_page  # noqa: E402
import config as C  # noqa: E402
import logger  # noqa: E402
import lookups  # noqa: E402
import pincode as pin_check  # noqa: E402
import predict  # noqa: E402
import wording  # noqa: E402

st.set_page_config(page_title="ABC Credit loan check", page_icon="🛵", layout="centered")
st.markdown("<style>#MainMenu{visibility:hidden}footer{visibility:hidden}[data-testid='stToolbar']{display:none}"
            "[data-testid='stAppDeployButton']{display:none}[data-testid='stDecoration']{display:none}</style>", unsafe_allow_html=True)


def clean_free(txt: str) -> str:
    """Optional brand and model text for an unlisted vehicle: letters, digits and a few signs only, no long digit runs."""
    t = " ".join(re.sub(r"[^A-Za-z0-9 .&'/\-]", "", txt or "").split())[:60]
    return "" if re.search(r"[0-9]{7,}", t.replace(" ", "")) else t


def scroll_to_bottom():
    """Keep the newest question in view. Streamlit otherwise jumps back to the top after menu steps."""
    st.html(f"<script>/* {time.time_ns()} */ (function(){{ const go=()=>{{const m=document.querySelector('[data-testid=\"stMain\"]')||document.querySelector('section.main'); "
            "if(m){m.scrollTo({top:m.scrollHeight});}}; go(); setTimeout(go,150); setTimeout(go,500); })();</script>", unsafe_allow_javascript=True)


def inr(x) -> str:
    """Rupees with Indian digit grouping (12,34,567)."""
    s = f"{int(round(x))}"
    if len(s) <= 3:
        return "₹" + s
    head, tail = s[:-3], s[-3:]
    head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
    return f"₹{head},{tail}"


def parse_number(text: str):
    return amounts.parse_amount(text)


CONSENT_TEXT = ("Before we start: I use your answers only to check this loan and to keep a masked record of the chat "
                "(pincode shortened, income as a band). Your name and phone are optional and are stored only as one-way hashes, "
                "never as typed. Do you agree?")

QUESTIONS = {
    "name": "First, what should I call you? This is optional and it does not affect your result, so feel free to skip.",
    "age": "Let's start with the quick one. How old are you, in years? (We lend to applicants aged 18 to 60.)",
    "pin": "Please share the pincode of the area you live in (6 digits).",
    "veh": "Nice. Which two-wheeler are you planning to buy? Pick the brand, then the model. If yours is not on the list, choose \"Other brand\" or \"My model is not listed\".",
    "price": "Great. What is the on-road price of the vehicle (the final price you pay, in rupees)? Please type numbers only.",
    "loan": "Next, how much of that would you like to borrow, in rupees? (We do not lend 100% of the price.)",
    "emp": "Tell me a little about your work. What is your employment type?",
    "inc": "What is your monthly income, in rupees? A rough figure is fine. Please type numbers only.",
    "act": "Last one. Do you have any loan running right now?",
}
TEXT_STEPS = {"name": "Your name (optional)", "pin": "e.g. 560001", "price": "e.g. 95000", "loan": "e.g. 70000", "inc": "e.g. 25000", "age": "e.g. 32"}

DOCS = {
    "SAL": "last 3 salary slips and 6 months of bank statements",
    "SEP": "latest income tax return and 6 months of bank statements",
    "NPP": "appointment or probation letter, latest salary slips and 6 months of bank statements",
    "AGR": "land records and 6 months of bank statements",
    "NREGI": "proof of your income sources and 6 months of bank statements",
    "PEN": "pension slip and 6 months of bank statements",
    "STU": "admission proof and the income documents of your co-applicant",
    "NONEARNMEM": "income documents of the earning family member who applies with you",
}


@st.cache_resource
def get_lookups():
    return lookups.load()


def reset():
    for k in list(st.session_state.keys()):  # keep the sidebar mode choice; clear() would wipe it
        if k != "mode":
            del st.session_state[k]
    st.session_state.sid = logger.start_session()
    st.session_state.hold = False
    st.session_state.pending = []
    st.session_state.step = "consent"
    st.session_state.phone_saved = False
    st.session_state.a = {}
    st.session_state.hist = []
    st.session_state.msgs = [("assistant", "Hi, welcome to ABC Credit! 👋 I will ask you a few quick questions "
                                           "(usually 2 to 3 minutes, sometimes fewer) and then tell you how likely your two-wheeler loan is to be approved."),
                             ("assistant", CONSENT_TEXT)]
    st.session_state.result = None
    for _, t in st.session_state.msgs:
        logger.log_turn(st.session_state.sid, "intro", "assistant", t)


def say(role, text, log=True):
    st.session_state.msgs.append((role, text))
    if not log:
        return
    if st.session_state.hold:  # keep the user's answer ahead of the next question in the log
        st.session_state.pending.append((role, text))
    else:
        logger.log_turn(st.session_state.sid, st.session_state.step, role, text)


def go(step):
    st.session_state.step = step
    if step in QUESTIONS:
        st.session_state.hist.append((step, len(st.session_state.msgs), dict(st.session_state.a)))
        q = QUESTIONS[step]
        if step == "loan":
            q += f" (up to {inr(st.session_state.a['price'])})"
        say("assistant", q)


BACK_STEPS = ("age", "pin", "veh", "price", "loan", "emp", "inc", "act")


def can_go_back():
    h = st.session_state.get("hist", [])
    return st.session_state.step in BACK_STEPS and len(h) >= 2 and h[-2][0] in BACK_STEPS


def go_back():
    """Return to the previous question and forget the answer to it."""
    h = st.session_state.hist
    if not can_go_back():
        return
    h.pop()
    step, n, a = h.pop()
    st.session_state.msgs = st.session_state.msgs[:n]
    st.session_state.a = a
    logger.log_turn(st.session_state.sid, step, "user", "[went back]")
    go(step)


def decide_or_go(next_step):
    """Adaptive questioning. After each answer we ask the model whether the remaining questions could still change the decision.
    If every plausible answer gives the same result, we stop and say so. Otherwise we ask the next question."""
    a = st.session_state.a
    if all(k in a for k in ("age", "price", "loan", "pincode")):
        b = predict.decision_bounds(a)
        if b["certain"]:
            return finish_early(b)
    return go(next_step)


def finish_early(b):
    a = st.session_state.a
    asked = [k for k in ("age", "price", "pincode", "make", "employment", "income") if k in a]
    st.session_state.step = "done"
    st.session_state.result = {"decision": b["certain"], "probability": b["mid"], "threshold": b["threshold"], "features": b["features"],
                               "model": b["model"], "saved_at": b["saved_at"], "latency_ms": 0.0, "early": True, "lo": b["lo"], "hi": b["hi"]}
    say("assistant", "Good news: I can already give you the answer, so I will stop here. "
                     "Whatever you would answer to my remaining questions, the model's result would be the same.")
    logger.finish_session(st.session_state.sid, st.session_state.result, status="finished_early")


def reenter_answer():
    """After a policy rejection, let the applicant correct a typing mistake (the age or the loan amount) without starting over."""
    h = st.session_state.hist
    if h and h[-1][0] in ("age", "loan"):
        step, n, a = h.pop()
        st.session_state.msgs = st.session_state.msgs[:n]
        st.session_state.a = a
        st.session_state.result = None
        logger.log_turn(st.session_state.sid, step, "user", f"[re-entered {step}]")
        go(step)


def finish():
    a = st.session_state.a
    st.session_state.step = "done"
    st.session_state.result = predict.score(a)
    logger.finish_session(st.session_state.sid, st.session_state.result)


def handle_text(step, raw):
    """Run the step, then log the masked answer followed by anything the bot said."""
    st.session_state.hold, st.session_state.pending = True, []
    _handle_text(step, raw)
    st.session_state.hold = False
    ok = st.session_state.step != step
    logger.log_turn(st.session_state.sid, step, "user", logger.mask_answer(step, raw, ok), valid=ok)
    for role, text in st.session_state.pending:
        logger.log_turn(st.session_state.sid, st.session_state.step, role, text)
    st.session_state.pending = []


def _handle_text(step, raw):
    a = st.session_state.a
    say("user", raw, log=False)
    num = parse_number(raw)
    err = None
    if step == "name":
        name = " ".join(raw.split())
        if 2 <= len(name) <= 60 and all(ch.isalpha() or ch in " .'-" for ch in name):
            logger.save_applicant(st.session_state.sid, name=name)
            say("assistant", "Thanks.")
            return go("age")
        err = "Please enter a name of 2 to 60 letters, or press Skip."
    elif step == "pin":
        pin = pin_check.extract(raw) or ""
        if pin_check.is_indian_pincode(pin):
            a["pincode"] = pin
            tier, note = predict.derive_tier(pin)
            tail = (f"ABC Credit files it under its {tier[3:]} area tier (our own grouping, not a place name)." if note is None
                    else "I could not find that exact pincode, so I will use an area-level estimate.")
            say("assistant", f"Thanks. That pincode is in {pin_check.place(pin)}. {tail}")
            return decide_or_go("veh")
        err = ("Hmm, I could not read an Indian pincode from that. Just the 6 digits will do, starting with a number from 1 to 8 "
               "(for example 400001). Could you try again?")
    elif step == "price":
        lo, hi = C.PRICE_ACCEPT
        if num is not None and lo <= num <= hi:
            a["price"] = float(num)
            tlo, thi = C.TRAIN_PRICE
            if not tlo <= num <= thi:
                say("assistant", f"Noted. This price is outside the range our model learned from ({inr(tlo)} to {inr(thi)}), "
                                 "so please treat the result as less certain.")
            return go("loan")
        err = f"Please type numbers only, between {inr(lo)} and {inr(hi)} (for example 95000 or 95,000)."
    elif step == "loan":
        if num is not None and C.MIN_LOAN <= num <= a["price"]:
            a["loan"] = float(num)
            reason = predict.policy_gate_ltv(a["price"], a["loan"])
            if reason:   # hard rule, like the age rule: reject up front and stop asking
                st.session_state.step = "policy"
                st.session_state.result = {"decision": "OUT_OF_POLICY", "kind": "loan",
                                           "reason": f"You asked for {inr(a['loan'])} on a vehicle priced at {inr(a['price'])}. {reason}"}
                logger.finish_session(st.session_state.sid, st.session_state.result, status="out_of_policy")
                return
            return go("pin")
        err = f"Please type numbers only, from {inr(C.MIN_LOAN)} up to the vehicle price ({inr(a['price'])})."
    elif step == "inc":
        if num is not None and 1000 <= num <= C.INCOME_MAX:
            a["income"] = float(num)
            return decide_or_go("act")
        err = f"Please type numbers only, between {inr(1000)} and {inr(C.INCOME_MAX)} (for example 25000)."
    elif step == "age":
        t = raw.strip()
        if re.fullmatch(r"[0-9]{1,3}", t) and 1 <= int(t) <= 120:
            a["age"] = int(t)
            reason = predict.policy_gate(a["age"])
            if reason:
                st.session_state.step = "policy"
                st.session_state.result = {"decision": "OUT_OF_POLICY", "kind": "age", "reason": f"You entered age {a['age']}. {reason}"}
                logger.finish_session(st.session_state.sid, st.session_state.result, status="out_of_policy")
                return
            return go("price")
        err = "Please type your age as a whole number of years, digits only (for example 32)."
    say("assistant", err)


def handle_choice(step, value, label):
    a = st.session_state.a
    say("user", label)
    if step == "consent":
        logger.log_consent(st.session_state.sid, bool(value))
        if value:
            return go("name")
        st.session_state.step = "no_consent"
        st.session_state.result = {"decision": "NO_CONSENT"}
        logger.finish_session(st.session_state.sid, st.session_state.result, status="no_consent")
        return say("assistant", "No problem. I have not kept your answers. You can start over any time.")
    if step == "name":
        return go("age")
    if step == "emp":
        a["employment"] = value
        if value in C.NO_INCOME_TYPES:
            say("assistant", "Got it. In that case I can skip the income question.")
            return decide_or_go("act")
        return decide_or_go("inc")
    if step == "act":
        a["active_loan"] = value
        return finish()


def explain(res, a):
    """Plain facts about the application. Not the decision, just context from our history."""
    ltv = res["features"]["LTV"]
    ltv_shown = int(ltv * 10) / 10 if ltv < 100 else ltv  # round down, so 99.998% never shows as 100%
    ltv_txt = f"{ltv_shown:.0f}" if ltv_shown == int(ltv_shown) else f"{ltv_shown:.1f}"
    pts = [f"Your loan covers **{ltv_txt}%** of the vehicle price."]
    if ltv > 90:
        pts.append("Loans above 90% of the price were declined more often in our past data.")
    if not a.get("active_loan", True) and ltv > 90:
        pts.append("A high loan share with no loan history on file was declined more often than the same share with an active loan.")
    return " ".join(pts)


def clean_phone(text: str):
    """10-digit Indian mobile number or None. Accepts +91, 91 and 0 prefixes, spaces and dashes."""
    d = re.sub(r"[\s\-()]", "", text)
    d = re.sub(r"^(\+91|91|0)(?=\d{10}$)", "", d)
    return d if re.fullmatch(r"[6-9]\d{9}", d) else None


def render_phone():
    if st.session_state.get("phone_saved"):
        st.success("Thanks. Our staff may call you about next steps.")
        return
    with st.expander("Optional: ask us to call you"):
        raw = st.text_input("Mobile number (10 digits)", key="phone_in", max_chars=20)
        if st.button("Save number"):
            ph = clean_phone(raw)
            if ph is None:
                st.error("Please enter a 10-digit mobile number.")
            else:
                logger.save_applicant(st.session_state.sid, phone=ph)
                logger.log_turn(st.session_state.sid, "phone", "user", "[phone provided]")
                st.session_state.phone_saved = True
                st.rerun()


def render_result():
    res, a = st.session_state.result, st.session_state.a
    if res["decision"] == "OUT_OF_POLICY":
        st.error("### Loan rejected")
        st.write(res["reason"] + " We stopped here, so there are no more questions. This is a fixed policy rule, not a model result.")
        c1, c2 = st.columns(2)
        c1.button("I typed the wrong age" if res.get("kind", "age") == "age" else "Change the loan amount", on_click=reenter_answer, width="stretch")
        c2.button("Start over", on_click=reset, width="stretch")
        return
    if res["decision"] == "APPROVED":
        st.success("### Approved")
        st.caption("Screening result from our model. The final offer follows document verification.")
        st.write(wording.rephrase(explain(res, a)))
        st.markdown("**Next steps.** Please bring: a photo ID, address proof, two passport photos, "
                    f"and {DOCS.get(a.get('employment'), 'the income documents for your type of work (our staff will tell you which)')}.")
    else:
        st.error("### Declined")
        st.caption("Screening result from our model. Please see the options below.")
        st.write(wording.rephrase(explain(res, a)))
        facts = [] if res.get("early") else predict.explain_factors(a)
        if facts:
            st.markdown("**What pushed the score up** (compared with a typical applicant): " + "; ".join(
                f"{lab.lower() if i else lab} (about {d * 100:.0f} points of risk)" for i, (lab, d) in enumerate(facts)) + ".")
        lower = predict.suggest_lower_loan_partial(a) if res.get("early") else predict.suggest_lower_loan(a)
        if lower:
            st.markdown(f"**You can try a smaller loan.** A loan of about **{inr(lower)}** "
                        f"({100 * lower / a['price']:.0f}% of the price) is likely to be approved. "
                        f"A bigger down payment of {inr(a['price'] - lower)} makes this possible.")
        else:
            st.markdown("**Next steps.** A smaller loan alone may not change this result. Ask our staff about a co-applicant "
                        "or a different vehicle.")
        st.markdown("You can also ask our staff about adding a co-applicant. Their details would be reviewed separately.")
    render_phone()
    with st.expander("How this result was made"):
        if res.get("early"):
            st.write(f"We stopped early. Over every plausible answer to the questions we did not ask, the decline risk stays between "
                     f"**{res['lo']:.1%}** and **{res['hi']:.1%}**, and we decline when it is **{res['threshold']:.1%}** or more, "
                     "so more answers could not change the result.")
        else:
            st.write(f"Estimated decline risk: **{res['probability']:.1%}**. We decline when it is **{res['threshold']:.1%}** or more. "
                     f"The result comes from a statistical model trained on past loan decisions, checked in {res['latency_ms']:.0f} ms. "
                     "It is a screening result, not a final offer.")
    st.button("Start over", on_click=reset)


# ---------------------------------------------------------------- file scoring page
@st.cache_data(show_spinner=False, max_entries=3)
def score_upload(data: bytes):
    """Score an uploaded CSV. Cached by file content so reruns do not rescore."""
    return predict.score_frame(predict.read_any_csv(data))


def render_upload():
    st.title("🛵 ABC Credit batch scoring")
    st.caption("Upload a CSV in the training-file layout. Each row gets APPROVED or DECLINED from the same model as the chat.")
    sample = ROOT_DIR / "sample_loan_applications.csv"
    if not sample.exists():
        sample = ROOT_DIR.parent / "reports" / "sample_loan_applications.csv"
    sample_bytes = sample.read_bytes() if sample.exists() else b""
    c1, c2 = st.columns(2)
    if sample_bytes:
        c1.download_button("Download a sample file", sample_bytes, file_name="sample_loan_applications.csv", mime="text/csv",
                           help="25 invented applicants: some clear approvals, some declines, and a few messy or out-of-policy rows.")
    use_sample = c2.toggle("Score the sample file now", value=False, disabled=not sample_bytes)
    f = st.file_uploader("CSV file", type=["csv", "txt"])
    if use_sample and f is None:
        data = sample_bytes
    elif f is not None:
        data = f.getvalue()
    else:
        data = None
    if data is None:
        st.info("The model reads: Age, Pincode, vehicle (Make_Code, Product_Code, Model_Variant), LTV and Loan_Amount, Employment_Type, Net_salary, "
                "PAST_LOANS_ACTIVE. Friendly headers are understood too: Income (monthly or annual, it checks), Cost of vehicle, Loan Required, Make, Model, "
                "Occupation, Existing loan. LTV is worked out from the cost and loan when it is missing. Extra columns such as CIBIL score or ITR fields are "
                "ignored with a warning. Anything missing is filled with a typical value and reported, so the result is less reliable. "
                "Read every warning shown after the upload.")
        return
    try:
        res, info = score_upload(data)
    except predict.FileFormatError as e:
        st.error(str(e))
        return
    except Exception:  # never show a traceback to a user
        st.error("This file could not be scored. Check that it is a CSV with a header row.")
        return
    for w in info["warnings"]:
        st.warning(w)
    n_app = int((res["Decision"] == "APPROVED").sum())
    n_rej = int(info["policy_rejected"])
    n_dec = int((res["Decision"] == "DECLINED").sum()) - n_rej
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{info['rows']:,}")
    c2.metric("Approved", f"{n_app:,}")
    c3.metric("Declined", f"{n_dec:,}")
    c4.metric("Rejected by policy rule", f"{n_rej:,}")
    st.caption(f"Threshold {info['threshold']:.3f}. Two fixed policy rules come first and cannot be overruled: age outside 18 to 60, and a loan of 100% or more of the "
               "vehicle price. Those rows are DECLINED with the reason shown. Every other row gets the model's APPROVED or DECLINED. "
               "Model_Decision shows what the model alone would say.")
    if "roc_auc" in info:
        st.success(f"The file has labels ({info['labelled_rows']:,} rows). ROC-AUC {info['roc_auc']}, PR-AUC {info['pr_auc']}, "
                   f"recall on DECLINED {info['recall_declined']:.1%}.")
    cols = st.radio("Columns in the download", ["Decision and probability", "Decision only", "Probability only"], horizontal=True)
    keep = {"Decision and probability": ["ID", "Decision", "Reason", "P_DECLINED", "Model_Decision", "policy_age_ok"], "Decision only": ["ID", "Decision", "Reason"],
            "Probability only": ["ID", "P_DECLINED"]}[cols]
    out = res[keep]
    st.dataframe(out.head(50), width="stretch", hide_index=True)
    if len(out) > 50:
        st.caption(f"Showing the first 50 of {len(out):,} rows. The download has all rows.")
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(), file_name="predictions.csv", mime="text/csv")


_mode = st.sidebar.radio("Mode", ["Loan check", "Score a file", "How it works"], key="mode")
if _mode == "Score a file":
    render_upload()
    st.stop()
if _mode == "How it works":
    explain_page.render()
    st.stop()

# ---------------------------------------------------------------- page
if "step" not in st.session_state:
    reset()

st.title("🛵 ABC Credit loan check")
st.caption("Two-wheeler loans. A few questions, one answer.")

for role, text in st.session_state.msgs:
    with st.chat_message(role):
        st.markdown(text)

step = st.session_state.step
lk = get_lookups()

if step == "consent":
    with st.chat_message("assistant"):
        c1, c2 = st.columns(2)
        if c1.button("I agree", width="stretch"):
            handle_choice("consent", True, "I agree")
            st.rerun()
        if c2.button("No, thanks", width="stretch"):
            handle_choice("consent", False, "No, thanks")
            st.rerun()
elif step in TEXT_STEPS:
    if step == "name":
        with st.chat_message("assistant"):
            if st.button("Skip"):
                handle_choice("name", None, "Skip")
                st.rerun()
    raw = st.chat_input(TEXT_STEPS[step], max_chars=60)
    if raw:
        handle_text(step, raw)
        st.rerun()
elif step == "veh":
    NOT_LISTED, OTHER_BRAND = "My model is not listed", "Other brand"
    labels = {(v["desc"] if v["desc"].upper().startswith("TVS ") else "TVS " + v["desc"]): v for v in lk["vehicles"]}
    with st.chat_message("assistant"):
        brand = st.selectbox("Brand", ["TVS", OTHER_BRAND])
        if brand == OTHER_BRAND:
            free = st.text_input("Optional: type the brand and model", max_chars=60, help="For your reference only. It does not change the result.")
            st.caption("The model has not seen this vehicle, so it will lean on the price, loan, income and area instead.")
            if st.button("Confirm vehicle"):
                st.session_state.a.update(make="OTHER", product="OTHER", variant="OTHER")
                note = clean_free(free)
                say("user", OTHER_BRAND + (f": {note}" if note else ""))
                decide_or_go("emp")
                st.rerun()
        else:
            choice = st.selectbox("Model", sorted(labels) + [NOT_LISTED], index=None, placeholder="Choose a model, or type to search")
            if st.button("Confirm vehicle"):
                if choice is None:
                    st.warning("Please choose a model first.")
                else:
                    if choice == NOT_LISTED:
                        st.session_state.a.update(make="OTHER", product="OTHER", variant="OTHER")
                    else:
                        v = labels[choice]
                        st.session_state.a.update(make=v["make"], product=v["product"], variant=v["variant"])
                    say("user", f"TVS: {choice}")
                    decide_or_go("emp")
                    st.rerun()
elif step == "emp":
    with st.chat_message("assistant"):
        code = st.selectbox("Employment type", list(C.EMPLOYMENT_LABELS), format_func=C.EMPLOYMENT_LABELS.get, index=None, placeholder="Choose your employment type")
        if st.button("Confirm"):
            if code is None:
                st.warning("Please choose your employment type first.")
            else:
                handle_choice("emp", code, C.EMPLOYMENT_LABELS[code])
                st.rerun()
elif step == "act":
    with st.chat_message("assistant"):
        c1, c2 = st.columns(2)
        if c1.button("Yes, I have an active loan", width="stretch"):
            handle_choice("act", True, "Yes")
            st.rerun()
        if c2.button("No active loan", width="stretch"):
            handle_choice("act", False, "No")
            st.rerun()
elif step == "no_consent":
    st.button("Start over", on_click=reset)
elif step in ("done", "policy"):
    with st.chat_message("assistant"):
        render_result()

if can_go_back():
    st.button("← Back to the previous question", on_click=go_back)

scroll_to_bottom()

st.divider()
st.caption("Results come from a trained prediction model checked against a fixed rule for age (18 to 60). "
           "Text assistance, if switched on, only rewords the explanation and never decides.")
