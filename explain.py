"""The "How it works" page: a plain-English walk through the data, the model, the threshold, the privacy rules,
and a live what-if tool that calls the real model. Written for someone who has never seen the project.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config as C
import predict

ROOT = Path(__file__).resolve().parent


def _find(name: str):
    """Find a data file or figure in the project layout (reports/, reports/figures/) or the flat deploy layout."""
    for base in (ROOT.parent / "reports" / "figures", ROOT.parent / "reports", ROOT):
        if (base / name).exists():
            return base / name
    return None


def _csv(name: str) -> pd.DataFrame:
    p = _find(name)
    return pd.read_csv(p) if p else pd.DataFrame()


def _img(name: str, caption: str):
    p = _find(name)
    if p:
        st.image(str(p), caption=caption, width="stretch")


DOT = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f3f4f6", fontname="Helvetica", fontsize=11];
  A [label="Applicant answers\\n7 short questions"];
  B [label="Policy check\\nage 18 to 60"];
  C [label="Feature builder\\nLTV, loan, tier, ..."];
  D [label="Trained model\\nrisk of decline (0 to 1)", fillcolor="#fde7c8"];
  E [label="Threshold\\nrisk >= 0.089 ?", fillcolor="#fde7c8"];
  F [label="Approve or Decline\\n+ next steps", fillcolor="#d6f0e0"];
  G [label="Masked log\\n(SQLite)", fillcolor="#e6e6fa"];
  A -> B -> C -> D -> E -> F; A -> G [style=dashed]; D -> G [style=dashed]; F -> G [style=dashed];
}
"""


def render():
    st.title("🛵 How this loan checker works")
    st.caption("A plain-English guide for anyone new to the project. No coding knowledge needed. "
               "Every number on this page comes from our own training run and hold-out test.")
    tabs = st.tabs(["1 Big picture", "2 Data and cleaning", "3 Fields we ask", "4 The model", "5 Approve or decline",
                    "6 Results", "7 Try the model", "8 Privacy and logging", "9 Limits", "10 More checks"])

    with tabs[0]:
        st.subheader("What this app does")
        st.write("ABC Credit wants to answer one question fast: **should this two-wheeler loan be approved?** "
                 "The app asks 7 short questions, turns the answers into numbers, and gives them to a trained model. "
                 "The model returns a **risk score** (the chance that a similar past application was declined). "
                 "If the score is at or above a cut-off, the answer is *Decline*. Otherwise it is *Approve*.")
        st.graphviz_chart(DOT, width="stretch")
        st.markdown(
            "**Two rules keep this safe and fair to audit.**\n"
            "1. **Only the model decides.** No language model and no hand-written rule changes the approve or decline result. "
            "The one exception is a policy rule: applicants must be 18 to 60, and that check runs before the model.\n"
            "2. **Predicting and talking are separate.** An optional text helper could reword explanations, but it is switched off "
            "and never sees the score or makes a decision.")
        st.info("Where did the model learn from? From **108,423 past loan applications**, each marked APPROVED or DECLINED.")

    with tabs[1]:
        st.subheader("The training data")
        c1, c2, c3 = st.columns(3)
        c1.metric("Past applications", "108,423")
        c2.metric("Declined", "5.25%")
        c3.metric("Approve-everyone accuracy", "94.75%")
        st.write("Only about 1 in 19 applications was declined. A lazy rule that approves everyone would be right 94.75% of the time "
                 "and would catch **zero** risky applicants. So **accuracy alone is a poor yardstick** here. We judge the model with "
                 "ROC-AUC, PR-AUC, and above all **recall on DECLINED** (how many risky applicants we catch) next to precision.")
        _img("01_class_balance.png", "Class balance: declined cases are rare.")
        st.subheader("Data problems we found and what we did")
        log = _csv("cleaning_log.csv")
        if len(log):
            st.dataframe(log.rename(columns={"issue": "Issue", "rows_affected": "Rows", "action": "What we did"}),
                         hide_index=True, width="stretch")
        st.markdown(
            "- **Blank values** are filled with the middle value (median) inside the model pipeline, and a flag records that something was blank.\n"
            "- **Garbled text** such as `â€“` is repaired, and spellings like `125 CC DISC` and `125CC DISC` are unified.\n"
            "- **Impossible values** (LTV of 0) are treated as missing. **Zero income** is real for students and non-earning members, so it is kept with a flag.\n"
            "- **Very large salaries** are capped at the 99.5th percentile (100,000) and log-scaled, so a few extreme values cannot dominate.\n"
            "- **Duplicates** are removed. Pairs with the same answers but different outcomes are dropped so the model is not taught a contradiction.\n"
            "- **New categories** (an unseen make or pincode) never crash the app. They fall back to a typical profile or a wider area.")

    with tabs[2]:
        st.subheader("Which fields we ask, and why")
        st.write("The data has 17 columns, but a long form drives customers away. We tested each group of fields by **removing it and "
                 "measuring how much the model got worse**. A big drop means the field matters.")
        imp = _csv("permutation_importance.csv")
        if len(imp):
            show = imp.copy()
            show["group"] = show["group"].map({"ltv": "Loan share of vehicle price (LTV)", "pincode": "Pincode and area tier", "loan": "Loan amount",
                                               "vehicle": "Vehicle make and model", "past_loans": "Active loan today", "age": "Age",
                                               "employment": "Employment type", "income": "Monthly income"}).fillna(show["group"])
            st.bar_chart(show.set_index("group")["AUC_drop"], horizontal=True)
            st.caption("Drop in ROC-AUC on the hold-out set when the group is scrambled. Longer bar = more important.")
        st.markdown(
            "| Question | What it gives the model |\n|---|---|\n"
            "| 1 Pincode | Area tier and the region's past decline rate |\n"
            "| 2 Vehicle | Brand and model (we read the product type and variant from the model) |\n"
            "| 3 Price and loan amount | Loan share of price (LTV), loan size, down payment |\n"
            "| 4 Employment type | Salaried, self-employed, student, pensioner, and so on |\n"
            "| 5 Monthly income | Income and loan-to-income ratio (skipped for students and non-earning members) |\n"
            "| 6 Age | Age, plus the 18 to 60 policy rule |\n"
            "| 7 Active loan | Whether the person already has a loan |")
        st.write("**Left out on purpose:** gender, qualification and residence type. Adding any of them changed the score by less than "
                 "0.002 AUC, which is noise, and gender is a sensitive field. ID is only a label and is never used.")
        st.write("**Honest note:** LTV is the strongest signal by far, then pincode. Income and employment add very little on their own. "
                 "We still ask them (the team chose to keep all 7 questions) because lenders expect them and they cost about 20 seconds.")

    with tabs[3]:
        st.subheader("The model, step by step")
        st.markdown(
            "**1. Split.** We keep a random 20% of applications (21,681) locked away as a **hold-out test**. The model never sees them while learning. "
            "The split keeps the same 5% decline mix in both parts.\n\n"
            "**2. Prepare.** Inside one pipeline: fill blanks, turn categories into columns of 0 and 1, and replace each pincode "
            "prefix by its past decline rate. This is binning: 14,000 pincodes are too many values, so we group them by their first 2 and 3 digits. "
            "The decline rate is worked out only from the training rows in each round, so it cannot leak the answer.\n\n"
            "**3. Learn.** A *gradient boosting* model builds many small decision trees one after another. Each new tree fixes the "
            "mistakes of the earlier ones. The result is a **risk score from 0 to 1**.\n\n"
            "**4. Check.** We compare models with 5-fold cross-validation (train on 4 parts, test on the 5th, rotate). "
            "A simpler logistic regression is the baseline.")
        st.info("**Did we balance the data?** No. We did not copy, delete or re-weight any rows, so the model learns from the real 5.25% decline mix. "
                "Instead we handled the imbalance in three ways: (1) judge the model with PR-AUC and recall on DECLINED, not accuracy; "
                "(2) keep the 20% hold-out and every cross-validation fold at the same 5% mix; (3) choose the decision threshold (0.089) from a cost curve. "
                "We also tested a class-balanced version (last row below). It did not beat the plain model on PR-AUC (0.167 against 0.171), so we kept the simpler one.")
        comp = _csv("model_comparison.csv")
        if len(comp):
            st.dataframe(comp.rename(columns={"secs": "seconds"}).drop(columns=["Brier"], errors="ignore"), hide_index=True, width="stretch")
            st.caption("Out-of-fold scores on the training part. Higher ROC-AUC and PR-AUC are better; lower LogLoss is better.")
        st.write("**Why this model?** It scores best or tied-best, trains in seconds, scores one applicant in about 5 milliseconds, "
                 "needs no special system libraries, and its scores are well calibrated (a score of 10% really means about 10% declined).")
        _img("15_calibration_holdout.png", "Calibration on the hold-out set: predicted risk against what really happened.")

    with tabs[4]:
        st.subheader("From risk score to Approve or Decline")
        st.write("The model gives a risk score. We still have to choose a **cut-off**. This is a business trade-off:")
        st.markdown("- **Too strict** (low cut-off): we decline many good customers. That is lost margin.\n"
                    "- **Too loose** (high cut-off): we approve many risky customers. That is credit loss.")
        st.write("We assumed a bad loan approved costs **10 times** as much as a good customer declined (ratio r = 10). "
                 "The cost curve gives the cut-off **0.089**. Decline when risk is 0.089 or more.")
        _img("11_cost_curve.png", "Expected cost at each cut-off. The lowest point is the chosen threshold.")
        cand = _csv("threshold_candidates.csv")
        if len(cand):
            st.dataframe(cand[["r (loss:margin)", "threshold", "declined_share", "recall_declined", "precision_declined"]]
                         .rename(columns={"r (loss:margin)": "Loss : margin ratio"}), hide_index=True, width="stretch")
        sweep = _csv("holdout_threshold_sweep.csv")
        if len(sweep):
            st.subheader("Move the cut-off yourself")
            t = st.slider("Cut-off (decline when risk is at or above this)", float(sweep.threshold.min()), float(sweep.threshold.max()), 0.09, 0.01)
            row = sweep.iloc[(sweep.threshold - t).abs().argmin()]
            a, b, c, d = st.columns(4)
            a.metric("Declined share", f"{row.declined_share:.1%}")
            b.metric("Recall on DECLINED", f"{row.recall:.1%}")
            c.metric("Precision", f"{row.precision:.1%}")
            d.metric("Accuracy", f"{row.accuracy:.1%}")
            st.caption(f"Hold-out set: {int(row.TP)} risky applicants caught, {int(row.FN)} missed, {int(row.FP)} good customers declined, "
                       f"{int(row.TN)} good customers approved. A stricter cut-off catches more risky applicants but declines more good ones.")

    with tabs[5]:
        st.subheader("How good is it? (hold-out test, scored once)")
        a, b, c, d = st.columns(4)
        a.metric("ROC-AUC", "0.783")
        b.metric("PR-AUC", "0.182", help="A random guesser scores 0.053.")
        c.metric("Recall on DECLINED", "54.4%")
        d.metric("Precision", "15.1%")
        st.markdown("At cut-off 0.089: accuracy **81.6%**, F1 **0.237**. Confusion matrix: 17,065 approved correctly, 3,477 good customers declined, "
                    "519 risky applicants missed, 620 risky applicants caught. 5-fold cross-validation gives recall 53.4% (plus or minus 1.2) "
                    "and ROC-AUC 0.779 (plus or minus 0.008), so the result is stable.")
        c1, c2 = st.columns(2)
        with c1:
            _img("12_roc_pr_holdout.png", "ROC and precision-recall curves.")
        with c2:
            _img("13_confusion_holdout.png", "Confusion matrix at the chosen cut-off.")
        st.write("**Reading these numbers fairly.** Accuracy looks lower than the approve-everyone 94.75% because we deliberately "
                 "decline about 19% of applications to catch more than half of the risky ones. Precision is low because declines are rare "
                 "in the first place. This is a screening tool: a decline means *look more closely*, not *this person will default*.")
        st.markdown("**How fast is it? (our values of X and Y)**")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Model scoring (median)", "5.4 ms", help="95th percentile 7.1 ms, over 100 scripted sessions.")
        e2.metric("Decision screen (median)", "27 ms", help="95th percentile 40 ms.")
        e3.metric("Click to decision, live app", "about 1.0 s", help="Our target Y is 2 seconds.")
        e4.metric("Time to finish the chat", "1 min 47 s", help="Average of three classmates (85, 105 and 130 seconds). Our target X is 2.5 minutes.")
        _img("02_decline_by_ltv.png", "The strongest pattern in the data: the larger the loan share of the price, the more often it was declined.")

    with tabs[6]:
        st.subheader("Try the real model")
        st.write("Change the inputs and watch the risk score move. This calls the same trained model as the chat. Nothing here is saved.")
        lk = predict._load()[2]
        makes = sorted({v["make"] for v in lk["vehicles"] if v["make"] != "MISSING"})
        c1, c2, c3 = st.columns(3)
        make = c1.selectbox("Vehicle make", makes, index=makes.index("JUPITER") if "JUPITER" in makes else 0, key="x_make")
        pin = c2.text_input("Pincode", "560001", max_chars=6, key="x_pin")
        emp = c3.selectbox("Employment", list(C.EMPLOYMENT_LABELS), format_func=C.EMPLOYMENT_LABELS.get, key="x_emp")
        c4, c5, c6 = st.columns(3)
        ltv = c4.slider("Loan as % of vehicle price", 40, 100, 80, key="x_ltv")
        age = c5.slider("Age", 18, 60, 32, key="x_age")
        active = c6.toggle("Has an active loan", value=False, key="x_act")
        income = 0 if emp in C.NO_INCOME_TYPES else st.number_input("Monthly income (rupees)", 1000, 1_000_000, 25000, 1000, key="x_inc")
        if not (len(pin) == 6 and pin.isdigit() and pin[0] != "0"):
            st.warning("Enter a 6-digit pincode that does not start with 0.")
            return
        v = next(x for x in lk["vehicles"] if x["make"] == make)
        base = dict(pincode=pin, make=v["make"], product=v["product"], variant=v["variant"], price=100000.0, employment=emp,
                    income=float(income), age=int(age), active_loan=bool(active))
        r = predict.score({**base, "loan": 100000.0 * ltv / 100})
        thr = r["threshold"]
        x1, x2 = st.columns(2)
        x1.metric("Risk of decline", f"{r['probability']:.1%}")
        x2.metric("Decision at cut-off " + f"{thr:.3f}", r["decision"])
        pts = pd.DataFrame({"Loan as % of price": list(range(40, 101, 5))})
        pts["Risk of decline"] = [predict.score({**base, "loan": 1000.0 * p})["probability"] for p in pts["Loan as % of price"]]
        pts["Cut-off"] = thr
        st.line_chart(pts.set_index("Loan as % of price"))
        st.caption("Risk rises as the loan covers more of the price. The flat line is the cut-off: points above it are declined.")

    with tabs[7]:
        st.subheader("Privacy and logging")
        st.markdown(
            "**What we log** (SQLite file, one row per session and one per message): time, consent choice, every question and the "
            "applicant's answer (masked), the feature values given to the model, the risk score, the cut-off, the decision, "
            "model version and response time.\n\n"
            "**What we never store:** the full pincode, an exact income, a name, a phone number, or any "
            "free text that failed validation.")
        st.table(pd.DataFrame([
            ["Pincode 560001", "560*** (first 3 digits)", "The model only needs the area, so the full code is dropped"],
            ["Income 25,000", "income band 20k-30k", "The exact figure is never stored. Even the saved feature record keeps only the band, so a score can be replayed only approximately"],
            ["Name 'Asha Rao'", "[name provided] + salted hash", "The hash is SHA-256 with a secret salt. It cannot be turned back into the name"],
            ["Phone 98765 43210", "[phone provided] + salted hash", "Same method. Only a 10-digit mobile number is accepted"],
            ["Typed junk or a phone number in the pincode box", "[invalid input, 10 chars]", "Wrong inputs are never saved as typed"],
        ], columns=["What the person typed", "What is stored", "Why"]))
        st.markdown(
            "**Consent first.** The chat starts by asking for consent. If the person says no, the session ends and no answers are kept. "
            "Name and phone are optional and are not used by the model.\n\n"
            "**Before any language model** (none is used today), text is scrubbed of long digit runs and emails.\n\n"
            "**Tested.** 344 automated checks confirm that a typed name, phone number and full pincode never reach the database.\n\n"
            "**Example of a stored session** (made-up values):")
        st.dataframe(pd.DataFrame([
            ["intro", "assistant", "Before we start: I use your answers only to check this loan..."],
            ["consent", "user", "I agree"],
            ["name", "user", "[name provided]"],
            ["pin", "user", "560***"],
            ["inc", "user", "income band 20k-30k"],
            ["age", "user", "30"],
            ["act", "user", "No"],
            ["done", "system", "risk 0.031, cut-off 0.089, APPROVED, 6 ms"],
        ], columns=["step", "who", "stored text"]), hide_index=True, width="stretch")
        st.caption("Note: on this public demo the log file lives only on the host and is erased when the app restarts.")

    with tabs[8]:
        st.subheader("Limits you should know")
        st.markdown(
            "- **It screens, it does not judge.** Precision is about 15%: most declines would have repaid. A decline is a prompt for human review.\n"
            "- **LTV dominates.** Loans above about 90% of the price carry most of the declines. Below 80% the model almost never declines.\n"
            "- **Self-reported answers.** Income and loan details are not verified by this app.\n"
            "- **Past patterns only.** The model repeats history, including any bias in past decisions. We left out gender for that reason.\n"
            "- **Hold-out is our own.** The professor's hidden test set may behave a little differently. Our 5-fold results suggest small swings.\n"
            "- **Cost ratio is an assumption.** r = 10 is a business judgement. Move the cut-off on the *Approve or decline* tab to see other trade-offs.")
        meta = json.loads((predict.C.MODEL_PATH.parent / "metadata.json").read_text())
        st.caption(f"Model: {meta['model']}. Saved {meta['saved_at']}. Trained on {meta['train_rows']:,} rows. Random seed {meta['random_state']}.")

    with tabs[9]:
        st.subheader("More checks: other models, overfitting, fairness")
        st.caption("These checks do not change the model. They test whether our choices hold up. Same data, same 5-fold method, fixed random seed.")
        st.markdown("**1. Is the data skewed? (why we fill blanks with the median)**")
        sk = _csv("extras_skewness.csv")
        if len(sk):
            st.dataframe(sk, hide_index=True, width="stretch")
            st.caption("A long tail pulls the mean away from the median. For skewed fields the median is the safer fill-in value.")
        st.markdown("**2. Do any fields repeat each other?**")
        st.write("Yes. Down payment is exactly vehicle value minus the loan, and LTV is the loan divided by the vehicle value, so these four fields "
                 "overlap. A linear model such as logistic regression can be confused by overlapping fields. Tree models are not, which is one reason "
                 "we chose one.")
        st.markdown("**3. Other models on the same yardstick**")
        mc = _csv("extras_model_comparison.csv")
        if len(mc):
            st.dataframe(mc.drop(columns=["Brier"], errors="ignore"), hide_index=True, width="stretch")
            _img("16_model_comparison.png", "PR-AUC for each model. The dashed line is a random guess.")
            st.caption("Logistic regression with L2 (Ridge) or L1 (Lasso) adds a penalty that shrinks weak inputs. L1 can set them to exactly zero. "
                       "The last column asks: if each model declined the riskiest 19%, how many risky applicants would it catch? "
                       "Random forest and gradient boosting are tied within the fold-to-fold noise, so we keep gradient boosting.")
        lt = _csv("extras_lasso_top.csv")
        if len(lt):
            st.markdown("**What Lasso kept (strongest inputs)**")
            st.dataframe(lt, hide_index=True, width="stretch")
        st.markdown("**4. Underfitting and overfitting (bias and variance)**")
        td = _csv("extras_tree_depth.csv")
        if len(td):
            st.dataframe(td, hide_index=True, width="stretch")
        _img("17_bias_variance.png", "Left: a very deep tree memorises the training rows (big gap). Right: more rows narrow the gap for gradient boosting.")
        st.markdown("**5. Fairness check on the hold-out set**")
        fa = _csv("extras_fairness.csv")
        if len(fa):
            _img("18_fairness.png", "Decline rate by gender and age band, history against the model.")
            st.dataframe(fa, hide_index=True, width="stretch")
            st.caption("Gender is not a model input. We check anyway, because other fields can stand in for it. "
                       "This is a descriptive audit, not a legal fairness test.")
