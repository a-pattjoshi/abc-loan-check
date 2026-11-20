# Handover: ABC Credit Loan Approval Chatbot (PGAI Group 11)

## Links
- Live app: https://pgai-group11-loan-approval.streamlit.app/
- Code (app and model): https://github.com/a-pattjoshi/abc-loan-check
- Short report: reports/Group11_Loan_Approval_Report_final.pdf (also in the GitHub repo)

## What the app does
Three modes in the sidebar: **Loan check** (consent, optional name, 7 questions, Approve or Decline, next steps), **Score a file** (upload a CSV, or use the built-in sample file, and download predictions), **How it works** (plain-English guide with a live what-if tool).
The decision comes only from the trained model (risk score against a threshold of 0.089). Age 18 to 60 is a policy rule that runs first. The optional language model helper is off.

## Brief deliverables and where they are
| Brief deliverable | Where |
|---|---|
| 1 Cleaning and preprocessing pipeline | src/clean.py, src/features.py, notebooks/01_eda_cleaning.ipynb, reports/cleaning_log.csv |
| 2 Trained model with reproducible code | notebooks/02_features_model.ipynb, notebooks/03_model_threshold.ipynb, src/train.py, models/model.joblib (seed 42) |
| 3 Chatbot with PII masking | src/app.py, src/predict.py, src/logger.py, src/explain.py |
| 4 Predictions on the held-out test set | `python src/predict.py <file.csv> --out predictions.csv` or the Score a file mode. Format still to confirm with the professor |
| 5 Short report | reports/Group11_Loan_Approval_Report_final.pdf (HTML source alongside) |
| Self-assessment against section 10 | reports/evaluation_vs_brief.md |
| Decisions made along the way | reports/decision_log.md (D1 to D25) |
| Numbers behind the report | reports/report_numbers.md, reports/figures/ |
| Masked log samples | reports/masked_log_sample_sessions.csv, reports/masked_log_sample_turns.csv |

## Run it
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run src/app.py                       # the app
.venv/bin/python src/predict.py data/mock_test_unlabeled.csv --out out.csv
OMP_NUM_THREADS=2 .venv/bin/python src/qa_checks.py      # 309 checks, about 5 minutes
.venv/bin/python src/latency_test.py                     # 100 scripted sessions
.venv/bin/python src/make_deploy.py                      # rebuild deploy_package/ for GitHub
```
Python 3.12. Packages are pinned in requirements.txt (the model file needs scikit-learn 1.9.1).
The raw training file is not in this zip. Put training_data.csv one folder above this one to rerun the notebooks.

## Key results (hold-out 21,681 applications, threshold 0.089)
ROC-AUC 0.783, PR-AUC 0.182, accuracy 81.6%, precision 15.1%, recall on DECLINED 54.4%, F1 0.237. Decision on screen 1.0 s after the last click on the live app; model scoring 5 ms.

## Open items
- Hidden test file format and deadline (ask the professor).
- X (2.5 minutes) needs timing with 3 people.
- ABC Credit's real documents-to-bring list (the app uses a generic list).
- The public demo log is temporary. Use local runs for log evidence.

## Privacy note for this zip
data/ holds rows derived from the course training file. Share this zip only with the course team and the professor.

## Notebook for testing a new CSV (for the professor)
`ABC_Credit_Loan_Approval_Final.ipynb` rebuilds the whole project from `training_data.csv` (cleaning, features, gradient boosting, threshold, hold-out check) in about a minute and then scores any CSV.
1. Put `training_data.csv` and your test file next to the notebook.
2. In Section 1 set `TEST_CSV` to your file name.
3. Run all cells. Predictions are saved to `predictions.csv` (ID, Decision, P_DECLINED, policy_age_ok, Final_Decision). If the file has a `Decision` column the notebook prints accuracy, precision, recall, F1, ROC-AUC and the confusion matrix.
Extra columns such as a CIBIL score or ITR fields are ignored and named in a warning. Try `sample_with_extra_columns.csv`. Needs Python 3.9+, pandas, numpy, scikit-learn 1.3+, matplotlib.
