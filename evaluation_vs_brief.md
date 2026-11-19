# Self-assessment against the project brief (version 2, 19 Nov 2026, after classmate feedback)

The brief gives no weights. We weighted the six assessment areas in section 10 as shown. Scores are our own honest estimate.

| Assessment area (brief section 10) | Weight | Score | Evidence | What costs points |
|---|---|---|---|---|
| Model performance on the held-out set | 20 | 15 | Hold-out ROC-AUC 0.783, PR-AUC 0.182 (random 0.053), recall on DECLINED 54.4%, precision 15.1%, F1 0.237. 5-fold CV is stable (recall 53.4%, sd 1.2). Calibrated scores. | Accuracy 81.6% is below the approve-everyone 94.75% (a deliberate recall trade-off). The professor's hidden set is not scored yet and its format is not confirmed. |
| Quality of exploratory analysis and data handling | 15 | 13 | 12 issues found and logged with row counts (empty row, blanks, age over 60, LTV 0, salary 0, outliers, mojibake, spelling variants, duplicates, conflicting duplicates, redundancy, pincode cardinality). Notebook 01, 8 charts, cleaning log, decision log D1 to D25. | Not every field has its own chart (gender, qualification, resident type are covered by the ablation only). |
| Quality of feature engineering and selection | 15 | 13 | Derived vehicle value, down payment, loan-to-income, log salary with cap, missing flags; smoothed target encoding of pincode prefixes inside CV; single-group, pair and permutation ablation; age over 60 test. | We kept income and employment although they add about 0 AUC (team decision to keep 7 questions). A 5-question version would predict about as well. |
| Chatbot functionality and field-count and latency trade-offs | 20 | 18 | Consent, optional name, 7 validated questions, dynamic skips (students and non-earning members skip income, age policy stops early), Back button, "vehicle not listed" choice, Indian-pincode check with city or state, price caution outside the training range, next steps (documents, smaller loan offer, co-applicant), amounts like "80k", back to start, file scoring mode with sample file, explainer page. Tested live: decision on screen 1.0 s after the last click. Latency in-process 27 ms median. | X (2.5 minutes) is a reasoned estimate, not yet timed with real people. The question sequence is only mildly dynamic. |
| Completeness of logging | 15 | 13 | SQLite log per session and per message: time, consent, every question and masked answer, feature values, risk score, threshold, decision, model version, latency. Masked samples exported in reports/masked_log_sample_*.csv. | The live demo log is temporary and cannot be viewed by the reviewer. Batch file scoring is not logged. The full pincode is dropped on purpose (privacy). |
| Clarity and reasoning in the report | 15 | 13 | Short report covers all six required items (fields, data issues, model choice, metrics, latency, X and Y) with figures and a threshold justification. | X timing is a fill-in. |
| **Total** | **100** | **85** | | |

## PII check (brief section 3: "ensure security and privacy of users by masking PII data")
| Requirement | Status | Evidence |
|---|---|---|
| Mask personal data in the log | Done | Pincode stored as first 3 digits plus stars. Income stored as a band. Invalid input stored only as a length. |
| Hash name and phone | Done (added 18 Nov) | Optional name and phone questions. Only a salted SHA-256 hash is stored (table applicants). Tests confirm the typed values appear nowhere in the database. |
| Consent | Done (added 18 Nov) | First question of the chat. Refusal ends the session and keeps no answers. |
| No personal data to a language model | Done | The language model helper is off. If switched on it receives text scrubbed of digit runs and emails. |
| Show the masking to the reviewer | Done | "How it works" tab, Privacy section, and the masked log samples in reports/. |
| Known residual risks | Open | Income is stored as a band only, in answers and in the feature record (decision D16, changed 2026-11-19). Price and loan amount answers are stored as typed (business data, not identifiers). The public demo log is temporary. |

## Changes since version 1 (19 Nov)
Income band only in the feature record (D16b). Back button, unlisted vehicles, wider price range with caution, warmer wording, Approved/Declined headline, Indian pincode check, vehicle-range labels (D29 to D37). File scorer now reads "25,000", "Rs. 25000", "30 years", Yes/No and survives negative values (D38). 321 automated checks.

## What would lift the score to about 90
1. Confirm the hidden test file format with the professor and deliver the predictions file (+3).
2. Time 3 people on the live app and fill in X in the report (+2).
3. Store income as a band in the feature record, or show a written reason for keeping it (+1).
4. Add a masked log viewer or a screenshot in the report (+1).
5. Add one chart per remaining field in the EDA notebook (+1).
