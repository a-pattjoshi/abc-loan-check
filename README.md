# ABC Credit two-wheeler loan check (PGAI Group 11)

Streamlit app with three modes:
- **Loan check**: consent, optional name, 7 questions, then Approve or Decline from a trained model, with next steps.
- **Score a file**: upload a CSV in the training-file layout (a sample file can be downloaded in the app) and download the predictions.
- **How it works**: a plain-English guide to the data, the model, the threshold, the privacy rules, and a live what-if tool.

The decision comes only from the model (risk score against a threshold of 0.089). Age 18 to 60 is a policy rule that runs first.
Answers are logged with PII masked; a name or phone is stored only as a salted hash.

Run locally: `pip install -r requirements.txt` then `streamlit run app.py`.
Full project (cleaning, notebooks, training code, tests, report) is in the handover zip.
