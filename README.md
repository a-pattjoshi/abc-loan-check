# ABC Credit two-wheeler loan check

Streamlit app: a 7-question chat that returns Likely approved or Likely declined, plus a "Score a file" mode for CSV files.
The decision comes only from a trained model (probability against a threshold). Age 18 to 60 is a policy rule that runs first.
Answers are logged with masking; name and phone are optional and stored only as salted hashes.

Run locally: `pip install -r requirements.txt` then `streamlit run app.py`.
`sample_upload.csv` holds 5 invented rows to try the file mode.
