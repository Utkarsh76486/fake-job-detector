# 🕵️ Fake Job Posting Detector

An end-to-end machine learning project that predicts whether a job posting is real or a scam.
It covers data cleaning, EDA, feature engineering, comparison of 5 algorithms, model tuning,
a FastAPI backend and a Streamlit frontend.

## Demo

![Streamlit demo](screenshots/demo.png)

## Problem

Fake job postings trick job seekers into paying fees or sharing personal data.
This project flags suspicious postings using the job text and posting metadata.

## Dataset

[Real / Fake Job Posting Prediction (EMSCAD)](https://www.kaggle.com/) from Kaggle:
17,880 postings, of which only about 4.8% are fraudulent (a heavily imbalanced dataset).
After removing 281 duplicates, 17,599 rows were used.
Download `fake_job_postings.csv` from Kaggle and place it in `data/`.

## Approach

1. **EDA:** checked missing values and tested which ones are actual fraud signals.
2. **Cleaning:** removed duplicates, filled text gaps, created missing-value flags *before* filling.
3. **Feature engineering:** `has_company_profile`, `filled_fields`, `profile_length`, `caps_ratio`,
   `suspicious_count` (scam keywords), `has_email_or_url`, `country`, plus TF-IDF on combined text.
4. **Split:** stratified 80/20 train/test split. All preprocessing sits inside a scikit-learn
   `Pipeline`, so nothing is fitted on test data (no data leakage).
5. **Models:** Logistic Regression, Linear SVM, Naive Bayes, Random Forest, XGBoost.
6. **Tuning:** `GridSearchCV` on SVM, `RandomizedSearchCV` on Random Forest, then decision-threshold tuning.
7. **Calibration:** the final SVM is wrapped in `CalibratedClassifierCV` to output real probabilities.

## Key findings from EDA

- Missing `company_profile`: **17.7% fake** vs 1.9% when present.
- No company logo: **15.9% fake** vs 2.0% with a logo.
- Surprise: postings that *show* a salary were more often fake (7.8% vs 4.3%).
- Country matters: AU 18.3% fake, US 6.9%, while GR, DE and NZ had 0%.
- A model that predicts "real" for everything gets 95% accuracy but 0% recall,
  so accuracy was ignored in favour of precision, recall, F1 and ROC-AUC.

## Model comparison (cross-validation on training data)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Linear SVM** | 0.765 | 0.831 | **0.796** | 0.979 |
| XGBoost* | 0.606 | 0.836 | 0.703 | 0.981 |
| Random Forest | 0.989 | 0.545 | 0.702 | 0.987 |
| Logistic Regression | 0.505 | 0.873 | 0.639 | 0.981 |
| Naive Bayes | 0.283 | 0.769 | 0.414 | 0.928 |

*XGBoost used 3-fold CV and lighter settings because of limited compute; the others used 5-fold.

Linear models beat tree models here because the signal lives mostly in sparse TF-IDF text features.

## Final model (evaluated once on the untouched test set)

**Calibrated Linear SVM** (C = 3, decision threshold = 0.417)

| Metric | Fake class |
|---|---|
| Precision | 0.90 |
| Recall | 0.78 |
| F1 | 0.84 |
| ROC-AUC | 0.988 |

Confusion matrix (3,520 test postings):

|  | Predicted real | Predicted fake |
|---|---|---|
| Actual real | 3335 | 14 |
| Actual fake | 38 | 133 |

## Limitations

- 38 of 171 fake postings in the test set were missed.
- The model leans heavily on company profile and logo, so a scam with a polished profile can slip through,
  and a genuine posting with no profile can be flagged.
- The dataset is mostly English and US-heavy, and may not reflect newer scam styles.
- Risk levels (Low / Medium / High) are a simple heuristic based on the decision threshold.
- The Streamlit demo uses dropdown templates for descriptions; the API accepts any free text.

## Run locally

```bash
git clone <your-repo-url>
cd fake-job-detector
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt -r requirements-ui.txt
```

Terminal 1 (API):
```bash
uvicorn app.main:app --reload
```

Terminal 2 (frontend):
```bash
streamlit run streamlit_app.py
```

API docs: http://127.0.0.1:8000/docs

### Docker (API only)
```bash
docker build -t fake-job-api .
docker run -p 8000:8000 fake-job-api
```

### Example request
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"title\": \"Data Entry Work From Home\", \"description\": \"URGENT hiring! Earn weekly pay of $5000. No experience needed.\"}"
```

## Project structure

```
fake-job-detector/
├── app/                 FastAPI backend (main.py, features.py)
├── models/              Saved model (joblib)
├── notebooks/           EDA, feature engineering, model training
├── screenshots/         Demo images
├── streamlit_app.py     Streamlit frontend
├── Dockerfile
├── requirements.txt
└── requirements-ui.txt
```

## Tech stack

Python, pandas, scikit-learn, XGBoost, FastAPI, Pydantic, Streamlit, Docker.