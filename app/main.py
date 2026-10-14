from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.features import build_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fake_job_model.joblib"
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
threshold = artifact["threshold"]

app = FastAPI(title="Fake Job Posting Detector", version="1.0")


class JobPosting(BaseModel):
    title: str = Field(..., min_length=2)
    description: str = Field(..., min_length=20)
    company_profile: Optional[str] = None
    requirements: Optional[str] = None
    benefits: Optional[str] = None
    location: Optional[str] = None
    salary_range: Optional[str] = None
    department: Optional[str] = None
    employment_type: Optional[str] = None
    required_experience: Optional[str] = None
    required_education: Optional[str] = None
    industry: Optional[str] = None
    function: Optional[str] = None
    telecommuting: int = Field(0, ge=0, le=1)
    has_company_logo: int = Field(0, ge=0, le=1)
    has_questions: int = Field(0, ge=0, le=1)


@app.get("/")
def health():
    return {"status": "ok", "message": "Fake Job Detector API chal rahi hai"}


@app.post("/predict")
def predict(job: JobPosting):
    X = build_features(job.model_dump(), artifact)
    prob = float(model.predict_proba(X)[0, 1])

    if prob >= threshold:
        risk = "High"
    elif prob >= threshold / 2:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "fake_probability_percent": round(prob * 100, 1),
        "prediction": "Fake" if prob >= threshold else "Real",
        "risk_level": risk,
    }