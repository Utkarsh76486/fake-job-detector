import re
import numpy as np
import pandas as pd

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]


def _clean(value):
    """None ya khali string ko NaN banao (training me missing = NaN tha)."""
    if value is None:
        return np.nan
    if isinstance(value, str) and value.strip() == "":
        return np.nan
    return value


def build_features(job: dict, art: dict) -> pd.DataFrame:
    raw = {k: _clean(v) for k, v in job.items()}

    has_company_profile = int(pd.notnull(raw.get("company_profile")))
    has_salary = int(pd.notnull(raw.get("salary_range")))
    has_employment_type = int(pd.notnull(raw.get("employment_type")))
    filled_fields = sum(int(pd.notnull(raw.get(c))) for c in art["info_cols"])

    t = {c: ("" if pd.isnull(raw.get(c)) else str(raw[c])) for c in TEXT_COLS}

    desc = t["description"]
    caps_ratio = sum(ch.isupper() for ch in desc) / (len(desc) + 1)

    text = " ".join([t["title"], t["company_profile"], t["description"],
                     t["requirements"], t["benefits"]]).lower()
    suspicious_count = sum(text.count(w) for w in art["suspicious"])
    has_email_or_url = int(re.search(r"#email_|#url_", text) is not None)

    loc = raw.get("location")
    country = "unknown" if pd.isnull(loc) else str(loc).split(",")[0].strip()

    row = {
        "filled_fields": filled_fields,
        "title_length": len(t["title"]),
        "desc_length": len(desc),
        "profile_length": len(t["company_profile"]),
        "req_length": len(t["requirements"]),
        "exclaim_count": desc.count("!"),
        "caps_ratio": caps_ratio,
        "suspicious_count": suspicious_count,
        "telecommuting": int(raw.get("telecommuting") or 0),
        "has_company_logo": int(raw.get("has_company_logo") or 0),
        "has_questions": int(raw.get("has_questions") or 0),
        "has_company_profile": has_company_profile,
        "has_salary": has_salary,
        "has_employment_type": has_employment_type,
        "has_email_or_url": has_email_or_url,
        "employment_type": raw.get("employment_type", np.nan),
        "required_experience": raw.get("required_experience", np.nan),
        "required_education": raw.get("required_education", np.nan),
        "country": country,
        "text": text,
    }

    cols = art["num_cols"] + art["bin_cols"] + art["cat_cols"] + ["text"]
    return pd.DataFrame([row])[cols]