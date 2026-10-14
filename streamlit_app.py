import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
OTHER = "Other (type manually)"

st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️")

# ---------- Dropdown options ----------
TITLES = [
    "Python Backend Developer",
    "Data Analyst",
    "Marketing Executive",
    "Customer Support Executive",
    "Sales Representative",
    "Data Entry Operator",
    "Work From Home Typist",
    OTHER,
]

DESCRIPTIONS = {
    "Software development role": (
        "You will design and maintain REST APIs, work with PostgreSQL, and "
        "collaborate with the product team on new features. You will write clean "
        "code, review pull requests and deploy services."
    ),
    "Data analysis role": (
        "You will analyse business data, build dashboards and share insights with "
        "the management team. You will work with SQL, Excel and visualization "
        "tools on a daily basis."
    ),
    "Marketing role": (
        "You will plan and run digital campaigns, manage social media pages and "
        "report campaign performance to the marketing manager every week."
    ),
    "Customer support role": (
        "You will answer customer queries over chat and email, resolve complaints "
        "and maintain records of every interaction in our support system."
    ),
    "Sales role": (
        "You will reach out to potential clients, present our products, negotiate "
        "contracts and meet monthly sales targets with the support of the sales team."
    ),
    "Simple typing / data entry work": (
        "Simple typing work. Enter data into our system and submit it daily. "
        "Flexible timing and quick payment."
    ),
    "Easy online work, quick payment": (
        "Work online in your free time. Get started today, flexible hours, "
        "and receive your payment every week."
    ),
}

PROFILES = {
    "Established company (detailed profile)": (
        "We are a 200-person technology company building software for small "
        "businesses across the US. Founded in 2012, we are known for a strong "
        "engineering culture, steady growth and long-term customers."
    ),
    "Small company / startup": (
        "We are a small startup building simple tools for local businesses."
    ),
    "No company profile": "",
    OTHER: None,
}

COUNTRIES = {
    "United States": "US", "United Kingdom": "GB", "Canada": "CA",
    "Australia": "AU", "India": "IN", "Germany": "DE", "Greece": "GR",
    "New Zealand": "NZ", "Philippines": "PH",
}

RED_FLAGS = {
    "Asks for a registration / security fee": "Small registration fee required.",
    "Says 'no experience needed'": "No experience needed.",
    "Uses urgent / immediate joining language": "URGENT hiring! Immediate start.",
    "Promises very high weekly earnings": "Earn weekly pay of $5000.",
    "Asks to contact through email / link": "Contact #EMAIL_abc123# now.",
}

EMPLOYMENT = ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"]
EXPERIENCE = ["", "Entry level", "Mid-Senior level", "Associate",
              "Executive", "Director", "Internship", "Not Applicable"]

# ---------- Form ----------
st.title("🕵️ Fake Job Posting Detector")
st.write("Choose from the dropdowns or select 'Other' to type manually. "
         "The more details you fill in, the more reliable the result.")

# 1. Job title
title_choice = st.selectbox("Job title *", TITLES)
if title_choice == OTHER:
    title = st.text_input("Enter job title")
else:
    title = title_choice

# 2. Description (standard options only)
description_choice = st.selectbox("Job description *", list(DESCRIPTIONS.keys()))
description = DESCRIPTIONS[description_choice]

# 3. Red flags
st.markdown("**Does this posting contain any of the following? (tick all that apply)**")
selected_flags = [text for label, text in RED_FLAGS.items() if st.checkbox(label)]

# 4. Company profile
profile_choice = st.selectbox("Company profile", list(PROFILES.keys()))
if profile_choice == OTHER:
    company_profile = st.text_area("Enter company profile", height=100)
else:
    company_profile = PROFILES[profile_choice]

# 5. Location
country_choice = st.selectbox("Country", list(COUNTRIES.keys()) + ["Unknown / not mentioned", OTHER])
if country_choice == OTHER:
    code = st.text_input("Enter country code (e.g. FR, JP)").strip().upper()
elif country_choice == "Unknown / not mentioned":
    code = ""
else:
    code = COUNTRIES[country_choice]
city = st.text_input("City (optional)")
location = f"{code}, , {city.strip()}" if code and city.strip() else code

# 6. Other details
c1, c2 = st.columns(2)
employment_type = c1.selectbox("Employment type", EMPLOYMENT)
required_experience = c2.selectbox("Experience level", EXPERIENCE)

salary_range = st.text_input("Salary range (optional, e.g. 40000-60000)")

# 7. Yes/No options
k1, k2, k3 = st.columns(3)
has_company_logo = k1.checkbox("Company logo present")
has_questions = k2.checkbox("Screening questions present")
telecommuting = k3.checkbox("Work from home")

# Add ticked red flags to the description
final_description = (description + " " + " ".join(selected_flags)).strip()

with st.expander("Description that will be sent to the model"):
    st.write(final_description)

# ---------- Check ----------
if st.button("Check job", type="primary"):
    payload = {
        "title": (title or "").strip(),
        "description": final_description,
        "company_profile": (company_profile or "").strip() or None,
        "location": location or None,
        "employment_type": employment_type or None,
        "required_experience": required_experience or None,
        "salary_range": salary_range.strip() or None,
        "has_company_logo": int(has_company_logo),
        "has_questions": int(has_questions),
        "telecommuting": int(telecommuting),
    }

    try:
        with st.spinner("Analysing..."):
            res = requests.post(API_URL, json=payload, timeout=30)
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Please start the FastAPI server first.")
        st.stop()

    if res.status_code == 422:
        st.warning("Please enter a valid job title (at least 2 characters).")
    elif not res.ok:
        st.error(f"API error: {res.status_code}")
    else:
        data = res.json()
        prob = data["fake_probability_percent"]

        st.subheader("Result")
        m1, m2 = st.columns(2)
        m1.metric("Fake probability", f"{prob}%")
        m2.metric("Prediction", data["prediction"])
        st.progress(min(prob / 100, 1.0))

        risk = data["risk_level"]
        if risk == "High":
            st.error("🚨 High risk: this job may be a scam. Verify the company before applying.")
        elif risk == "Medium":
            st.warning("⚠️ Medium risk: be cautious and check the company's details.")
        else:
            st.success("✅ Low risk: this job looks genuine.")

        if not payload["company_profile"] or not has_company_logo:
            st.info("Note: the company profile or logo is missing. The model treats this "
                    "as a strong warning sign, so the risk score may increase.")

st.caption("Model: Calibrated Linear SVM + TF-IDF | Test F1: 0.84 | "
           "This is an ML prediction. Always verify a job yourself before applying.")