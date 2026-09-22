"""
app/app.py
----------
Streamlit app for live churn prediction with SHAP explainability.
Run: streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Telecom Customer Churn Predictor")
st.caption("XGBoost model with SHAP explainability · IBM Telco Dataset")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model        = joblib.load("models/xgb_churn_model.pkl")
    explainer    = joblib.load("models/shap_explainer.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, explainer, feature_names

try:
    model, explainer, feature_names = load_model()
except FileNotFoundError:
    st.error("Model not found. Run `python src/train.py` first to train the model.")
    st.stop()

# ── Sidebar — customer inputs ─────────────────────────────────────────────────
st.sidebar.header("Customer Profile")

tenure           = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges  = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
contract         = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method   = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
paperless        = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
senior           = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner          = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents       = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
phone_service    = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines   = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
online_security  = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup    = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection= st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support     = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv     = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])


# ── Build input row ───────────────────────────────────────────────────────────

def build_input() -> pd.DataFrame:
    total_charges = monthly_charges * tenure if tenure > 0 else monthly_charges

    service_vals = [
        phone_service, multiple_lines, internet_service,
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies
    ]
    num_services = sum(
        v not in ["No", "No internet service", "No phone service"]
        for v in service_vals
    )

    tenure_bucket = (
        "0-12m"   if tenure <= 12 else
        "13-24m"  if tenure <= 24 else
        "25-48m"  if tenure <= 48 else
        "49-72m"
    )

    row = {
        "tenure":            tenure,
        "MonthlyCharges":    monthly_charges,
        "TotalCharges":      total_charges,
        "SeniorCitizen":     senior,
        "Partner":           partner,
        "Dependents":        dependents,
        "PhoneService":      phone_service,
        "MultipleLines":     multiple_lines,
        "InternetService":   internet_service,
        "OnlineSecurity":    online_security,
        "OnlineBackup":      online_backup,
        "DeviceProtection":  device_protection,
        "TechSupport":       tech_support,
        "StreamingTV":       streaming_tv,
        "StreamingMovies":   streaming_movies,
        "Contract":          contract,
        "PaperlessBilling":  paperless,
        "PaymentMethod":     payment_method,
        "revenue_per_tenure": total_charges / tenure if tenure > 0 else monthly_charges,
        "num_services":      num_services,
        "tenure_bucket":     tenure_bucket,
        "high_spend":        int(monthly_charges > 64.76),
        "has_support":       int(any(v == "Yes" for v in [online_security, online_backup, tech_support])),
    }
    df = pd.DataFrame([row])
    df = pd.get_dummies(df)

    # Align with training features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names].astype(float)
    return df


# ── Predict ───────────────────────────────────────────────────────────────────

input_df   = build_input()
churn_prob = model.predict_proba(input_df)[0][1]
churn_pred = churn_prob >= 0.5

# ── Main panel ────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)

col1.metric("Churn Probability", f"{churn_prob:.1%}")
col2.metric("Prediction", "⚠️ WILL CHURN" if churn_pred else "✅ WILL STAY")
col3.metric("Monthly Charges", f"${monthly_charges:.2f}")

# Risk gauge
st.subheader("Risk Level")
if churn_prob < 0.3:
    st.success(f"🟢 Low Risk ({churn_prob:.1%}) — Customer is likely to stay.")
elif churn_prob < 0.6:
    st.warning(f"🟡 Medium Risk ({churn_prob:.1%}) — Monitor this customer.")
else:
    st.error(f"🔴 High Risk ({churn_prob:.1%}) — Immediate retention action needed.")

st.divider()

# ── SHAP explanation ──────────────────────────────────────────────────────────

st.subheader("Why this prediction? (SHAP Explanation)")
st.caption("Features pushing the prediction toward churn (red) or staying (blue)")

shap_values = explainer.shap_values(input_df.values)

fig, ax = plt.subplots(figsize=(10, 5))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.values[0],
        feature_names=feature_names,
    ),
    max_display=15,
    show=False,
)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# ── Retention recommendations ─────────────────────────────────────────────────

st.subheader("💡 Retention Recommendations")

recs = []
if contract == "Month-to-month":
    recs.append("🔒 Offer a discounted annual or 2-year contract to increase switching costs.")
if monthly_charges > 80:
    recs.append("💰 Customer is high-spend — consider a loyalty discount or bundle offer.")
if tenure < 12:
    recs.append("🆕 Early-tenure customer — onboarding check-in call recommended.")
if online_security == "No" and internet_service != "No":
    recs.append("🛡️ Offer Online Security add-on — reduces churn significantly.")
if tech_support == "No" and internet_service != "No":
    recs.append("🛠️ Offer Tech Support — customers with it churn less.")
if payment_method == "Electronic check":
    recs.append("💳 Encourage auto-payment — electronic check users churn more.")
if num_services := input_df.get("num_services", [0])[0] if hasattr(input_df.get("num_services", [0]), '__getitem__') else 0:
    if num_services < 3:
        recs.append("📦 Cross-sell additional services — more services = more stickiness.")

if not recs:
    st.success("No immediate actions needed. Customer profile is stable.")
else:
    for r in recs:
        st.markdown(f"- {r}")

st.divider()

# ── Model performance summary ─────────────────────────────────────────────────

with st.expander("📊 Model Performance Summary"):
    import json
    try:
        with open("models/metrics.json") as f:
            metrics = json.load(f)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Logistic Regression**")
            st.metric("ROC-AUC", metrics["logistic_regression"]["roc_auc"])
            st.metric("PR-AUC",  metrics["logistic_regression"]["pr_auc"])
        with col2:
            st.markdown("**XGBoost** ✅ (selected)")
            st.metric("ROC-AUC", metrics["xgboost"]["roc_auc"])
            st.metric("PR-AUC",  metrics["xgboost"]["pr_auc"])
        st.caption("XGBoost selected for higher ROC-AUC and better handling of class imbalance.")
    except FileNotFoundError:
        st.info("Run training to see metrics.")

with st.expander("🔍 Raw input features sent to model"):
    st.dataframe(input_df.T.rename(columns={0: "value"}), use_container_width=True)
