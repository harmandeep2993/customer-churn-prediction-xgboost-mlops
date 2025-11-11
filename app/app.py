import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# =====================================================
# Path setup (so we can import from src)
# =====================================================
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.predict import load_model, predict_churn

# =====================================================
# Load model
# =====================================================
MODEL_PATH = "models/xgb_churn_full_tuned.pkl"
model = load_model(MODEL_PATH)

# =====================================================
# App header
# =====================================================
st.title("📊 Customer Churn Prediction App")
st.markdown("Use this app to predict whether a customer is likely to churn based on their account information.")

# =====================================================
# Input section with 2 columns
# =====================================================
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["Yes", "No"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with col2:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 2500.0)

    submitted = st.form_submit_button("Predict Churn")

# =====================================================
# Prediction section
# =====================================================
if submitted:
    input_dict = {
        "gender": gender,
        "SeniorCitizen": "Yes" if senior == "Yes" else "No",
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([input_dict])
    pred, prob = predict_churn(input_df, model)

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if pred[0] == 1:
        st.error("🟥 **Churn Likely**")
    else:
        st.success("🟩 **No Churn**")

    st.metric("Churn Probability", f"{prob[0]*100:.2f}%")