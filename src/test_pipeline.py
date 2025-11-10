import pandas as pd
from src.predict import load_model, predict_churn

# Load model
model_path = 'models/xgb_churn_full_tuned.pkl'
model = load_model(model_path)

# Sample input
sample_input = pd.DataFrame([{
    'gender': 'Female',
    'SeniorCitizen': 'Yes',
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 2,
    'PhoneService': 'Yes',
    'MultipleLines': 'Yes',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'Yes',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 95.0,
    'TotalCharges': 95.0
}])

# Predict
pred, prob = predict_churn(sample_input, model)

print('=== Sanity Test ===')
print(f'Prediction: {"Churn" if pred[0] == 1 else "No Churn"}')
print(f'Probability: {prob[0]:.2f}')