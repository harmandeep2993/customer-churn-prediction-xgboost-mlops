from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.predict import load_model, predict_churn

app = FastAPI()
model = load_model()

# Define input schema
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str

# Create Api to predict
@app.post("/predict")
def predict(input_data: CustomerInput):
    data = pd.DataFrame([input_data.dict()])
    pred, prob = predict_churn(data, model)
    return {
        "prediction": int(pred[0]),
        "probability": float(prob[0])
    }