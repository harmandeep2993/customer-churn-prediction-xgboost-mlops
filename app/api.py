from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.pipeline import full_preprocess
from src.predict import load_model, predict

# Initialize app and load model
app = FastAPI()
model = load_model("models/xgb_churn_model.pkl")

# Define input schema
class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str  # handled as numeric later

# Define API route
@app.post("/predict")
def predict_churn(data: ChurnInput):
    df = pd.DataFrame([data.dict()])
    df = full_preprocess(df)
    prediction = predict(model, df)[0]
    return {"prediction": int(prediction)}