import joblib
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.components.data_preprocessing import DataPreprocessing

class ModelPredictor:
    """Handles loading model and making churn predictions."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)
        self.model_path = self.config["model"]["path"]
        self.columns_path = self.config["model"]["columns"]
        self.encoder_path = self.config["model"]["encoder"]

    def load_model(self):
        """Load trained churn prediction model."""
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            return joblib.load(self.model_path)
        except FileNotFoundError:
            self.logger.error(f"Model not found at {self.model_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def predict_churn(self, input_df: pd.DataFrame, model):
        """
        Preprocess input data and predict churn probability.
        Ensures exact same structure as training data.
        """
        try:
            self.logger.info("Starting churn prediction.")
            encoder = joblib.load(self.encoder_path)
            train_columns = joblib.load(self.columns_path)

            preprocessor = DataPreprocessing()
            processed = preprocessor.full_preprocess_pipeline(
                input_df.copy(), encoder=encoder, fit_encoder=False
            )

            # Align features with training columns
            processed = processed.reindex(columns=train_columns, fill_value=0)

            # Check if all zeros (means mismatch)
            if processed.sum().sum() == 0:
                self.logger.warning("Processed input contains all zeros — possible feature mismatch.")
                print("⚠️ Feature mismatch: check category names and preprocessing consistency.")

            print("\n=== DEBUG SHAPES ===")
            print("Processed shape:", processed.shape)
            print("Non-zero count:", (processed != 0).sum().sum())
            print("First row preview:\n", processed.head(1))

            # Predict
            pred = model.predict(processed)
            prob = model.predict_proba(processed)[:, 1]

            self.logger.info("Prediction complete.")
            return pred, prob

        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise

if __name__ == "__main__":
    predictor = ModelPredictor()
    model = predictor.load_model()

    sample_input = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5,
        "TotalCharges": 280.0
    }])

    pred, prob = predictor.predict_churn(sample_input, model)
    print(f"Prediction: {'Churn' if pred[0] else 'No Churn'} | Probability: {prob[0]:.2f}")