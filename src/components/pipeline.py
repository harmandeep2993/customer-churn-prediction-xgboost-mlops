import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluator import ModelEvaluator


class ChurnPipeline:
    """End-to-end pipeline for training, evaluating, and saving churn model."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)

        self.ingestion = DataIngestion(config_path)
        self.preprocessor = DataPreprocessing()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()

    def run(self):
        """Execute the full training pipeline."""
        try:
            self.logger.info("=== Starting Churn Prediction Training Pipeline ===")

            # 1. Load raw data
            df = self.ingestion.load_dataset()

            # 2. Preprocess
            df, encoder = self.preprocessor.full_preprocess_pipeline(df, fit_encoder=True)
            self.ingestion.save_processed_data(df, self.config["data"]["processed_path"])
            joblib.dump(encoder, self.config["model"]["encoder"])
            self.logger.info("Preprocessing complete. Encoder saved.")

            # 3. Split
            X = df.drop("Churn", axis=1)
            # ✅ FIX: Convert target variable to numeric
            y = df["Churn"].replace({"Yes": 1, "No": 0})

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config["train"]["test_size"],
                stratify=y,
                random_state=self.config["train"]["random_state"]
            )
            self.logger.info("Data split into training and test sets.")

            # 4. Define hyperparameter space
            param_dist = self.config["train"]["param_dist"]

            # 5. Train and tune
            model, best_params = self.trainer.train_xgb_model(
                X_train=X_train, y_train=y_train, n_iter=15, scoring='recall'
            )
            self.logger.info(f"Model trained successfully. Best params: {best_params}")

            # 6. Save artifacts
            joblib.dump(model, self.config["model"]["path"])
            joblib.dump(X_train.columns.tolist(), self.config["model"]["columns"])
            self.logger.info("Model and feature list saved.")

            # 7. Evaluate
            self.logger.info("Starting model evaluation.")
            metrics, cm = self.evaluator.evaluate_model(model, X_test, y_test)

            self.logger.info("=== Pipeline Completed Successfully ===")
            return metrics, cm

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

if __name__ == "__main__":
    pipeline = ChurnPipeline()
    metrics, cm = pipeline.run()
    print("\nFinal Evaluation Metrics:")
    print(metrics)