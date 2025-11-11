import pandas as pd
import os
from src.utils.logger import get_logger
from src.utils.config_loader import load_config


class DataIngestion:
    """Handles loading and inspecting raw customer churn data."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)
        self.raw_data_path = self.config["data"]["raw_path"]

    def load_dataset(self) -> pd.DataFrame:
        """Load dataset from CSV file."""
        if not os.path.exists(self.raw_data_path):
            self.logger.error(f"Raw data not found at {self.raw_data_path}")
            raise FileNotFoundError(f"{self.raw_data_path} not found")

        df = pd.read_csv(self.raw_data_path)
        self.logger.info(f"Loaded dataset with shape {df.shape}")
        return df

    @staticmethod
    def df_overview(df: pd.DataFrame):
        """Print dataset overview: shape, types, and missing values."""
        print("\n=== Shape ===")
        print(df.shape)

        print("\n=== Dtypes ===")
        print(df.dtypes)

        print("\n=== Missing Values ===")
        print(df.isna().sum().sort_values(ascending=False).head(20))


if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.load_dataset()
    ingestion.df_overview(df)