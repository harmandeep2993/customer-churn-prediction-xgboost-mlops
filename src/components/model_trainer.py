from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from src.utils.logger import get_logger


class ModelTrainer:
    """Handles training and tuning of XGBoost model."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def train_xgb_model(self, X_train, y_train, param_dist, n_iter=10, scoring='f1'):
        """
        Train an XGBoost classifier using RandomizedSearchCV.

        Parameters:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            param_dist (dict): Hyperparameter search space.
            n_iter (int): Number of search iterations.
            scoring (str): Metric for model evaluation.

        Returns:
            tuple: (best_model, best_params)
        """
        self.logger.info("Initializing XGBoost model training with hyperparameter tuning.")

        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )

        try:
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            self.logger.info(f"Model training complete. Best parameters: {best_params}")
            return best_model, best_params
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise

if __name__ == "__main__":
    print("ModelTrainer module is ready for import.")