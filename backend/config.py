import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MODEL_NAME = "xgboost_model.pkl"
SCALER_NAME = "feature_scaler.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"

API_TITLE = "Real Estate Price Prediction API"
API_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False") == "True"

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}
