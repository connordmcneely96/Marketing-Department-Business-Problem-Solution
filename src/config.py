"""
Configuration file for ML project paths and parameters.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# File paths
# TODO: Update these based on your specific project
RAW_DATA_FILE = RAW_DATA_DIR / "data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.csv"
MODEL_FILE = MODELS_DIR / "model.joblib"

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1  # Optional, for train/val/test split

# Feature engineering params
# TODO: Update based on project requirements
NUMERIC_FEATURES = []
CATEGORICAL_FEATURES = []
TARGET_COLUMN = "target"

# Model hyperparameters (defaults)
# TODO: Update based on project type
MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": RANDOM_SEED,
    },
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
    },
    "xgboost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "random_state": RANDOM_SEED,
    },
}

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
