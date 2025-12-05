"""
Configuration file for Customer Segmentation ML project.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# File paths
RAW_DATA_FILE = RAW_DATA_DIR / "Marketing_data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.csv"
SCALED_DATA_FILE = PROCESSED_DATA_DIR / "scaled_data.csv"

# Model files
KMEANS_MODEL_FILE = MODELS_DIR / "kmeans_model.joblib"
AUTOENCODER_MODEL_FILE = MODELS_DIR / "autoencoder_model.h5"
SCALER_FILE = MODELS_DIR / "scaler.joblib"
PCA_FILE = MODELS_DIR / "pca_model.joblib"

# General parameters
RANDOM_SEED = 42

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Column to drop (not useful for clustering)
ID_COLUMN = "CUST_ID"

# All numeric features (17 features after dropping CUST_ID)
NUMERIC_FEATURES = [
    "BALANCE",
    "BALANCE_FREQUENCY",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
    "TENURE",
]

# No categorical features in this dataset
CATEGORICAL_FEATURES = []

# No target column (unsupervised learning)
TARGET_COLUMN = None

# ============================================================================
# CLUSTERING HYPERPARAMETERS
# ============================================================================

# K-Means parameters
KMEANS_PARAMS = {
    "n_clusters": 7,  # Optimal based on elbow method from notebook
    "random_state": RANDOM_SEED,
    "n_init": 10,
    "max_iter": 300,
}

# Range for elbow method analysis
ELBOW_RANGE = range(1, 20)

# ============================================================================
# AUTOENCODER HYPERPARAMETERS
# ============================================================================

AUTOENCODER_PARAMS = {
    "input_dim": len(NUMERIC_FEATURES),  # 17
    "encoding_dim": 10,  # Bottleneck layer
    "hidden_layers": [7, 500, 500, 2000],  # Encoder architecture
    "activation": "relu",
    "optimizer": "adam",
    "loss": "mean_squared_error",
    "epochs": 25,
    "batch_size": 128,
    "validation_split": 0.1,
}

# ============================================================================
# PCA PARAMETERS
# ============================================================================

PCA_PARAMS = {
    "n_components": 2,  # For 2D visualization
}

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

# Missing value imputation strategy
IMPUTATION_STRATEGY = "mean"  # Fill with column mean

# Scaling method
SCALING_METHOD = "standard"  # StandardScaler

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

VISUALIZATION_PARAMS = {
    "figsize": (10, 8),
    "color_palette": "viridis",
    "scatter_alpha": 0.6,
    "scatter_size": 50,
}

# ============================================================================
# BUSINESS SEGMENTS (Cluster Interpretations)
# ============================================================================

# These will be populated after clustering analysis
CLUSTER_NAMES = {
    0: "Transactors",
    1: "Revolvers",
    2: "VIP/Prime",
    3: "Low Activity",
    4: "Cash Advancers",
    5: "Installment Buyers",
    6: "Moderate Users",
}

CLUSTER_DESCRIPTIONS = {
    0: "Customers with low balance, low cash advance, and some full payments",
    1: "Customers who don't pay off balance in full, revolve credit",
    2: "High credit limit, high purchases, premium customers",
    3: "Low engagement, minimal transactions",
    4: "High cash advance usage, financial stress signals",
    5: "Prefer installment purchases over one-off",
    6: "Balanced usage across all features",
}

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
