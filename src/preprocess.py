"""
Data preprocessing pipeline for ML project.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    RANDOM_SEED,
    TEST_SIZE,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)


def load_raw_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    Args:
        path: Optional path to data file. If None, uses default from config.
    
    Returns:
        DataFrame containing raw data
    """
    file_path = path if path is not None else RAW_DATA_FILE
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded data from {file_path}")
        print(f"  Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        raise
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise


def handle_missing_values(
    df: pd.DataFrame, 
    numeric_strategy: str = "mean",
    categorical_strategy: str = "most_frequent"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        numeric_strategy: Strategy for numeric features ('mean', 'median', 'constant')
        categorical_strategy: Strategy for categorical features ('most_frequent', 'constant')
    
    Returns:
        Tuple of (processed DataFrame, dict of imputers)
    """
    imputers = {}
    df_copy = df.copy()
    
    # Identify numeric and categorical columns
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_copy.select_dtypes(include=['object']).columns.tolist()
    
    # Impute numeric features
    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        df_copy[numeric_cols] = numeric_imputer.fit_transform(df_copy[numeric_cols])
        imputers['numeric'] = numeric_imputer
    
    # Impute categorical features
    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy=categorical_strategy)
        df_copy[categorical_cols] = categorical_imputer.fit_transform(df_copy[categorical_cols])
        imputers['categorical'] = categorical_imputer
    
    return df_copy, imputers


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: list,
    encoding_type: str = "onehot"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical features.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
        encoding_type: 'onehot' or 'label'
    
    Returns:
        Tuple of (processed DataFrame, dict of encoders)
    """
    encoders = {}
    df_copy = df.copy()
    
    if not categorical_cols:
        return df_copy, encoders
    
    if encoding_type == "label":
        for col in categorical_cols:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            encoders[col] = le
    
    elif encoding_type == "onehot":
        # Use pandas get_dummies for simplicity
        df_copy = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=True)
        encoders['onehot_columns'] = categorical_cols
    
    return df_copy, encoders


def scale_numeric_features(
    df: pd.DataFrame,
    numeric_cols: list,
    scaler_type: str = "standard"
) -> Tuple[pd.DataFrame, Any]:
    """
    Scale numeric features.
    
    Args:
        df: Input DataFrame
        numeric_cols: List of numeric column names
        scaler_type: Type of scaler ('standard', 'minmax', etc.)
    
    Returns:
        Tuple of (processed DataFrame, fitted scaler)
    """
    df_copy = df.copy()
    
    if not numeric_cols:
        return df_copy, None
    
    if scaler_type == "standard":
        scaler = StandardScaler()
        df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
    else:
        # Add other scaler types as needed
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
    
    return df_copy, scaler


def preprocess(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    numeric_features: Optional[list] = None,
    categorical_features: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
    
    Returns:
        Tuple of (X features DataFrame, y target Series, transformers dict)
    """
    transformers = {}
    
    # Handle missing values
    df_clean, imputers = handle_missing_values(df)
    transformers['imputers'] = imputers
    
    # Separate features and target
    if target_col in df_clean.columns:
        y = df_clean[target_col]
        X = df_clean.drop(columns=[target_col])
    else:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Encode categorical features if specified
    if categorical_features:
        X, encoders = encode_categorical_features(X, categorical_features)
        transformers['encoders'] = encoders
    
    # Scale numeric features if specified
    if numeric_features:
        # Filter to only those that still exist (after encoding)
        existing_numeric = [col for col in numeric_features if col in X.columns]
        if existing_numeric:
            X, scaler = scale_numeric_features(X, existing_numeric)
            transformers['scaler'] = scaler
    
    print(f"✓ Preprocessing complete")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    return X, y, transformers


def make_train_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Preprocess data and create train/test split.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified splitting
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, transformers)
    """
    # Preprocess the data
    X, y, transformers = preprocess(df, target_col=target_col)
    
    # Create train/test split
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"✓ Train/test split complete")
    print(f"  Train size: {X_train.shape[0]} samples")
    print(f"  Test size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, transformers


def save_processed_data(df: pd.DataFrame, path: Optional[str] = None) -> None:
    """
    Save processed data to CSV.
    
    Args:
        df: DataFrame to save
        path: Optional path. If None, uses default from config.
    """
    file_path = path if path is not None else PROCESSED_DATA_FILE
    df.to_csv(file_path, index=False)
    print(f"✓ Saved processed data to {file_path}")
