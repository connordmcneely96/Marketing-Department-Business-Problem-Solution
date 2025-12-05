"""
Data preprocessing pipeline for Customer Segmentation.

This module handles data loading, cleaning, and feature engineering
for the credit card customer clustering project.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

from src.config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    SCALED_DATA_FILE,
    SCALER_FILE,
    NUMERIC_FEATURES,
    ID_COLUMN,
    IMPUTATION_STRATEGY,
)


def load_raw_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw credit card customer data from CSV file.

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
        print(f"  Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        raise
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise


def explore_data(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.

    Args:
        df: Input DataFrame
    """
    print("\n" + "=" * 80)
    print("DATA EXPLORATION")
    print("=" * 80)

    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.shape[1]}")
    print(f"Rows: {df.shape[0]}")

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values")

    print("\nBasic Statistics:")
    print(df.describe())


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = IMPUTATION_STRATEGY
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle missing values in the dataset.

    Based on notebook analysis:
    - MINIMUM_PAYMENTS has missing values
    - CREDIT_LIMIT has missing values
    - Strategy: Fill with column mean

    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', 'most_frequent')

    Returns:
        Tuple of (processed DataFrame, imputers dict)
    """
    df_copy = df.copy()
    imputers = {}

    # Check for missing values
    missing_before = df_copy.isnull().sum().sum()
    print(f"\n[Preprocessing] Missing values before: {missing_before}")

    if missing_before > 0:
        print(f"  Columns with missing values:")
        missing_cols = df_copy.columns[df_copy.isnull().any()].tolist()
        for col in missing_cols:
            missing_count = df_copy[col].isnull().sum()
            print(f"    - {col}: {missing_count} ({missing_count/len(df_copy)*100:.2f}%)")

        # Fill missing values with mean
        for col in missing_cols:
            if df_copy[col].dtype in [np.float64, np.int64]:
                imputer = SimpleImputer(strategy=strategy)
                df_copy[[col]] = imputer.fit_transform(df_copy[[col]])
                imputers[col] = imputer

    missing_after = df_copy.isnull().sum().sum()
    print(f"  Missing values after: {missing_after}")
    print(f"✓ Missing values handled")

    return df_copy, imputers


def drop_id_column(df: pd.DataFrame, id_col: str = ID_COLUMN) -> pd.DataFrame:
    """
    Drop customer ID column as it's not useful for clustering.

    Args:
        df: Input DataFrame
        id_col: Name of ID column to drop

    Returns:
        DataFrame without ID column
    """
    if id_col in df.columns:
        df = df.drop(columns=[id_col])
        print(f"✓ Dropped ID column: {id_col}")
    else:
        print(f"⚠️  ID column '{id_col}' not found in dataset")

    return df


def scale_features(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numeric features using StandardScaler.

    Critical for K-Means clustering as it's distance-based.

    Args:
        df: Input DataFrame
        feature_cols: List of columns to scale. If None, scales all numeric columns.
        scaler: Pre-fitted scaler. If None, creates new one.
        fit: Whether to fit the scaler (True for training, False for inference)

    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df_scaled = df.copy()

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        scaled_values = scaler.fit_transform(df[feature_cols])
        print(f"✓ Fitted and transformed {len(feature_cols)} features")
    else:
        scaled_values = scaler.transform(df[feature_cols])
        print(f"✓ Transformed {len(feature_cols)} features using pre-fitted scaler")

    df_scaled[feature_cols] = scaled_values

    return df_scaled, scaler


def preprocess_pipeline(
    df: pd.DataFrame,
    drop_id: bool = True,
    handle_missing: bool = True,
    scale: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete preprocessing pipeline for customer segmentation.

    Steps:
    1. Handle missing values (fill with mean)
    2. Drop ID column
    3. Scale numeric features

    Args:
        df: Input DataFrame
        drop_id: Whether to drop ID column
        handle_missing: Whether to handle missing values
        scale: Whether to scale features

    Returns:
        Tuple of (preprocessed DataFrame, transformers dict)
    """
    transformers = {}

    print("\n" + "=" * 80)
    print("PREPROCESSING PIPELINE")
    print("=" * 80)

    # Step 1: Handle missing values
    if handle_missing:
        df, imputers = handle_missing_values(df)
        transformers['imputers'] = imputers

    # Step 2: Drop ID column
    if drop_id:
        df = drop_id_column(df)

    # Step 3: Scale features
    if scale:
        df_scaled, scaler = scale_features(df, feature_cols=NUMERIC_FEATURES)
        transformers['scaler'] = scaler
    else:
        df_scaled = df

    print(f"\n✓ Preprocessing complete")
    print(f"  Final shape: {df_scaled.shape}")
    print(f"  Features: {list(df_scaled.columns)}")

    return df_scaled, transformers


def save_processed_data(
    df: pd.DataFrame,
    path: Optional[str] = None,
    scaled: bool = False,
) -> None:
    """
    Save processed data to CSV.

    Args:
        df: DataFrame to save
        path: Optional save path. If None, uses default from config.
        scaled: Whether this is scaled data (affects default path)
    """
    if path is None:
        path = SCALED_DATA_FILE if scaled else PROCESSED_DATA_FILE

    df.to_csv(path, index=False)
    print(f"✓ Saved processed data to {path}")


def save_transformers(transformers: Dict[str, Any]) -> None:
    """
    Save preprocessing transformers (scaler, imputers) to disk.

    Args:
        transformers: Dictionary of fitted transformers
    """
    if 'scaler' in transformers:
        joblib.dump(transformers['scaler'], SCALER_FILE)
        print(f"✓ Saved scaler to {SCALER_FILE}")

    # Save imputers if needed (though usually we use scaler in inference)
    for key, value in transformers.items():
        if key != 'scaler' and hasattr(value, 'transform'):
            filepath = SCALER_FILE.parent / f"{key}.joblib"
            joblib.dump(value, filepath)
            print(f"✓ Saved {key} to {filepath}")


def load_scaler(path: Optional[str] = None) -> StandardScaler:
    """
    Load saved scaler from disk.

    Args:
        path: Optional path to scaler file

    Returns:
        Loaded StandardScaler
    """
    scaler_path = path if path is not None else SCALER_FILE

    try:
        scaler = joblib.load(scaler_path)
        print(f"✓ Loaded scaler from {scaler_path}")
        return scaler
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            "Please run preprocessing first."
        )


def preprocess_for_inference(
    df: pd.DataFrame,
    scaler_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Preprocess new data for inference using saved scaler.

    Args:
        df: Input DataFrame
        scaler_path: Optional path to saved scaler

    Returns:
        Preprocessed DataFrame ready for clustering
    """
    # Load scaler
    scaler = load_scaler(scaler_path)

    # Handle missing values
    df_clean, _ = handle_missing_values(df)

    # Drop ID if present
    if ID_COLUMN in df_clean.columns:
        df_clean = drop_id_column(df_clean)

    # Scale using pre-fitted scaler
    df_scaled, _ = scale_features(
        df_clean,
        feature_cols=NUMERIC_FEATURES,
        scaler=scaler,
        fit=False,
    )

    return df_scaled


if __name__ == "__main__":
    # Example usage: Run full preprocessing pipeline
    print("Running preprocessing pipeline...")

    # Load data
    df = load_raw_data()

    # Explore
    explore_data(df)

    # Preprocess
    df_processed, transformers = preprocess_pipeline(df)

    # Save
    save_processed_data(df_processed, scaled=True)
    save_transformers(transformers)

    print("\n✓ Preprocessing complete!")
