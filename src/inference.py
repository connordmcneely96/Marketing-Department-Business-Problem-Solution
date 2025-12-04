"""
Model inference utilities for making predictions.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from src.config import MODEL_FILE

# Global cache for loaded model
_MODEL_CACHE = None


def load_model(path: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
    """
    Load model bundle from disk with caching.
    
    Args:
        path: Optional path to model file. If None, uses default from config.
        force_reload: Force reload even if cached
    
    Returns:
        Dictionary containing 'model', 'transformers', and 'metadata'
    """
    global _MODEL_CACHE
    
    # Return cached model if available
    if _MODEL_CACHE is not None and not force_reload:
        return _MODEL_CACHE
    
    model_path = path if path is not None else MODEL_FILE
    
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please train a model first using: python -m src.train"
        )
    
    try:
        bundle = joblib.load(model_path)
        _MODEL_CACHE = bundle
        print(f"✓ Model loaded from {model_path}")
        
        # Print model info if available
        metadata = bundle.get("metadata", {})
        if metadata:
            print(f"  Model: {metadata.get('model_name', 'Unknown')}")
            print(f"  Type: {metadata.get('model_type', 'Unknown')}")
            print(f"  Features: {metadata.get('n_features', 'Unknown')}")
        
        return bundle
    
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def preprocess_input(
    features: Union[Dict[str, Any], List[float], pd.DataFrame],
    transformers: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Preprocess input features using saved transformers.
    
    Args:
        features: Input features as dict, list, or DataFrame
        transformers: Dictionary of preprocessing transformers
        feature_names: Optional list of feature names (for validation)
    
    Returns:
        Preprocessed feature array ready for prediction
    """
    # Convert input to DataFrame
    if isinstance(features, dict):
        df = pd.DataFrame([features])
    elif isinstance(features, list):
        if feature_names:
            df = pd.DataFrame([features], columns=feature_names)
        else:
            df = pd.DataFrame([features])
    elif isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        raise ValueError(f"Unsupported input type: {type(features)}")
    
    # Apply transformers in the same order as training
    # TODO: This is a simplified version. Adapt based on your preprocessing pipeline.
    
    # Handle missing values
    if "imputers" in transformers:
        imputers = transformers["imputers"]
        if "numeric" in imputers:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df[numeric_cols] = imputers["numeric"].transform(df[numeric_cols])
        
        if "categorical" in imputers:
            categorical_cols = df.select_dtypes(include=["object"]).columns
            if len(categorical_cols) > 0:
                df[categorical_cols] = imputers["categorical"].transform(df[categorical_cols])
    
    # Encode categorical features
    if "encoders" in transformers:
        encoders = transformers["encoders"]
        if "onehot_columns" in encoders:
            # Use same encoding as training
            df = pd.get_dummies(df, columns=encoders["onehot_columns"], drop_first=True)
    
    # Scale numeric features
    if "scaler" in transformers and transformers["scaler"] is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = transformers["scaler"].transform(df[numeric_cols])
    
    return df.values


def predict_classification(
    features: Union[Dict[str, Any], List[float], pd.DataFrame],
    return_proba: bool = False,
    model_path: Optional[str] = None,
) -> Union[int, float, Dict[str, Any]]:
    """
    Make a classification prediction.
    
    Args:
        features: Input features (dict, list, or DataFrame)
        return_proba: Whether to return class probabilities
        model_path: Optional path to model file
    
    Returns:
        Predicted class (int) or probabilities (dict) or detailed dict
    """
    # Load model bundle
    bundle = load_model(model_path)
    model = bundle["model"]
    transformers = bundle["transformers"]
    metadata = bundle.get("metadata", {})
    
    # Preprocess input
    feature_names = metadata.get("feature_names")
    X = preprocess_input(features, transformers, feature_names)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get probabilities if available and requested
    result = {"prediction": int(prediction)}
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        result["probabilities"] = probabilities.tolist()
        result["confidence"] = float(max(probabilities))
    
    if return_proba:
        return result
    else:
        return int(prediction)


def predict_regression(
    features: Union[Dict[str, Any], List[float], pd.DataFrame],
    model_path: Optional[str] = None,
) -> float:
    """
    Make a regression prediction.
    
    Args:
        features: Input features (dict, list, or DataFrame)
        model_path: Optional path to model file
    
    Returns:
        Predicted value (float)
    """
    # Load model bundle
    bundle = load_model(model_path)
    model = bundle["model"]
    transformers = bundle["transformers"]
    metadata = bundle.get("metadata", {})
    
    # Preprocess input
    feature_names = metadata.get("feature_names")
    X = preprocess_input(features, transformers, feature_names)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    return float(prediction)


def predict_from_text(
    text: str,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Make prediction from text input (for NLP tasks).
    
    Args:
        text: Input text
        model_path: Optional path to model file
    
    Returns:
        Dictionary with prediction and metadata
    """
    # TODO: Implement text preprocessing and prediction
    # This will be customized based on the NLP project
    raise NotImplementedError("Text prediction not yet implemented. Customize for your NLP project.")


def predict_from_image(
    image_path: Union[str, Path],
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Make prediction from image input (for computer vision tasks).
    
    Args:
        image_path: Path to image file
        model_path: Optional path to model file
    
    Returns:
        Dictionary with prediction and metadata
    """
    # TODO: Implement image preprocessing and prediction
    # This will be customized based on the CV project
    raise NotImplementedError("Image prediction not yet implemented. Customize for your CV project.")


def batch_predict(
    features_list: List[Union[Dict, List]],
    model_path: Optional[str] = None,
) -> List[Any]:
    """
    Make predictions for multiple samples.
    
    Args:
        features_list: List of feature dicts or lists
        model_path: Optional path to model file
    
    Returns:
        List of predictions
    """
    # Load model once
    bundle = load_model(model_path)
    model = bundle["model"]
    transformers = bundle["transformers"]
    metadata = bundle.get("metadata", {})
    
    # Convert all features to DataFrame
    if isinstance(features_list[0], dict):
        df = pd.DataFrame(features_list)
    else:
        feature_names = metadata.get("feature_names")
        df = pd.DataFrame(features_list, columns=feature_names)
    
    # Preprocess
    X = preprocess_input(df, transformers, metadata.get("feature_names"))
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions.tolist()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_info(model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about the loaded model.
    
    Args:
        model_path: Optional path to model file
    
    Returns:
        Dictionary with model metadata
    """
    bundle = load_model(model_path)
    return bundle.get("metadata", {})


def clear_model_cache() -> None:
    """Clear the cached model from memory."""
    global _MODEL_CACHE
    _MODEL_CACHE = None
    print("✓ Model cache cleared")
