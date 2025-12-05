"""
Model inference utilities for Customer Segmentation.

This module provides functions to predict cluster assignments
for new credit card customers.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from src.config import (
    KMEANS_MODEL_FILE,
    SCALER_FILE,
    CLUSTER_NAMES,
    CLUSTER_DESCRIPTIONS,
    NUMERIC_FEATURES,
)
from src.preprocess import preprocess_for_inference

# Global cache for loaded models
_MODEL_CACHE = {}


def load_clustering_models(
    kmeans_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """
    Load K-Means model and scaler with caching.

    Args:
        kmeans_path: Optional path to K-Means model
        scaler_path: Optional path to scaler
        force_reload: Force reload even if cached

    Returns:
        Dictionary containing 'kmeans' and 'scaler'
    """
    global _MODEL_CACHE

    # Return cached models if available
    if _MODEL_CACHE and not force_reload:
        return _MODEL_CACHE

    # Load K-Means model
    model_path = kmeans_path if kmeans_path is not None else KMEANS_MODEL_FILE

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"K-Means model not found at {model_path}. "
            "Please train a model first using: python -m src.train"
        )

    try:
        kmeans = joblib.load(model_path)
        print(f"✓ K-Means model loaded from {model_path}")
        print(f"  Number of clusters: {kmeans.n_clusters}")
    except Exception as e:
        raise RuntimeError(f"Error loading K-Means model: {e}")

    # Load scaler
    scaler_file = scaler_path if scaler_path is not None else SCALER_FILE

    if not Path(scaler_file).exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_file}. "
            "Please train models first."
        )

    try:
        scaler = joblib.load(scaler_file)
        print(f"✓ Scaler loaded from {scaler_file}")
    except Exception as e:
        raise RuntimeError(f"Error loading scaler: {e}")

    # Cache models
    _MODEL_CACHE = {
        "kmeans": kmeans,
        "scaler": scaler,
    }

    return _MODEL_CACHE


def predict_cluster(
    features: Union[Dict[str, float], pd.DataFrame],
    return_details: bool = True,
    kmeans_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
) -> Union[int, Dict[str, Any]]:
    """
    Predict cluster assignment for a customer.

    Args:
        features: Customer features as dict or DataFrame
        return_details: Whether to return detailed cluster information
        kmeans_path: Optional path to K-Means model
        scaler_path: Optional path to scaler

    Returns:
        Cluster ID (int) or detailed dictionary
    """
    # Load models
    models = load_clustering_models(kmeans_path, scaler_path)
    kmeans = models["kmeans"]
    scaler = models["scaler"]

    # Convert features to DataFrame if dict
    if isinstance(features, dict):
        df = pd.DataFrame([features])
    elif isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        raise ValueError(f"Unsupported input type: {type(features)}")

    # Ensure all required features are present
    missing_features = set(NUMERIC_FEATURES) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}. "
            f"Expected features: {NUMERIC_FEATURES}"
        )

    # Preprocess (scale) features
    df_scaled = pd.DataFrame(
        scaler.transform(df[NUMERIC_FEATURES]),
        columns=NUMERIC_FEATURES,
        index=df.index
    )

    # Predict cluster
    cluster_id = kmeans.predict(df_scaled.values)[0]

    if not return_details:
        return int(cluster_id)

    # Get cluster distances (distance to each cluster center)
    distances = kmeans.transform(df_scaled.values)[0]
    closest_clusters = np.argsort(distances)[:3]  # Top 3 closest clusters

    # Build detailed result
    result = {
        "cluster_id": int(cluster_id),
        "cluster_name": CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}"),
        "cluster_description": CLUSTER_DESCRIPTIONS.get(cluster_id, ""),
        "distance_to_center": float(distances[cluster_id]),
        "confidence_score": _calculate_confidence(distances, cluster_id),
        "closest_clusters": [
            {
                "cluster_id": int(cid),
                "cluster_name": CLUSTER_NAMES.get(cid, f"Cluster {cid}"),
                "distance": float(distances[cid]),
            }
            for cid in closest_clusters
        ],
        "customer_features": features if isinstance(features, dict) else features.iloc[0].to_dict(),
    }

    return result


def _calculate_confidence(distances: np.ndarray, assigned_cluster: int) -> float:
    """
    Calculate confidence score based on distance to cluster centers.

    Confidence is higher when the customer is much closer to their assigned
    cluster than to other clusters.

    Args:
        distances: Array of distances to each cluster center
        assigned_cluster: The assigned cluster ID

    Returns:
        Confidence score between 0 and 1
    """
    assigned_distance = distances[assigned_cluster]
    other_distances = np.delete(distances, assigned_cluster)

    if len(other_distances) == 0:
        return 1.0

    # Average distance to other clusters
    avg_other_distance = np.mean(other_distances)

    # Confidence: how much closer we are to assigned vs others
    # Higher when assigned_distance << avg_other_distance
    if avg_other_distance == 0:
        return 1.0

    confidence = 1.0 - (assigned_distance / avg_other_distance)
    confidence = np.clip(confidence, 0.0, 1.0)

    return float(confidence)


def predict_batch(
    features_df: pd.DataFrame,
    return_details: bool = False,
    kmeans_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
) -> Union[np.ndarray, List[Dict[str, Any]]]:
    """
    Predict cluster assignments for multiple customers.

    Args:
        features_df: DataFrame with customer features
        return_details: Whether to return detailed information
        kmeans_path: Optional path to K-Means model
        scaler_path: Optional path to scaler

    Returns:
        Array of cluster IDs or list of detailed dictionaries
    """
    # Load models
    models = load_clustering_models(kmeans_path, scaler_path)
    kmeans = models["kmeans"]
    scaler = models["scaler"]

    # Ensure all features are present
    missing_features = set(NUMERIC_FEATURES) - set(features_df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Preprocess
    df_scaled = pd.DataFrame(
        scaler.transform(features_df[NUMERIC_FEATURES]),
        columns=NUMERIC_FEATURES,
        index=features_df.index
    )

    # Predict
    cluster_ids = kmeans.predict(df_scaled.values)

    if not return_details:
        return cluster_ids

    # Build detailed results
    distances = kmeans.transform(df_scaled.values)
    results = []

    for i, cluster_id in enumerate(cluster_ids):
        result = {
            "cluster_id": int(cluster_id),
            "cluster_name": CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}"),
            "cluster_description": CLUSTER_DESCRIPTIONS.get(cluster_id, ""),
            "distance_to_center": float(distances[i][cluster_id]),
            "confidence_score": _calculate_confidence(distances[i], cluster_id),
        }
        results.append(result)

    return results


def get_cluster_info(cluster_id: int) -> Dict[str, Any]:
    """
    Get information about a specific cluster.

    Args:
        cluster_id: Cluster ID

    Returns:
        Dictionary with cluster information
    """
    return {
        "cluster_id": cluster_id,
        "cluster_name": CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}"),
        "cluster_description": CLUSTER_DESCRIPTIONS.get(cluster_id, "No description available"),
    }


def get_all_clusters_info() -> List[Dict[str, Any]]:
    """
    Get information about all clusters.

    Returns:
        List of cluster information dictionaries
    """
    # Load model to get number of clusters
    models = load_clustering_models()
    kmeans = models["kmeans"]

    clusters_info = []
    for cluster_id in range(kmeans.n_clusters):
        clusters_info.append(get_cluster_info(cluster_id))

    return clusters_info


def clear_model_cache() -> None:
    """Clear the cached models from memory."""
    global _MODEL_CACHE
    _MODEL_CACHE = {}
    print("✓ Model cache cleared")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_customer_profile(
    balance: float = 0.0,
    balance_frequency: float = 0.0,
    purchases: float = 0.0,
    oneoff_purchases: float = 0.0,
    installments_purchases: float = 0.0,
    cash_advance: float = 0.0,
    purchases_frequency: float = 0.0,
    oneoff_purchases_frequency: float = 0.0,
    purchases_installments_frequency: float = 0.0,
    cash_advance_frequency: float = 0.0,
    cash_advance_trx: int = 0,
    purchases_trx: int = 0,
    credit_limit: float = 0.0,
    payments: float = 0.0,
    minimum_payments: float = 0.0,
    prc_full_payment: float = 0.0,
    tenure: int = 12,
) -> Dict[str, float]:
    """
    Create a customer profile dictionary with all required features.

    This is a helper function to make it easier to create feature inputs.
    All parameters default to sensible values.

    Args:
        All 17 credit card features as named parameters

    Returns:
        Dictionary with all features
    """
    return {
        "BALANCE": float(balance),
        "BALANCE_FREQUENCY": float(balance_frequency),
        "PURCHASES": float(purchases),
        "ONEOFF_PURCHASES": float(oneoff_purchases),
        "INSTALLMENTS_PURCHASES": float(installments_purchases),
        "CASH_ADVANCE": float(cash_advance),
        "PURCHASES_FREQUENCY": float(purchases_frequency),
        "ONEOFF_PURCHASES_FREQUENCY": float(oneoff_purchases_frequency),
        "PURCHASES_INSTALLMENTS_FREQUENCY": float(purchases_installments_frequency),
        "CASH_ADVANCE_FREQUENCY": float(cash_advance_frequency),
        "CASH_ADVANCE_TRX": int(cash_advance_trx),
        "PURCHASES_TRX": int(purchases_trx),
        "CREDIT_LIMIT": float(credit_limit),
        "PAYMENTS": float(payments),
        "MINIMUM_PAYMENTS": float(minimum_payments),
        "PRC_FULL_PAYMENT": float(prc_full_payment),
        "TENURE": int(tenure),
    }


if __name__ == "__main__":
    # Example usage
    print("Customer Segmentation Inference Module")
    print("\nExample: Predict cluster for a sample customer")

    # Create sample customer
    sample_customer = create_customer_profile(
        balance=2000.0,
        balance_frequency=0.9,
        purchases=1500.0,
        oneoff_purchases=800.0,
        installments_purchases=700.0,
        cash_advance=500.0,
        purchases_frequency=0.8,
        credit_limit=5000.0,
        payments=2500.0,
        prc_full_payment=0.3,
        tenure=12,
    )

    try:
        # Predict cluster
        result = predict_cluster(sample_customer, return_details=True)

        print("\nPrediction Result:")
        print(f"  Cluster: {result['cluster_name']} (ID: {result['cluster_id']})")
        print(f"  Description: {result['cluster_description']}")
        print(f"  Confidence: {result['confidence_score']:.2%}")
        print(f"  Distance to center: {result['distance_to_center']:.4f}")

    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("Please train the model first: python -m src.train")
