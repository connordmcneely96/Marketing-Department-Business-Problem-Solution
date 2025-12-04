"""
Model evaluation utilities for different ML tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    # Classification metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    # Regression metrics
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    # Clustering metrics
    silhouette_score,
)


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    average: str = "binary",
) -> Dict[str, float]:
    """
    Evaluate classification model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for ROC AUC)
        average: Averaging strategy for multi-class ('binary', 'macro', 'weighted')
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # Add ROC AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                # Multi-class classification
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_pred_proba, multi_class="ovr", average=average
                )
        except Exception as e:
            print(f"Warning: Could not compute ROC AUC: {e}")
            metrics["roc_auc"] = None
    
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print detailed classification report."""
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def evaluate_classification_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    average: str = "binary",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple classification models.
    
    Args:
        models: Dictionary of {model_name: fitted_model}
        X_test: Test features
        y_test: Test target
        average: Averaging strategy for metrics
    
    Returns:
        Dictionary of {model_name: metrics_dict}
    """
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
        
        metrics = evaluate_classification(y_test, y_pred, y_pred_proba, average=average)
        results[name] = metrics
        
        print(f"\n{name}:")
        for metric, value in metrics.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
    
    return results


# ============================================================================
# REGRESSION METRICS
# ============================================================================

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate regression model performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }
    
    return metrics


def evaluate_regression_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple regression models.
    
    Args:
        models: Dictionary of {model_name: fitted_model}
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary of {model_name: metrics_dict}
    """
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)
        results[name] = metrics
        
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return results


# ============================================================================
# CLUSTERING METRICS
# ============================================================================

def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    model: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Evaluate clustering model performance.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        model: Optional fitted clustering model (for inertia)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Silhouette score
    if len(np.unique(labels)) > 1:
        metrics["silhouette_score"] = silhouette_score(X, labels)
    
    # Inertia (for KMeans-like models)
    if model is not None and hasattr(model, "inertia_"):
        metrics["inertia"] = model.inertia_
    
    return metrics


# ============================================================================
# TIME SERIES / FORECASTING METRICS
# ============================================================================

def evaluate_forecasting(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate forecasting model performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # SMAPE (Symmetric MAPE)
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
    }
    
    return metrics


# ============================================================================
# GENERAL PURPOSE EVALUATION
# ============================================================================

def get_best_model(
    results: Dict[str, Dict[str, float]],
    metric: str = "accuracy",
    higher_is_better: bool = True,
) -> str:
    """
    Get the best model based on a specific metric.
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
        metric: Metric to compare
        higher_is_better: Whether higher metric values are better
    
    Returns:
        Name of best model
    """
    model_scores = {name: metrics.get(metric, float('-inf' if higher_is_better else 'inf'))
                    for name, metrics in results.items()}
    
    if higher_is_better:
        best_model = max(model_scores, key=model_scores.get)
    else:
        best_model = min(model_scores, key=model_scores.get)
    
    return best_model


def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted table of model results.
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
    """
    if not results:
        print("No results to display")
        return
    
    # Convert to DataFrame for nice formatting
    df_results = pd.DataFrame(results).T
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(df_results.to_string())
    print("=" * 80)
