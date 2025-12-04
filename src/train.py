"""
Model training pipeline.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import MODEL_FILE, MODEL_PARAMS, RANDOM_SEED, TARGET_COLUMN
from src.preprocess import load_raw_data, make_train_test_split
from src.evaluate import (
    evaluate_classification_models,
    get_best_model,
    print_results_table,
    print_classification_report,
)


def get_classification_models() -> Dict[str, Any]:
    """
    Initialize classification models with default hyperparameters.
    
    Returns:
        Dictionary of {model_name: model_instance}
    """
    models = {
        "Logistic Regression": LogisticRegression(**MODEL_PARAMS.get("logistic_regression", {})),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(**MODEL_PARAMS.get("random_forest", {})),
        "XGBoost": XGBClassifier(**MODEL_PARAMS.get("xgboost", {})),
    }
    return models


def train_single_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Model",
) -> Any:
    """
    Train a single model.
    
    Args:
        model: Sklearn-compatible model instance
        X_train: Training features
        y_train: Training target
        model_name: Name for logging
    
    Returns:
        Fitted model
    """
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    print(f"✓ {model_name} trained successfully")
    return model


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train multiple classification models and compare performance.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: Optional dictionary of models. If None, uses defaults.
    
    Returns:
        Dictionary of {model_name: fitted_model}
    """
    if models is None:
        models = get_classification_models()
    
    # Train all models
    trained_models = {}
    for name, model in models.items():
        trained_models[name] = train_single_model(model, X_train, y_train, name)
    
    # Evaluate all models
    print("\n" + "=" * 80)
    print("EVALUATING MODELS")
    print("=" * 80)
    
    # Determine if binary or multiclass
    n_classes = len(np.unique(y_train))
    average = "binary" if n_classes == 2 else "weighted"
    
    results = evaluate_classification_models(trained_models, X_test, y_test, average=average)
    
    # Print results table
    print_results_table(results)
    
    # Get best model
    best_model_name = get_best_model(results, metric="f1_score", higher_is_better=True)
    print(f"\n🏆 Best model: {best_model_name}")
    
    # Print detailed report for best model
    best_model = trained_models[best_model_name]
    y_pred = best_model.predict(X_test)
    print_classification_report(y_test, y_pred)
    
    return trained_models, results, best_model_name


def save_model_bundle(
    model: Any,
    transformers: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> None:
    """
    Save model, transformers, and metadata as a single bundle.
    
    Args:
        model: Trained model
        transformers: Dictionary of preprocessing transformers
        metadata: Optional metadata (metrics, feature names, etc.)
        path: Optional save path. If None, uses default from config.
    """
    bundle = {
        "model": model,
        "transformers": transformers,
        "metadata": metadata or {},
    }
    
    save_path = path if path is not None else MODEL_FILE
    joblib.dump(bundle, save_path)
    print(f"\n✓ Model bundle saved to {save_path}")


def train_models(
    data_path: Optional[str] = None,
    target_col: str = TARGET_COLUMN,
    model_type: str = "classification",
) -> None:
    """
    Main training pipeline.
    
    Args:
        data_path: Optional path to data file
        target_col: Name of target column
        model_type: Type of ML task ('classification', 'regression', etc.)
    """
    print("=" * 80)
    print("STARTING TRAINING PIPELINE")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_raw_data(data_path)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Target: {target_col}")
    
    # Preprocess and split
    print("\n[2/5] Preprocessing and splitting data...")
    # TODO: Update stratify parameter based on project type
    X_train, X_test, y_train, y_test, transformers = make_train_test_split(
        df, target_col=target_col, stratify=False
    )
    
    # Train models
    print("\n[3/5] Training models...")
    if model_type == "classification":
        trained_models, results, best_model_name = train_classification_models(
            X_train, y_train, X_test, y_test
        )
        best_model = trained_models[best_model_name]
        
        # Save best model
        print("\n[4/5] Saving best model...")
        metadata = {
            "model_name": best_model_name,
            "model_type": model_type,
            "metrics": results[best_model_name],
            "feature_names": list(X_train.columns),
            "n_features": X_train.shape[1],
            "target_column": target_col,
        }
        save_model_bundle(best_model, transformers, metadata)
    
    else:
        # TODO: Implement other model types (regression, clustering, etc.)
        raise NotImplementedError(f"Model type '{model_type}' not yet implemented")
    
    print("\n[5/5] Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Run training pipeline
    train_models()
