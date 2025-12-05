"""
Model training pipeline for Customer Segmentation.

This module implements K-Means clustering with optional autoencoder
dimensionality reduction for credit card customer segmentation.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# TensorFlow/Keras imports
try:
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️  TensorFlow/Keras not available. Autoencoder functionality will be disabled.")

from src.config import (
    KMEANS_MODEL_FILE,
    AUTOENCODER_MODEL_FILE,
    PCA_FILE,
    KMEANS_PARAMS,
    AUTOENCODER_PARAMS,
    PCA_PARAMS,
    ELBOW_RANGE,
    RANDOM_SEED,
)
from src.preprocess import load_raw_data, preprocess_pipeline, save_transformers
from src.evaluate import evaluate_clustering, plot_elbow_curve, plot_clusters_2d


# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================

def elbow_method_analysis(
    X: pd.DataFrame,
    k_range: range = ELBOW_RANGE,
    plot: bool = True,
) -> List[float]:
    """
    Perform elbow method analysis to find optimal number of clusters.

    Args:
        X: Scaled feature DataFrame
        k_range: Range of k values to try
        plot: Whether to plot the elbow curve

    Returns:
        List of WCSS (inertia) scores for each k
    """
    print("\n" + "=" * 80)
    print("ELBOW METHOD ANALYSIS")
    print("=" * 80)

    wcss_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        kmeans.fit(X)
        wcss_scores.append(kmeans.inertia_)

        if k % 5 == 0:
            print(f"  k={k}: WCSS={kmeans.inertia_:.2f}")

    print(f"\n✓ Elbow analysis complete for k={k_range.start} to {k_range.stop-1}")

    if plot:
        plot_elbow_curve(list(k_range), wcss_scores, title="Elbow Method - Original Data")

    return wcss_scores


def train_kmeans(
    X: pd.DataFrame,
    n_clusters: int = KMEANS_PARAMS["n_clusters"],
    **kwargs
) -> KMeans:
    """
    Train K-Means clustering model.

    Args:
        X: Scaled feature DataFrame
        n_clusters: Number of clusters
        **kwargs: Additional KMeans parameters

    Returns:
        Fitted KMeans model
    """
    print(f"\n[Training K-Means] n_clusters={n_clusters}")

    # Merge params
    params = {**KMEANS_PARAMS, **kwargs}
    params['n_clusters'] = n_clusters

    # Train model
    kmeans = KMeans(**params)
    kmeans.fit(X)

    print(f"✓ K-Means trained successfully")
    print(f"  Inertia (WCSS): {kmeans.inertia_:.2f}")
    print(f"  Iterations: {kmeans.n_iter_}")

    # Get cluster assignments
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)

    print(f"  Cluster distribution:")
    for cluster_id, count in zip(unique, counts):
        print(f"    Cluster {cluster_id}: {count} customers ({count/len(labels)*100:.1f}%)")

    return kmeans


def save_kmeans_model(model: KMeans, path: Optional[str] = None) -> None:
    """
    Save K-Means model to disk.

    Args:
        model: Fitted KMeans model
        path: Optional save path
    """
    save_path = path if path is not None else KMEANS_MODEL_FILE
    joblib.dump(model, save_path)
    print(f"✓ K-Means model saved to {save_path}")


def load_kmeans_model(path: Optional[str] = None) -> KMeans:
    """
    Load K-Means model from disk.

    Args:
        path: Optional model path

    Returns:
        Loaded KMeans model
    """
    model_path = path if path is not None else KMEANS_MODEL_FILE
    model = joblib.load(model_path)
    print(f"✓ K-Means model loaded from {model_path}")
    return model


# ============================================================================
# AUTOENCODER
# ============================================================================

def build_autoencoder(
    input_dim: int = AUTOENCODER_PARAMS["input_dim"],
    encoding_dim: int = AUTOENCODER_PARAMS["encoding_dim"],
    hidden_layers: List[int] = AUTOENCODER_PARAMS["hidden_layers"],
    activation: str = AUTOENCODER_PARAMS["activation"],
) -> Tuple[Model, Model]:
    """
    Build autoencoder model for dimensionality reduction.

    Architecture from notebook:
    - Encoder: 17 → 7 → 500 → 500 → 2000 → 10 (bottleneck)
    - Decoder: 10 → 2000 → 500 → 17

    Args:
        input_dim: Input dimension (number of features)
        encoding_dim: Bottleneck dimension
        hidden_layers: List of hidden layer sizes
        activation: Activation function

    Returns:
        Tuple of (autoencoder model, encoder model)
    """
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required for autoencoder functionality")

    print("\n[Building Autoencoder]")
    print(f"  Input dim: {input_dim}")
    print(f"  Encoding dim: {encoding_dim}")
    print(f"  Hidden layers: {hidden_layers}")

    # Input layer
    input_layer = Input(shape=(input_dim,))

    # Encoder
    x = input_layer
    for i, units in enumerate(hidden_layers):
        x = Dense(units, activation=activation, kernel_initializer='glorot_uniform')(x)

    # Bottleneck
    encoded = Dense(encoding_dim, activation=activation, name='encoding')(x)

    # Decoder (reverse of encoder, excluding first layer)
    x = encoded
    for units in reversed(hidden_layers[1:]):  # Skip first small layer (7)
        x = Dense(units, activation=activation, kernel_initializer='glorot_uniform')(x)

    # Output layer
    decoded = Dense(input_dim, kernel_initializer='glorot_uniform')(x)

    # Models
    autoencoder = Model(input_layer, decoded, name='autoencoder')
    encoder = Model(input_layer, encoded, name='encoder')

    print(f"✓ Autoencoder built")
    print(f"  Total parameters: {autoencoder.count_params():,}")

    return autoencoder, encoder


def train_autoencoder(
    X: pd.DataFrame,
    autoencoder: Model,
    epochs: int = AUTOENCODER_PARAMS["epochs"],
    batch_size: int = AUTOENCODER_PARAMS["batch_size"],
    validation_split: float = AUTOENCODER_PARAMS["validation_split"],
) -> Dict[str, Any]:
    """
    Train autoencoder model.

    Args:
        X: Scaled feature DataFrame
        autoencoder: Autoencoder model
        epochs: Number of training epochs
        batch_size: Batch size
        validation_split: Validation split ratio

    Returns:
        Training history
    """
    print(f"\n[Training Autoencoder]")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")

    # Compile
    autoencoder.compile(
        optimizer=AUTOENCODER_PARAMS["optimizer"],
        loss=AUTOENCODER_PARAMS["loss"]
    )

    # Train
    history = autoencoder.fit(
        X.values, X.values,  # Input and output are the same (reconstruction)
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )

    print(f"✓ Autoencoder trained")
    print(f"  Final loss: {history.history['loss'][-1]:.4f}")
    if validation_split > 0:
        print(f"  Final val_loss: {history.history['val_loss'][-1]:.4f}")

    return history


def save_autoencoder(model: Model, path: Optional[str] = None) -> None:
    """
    Save autoencoder model to disk.

    Args:
        model: Trained autoencoder
        path: Optional save path
    """
    save_path = path if path is not None else AUTOENCODER_MODEL_FILE
    model.save(save_path)
    print(f"✓ Autoencoder saved to {save_path}")


def load_autoencoder(path: Optional[str] = None) -> Model:
    """
    Load autoencoder from disk.

    Args:
        path: Optional model path

    Returns:
        Loaded autoencoder model
    """
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required for autoencoder functionality")

    model_path = path if path is not None else AUTOENCODER_MODEL_FILE
    model = keras.models.load_model(model_path)
    print(f"✓ Autoencoder loaded from {model_path}")
    return model


# ============================================================================
# PCA DIMENSIONALITY REDUCTION
# ============================================================================

def apply_pca(
    X: pd.DataFrame,
    n_components: int = PCA_PARAMS["n_components"],
) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA for dimensionality reduction and visualization.

    Args:
        X: Input data
        n_components: Number of components

    Returns:
        Tuple of (transformed data, fitted PCA model)
    """
    print(f"\n[Applying PCA] n_components={n_components}")

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)

    print(f"✓ PCA applied")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    return X_pca, pca


def save_pca(model: PCA, path: Optional[str] = None) -> None:
    """Save PCA model to disk."""
    save_path = path if path is not None else PCA_FILE
    joblib.dump(model, save_path)
    print(f"✓ PCA model saved to {save_path}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_clustering_pipeline(
    data_path: Optional[str] = None,
    use_autoencoder: bool = False,
    run_elbow_analysis: bool = True,
) -> Dict[str, Any]:
    """
    Main training pipeline for customer segmentation.

    Workflow:
    1. Load and preprocess data
    2. Optional: Run elbow method analysis
    3. Optional: Train autoencoder and use encoded features
    4. Train K-Means clustering
    5. Apply PCA for visualization
    6. Save all models

    Args:
        data_path: Optional path to data file
        use_autoencoder: Whether to use autoencoder for dimensionality reduction
        run_elbow_analysis: Whether to run elbow method analysis

    Returns:
        Dictionary containing trained models and results
    """
    print("=" * 80)
    print("CUSTOMER SEGMENTATION TRAINING PIPELINE")
    print("=" * 80)

    # Step 1: Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    df = load_raw_data(data_path)
    df_scaled, transformers = preprocess_pipeline(df)
    save_transformers(transformers)

    # Step 2: Elbow method analysis
    wcss_scores = None
    if run_elbow_analysis:
        print("\n[2/6] Running elbow method analysis...")
        wcss_scores = elbow_method_analysis(df_scaled)

    # Step 3: Optional autoencoder
    encoder = None
    X_features = df_scaled

    if use_autoencoder and KERAS_AVAILABLE:
        print("\n[3/6] Training autoencoder...")
        autoencoder, encoder = build_autoencoder()
        train_autoencoder(df_scaled, autoencoder)
        save_autoencoder(autoencoder)

        # Use encoded features for clustering
        X_features = pd.DataFrame(
            encoder.predict(df_scaled.values),
            index=df_scaled.index
        )
        print(f"✓ Using encoded features: {X_features.shape}")

    # Step 4: Train K-Means
    print(f"\n[4/6] Training K-Means clustering...")
    kmeans = train_kmeans(X_features)
    save_kmeans_model(kmeans)

    # Step 5: PCA for visualization
    print(f"\n[5/6] Applying PCA for visualization...")
    X_pca, pca = apply_pca(X_features)
    save_pca(pca)

    # Step 6: Visualization
    print(f"\n[6/6] Generating visualizations...")
    labels = kmeans.labels_
    plot_clusters_2d(X_pca, labels, title="Customer Segments (PCA Visualization)")

    # Evaluate
    metrics = evaluate_clustering(X_features.values, labels, kmeans)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"✓ K-Means model: {KMEANS_MODEL_FILE}")
    if use_autoencoder:
        print(f"✓ Autoencoder: {AUTOENCODER_MODEL_FILE}")
    print(f"✓ PCA model: {PCA_FILE}")
    print(f"✓ Scaler: models/scaler.joblib")

    return {
        "kmeans": kmeans,
        "encoder": encoder,
        "pca": pca,
        "labels": labels,
        "metrics": metrics,
        "wcss_scores": wcss_scores,
        "X_pca": X_pca,
    }


if __name__ == "__main__":
    # Train clustering models
    results = train_clustering_pipeline(
        use_autoencoder=False,  # Set to True to use autoencoder
        run_elbow_analysis=True,
    )

    print("\n✓ Training pipeline complete!")
    print(f"  Number of clusters: {KMEANS_PARAMS['n_clusters']}")
    print(f"  WCSS: {results['metrics']['wcss']:.2f}")
    if 'silhouette' in results['metrics']:
        print(f"  Silhouette Score: {results['metrics']['silhouette']:.3f}")
