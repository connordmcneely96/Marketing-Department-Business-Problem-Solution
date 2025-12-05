"""
Model evaluation utilities for Clustering (Customer Segmentation).

This module provides metrics and visualization functions specifically
for unsupervised learning (K-Means clustering).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans

from src.config import VISUALIZATION_PARAMS, CLUSTER_NAMES, CLUSTER_DESCRIPTIONS


# ============================================================================
# CLUSTERING METRICS
# ============================================================================

def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    model: Optional[KMeans] = None,
) -> Dict[str, float]:
    """
    Evaluate clustering performance using multiple metrics.

    Metrics:
    - WCSS (Within-Cluster Sum of Squares) / Inertia: Lower is better
    - Silhouette Score: Range [-1, 1], higher is better
    - Davies-Bouldin Index: Lower is better
    - Calinski-Harabasz Score: Higher is better

    Args:
        X: Feature matrix
        labels: Cluster labels
        model: Optional fitted KMeans model (for inertia)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # WCSS / Inertia (from model)
    if model is not None and hasattr(model, 'inertia_'):
        metrics["wcss"] = model.inertia_

    # Silhouette Score
    if len(np.unique(labels)) > 1:
        try:
            metrics["silhouette"] = silhouette_score(X, labels)
        except Exception as e:
            print(f"⚠️  Could not compute silhouette score: {e}")
            metrics["silhouette"] = None

    # Davies-Bouldin Index
    if len(np.unique(labels)) > 1:
        try:
            metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
        except Exception as e:
            print(f"⚠️  Could not compute Davies-Bouldin index: {e}")
            metrics["davies_bouldin"] = None

    # Calinski-Harabasz Score
    if len(np.unique(labels)) > 1:
        try:
            metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
        except Exception as e:
            print(f"⚠️  Could not compute Calinski-Harabasz score: {e}")
            metrics["calinski_harabasz"] = None

    # Cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    metrics["n_clusters"] = len(unique)
    metrics["cluster_sizes"] = dict(zip(unique.tolist(), counts.tolist()))

    return metrics


def print_clustering_metrics(metrics: Dict[str, Any]) -> None:
    """
    Print clustering metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics from evaluate_clustering
    """
    print("\n" + "=" * 80)
    print("CLUSTERING METRICS")
    print("=" * 80)

    if "wcss" in metrics and metrics["wcss"] is not None:
        print(f"WCSS (Inertia):         {metrics['wcss']:.2f} (lower is better)")

    if "silhouette" in metrics and metrics["silhouette"] is not None:
        print(f"Silhouette Score:       {metrics['silhouette']:.4f} (range [-1,1], higher is better)")

    if "davies_bouldin" in metrics and metrics["davies_bouldin"] is not None:
        print(f"Davies-Bouldin Index:   {metrics['davies_bouldin']:.4f} (lower is better)")

    if "calinski_harabasz" in metrics and metrics["calinski_harabasz"] is not None:
        print(f"Calinski-Harabasz:      {metrics['calinski_harabasz']:.2f} (higher is better)")

    if "n_clusters" in metrics:
        print(f"\nNumber of Clusters:     {metrics['n_clusters']}")

    if "cluster_sizes" in metrics:
        print("\nCluster Distribution:")
        for cluster_id, size in sorted(metrics["cluster_sizes"].items()):
            total = sum(metrics["cluster_sizes"].values())
            percentage = size / total * 100
            print(f"  Cluster {cluster_id}: {size:,} customers ({percentage:.1f}%)")

    print("=" * 80)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_elbow_curve(
    k_values: List[int],
    wcss_scores: List[float],
    title: str = "Elbow Method - Optimal K",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot elbow curve for K-Means clustering.

    Args:
        k_values: List of k values tested
        wcss_scores: List of WCSS scores for each k
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=VISUALIZATION_PARAMS["figsize"])

    plt.plot(k_values, wcss_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)

    # Annotate some points
    for i in [0, len(k_values)//2, -1]:
        plt.annotate(
            f'k={k_values[i]}\nWCSS={wcss_scores[i]:.0f}',
            (k_values[i], wcss_scores[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Elbow curve saved to {save_path}")

    plt.show()


def plot_clusters_2d(
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "Customer Segments",
    save_path: Optional[str] = None,
    show_centers: bool = False,
    centers_2d: Optional[np.ndarray] = None,
) -> None:
    """
    Plot 2D scatter plot of clusters (after PCA).

    Args:
        X_2d: 2D data (e.g., after PCA)
        labels: Cluster labels
        title: Plot title
        save_path: Optional path to save figure
        show_centers: Whether to show cluster centers
        centers_2d: Optional 2D cluster centers
    """
    plt.figure(figsize=VISUALIZATION_PARAMS["figsize"])

    # Create scatter plot
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap=VISUALIZATION_PARAMS["color_palette"],
        alpha=VISUALIZATION_PARAMS["scatter_alpha"],
        s=VISUALIZATION_PARAMS["scatter_size"],
        edgecolors='k',
        linewidths=0.5,
    )

    # Plot cluster centers if provided
    if show_centers and centers_2d is not None:
        plt.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            c='red',
            marker='X',
            s=300,
            edgecolors='black',
            linewidths=2,
            label='Cluster Centers',
            zorder=10,
        )
        plt.legend()

    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Cluster plot saved to {save_path}")

    plt.show()


def plot_cluster_distribution(
    labels: np.ndarray,
    title: str = "Cluster Distribution",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot bar chart of cluster sizes.

    Args:
        labels: Cluster labels
        title: Plot title
        save_path: Optional path to save figure
    """
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(10, 6))

    bars = plt.bar(unique, counts, color='skyblue', edgecolor='black', alpha=0.7)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{count:,}\n({count/len(labels)*100:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(unique)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Distribution plot saved to {save_path}")

    plt.show()


def plot_feature_comparison_by_cluster(
    df: pd.DataFrame,
    labels: np.ndarray,
    features: List[str],
    title: str = "Feature Comparison by Cluster",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot feature means for each cluster.

    Args:
        df: DataFrame with features
        labels: Cluster labels
        features: List of feature names to plot
        title: Plot title
        save_path: Optional path to save figure
    """
    # Add cluster labels to dataframe
    df_with_labels = df.copy()
    df_with_labels['Cluster'] = labels

    # Calculate mean values per cluster
    cluster_means = df_with_labels.groupby('Cluster')[features].mean()

    # Plot
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 3*len(features)))

    if len(features) == 1:
        axes = [axes]

    for idx, feature in enumerate(features):
        axes[idx].bar(
            cluster_means.index,
            cluster_means[feature],
            color='steelblue',
            alpha=0.7,
            edgecolor='black'
        )
        axes[idx].set_title(f'{feature} by Cluster', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Cluster ID')
        axes[idx].set_ylabel(f'Mean {feature}')
        axes[idx].grid(axis='y', alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature comparison saved to {save_path}")

    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Feature Correlation Heatmap",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot correlation heatmap of features.

    Args:
        df: DataFrame with features
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(12, 10))

    corr = df.corr()

    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Correlation heatmap saved to {save_path}")

    plt.show()


# ============================================================================
# CLUSTER INTERPRETATION
# ============================================================================

def get_cluster_profile(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_id: int,
    top_n_features: int = 5,
) -> Dict[str, Any]:
    """
    Get profile (characteristics) of a specific cluster.

    Args:
        df: DataFrame with original (unscaled) features
        labels: Cluster labels
        cluster_id: ID of cluster to profile
        top_n_features: Number of top features to include

    Returns:
        Dictionary with cluster statistics
    """
    cluster_mask = labels == cluster_id
    cluster_data = df[cluster_mask]

    profile = {
        "cluster_id": cluster_id,
        "size": cluster_mask.sum(),
        "percentage": cluster_mask.sum() / len(labels) * 100,
        "mean_values": cluster_data.mean().to_dict(),
        "median_values": cluster_data.median().to_dict(),
        "std_values": cluster_data.std().to_dict(),
    }

    # Add cluster name and description if available
    if cluster_id in CLUSTER_NAMES:
        profile["name"] = CLUSTER_NAMES[cluster_id]

    if cluster_id in CLUSTER_DESCRIPTIONS:
        profile["description"] = CLUSTER_DESCRIPTIONS[cluster_id]

    return profile


def print_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> None:
    """
    Print profiles for all clusters.

    Args:
        df: DataFrame with original features
        labels: Cluster labels
    """
    print("\n" + "=" * 80)
    print("CLUSTER PROFILES")
    print("=" * 80)

    for cluster_id in sorted(np.unique(labels)):
        profile = get_cluster_profile(df, labels, cluster_id)

        print(f"\n{'─' * 80}")
        print(f"CLUSTER {cluster_id}")
        if "name" in profile:
            print(f"Name: {profile['name']}")
        if "description" in profile:
            print(f"Description: {profile['description']}")

        print(f"Size: {profile['size']:,} customers ({profile['percentage']:.1f}%)")

        print(f"\nKey Characteristics (Mean Values):")
        # Sort features by mean value and show top/bottom
        mean_values = profile['mean_values']
        sorted_features = sorted(mean_values.items(), key=lambda x: x[1], reverse=True)

        for feature, value in sorted_features[:5]:
            print(f"  {feature}: {value:.2f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example: Generate sample clustering metrics
    print("Clustering evaluation module loaded successfully!")
    print("\nAvailable functions:")
    print("  - evaluate_clustering()")
    print("  - plot_elbow_curve()")
    print("  - plot_clusters_2d()")
    print("  - plot_cluster_distribution()")
    print("  - get_cluster_profile()")
