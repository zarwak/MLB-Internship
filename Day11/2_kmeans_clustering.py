"""
K-Means Clustering Script
Applies K-Means clustering with Elbow Method and visualizes results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import os

os.makedirs('images', exist_ok=True)

def apply_kmeans():
    """Apply K-Means clustering with Elbow Method"""
    
    # Load and prepare data
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("=" * 60)
    print("K-MEANS CLUSTERING ON IRIS DATASET")
    print("=" * 60)
    
    # Elbow Method
    wcss = []
    silhouette_scores = []
    k_range = range(2, 10)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Plot Elbow Method
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # WCSS plot
    axes[0].plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(x=3, color='red', linestyle='--', label='Optimal k=3')
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    axes[0].set_title('Elbow Method for Optimal k', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Silhouette scores
    axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    axes[1].axvline(x=3, color='blue', linestyle='--', label='Optimal k=3')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score for Different k', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('images/elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Apply K-Means with optimal k=3
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create DataFrame with results
    df = pd.DataFrame(X, columns=feature_names)
    df['true_label'] = y
    df['true_name'] = df['true_label'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    df['cluster'] = clusters
    df['cluster_name'] = df['cluster'].map({0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'})
    
    # Evaluate clustering
    print("\n📊 Clustering Evaluation:")
    print(f"Adjusted Rand Index: {adjusted_rand_score(y, clusters):.3f}")
    print(f"Normalized Mutual Info: {normalized_mutual_info_score(y, clusters):.3f}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, clusters):.3f}")
    
    # Confusion matrix
    confusion = pd.crosstab(df['cluster'], df['true_name'])
    print("\n📊 Confusion Matrix (Cluster vs True Species):")
    print(confusion)
    
    # Visualize clusters
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Clusters using first two features
    scatter1 = axes[0, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                  c=clusters, cmap='viridis', alpha=0.7, s=50)
    axes[0, 0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                      c='red', marker='X', s=200, label='Centroids')
    axes[0, 0].set_xlabel('Sepal Length (scaled)', fontsize=11)
    axes[0, 0].set_ylabel('Sepal Width (scaled)', fontsize=11)
    axes[0, 0].set_title('K-Means Clusters (Sepal Features)', fontsize=13)
    axes[0, 0].legend()
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # 2. Clusters using last two features
    scatter2 = axes[0, 1].scatter(X_scaled[:, 2], X_scaled[:, 3], 
                                  c=clusters, cmap='viridis', alpha=0.7, s=50)
    axes[0, 1].scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], 
                      c='red', marker='X', s=200, label='Centroids')
    axes[0, 1].set_xlabel('Petal Length (scaled)', fontsize=11)
    axes[0, 1].set_ylabel('Petal Width (scaled)', fontsize=11)
    axes[0, 1].set_title('K-Means Clusters (Petal Features)', fontsize=13)
    axes[0, 1].legend()
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # 3. True labels comparison
    scatter3 = axes[1, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                  c=y, cmap='Set1', alpha=0.7, s=50)
    axes[1, 0].set_xlabel('Sepal Length (scaled)', fontsize=11)
    axes[1, 0].set_ylabel('Sepal Width (scaled)', fontsize=11)
    axes[1, 0].set_title('True Labels (Sepal Features)', fontsize=13)
    plt.colorbar(scatter3, ax=axes[1, 0])
    
    # 4. True labels comparison (petal features)
    scatter4 = axes[1, 1].scatter(X_scaled[:, 2], X_scaled[:, 3], 
                                  c=y, cmap='Set1', alpha=0.7, s=50)
    axes[1, 1].set_xlabel('Petal Length (scaled)', fontsize=11)
    axes[1, 1].set_ylabel('Petal Width (scaled)', fontsize=11)
    axes[1, 1].set_title('True Labels (Petal Features)', fontsize=13)
    plt.colorbar(scatter4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('images/kmeans_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print cluster characteristics
    print("\n📊 Cluster Characteristics:")
    for cluster_num in range(optimal_k):
        cluster_data = df[df['cluster'] == cluster_num]
        majority_class = cluster_data['true_name'].mode().iloc[0]
        print(f"\nCluster {cluster_num}:")
        print(f"  Size: {len(cluster_data)} samples")
        print(f"  Majority species: {majority_class}")
        print(f"  Mean values:")
        for feature in feature_names:
            print(f"    {feature}: {cluster_data[feature].mean():.3f}")
    
    return df, kmeans, scaler, X_scaled

if __name__ == "__main__":
    df, kmeans, scaler, X_scaled = apply_kmeans()
    print("\n✅ K-Means clustering complete! Check the 'images' folder for visualizations.")