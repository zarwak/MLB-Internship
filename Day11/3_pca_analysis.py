"""
PCA Analysis Script
Applies PCA for dimensionality reduction and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

os.makedirs('images', exist_ok=True)

def apply_pca():
    """Apply PCA for dimensionality reduction"""
    
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
    print("PCA DIMENSIONALITY REDUCTION")
    print("=" * 60)
    
    # Apply PCA
    pca = PCA()
    X_pca_all = pca.fit_transform(X_scaled)
    
    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print("\n📊 Explained Variance per Component:")
    for i, var in enumerate(explained_variance[:4], 1):
        print(f"PC{i}: {var:.2%}")
    print(f"\nCumulative variance for first 2 components: {cumulative_variance[1]:.2%}")
    print(f"Cumulative variance for first 3 components: {cumulative_variance[2]:.2%}")
    
    # Plot explained variance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual explained variance
    axes[0].bar(range(1, 5), explained_variance[:4], alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0].set_title('Individual Component Variance', fontsize=14)
    axes[0].set_xticks(range(1, 5))
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative explained variance
    axes[1].plot(range(1, 5), cumulative_variance[:4], 'ro-', linewidth=2, markersize=8)
    axes[1].axhline(y=0.95, color='green', linestyle='--', label='95% threshold')
    axes[1].axvline(x=2, color='red', linestyle='--', label='n_components=2')
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[1].set_title('Cumulative Variance Explained', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/pca_explained_variance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Reduce to 2 components
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)
    
    # Visualize PCA results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. PCA with true labels
    for i, target_name in enumerate(target_names):
        axes[0, 0].scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                          label=target_name, alpha=0.7, s=50)
    axes[0, 0].set_xlabel('First Principal Component', fontsize=11)
    axes[0, 0].set_ylabel('Second Principal Component', fontsize=11)
    axes[0, 0].set_title('PCA - True Labels', fontsize=13)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. PCA with K-Means clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    scatter = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], 
                                 c=clusters, cmap='viridis', alpha=0.7, s=50)
    axes[0, 1].set_xlabel('First Principal Component', fontsize=11)
    axes[0, 1].set_ylabel('Second Principal Component', fontsize=11)
    axes[0, 1].set_title('PCA - K-Means Clusters', fontsize=13)
    plt.colorbar(scatter, ax=axes[0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. PCA with loadings
    loadings = pca_2d.components_.T
    for i, feature in enumerate(feature_names):
        axes[1, 0].arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                        head_width=0.05, head_length=0.05, 
                        fc='red', ec='red', alpha=0.7)
        axes[1, 0].text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, 
                       feature, fontsize=10)
    axes[1, 0].set_xlim(-1, 1)
    axes[1, 0].set_ylim(-1, 1)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.2)
    axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.2)
    axes[1, 0].set_xlabel('First Principal Component', fontsize=11)
    axes[1, 0].set_ylabel('Second Principal Component', fontsize=11)
    axes[1, 0].set_title('PCA Feature Loadings', fontsize=13)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 3D PCA visualization (first 3 components)
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
    for i, target_name in enumerate(target_names):
        ax_3d.scatter(X_pca_3d[y == i, 0], X_pca_3d[y == i, 1], X_pca_3d[y == i, 2],
                     label=target_name, alpha=0.7, s=30)
    ax_3d.set_xlabel('PC1', fontsize=9)
    ax_3d.set_ylabel('PC2', fontsize=9)
    ax_3d.set_zlabel('PC3', fontsize=9)
    ax_3d.set_title('3D PCA Visualization', fontsize=13)
    ax_3d.legend()
    
    plt.tight_layout()
    plt.savefig('images/pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation between original features and principal components
    correlation_df = pd.DataFrame(
        np.corrcoef(X.T, X_pca.T)[:4, 4:],
        index=feature_names,
        columns=['PC1', 'PC2']
    )
    
    print("\n📊 Correlation with Principal Components:")
    print(correlation_df)
    
    # Heatmap of component loadings
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', linewidths=2, linecolor='white')
    plt.title('Feature Correlations with Principal Components', fontsize=14)
    plt.tight_layout()
    plt.savefig('images/pca_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n📊 PCA Component Interpretation:")
    for i, comp in enumerate(['PC1', 'PC2']):
        print(f"\n{comp}:")
        top_features = correlation_df[comp].abs().sort_values(ascending=False).head(2)
        for feature, corr in top_features.items():
            sign = '+' if corr > 0 else '-'
            print(f"  {sign} {feature} (correlation: {corr:.3f})")
    
    return X_pca, pca_2d

if __name__ == "__main__":
    X_pca, pca_2d = apply_pca()
    print("\n✅ PCA analysis complete! Check the 'images' folder for visualizations.")