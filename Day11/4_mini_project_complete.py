"""
Complete Mini Project: Iris Flower Clustering & Visualization
Combines all analyses into one complete script
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

class IrisAnalysis:
    """Complete analysis pipeline for Iris dataset"""
    
    def __init__(self):
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.feature_names = self.iris.feature_names
        self.target_names = self.iris.target_names
        
        # Standardize data
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Initialize results
        self.df = None
        self.kmeans = None
        self.clusters = None
        self.X_pca = None
        self.pca = None
        
    def explore(self):
        """Explore dataset"""
        print("\n" + "=" * 70)
        print("IRIS FLOWER CLUSTERING & VISUALIZATION")
        print("=" * 70)
        
        # Create DataFrame
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df['species'] = self.y
        self.df['species_name'] = self.df['species'].map(
            {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        )
        
        print("\n📊 Dataset Overview:")
        print(f"• Samples: {len(self.df)}")
        print(f"• Features: {len(self.feature_names)}")
        print(f"• Species: {', '.join(self.target_names)}")
        
        print("\n📊 Statistical Summary:")
        print(self.df[self.feature_names].describe().round(3))
        
        return self.df
    
    def apply_kmeans(self):
        """Apply K-Means clustering"""
        print("\n" + "=" * 70)
        print("K-MEANS CLUSTERING")
        print("=" * 70)
        
        # Elbow Method
        wcss = []
        silhouette_scores = []
        k_range = range(2, 9)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_scaled, kmeans.labels_))
        
        # Determine optimal k
        optimal_k = 3  # Based on elbow method and silhouette score
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.clusters = self.kmeans.fit_predict(self.X_scaled)
        self.df['cluster'] = self.clusters
        
        # Evaluate
        ari = adjusted_rand_score(self.y, self.clusters)
        sil_score = silhouette_score(self.X_scaled, self.clusters)
        
        print(f"\n📊 Clustering Results (k={optimal_k}):")
        print(f"• Adjusted Rand Index: {ari:.3f}")
        print(f"• Silhouette Score: {sil_score:.3f}")
        
        # Confusion matrix
        confusion = pd.crosstab(self.df['cluster'], self.df['species_name'])
        print("\n📊 Confusion Matrix (Cluster vs Species):")
        print(confusion)
        
        return self.df, self.kmeans
    
    def apply_pca(self):
        """Apply PCA for dimensionality reduction"""
        print("\n" + "=" * 70)
        print("PCA DIMENSIONALITY REDUCTION")
        print("=" * 70)
        
        self.pca = PCA(n_components=2)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        explained_var = self.pca.explained_variance_ratio_
        print(f"\n📊 Explained Variance:")
        print(f"• PC1: {explained_var[0]:.2%}")
        print(f"• PC2: {explained_var[1]:.2%}")
        print(f"• Total: {sum(explained_var):.2%}")
        
        return self.X_pca, self.pca
    
    def visualize(self):
        """Create comprehensive visualizations"""
        print("\n" + "=" * 70)
        print("VISUALIZATIONS")
        print("=" * 70)
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Original Data (first two features)
        ax1 = fig.add_subplot(gs[0, 0])
        for i, species in enumerate(self.target_names):
            mask = self.df['species_name'] == species
            ax1.scatter(self.X[mask, 0], self.X[mask, 1], 
                       label=species, alpha=0.7, s=40)
        ax1.set_xlabel('Sepal Length')
        ax1.set_ylabel('Sepal Width')
        ax1.set_title('Original Data (2D)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. K-Means Clusters
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(self.X_scaled[:, 0], self.X_scaled[:, 1], 
                             c=self.clusters, cmap='viridis', alpha=0.7, s=40)
        ax2.scatter(self.kmeans.cluster_centers_[:, 0], 
                   self.kmeans.cluster_centers_[:, 1], 
                   c='red', marker='X', s=200, label='Centroids')
        ax2.set_xlabel('Sepal Length (scaled)')
        ax2.set_ylabel('Sepal Width (scaled)')
        ax2.set_title('K-Means Clusters')
        ax2.legend()
        plt.colorbar(scatter, ax=ax2)
        ax2.grid(True, alpha=0.3)
        
        # 3. PCA Visualization
        ax3 = fig.add_subplot(gs[0, 2])
        for i, species in enumerate(self.target_names):
            mask = self.y == i
            ax3.scatter(self.X_pca[mask, 0], self.X_pca[mask, 1], 
                       label=species, alpha=0.7, s=40)
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.set_title(f'PCA (2D) - {sum(self.pca.explained_variance_ratio_):.1%} variance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Elbow Method
        ax4 = fig.add_subplot(gs[1, 0])
        wcss = []
        for k in range(2, 9):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            wcss.append(kmeans.inertia_)
        ax4.plot(range(2, 9), wcss, 'bo-', linewidth=2, markersize=8)
        ax4.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='k=3')
        ax4.set_xlabel('Number of Clusters (k)')
        ax4.set_ylabel('WCSS')
        ax4.set_title('Elbow Method')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Confusion Matrix Heatmap
        ax5 = fig.add_subplot(gs[1, 1])
        confusion = pd.crosstab(self.df['cluster'], self.df['species_name'])
        sns.heatmap(confusion, annot=True, fmt='d', cmap='YlOrRd', 
                   ax=ax5, cbar=False)
        ax5.set_xlabel('True Species')
        ax5.set_ylabel('Cluster')
        ax5.set_title('Cluster Confusion Matrix')
        
        # 6. Feature Correlations with PCA
        ax6 = fig.add_subplot(gs[1, 2])
        correlations = np.corrcoef(self.X.T, self.X_pca.T)[:4, 4:]
        corr_df = pd.DataFrame(correlations, 
                              index=self.feature_names, 
                              columns=['PC1', 'PC2'])
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0,
                   ax=ax6, fmt='.2f', linewidths=1, linecolor='white')
        ax6.set_title('Feature-PCA Correlations')
        
        # 7. Cluster Distribution
        ax7 = fig.add_subplot(gs[2, 0])
        cluster_counts = self.df['cluster'].value_counts()
        ax7.bar(cluster_counts.index, cluster_counts.values, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax7.set_xlabel('Cluster')
        ax7.set_ylabel('Count')
        ax7.set_title('Cluster Distribution')
        ax7.set_xticks(range(3))
        
        # 8. Silhouette Score by Cluster
        ax8 = fig.add_subplot(gs[2, 1])
        from sklearn.metrics import silhouette_samples
        silhouette_vals = silhouette_samples(self.X_scaled, self.clusters)
        y_lower = 10
        for i in range(3):
            cluster_sil_vals = silhouette_vals[self.clusters == i]
            cluster_sil_vals.sort()
            size = len(cluster_sil_vals)
            y_upper = y_lower + size
            ax8.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_vals,
                             alpha=0.7, label=f'Cluster {i}')
            y_lower = y_upper + 10
        ax8.axvline(x=silhouette_score(self.X_scaled, self.clusters), 
                   color='red', linestyle='--', label='Overall')
        ax8.set_xlabel('Silhouette Score')
        ax8.set_ylabel('Cluster')
        ax8.set_title('Silhouette Scores by Cluster')
        ax8.legend()
        
        # 9. PCA with K-Means
        ax9 = fig.add_subplot(gs[2, 2])
        scatter = ax9.scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
                             c=self.clusters, cmap='viridis', alpha=0.6, s=40)
        ax9.scatter(self.pca.transform(self.kmeans.cluster_centers_)[:, 0],
                   self.pca.transform(self.kmeans.cluster_centers_)[:, 1],
                   c='red', marker='X', s=200, label='Centroids')
        ax9.set_xlabel('PC1')
        ax9.set_ylabel('PC2')
        ax9.set_title('PCA with K-Means Centroids')
        ax9.legend()
        plt.colorbar(scatter, ax=ax9)
        ax9.grid(True, alpha=0.3)
        
        plt.suptitle('Iris Dataset: Clustering & PCA Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('images/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Visualizations saved to 'images/comprehensive_analysis.png'")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        self.explore()
        self.apply_kmeans()
        self.apply_pca()
        self.visualize()
        
        # Summary
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        print("\n📌 Key Findings:")
        print(f"• Optimal number of clusters: 3 (by Elbow Method)")
        print(f"• K-Means accuracy (ARI): {adjusted_rand_score(self.y, self.clusters):.3f}")
        print(f"• PCA preserves {sum(self.pca.explained_variance_ratio_):.1%} variance in 2D")
        print(f"• Silhouette Score: {silhouette_score(self.X_scaled, self.clusters):.3f}")
        
        print("\n📌 Cluster Interpretation:")
        for cluster_num in range(3):
            cluster_data = self.df[self.df['cluster'] == cluster_num]
            majority = cluster_data['species_name'].mode().iloc[0]
            print(f"• Cluster {cluster_num}: {len(cluster_data)} samples, Majority: {majority}")
        
        return self.df

if __name__ == "__main__":
    # Run complete analysis
    analyzer = IrisAnalysis()
   