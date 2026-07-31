"""
Dataset Exploration Script
Analyzes the Iris dataset structure, statistics, and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

def explore_dataset():
    """Load and explore the Iris dataset"""
    
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = y
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("=" * 60)
    print("IRIS DATASET EXPLORATION")
    print("=" * 60)
    
    # Basic information
    print("\n📊 Dataset Information:")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Target classes: {target_names}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Statistical summary
    print("\n📈 Statistical Summary:")
    print(df[feature_names].describe())
    
    # Check class distribution
    print("\n🎯 Class Distribution:")
    print(df['species_name'].value_counts())
    
    # Correlation matrix
    correlation_matrix = df[feature_names].corr()
    print("\n🔄 Correlation Matrix:")
    print(correlation_matrix)
    
    # Create visualization grid
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Histograms of all features
    for i, feature in enumerate(feature_names, 1):
        ax = plt.subplot(2, 2, i)
        for species in target_names:
            species_data = df[df['species_name'] == species][feature]
            ax.hist(species_data, bins=15, alpha=0.7, label=species)
        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('images/dataset_exploration_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Pairplot
    plt.figure(figsize=(12, 10))
    pairplot = sns.pairplot(df, hue='species_name', vars=feature_names, 
                            diag_kind='kde', height=2.5)
    pairplot.fig.suptitle('Pairplot of Iris Features', y=1.02, fontsize=14)
    pairplot.savefig('images/dataset_exploration_pairplot.png', dpi=300)
    plt.show()
    
    # 3. Box plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, feature in enumerate(feature_names):
        row, col = i // 2, i % 2
        df.boxplot(column=feature, by='species_name', ax=axes[row, col], 
                  color=dict(boxes='blue', whiskers='green', medians='red'))
        axes[row, col].set_title(f'Boxplot of {feature}')
        axes[row, col].set_xlabel('')
        axes[row, col].set_ylabel(feature)
    plt.suptitle('Boxplots by Species', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('images/dataset_exploration_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df, feature_names, target_names

if __name__ == "__main__":
    df, feature_names, target_names = explore_dataset()
    print("\n✅ Dataset exploration complete! Check the 'images' folder for visualizations.")