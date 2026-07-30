'''1. Dataset Exploration Script (1_exploration.py)
This script loads the data, converts it to a DataFrame,
 and provides an initial EDA.'''

#EDA and visualization

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Exploration
print("="*50)
print("DATASET EXPLORATION")
print("="*50)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nTarget Distribution (0 = Malignant, 1 = Benign):")
print(df['target'].value_counts())

# Visualize target distribution
sns.countplot(x='target', data=df)
plt.title('Distribution of Target Classes')
plt.savefig('target_distribution.png')
print("\n✅ Target distribution plot saved as 'target_distribution.png'")