import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Generate 200 custom samples
n_samples = 200
n_features = 30

# Generate random feature values (mean ~15, std ~5, all positive)
X = np.abs(np.random.randn(n_samples, n_features) * 5 + 15)

# Generate target labels (0 = Malignant, 1 = Benign)
# Make it a balanced split (50% each)
y = np.random.randint(0, 2, n_samples)

# Make the data slightly more realistic:
# Malignant tumors (class 0) usually have larger measurements.
# Let's add a bias: if class is 0, increase feature values by 20%
for i in range(n_samples):
    if y[i] == 0:
        X[i] = X[i] * 1.2  # Larger values for Malignant

# Create column names (feature_1 to feature_30)
columns = [f'feature_{i+1}' for i in range(n_features)]

# Build DataFrame
df = pd.DataFrame(X, columns=columns)
df['target'] = y  # 0 = Malignant, 1 = Benign

# Save to CSV
df.to_csv('custom_breast_cancer_data.csv', index=False)

print("✅ Custom dataset 'custom_breast_cancer_data.csv' generated successfully!")
print(f"   - Rows: {n_samples}")
print(f"   - Features: {n_features}")
print(f"   - Target distribution:\n{df['target'].value_counts()}")