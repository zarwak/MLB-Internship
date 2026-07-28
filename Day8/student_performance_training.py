import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------
# 1. LOAD THE DATASET
# ------------------------------
# Since the data is provided in the prompt, we'll read it directly from a string.
from io import StringIO

data_string = """Student_ID,Name,Age,Program,Python,Mathematics,Statistics,Machine_Learning,Attendance
S001,Ali Khan,20,AI,85,78,92,88,95
S002,Sara Ahmed,21,AI,72,75,70,80,90
S003,Ahmed Raza,22,SE,90,88,91,93,96
S004,Fatima Noor,20,DS,65,70,68,72,85
S005,Usman Ali,21,AI,78,82,80,76,88
S006,Ayesha Malik,22,SE,95,94,96,97,99
S007,Hassan Tariq,20,DS,55,60,58,62,75
S008,Zainab Iqbal,21,AI,88,86,90,91,94
S009,Bilal Ahmed,23,SE,73,77,75,79,82
S010,Maryam Khan,20,DS,81,84,79,85,91
S011,Hamza Siddiqui,22,AI,69,71,74,70,87
S012,Noor Fatima,21,SE,92,90,93,95,98
S013,Talha Javed,20,DS,76,74,78,80,89
S014,Iqra Aslam,22,AI,84,83,86,88,92
S015,Danish Ali,23,SE,61,65,63,67,78
S016,Hira Shah,21,DS,89,91,88,90,96
S017,Omar Farooq,20,AI,74,72,76,78,84
S018,Laiba Khan,22,SE,97,95,98,99,100
S019,Abdullah,21,DS,68,66,70,72,80
S020,Mehwish,20,AI,86,89,87,90,93"""

df = pd.read_csv(StringIO(data_string))

# Drop the 'Student_ID' and 'Name' columns – they are unique identifiers, not features.
df.drop(['Student_ID', 'Name'], axis=1, inplace=True)

print("📊 First 5 rows of raw data:")
print(df.head())

# ------------------------------
# 2. DATA PREPROCESSING
# ------------------------------

# 2.1 Create the Target Column (Average_Score)
# Average of the four core subjects
df['Average_Score'] = df[['Python', 'Mathematics', 'Statistics', 'Machine_Learning']].mean(axis=1)

# 2.2 Encode Categorical Columns
# We only have 'Program' – it's nominal (AI, SE, DS). Let's One-Hot Encode it.
# We'll use pandas get_dummies for simplicity, dropping the first to avoid multicollinearity.
df_encoded = pd.get_dummies(df, columns=['Program'], drop_first=True)

print("\n🔢 Columns after encoding:")
print(df_encoded.columns.tolist())

# ------------------------------
# 3. SELECT FEATURES (X) AND TARGET (y)
# ------------------------------
# We will NOT use the individual subject scores as features.
# We want to predict Average_Score using Age, Attendance, and the Program type.
# This makes the regression challenge real and avoids trivial perfect predictions.

feature_columns = ['Age', 'Attendance'] + [col for col in df_encoded.columns if col.startswith('Program_')]
X = df_encoded[feature_columns]
y = df_encoded['Average_Score']  # Target

print(f"\n✅ Features (X): {X.columns.tolist()}")
print(f"🎯 Target (y): Average_Score")

# ------------------------------
# 4. TRAIN-TEST SPLIT (80% / 20%)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📦 Training set size: {X_train.shape[0]} rows")
print(f"📦 Testing set size:  {X_test.shape[0]} rows")

# ------------------------------
# 5. FEATURE SCALING (Standardization)
# ------------------------------
# We scale Age and Attendance because they are on different scales (20-23 vs 75-100).
# For the binary one-hot columns, scaling isn't strictly necessary, but we'll scale all numeric columns.
# IMPORTANT: Fit ONLY on the training set to prevent Data Leakage!

scaler = StandardScaler()

# Fit the scaler on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the SAME scaler
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for readability (optional)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\n⚙️ Feature scaling applied (mean=0, std=1) using StandardScaler.")

# ------------------------------
# 6. BUILD & TRAIN LINEAR REGRESSION MODEL
# ------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\n🚀 Model training complete!")
print(f"📈 Intercept (beta_0): {model.intercept_:.4f}")
print("📉 Coefficients:")
for col, coef in zip(X_train.columns, model.coef_):
    print(f"   {col}: {coef:.4f}")

# ------------------------------
# 7. MAKE PREDICTIONS ON TEST SET
# ------------------------------
y_pred = model.predict(X_test_scaled)

# ------------------------------
# 8. COMPARE ACTUAL vs PREDICTED
# ------------------------------
comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
comparison_df['Difference'] = comparison_df['Actual'] - comparison_df['Predicted']

print("\n📋 Actual vs Predicted (Test Set):")
print(comparison_df.to_string(index=False))

# ------------------------------
# 9. EVALUATION METRICS
# ------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error (more interpretable)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance Metrics:")
print(f"🔹 Mean Absolute Error (MAE)     : {mae:.4f}")
print(f"🔹 Mean Squared Error (MSE)      : {mse:.4f}")
print(f"🔹 Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"🔹 R² Score                      : {r2:.4f}")

# Bonus: Interpretation
if r2 > 0.7:
    print("\n✅ The model explains a good amount of variance. Age, Program, and Attendance are strong predictors!")
else:
    print("\n⚠️ The model explains less than 70% of variance. We might need more features.")