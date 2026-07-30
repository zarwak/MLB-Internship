# Full Streamlit deployment

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

st.title("🩺 Breast Cancer Prediction System")
st.markdown("Upload your dataset or use the default Wisconsin Breast Cancer dataset.")

# Load default dataset function
@st.cache_data
def load_default_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.target_names

# Sidebar for data selection
option = st.sidebar.radio("Choose Data Source:", ("Use Default Scikit-learn Dataset", "Upload your own CSV"))

if option == "Upload your own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
        # Assume the target column is named 'target' or last column
        if 'target' in df.columns:
            y = df['target'].values
            X = df.drop('target', axis=1).values
        else:
            st.error("CSV must contain a 'target' column (0 for Malignant, 1 for Benign).")
            st.stop()
        target_names = ['Malignant', 'Benign']
    else:
        st.warning("Please upload a CSV file.")
        st.stop()
else:
    df, target_names = load_default_data()
    X = df.drop('target', axis=1).values
    y = df['target'].values
    st.sidebar.success("Loaded default dataset!")

# Display dataset preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

st.subheader("📈 Target Distribution")
fig, ax = plt.subplots()
sns.countplot(x='target', data=df, ax=ax)
ax.set_xticklabels(target_names)
st.pyplot(fig)

# Train/Test Split Button
if st.button("🚀 Run Full Pipeline (Baseline + Tuning)"):
    with st.spinner("Training models... This may take a moment."):
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # --- BASELINE ---
        baseline = LogisticRegression(max_iter=1000, random_state=42)
        baseline.fit(X_train, y_train)
        y_pred_base = baseline.predict(X_test)
        
        # --- TUNED (GridSearch) ---
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__solver': ['liblinear', 'lbfgs']
        }
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred_tuned = best_model.predict(X_test)
        
        # Metrics
        def get_metrics(y_true, y_pred):
            return {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred),
                "Recall": recall_score(y_true, y_pred),
                "F1-Score": f1_score(y_true, y_pred)
            }
        
        metrics_base = get_metrics(y_test, y_pred_base)
        metrics_tuned = get_metrics(y_test, y_pred_tuned)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("❌ Baseline Model")
            st.metric("Accuracy", f"{metrics_base['Accuracy']:.4f}")
            st.metric("Precision", f"{metrics_base['Precision']:.4f}")
            st.metric("Recall", f"{metrics_base['Recall']:.4f}")
            st.metric("F1-Score", f"{metrics_base['F1-Score']:.4f}")
            cm_base = confusion_matrix(y_test, y_pred_base)
            fig1, ax1 = plt.subplots()
            sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title("Baseline CM")
            st.pyplot(fig1)
        
        with col2:
            st.subheader("✅ Tuned Model (GridSearchCV)")
            st.write(f"**Best Params:** `{grid.best_params_}`")
            st.metric("Accuracy", f"{metrics_tuned['Accuracy']:.4f}")
            st.metric("Precision", f"{metrics_tuned['Precision']:.4f}")
            st.metric("Recall", f"{metrics_tuned['Recall']:.4f}")
            st.metric("F1-Score", f"{metrics_tuned['F1-Score']:.4f}")
            cm_tuned = confusion_matrix(y_test, y_pred_tuned)
            fig2, ax2 = plt.subplots()
            sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Greens', ax=ax2)
            ax2.set_title("Tuned CM")
            st.pyplot(fig2)
        
        # Improvement
        st.subheader("📈 Improvement Summary")
        improvement = metrics_tuned["Accuracy"] - metrics_base["Accuracy"]
        if improvement > 0:
            st.success(f"✅ Tuned model improved accuracy by {improvement:.4f} ({improvement*100:.2f}%)")
        elif improvement < 0:
            st.warning(f"⚠️ Tuned model performed slightly worse by {abs(improvement):.4f}")
        else:
            st.info("No significant change.")

        # Save model
        with open('app_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        st.download_button("📥 Download Trained Model", data=open('app_model.pkl', 'rb'), file_name='breast_cancer_model.pkl')