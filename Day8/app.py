"""
Day 8 — Streamlit app for the Student Score Prediction System.

Same model as student_score_prediction.py, but with sliders instead of a terminal:
move the inputs around and watch the predicted score change.

    streamlit run app.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_preprocessing import RAW_FILE, prepare_everything
from linear_regression_model import (
    BLUE,
    ORANGE,
    PLOT_STYLE,
    coefficient_table,
    comparison_table,
    evaluate_model,
    train_model,
)
from student_score_prediction import predict_new_student

st.set_page_config(
    page_title="Student Score Prediction",
    page_icon="📈",
    layout="wide",
)


# @st.cache_resource runs this once and reuses the result. Without it, Streamlit
# would re-read the CSV and retrain the model on every single slider move.
@st.cache_resource
def load_everything():
    data = prepare_everything(RAW_FILE)
    model = train_model(data["X_train"], data["y_train"])
    y_pred, metrics = evaluate_model(model, data["X_test"], data["y_test"])
    coefs = coefficient_table(model, data["X_train"].columns)
    return data, model, y_pred, metrics, coefs


data, model, y_pred, metrics, coefs = load_everything()

st.title("📈 Student Score Prediction System")
st.caption(
    "Day 8 — Linear Regression on student study habits. "
    "The model predicts a student's average score across Python, Mathematics, "
    "Statistics and Machine Learning."
)

# ---------------------------------------------------------------- metrics row
st.subheader("Model performance on the test set")
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{metrics['MAE']:.2f}", help="Average miss, in marks")
c2.metric("MSE", f"{metrics['MSE']:.2f}", help="Errors squared then averaged")
c3.metric("RMSE", f"{metrics['RMSE']:.2f}", help="MSE brought back into marks")
c4.metric("R² Score", f"{metrics['R2']:.4f}",
          help="Share of the variation in scores the model explains")

st.info(
    f"Trained on **{len(data['X_train'])}** students, tested on "
    f"**{len(data['X_test'])}** it had never seen. The model explains "
    f"**{metrics['R2'] * 100:.1f}%** of the variation in average scores, and its typical "
    f"miss is about **{metrics['MAE']:.1f} marks**."
)

# ---------------------------------------------------------------- predictor
st.divider()
st.subheader("🎯 Predict a student's score")
st.write("Set the inputs and the model predicts the average score straight away.")

left, right = st.columns([2, 1])

with left:
    a, b = st.columns(2)
    with a:
        age = st.slider("Age", 18, 25, 21)
        study_hours = st.slider("Study hours per day", 0.5, 10.0, 4.0, 0.5)
        attendance = st.slider("Attendance (%)", 50.0, 100.0, 85.0, 1.0)
        sleep_hours = st.slider("Sleep hours per night", 4.0, 10.0, 7.0, 0.5)
    with b:
        previous_score = st.slider("Previous score", 35.0, 100.0, 70.0, 1.0)
        program = st.selectbox("Program", ["AI", "DS", "SE", "CS"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        internet = st.radio("Internet access at home", ["Yes", "No"], horizontal=True)

new_student = {
    "Age": age,
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Previous_Score": previous_score,
    "Sleep_Hours": sleep_hours,
    "Internet_Access": 1 if internet == "Yes" else 0,
    "Gender": gender,
    "Program": program,
}
predicted = predict_new_student(model, data["scaler"], data["X_train"].columns, new_student)

with right:
    st.metric("Predicted average score", f"{predicted:.2f} / 100")
    st.caption(f"± about {metrics['RMSE']:.1f} marks (the model's RMSE)")
    if predicted >= 85:
        st.success("Excellent — top of the class.")
    elif predicted >= 70:
        st.info("Good — solid, steady performance.")
    elif predicted >= 60:
        st.warning("Average — more study hours would move this up.")
    else:
        st.error("At risk — needs support with attendance and study time.")

# ---------------------------------------------------------------- tabs
st.divider()
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Actual vs Predicted", "🧮 Comparison table", "🔍 What matters", "📂 The data"]
)

with tab1:
    st.write(
        "Each dot is one test student. The dashed line is a perfect prediction — "
        "the tighter the dots hug it, the better the model."
    )
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data["y_test"], y_pred, s=70, color=BLUE, alpha=0.7,
               edgecolor="white", linewidth=0.8, label="Test students")
    lo = min(data["y_test"].min(), y_pred.min()) - 3
    hi = max(data["y_test"].max(), y_pred.max()) + 3
    ax.plot([lo, hi], [lo, hi], "--", color=ORANGE, linewidth=2,
            label="Perfect prediction")
    ax.set_xlabel("Actual Average Score")
    ax.set_ylabel("Predicted Average Score")
    ax.set_title("Actual vs Predicted Student Average Scores", fontweight="bold")
    ax.legend()
    st.pyplot(fig)

with tab2:
    table = comparison_table(data["y_test"], y_pred)
    st.write(f"All {len(table)} test students, actual score next to the prediction.")
    st.dataframe(table, use_container_width=True, height=420)
    within_5 = (table["Abs_Error"] <= 5).mean() * 100
    st.caption(
        f"{within_5:.1f}% of predictions land within 5 marks. "
        f"Worst miss: {table['Abs_Error'].max():.2f} marks."
    )

with tab3:
    st.write(
        "Every feature was standardised before training, so these coefficients are "
        "directly comparable. Green pushes the predicted score up, red pulls it down."
    )
    plt.style.use(PLOT_STYLE)
    ordered = coefs.sort_values("Coefficient")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = ["#dc2626" if c < 0 else "#16a34a" for c in ordered["Coefficient"]]
    ax2.barh(ordered["Feature"], ordered["Coefficient"], color=colors, alpha=0.85)
    ax2.axvline(0, color="#334155", linewidth=1)
    ax2.set_xlabel("Coefficient (effect on predicted score)")
    ax2.set_title("What the model thinks matters", fontweight="bold")
    st.pyplot(fig2)
    st.dataframe(coefs[["Feature", "Coefficient"]], use_container_width=True)

with tab4:
    st.write("The cleaned dataset the model was trained on (200 students).")
    st.dataframe(data["clean_df"], use_container_width=True, height=420)
    st.write("Summary statistics:")
    st.dataframe(data["clean_df"].describe().round(2), use_container_width=True)

st.divider()
st.caption("Day 8 · MLB Internship · Linear Regression with scikit-learn")
