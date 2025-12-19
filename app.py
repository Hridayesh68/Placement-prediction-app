import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from huggingface_hub import hf_hub_download

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Student Placement Prediction",
    layout="wide"
)

st.title("🎓 Student Placement Prediction System")

# -----------------------------
# Hugging Face Repo Info
# -----------------------------
REPO_ID = "hridayeshdebsarma6/placement-prediction"

MODEL_FILES = {
    "lr": "linear_model.pkl",
    "rf": "rf_model.pkl",
    "xgb": "xgb_model.pkl",
    "scaler": "scaler.pkl"
}

# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_models():
    try:
        lr_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILES["lr"],
            token=True
        )
        rf_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILES["rf"],
            token=True
        )
        xgb_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILES["xgb"],
            token=True
        )
        scaler_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILES["scaler"],
            token=True
        )

        lr = joblib.load(lr_path)
        rf = joblib.load(rf_path)
        xgb = joblib.load(xgb_path)
        scaler = joblib.load(scaler_path)

        return lr, rf, xgb, scaler

    except Exception as e:
        st.error("❌ Error loading models from Hugging Face.")
        st.exception(e)
        st.stop()


lr, rf, xgb, scaler = load_models()

# -----------------------------
# Sidebar - Model Selection
# -----------------------------
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Random Forest", "XGBoost"]
)

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("📥 Enter Student Details")

cgpa = st.number_input("CGPA", 0.0, 10.0, step=0.1)
internships = st.number_input("Internships", 0, 10)
projects = st.number_input("Projects", 0, 10)
certifications = st.number_input("Workshops / Certifications", 0, 10)
aptitude = st.slider("Aptitude Test Score", 0, 100)
soft_skills = st.slider("Soft Skills Rating", 0.0, 5.0)
extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
training = st.selectbox("Placement Training", ["Yes", "No"])
ssc = st.number_input("SSC Marks", 0, 100)
hsc = st.number_input("HSC Marks", 0, 100)

extra = 1 if extra == "Yes" else 0
training = 1 if training == "Yes" else 0

features = np.array([[
    cgpa, internships, projects, certifications,
    aptitude, soft_skills, extra, training,
    ssc, hsc
]])

scaled_features = scaler.transform(features)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Placement"):

    if model_choice == "Linear Regression":
        raw_score = lr.predict(scaled_features)[0]
        probability = float(np.clip(raw_score, 0, 1)) * 100
        prediction = 1 if raw_score >= 0.5 else 0
        model_used = "Linear Regression"

    elif model_choice == "Random Forest":
        probability = rf.predict_proba(scaled_features)[0][1] * 100
        prediction = rf.predict(scaled_features)[0]
        model_used = "Random Forest"

    else:
        probability = xgb.predict_proba(scaled_features)[0][1] * 100
        prediction = xgb.predict(scaled_features)[0]
        model_used = "XGBoost"

    # -----------------------------
    # Result
    # -----------------------------
    if prediction == 1:
        st.success(f"🎉 PLACED | {model_used} | Confidence: {probability:.2f}%")
    else:
        st.error(f"❌ NOT PLACED | {model_used} | Confidence: {100 - probability:.2f}%")

    # -----------------------------
    # Pie Chart
    # -----------------------------
    st.subheader("📊 Placement Probability")

    fig, ax = plt.subplots()
    ax.pie(
        [probability, 100 - probability],
        labels=["Placed", "Not Placed"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)

    # -----------------------------
    # Feature Importance
    # -----------------------------
    feature_names = [
        "CGPA", "Internships", "Projects", "Certifications",
        "Aptitude", "Soft Skills", "Extracurricular",
        "Placement Training", "SSC", "HSC"
    ]

    if model_choice != "Linear Regression":
        st.subheader("📌 Feature Importance")

        importance = (
            rf.feature_importances_
            if model_choice == "Random Forest"
            else xgb.feature_importances_
        )

        fig2, ax2 = plt.subplots()
        ax2.barh(feature_names, importance)
        ax2.set_title(f"{model_used} Feature Importance")
        st.pyplot(fig2)
    else:
        st.info("Feature importance not available for Linear Regression.")

    # -----------------------------
    # Download Report
    # -----------------------------
    report = pd.DataFrame({
        "Feature": feature_names,
        "Value": features[0]
    })

    report["Model Used"] = model_used
    report["Prediction"] = "Placed" if prediction == 1 else "Not Placed"
    report["Probability (%)"] = probability

    csv = report.to_csv(index=False)

    st.download_button(
        label="⬇️ Download Prediction Report",
        data=csv,
        file_name="placement_prediction_report.csv",
        mime="text/csv"
    )
