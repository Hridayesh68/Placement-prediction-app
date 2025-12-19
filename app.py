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
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Placement Prediction System")
st.markdown("---")

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
    # Access the token from st.secrets
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except KeyError:
        st.error("🔑 HF_TOKEN not found in Secrets. Please add it to continue.")
        st.stop()

    try:
        loaded_objs = {}
        for key, filename in MODEL_FILES.items():
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                token=hf_token
            )
            loaded_objs[key] = joblib.load(path)
        
        return loaded_objs["lr"], loaded_objs["rf"], loaded_objs["xgb"], loaded_objs["scaler"]

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

lr, rf, xgb, scaler = load_models()

# -----------------------------
# Sidebar - Model Selection
# -----------------------------
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Choose Model",
        ["Random Forest", "XGBoost", "Linear Regression"]
    )
    st.info("Note: Random Forest and XGBoost usually provide better classification accuracy.")

# -----------------------------
# User Inputs (Organized in Columns)
# -----------------------------
st.subheader("📥 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input("CGPA (0.0 - 10.0)", 0.0, 10.0, 7.5, step=0.1)
    internships = st.number_input("Number of Internships", 0, 10, 1)
    projects = st.number_input("Number of Projects", 0, 10, 2)
    certifications = st.number_input("Workshops / Certifications", 0, 10, 1)
    aptitude = st.slider("Aptitude Test Score", 0, 100, 75)

with col2:
    soft_skills = st.slider("Soft Skills Rating", 0.0, 5.0, 4.0)
    ssc = st.number_input("SSC Marks (%)", 0, 100, 80)
    hsc = st.number_input("HSC Marks (%)", 0, 100, 80)
    extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    training = st.selectbox("Placement Training", ["Yes", "No"])

# Convert Categorical to Binary
extra_val = 1 if extra == "Yes" else 0
training_val = 1 if training == "Yes" else 0

# -----------------------------
# Preprocessing
# -----------------------------
# Define feature names exactly as they were during scaler.fit()
# Replace these names if they differ in your training dataset!
# -----------------------------
# Preprocessing
# -----------------------------
# Updated feature names to match your training 'fit' exactly
feature_names = [
    "CGPA", "Internships", "Projects", "Certifications",
    "AptitudeTestScore", "SoftSkillsRating", "ExtracurricularActivities",
    "PlacementTraining", "SSC_Marks", "HSC_Marks"
]

input_data = pd.DataFrame([[
    cgpa, internships, projects, certifications,
    aptitude, soft_skills, extra_val, training_val,
    ssc, hsc
]], columns=feature_names)

# Now the scaler will recognize "Certifications" and won't throw a ValueError
scaled_features = scaler.transform(input_data)

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("🔮 Predict Placement Status", use_container_width=True):
    
    if model_choice == "Linear Regression":
        raw_score = lr.predict(scaled_features)[0]
        probability = float(np.clip(raw_score, 0, 1)) * 100
        prediction = 1 if raw_score >= 0.5 else 0
    elif model_choice == "Random Forest":
        probability = rf.predict_proba(scaled_features)[0][1] * 100
        prediction = rf.predict(scaled_features)[0]
    else: # XGBoost
        probability = xgb.predict_proba(scaled_features)[0][1] * 100
        prediction = xgb.predict(scaled_features)[0]

    st.markdown("---")
    
    # Result Display
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.subheader("Result")
        if prediction == 1:
            st.success(f"### 🎉 Result: PLACED")
            st.write(f"**Model:** {model_choice}")
            st.write(f"**Confidence:** {probability:.2f}%")
        else:
            st.error(f"### ❌ Result: NOT PLACED")
            st.write(f"**Model:** {model_choice}")
            st.write(f"**Likelihood of non-placement:** {100 - probability:.2f}%")

        # CSV Download
        report = input_data.copy()
        report["Model"] = model_choice
        report["Prediction"] = "Placed" if prediction == 1 else "Not Placed"
        report["Probability"] = f"{probability:.2f}%"
        
        csv = report.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download This Report", data=csv, file_name="prediction.csv", mime="text/csv")

    with res_col2:
        # Pie Chart Visualization
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            [probability, 100 - probability],
            labels=["Placed", "Not Placed"],
            autopct="%1.1f%%",
            startangle=90,
            colors=['#2ecc71', '#e74c3c']
        )
        ax.set_title("Success Probability")
        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------
    # Feature Importance
    # -----------------------------
    if model_choice != "Linear Regression":
        st.subheader("📌 Impact of Features")
        importance = rf.feature_importances_ if model_choice == "Random Forest" else xgb.feature_importances_
        
        # Sort importance for better visual
        feat_imp = pd.Series(importance, index=feature_names).sort_values()
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        feat_imp.plot(kind='barh', ax=ax2, color='skyblue')
        ax2.set_title(f"Key Factors Influencing {model_choice} Prediction")
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("Feature importance visual is not available for Linear Regression.")