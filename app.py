import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
from huggingface_hub import hf_hub_download

# -----------------------------
# 1. Configuration & Styling
# -----------------------------
st.set_page_config(
    page_title="Placement Predictor Pro",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3em; 
        background-color: #4CAF50; 
        color: white; 
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover { background-color: #45a049; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Student Placement Prediction System")
st.markdown("### AI-Driven Career Insights & Probability Analysis")
st.markdown("---")

# -----------------------------
# 2. Load Models from Hugging Face
# -----------------------------
REPO_ID = "hridayeshdebsarma6/placement-prediction"

# Renamed function to force cache reset
@st.cache_resource(show_spinner="Downloading models...")
def load_models_v2():
    """Downloads models one by one. Returns a dictionary."""
    models = {}
    hf_token = st.secrets.get("HF_TOKEN", None)

    try:
        # 1. Load Scaler
        path_s = hf_hub_download(repo_id=REPO_ID, filename="scaler.pkl", token=hf_token)
        models["scaler"] = joblib.load(path_s)
        
        # 2. Load Logistic Regression
        path_lr = hf_hub_download(repo_id=REPO_ID, filename="logistic_regression.pkl", token=hf_token)
        models["Logistic Regression"] = joblib.load(path_lr)

        # 3. Load Random Forest
        path_rf = hf_hub_download(repo_id=REPO_ID, filename="random_forest.pkl", token=hf_token)
        models["Random Forest"] = joblib.load(path_rf)

        # 4. Load XGBoost
        path_xgb = hf_hub_download(repo_id=REPO_ID, filename="xgboost.pkl", token=hf_token)
        models["XGBoost"] = joblib.load(path_xgb)

        return models

    except Exception as e:
        st.error(f"❌ Failed to load a file. Error: {e}")
        st.stop()

# Load all artifacts
loaded_artifacts = load_models_v2()

# Safe Extraction (No .pop() to prevent cache errors)
if "scaler" in loaded_artifacts:
    scaler = loaded_artifacts["scaler"]
else:
    st.error("❌ Critical Error: 'scaler' not found in loaded artifacts.")
    st.write("Keys found:", list(loaded_artifacts.keys()))
    st.stop()

# -----------------------------
# 3. Sidebar - Model Selection
# -----------------------------
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    
    # User selects which model to use
    selected_model_name = st.selectbox(
        "Choose AI Model", 
        ["XGBoost", "Random Forest", "Logistic Regression"]
    )
    
    if selected_model_name in loaded_artifacts:
        current_model = loaded_artifacts[selected_model_name]
    else:
        st.error(f"Selected model {selected_model_name} not found in loaded artifacts!")
        st.stop()
    
    st.info(f"Using: **{selected_model_name}**")

# -----------------------------
# 4. User Inputs
# -----------------------------
st.subheader("📝 Enter Candidate Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 📚 Academic Scores")
    cgpa = st.number_input("CGPA (0-10)", 0.0, 10.0, 7.5, step=0.1)
    ssc = st.number_input("10th Marks (%)", 0, 100, 85)
    hsc = st.number_input("12th Marks (%)", 0, 100, 85)

with col2:
    st.markdown("##### 💼 Experience")
    internships = st.number_input("Internships", 0, 10, 1)
    projects = st.number_input("Projects", 0, 10, 2)
    workshops = st.number_input("Workshops/Certs", 0, 10, 1)

with col3:
    st.markdown("##### 🧠 Skills & Others")
    aptitude = st.slider("Aptitude Score", 0, 100, 70)
    soft_skills = st.slider("Soft Skills (0-5)", 0.0, 5.0, 3.5, 0.1)
    extra = st.selectbox("Extracurriculars", ["Yes", "No"])
    training = st.selectbox("Placement Training", ["Yes", "No"])

# -----------------------------
# 5. Preprocessing
# -----------------------------
extra_val = 1 if extra == "Yes" else 0
training_val = 1 if training == "Yes" else 0

academic_score = (ssc + hsc + (cgpa * 10)) / 3
experience_score = internships + projects + workshops

feature_cols = [
    "CGPA", "Internships", "Projects", "Workshops_Certifications", 
    "AptitudeTestScore", "SoftSkillsRating", 
    "ExtracurricularActivities", "PlacementTraining", 
    "SSC_Marks", "HSC_Marks", 
    "Academic_Score", "Experience_Score"
]

input_data = pd.DataFrame([[
    cgpa, internships, projects, workshops,
    aptitude, soft_skills, 
    extra_val, training_val, 
    ssc, hsc, 
    academic_score, experience_score
]], columns=feature_cols)

# -----------------------------
# 6. Prediction Logic
# -----------------------------
st.markdown("---")

if st.button("🚀 Analyze Placement Probability"):
    
    # 1. Scale Data
    scaled_data = scaler.transform(input_data)
    
    # 2. Predict
    prediction = current_model.predict(scaled_data)[0]
    probability = current_model.predict_proba(scaled_data)[0][1] * 100
    
    # 3. Display Results
    r_col1, r_col2 = st.columns([1, 1])
    
    with r_col1:
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"### 🎉 Likely to be Placed!")
            st.metric(label="Placement Probability", value=f"{probability:.2f}%", delta="High Chance")
            st.balloons()
            status_text = "High Probability of Placement"
        else:
            st.error(f"### ⚠️ Needs Improvement")
            st.metric(label="Placement Probability", value=f"{probability:.2f}%", delta="- Low Chance")
            st.markdown(f"**Action Plan:** Focus on increasing projects or Aptitude scores.")
            status_text = "Improvement Needed"

    with r_col2:
        st.subheader("Model Insights")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie([100-probability, probability], labels=['Risk', 'Success'], 
               autopct='%1.1f%%', startangle=90, colors=['#e74c3c', '#2ecc71'], 
               wedgeprops={'width': 0.4})
        st.pyplot(fig)

    # 4. Feature Importance
    if selected_model_name in ["Random Forest", "XGBoost"]:
        st.subheader("📊 Key Factors")
        importances = current_model.feature_importances_
        indices = np.argsort(importances)[-5:] 
        
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh(range(len(indices)), importances[indices], align='center', color='#4CAF50')
        ax2.set_yticks(range(len(indices)))
        ax2.set_yticklabels([feature_cols[i] for i in indices])
        st.pyplot(fig2)

    # -----------------------------
    # 7. Generate & Download Report
    # -----------------------------
    st.markdown("---")
    st.subheader("📄 Download Report")

    # Create Report Content
    report_content = f"""
    🎓 PLACEMENT PREDICTION REPORT
    =========================================
    Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
    Model Used: {selected_model_name}

    ---------------
    CANDIDATE PROFILE
    ---------------
    CGPA: {cgpa}
    10th Marks: {ssc}%
    12th Marks: {hsc}%
    Projects: {projects}
    Internships: {internships}
    Aptitude Score: {aptitude}
    Soft Skills: {soft_skills}/5.0

    ---------------
    AI ANALYSIS RESULT
    ---------------
    Prediction: {status_text}
    Probability Score: {probability:.2f}%

    ---------------
    RECOMMENDATION
    ---------------
    {"Keep up the great work! Your profile is strong." if prediction == 1 else "Focus on building more projects and improving your aptitude score."}
    
    Generated by Placement Predictor Pro
    """

    st.download_button(
        label="📥 Download Analysis Report (TXT)",
        data=report_content,
        file_name=f"placement_report_{int(time.time())}.txt",
        mime="text/plain"
    )