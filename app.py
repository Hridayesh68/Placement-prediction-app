import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ----------------------------
# Load models
# ----------------------------
lr = joblib.load("linear_model.pkl")
rf = joblib.load("rf_model.pkl")
xgb = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Placement Prediction",
    layout="wide"
)

st.title("🎓 Student Placement Prediction System")

# ----------------------------
# Model selection
# ----------------------------
model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    ["Linear Regression", "Random Forest", "XGBoost"]
)

# ----------------------------
# User Inputs
# ----------------------------
cgpa = st.number_input("CGPA", 0.0, 10.0, step=0.1)
internships = st.number_input("Internships", 0, 10)
projects = st.number_input("Projects", 0, 10)
certifications = st.number_input("Workshops / Certifications", 0, 10)
aptitude = st.slider("Aptitude Test Score", 0, 100)
soft_skills = st.slider("Soft Skills Rating", 0.0, 5.0)
extra = st.selectbox("Extracurricular Activities", ["Yes","No"])
training = st.selectbox("Placement Training", ["Yes","No"])


ssc = st.number_input("Class 10 Marks", min_value=0, max_value=100)
hsc = st.number_input("Class 12 Marks", min_value=0, max_value=100)
input_data = {
    "SSC_Marks": ssc,
    "HSC_Marks": hsc
}


extra = 1 if extra == "Yes" else 0
training = 1 if training == "Yes" else 0

features = np.array([[cgpa, internships, projects, certifications,
                      aptitude, soft_skills, extra, training,
                      ssc, hsc]])

scaled_features = scaler.transform(features)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Placement"):

    if model_choice == "Linear Regression":
        raw_score = lr.predict(scaled_features)[0]
        probability = min(max(raw_score, 0), 1) * 100
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

    # ----------------------------
    # Output result
    # ----------------------------
    if prediction == 1:
        st.success(f"🎉 PLACED ({model_used}) | Confidence: {probability:.2f}%")
    else:
        st.error(f"❌ NOT PLACED ({model_used}) | Confidence: {100-probability:.2f}%")

    # ----------------------------
    # PIE CHART
    # ----------------------------
    fig, ax = plt.subplots()
    ax.pie(
        [probability, 100-probability],
        labels=["Placed", "Not Placed"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Placement Probability")
    st.pyplot(fig)

    # ----------------------------
    # FEATURE IMPORTANCE (Tree models only)
    # ----------------------------
    if model_choice != "Linear Regression":
        st.subheader("📊 Feature Importance")

        importance = (
            rf.feature_importances_
            if model_choice == "Random Forest"
            else xgb.feature_importances_
        )

        feature_names = [
            "CGPA","Internships","Projects","Certifications",
            "Aptitude","Soft Skills","Extracurricular",
            "Placement Training","SSC","HSC"
        ]

        fig2, ax2 = plt.subplots()
        ax2.barh(feature_names, importance)
        ax2.set_title(f"{model_used} Feature Importance")
        st.pyplot(fig2)

    else:
        st.info("Feature importance not available for Linear Regression.")

    # ----------------------------
    # DOWNLOAD REPORT
    # ----------------------------
    report = pd.DataFrame({
        "Feature": [
            "CGPA","Internships","Projects","Certifications",
            "Aptitude","Soft Skills","Extracurricular",
            "Placement Training","SSC","HSC"
        ],
        "Value": features[0]
    })

    report["Model Used"] = model_used
    report["Prediction"] = "Placed" if prediction == 1 else "Not Placed"
    report["Probability (%)"] = probability

    csv = report.to_csv(index=False)

    st.download_button(
        "⬇️ Download Prediction Report",
        csv,
        "placement_prediction_report.csv",
        "text/csv"
    )


