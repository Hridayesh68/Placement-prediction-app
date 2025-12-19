import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "placementdata.csv"

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(DATA_PATH)

print("Original columns:")
print(df.columns.tolist())

# ----------------------------
# Rename problematic column
# ----------------------------
df.rename(
    columns={
        "Workshops/Certifications": "Certifications"
    },
    inplace=True
)

# ----------------------------
# Encode categorical columns
# ----------------------------
df["ExtracurricularActivities"] = df["ExtracurricularActivities"].map(
    {"Yes": 1, "No": 0}
)
df["PlacementTraining"] = df["PlacementTraining"].map(
    {"Yes": 1, "No": 0}
)
df["PlacementStatus"] = df["PlacementStatus"].map(
    {"Placed": 1, "NotPlaced": 0}
)

# ----------------------------
# Drop ID column
# ----------------------------
df.drop("StudentID", axis=1, inplace=True)

# ----------------------------
# Feature order (LOCKED – MUST match app.py)
# ----------------------------
FEATURE_COLUMNS = [
    "CGPA",
    "Internships",
    "Projects",
    "Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "ExtracurricularActivities",
    "PlacementTraining",
    "SSC_Marks",
    "HSC_Marks",
]

# ----------------------------
# Safety check
# ----------------------------
missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

# ----------------------------
# Split features & target
# ----------------------------
X = df[FEATURE_COLUMNS]
y = df["PlacementStatus"]

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# Feature scaling
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Linear Regression (used as classifier)
# ----------------------------
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

lr_preds = (lr.predict(X_test_scaled) >= 0.5).astype(int)
print("Linear Regression Accuracy:",
      accuracy_score(y_test, lr_preds))

# ----------------------------
# Random Forest
# ----------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

print("Random Forest Accuracy:",
      accuracy_score(y_test, rf.predict(X_test_scaled)))

# ----------------------------
# XGBoost (Streamlit-safe)
# ----------------------------
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)
xgb.fit(X_train_scaled, y_train)

print("XGBoost Accuracy:",
      accuracy_score(y_test, xgb.predict(X_test_scaled)))

# ----------------------------
# Save models to ROOT
# ----------------------------
joblib.dump(lr, BASE_DIR / "linear_model.pkl")
joblib.dump(rf, BASE_DIR / "rf_model.pkl")
joblib.dump(xgb, BASE_DIR / "xgb_model.pkl")
joblib.dump(scaler, BASE_DIR / "scaler.pkl")

print("✅ Models saved successfully")
