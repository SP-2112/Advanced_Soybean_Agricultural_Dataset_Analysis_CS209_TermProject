import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# 📦 Load XGBoost Classification Model + Expected Features
# -------------------------------
model = joblib.load("xgboost_classifier_kmeans_allparams.pkl")  # <-- Change filename if different
expected_features = joblib.load("xgboost_classifier_kmeans_allparams_features.pkl")  # list of features during training

# -------------------------------
# 🌾 App Title
# -------------------------------
st.set_page_config(page_title="Soybean Productivity Classifier 🌿", layout="wide")
st.title("🌾 Soybean Productivity Classifier")
st.markdown("This tool predicts whether a given soybean experiment setup will lead to **Good Production (1)** or **Poor Production (0)**.")
st.markdown("---")

# -------------------------------
# 🎛️ Input Sidebar
# -------------------------------
st.sidebar.header("🧪 Input Parameters")

# 🌱 Feature Inputs — Editable as per your final feature set
G = st.sidebar.selectbox("Genotype (G)", [1, 2, 3, 4, 5, 6])
S = st.sidebar.selectbox("Salicylic Acid (S)", [1, 2, 3])
C = st.sidebar.selectbox("Water Stress (C)", [1, 2])
PH = st.sidebar.slider("Plant Height (PH)", 20.0, 120.0, 70.0)
BW = st.sidebar.slider("Biological Weight (BW)", 50.0, 400.0, 200.0)
CHL_A = st.sidebar.slider("Chlorophyll A663", 0.0, 15.0, 4.0)
CHL_B = st.sidebar.slider("Chlorophyll b649", 0.0, 8.0, 2.0)
RWCL = st.sidebar.slider("Relative Water Content in Leaves (RWCL)", 0.3, 1.0, 0.7)
LAI = st.sidebar.slider("Leaf Area Index (LAI)", 0.0, 6.0, 2.0)
NP = st.sidebar.slider("Number of Pods (NP)", 50, 400, 200)
NSP = st.sidebar.slider("Number of Seeds per Pod (NSP)", 1.0, 3.0, 2.0)
PPE = st.sidebar.slider("Protein Percentage (PPE)", 25.0, 50.0, 35.0)
PCO = st.sidebar.slider("Protein Content (PCO)", 5.0, 20.0, 10.0)
SU = st.sidebar.slider("Sugars (Su)", 0.1, 1.0, 0.5)
W3S = st.sidebar.slider("Weight of 300 Seeds (W3S)", 30.0, 100.0, 60.0)
SYUA = st.sidebar.slider("Seed Yield per Unit Area (SYUA)", 20.0, 120.0, 60.0)

# -------------------------------
# 🧩 Construct Feature Vector
# -------------------------------
input_dict = {
    "G": G,
    "S": S,
    "C": C,
    "Plant Height (PH)": PH,
    "Biological Weight (BW)": BW,
    "ChlorophyllA663": CHL_A,
    "Chlorophyllb649": CHL_B,
    "Relative Water Content in Leaves (RWCL)": RWCL,
    "Leaf Area Index (LAI)": LAI,
    "Number of Pods (NP)": NP,
    "Number of Seeds per Pod (NSP)": NSP,
    "Protein Percentage (PPE)": PPE,
    "Protein Content (PCO)": PCO,
    "Sugars (Su)": SU,
    "Weight of 300 Seeds (W3S)": W3S,
    "Seed Yield per Unit Area (SYUA)": SYUA
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# One-hot encode G, S, C
input_df_encoded = pd.get_dummies(input_df, columns=["G", "S", "C"])

# Fill missing dummy cols with 0
for col in expected_features:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

# Reorder columns
input_df_encoded = input_df_encoded[expected_features]

# -------------------------------
# 🚀 Predict
# -------------------------------
if st.button("🔍 Predict Productivity Class"):
    prediction = model.predict(input_df_encoded)[0]
    prob = model.predict_proba(input_df_encoded)[0][int(prediction)]

    st.markdown("### ✅ Prediction:")
    if prediction == 1:
        st.success(f"🌟 Predicted Class: **Good Production (1)**")
        st.markdown(f"**Confidence**: `{prob * 100:.2f}%`")
        # st.balloons()
    else:
        st.error(f"🔻 Predicted Class: **Poor Production (0)**")
        st.markdown(f"**Confidence**: `{prob * 100:.2f}%`")

# -------------------------------
# 📌 Footer
# -------------------------------
st.markdown("---")
st.caption("Built by Team 3 • Soybean Productivity Classifier 🌱")