import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# 📦 Load model + scaler + columns
# -------------------------------
model = joblib.load("regression_model_composite_yield.pkl")
scaler = joblib.load("minmax_scaler_wyua_ppe.pkl")
expected_features = joblib.load("feature_names_expected.pkl")

# -------------------------------
# 🌾 App Title
# -------------------------------
st.set_page_config(page_title="Soybean Composite Predictor 🌱", layout="wide")
st.title("🌾 Composite Yield & Protein Score Predictor")
st.markdown("This tool predicts the **Composite Yield Metric** (yield + protein) based on soybean growth conditions.")

st.markdown("---")

# -------------------------------
# 🎛️ Input Sidebar
# -------------------------------
st.sidebar.header("🧪 Input Parameters")

# Dropdowns for G/S/C
G = st.sidebar.selectbox("Genotype (G)", [1, 2, 3, 4, 5, 6])
S = st.sidebar.selectbox("Salicylic Acid Level (S)", [1, 2, 3])
C = st.sidebar.selectbox("Water Stress (C)", [1, 2])

# Sliders for remaining important inputs
PH = st.sidebar.slider("Plant Height (PH)", 20.0, 120.0, 70.0)
BW = st.sidebar.slider("Biological Weight (BW)", 50.0, 400.0, 200.0)
CHL_A = st.sidebar.slider("Chlorophyll A663", 0.0, 15.0, 4.0)
CHL_B = st.sidebar.slider("Chlorophyll b649", 0.0, 8.0, 2.0)
RWCL = st.sidebar.slider("Relative Water Content in Leaves", 0.3, 1.0, 0.7)
LAI = st.sidebar.slider("Leaf Area Index (LAI)", 0.5, 6.0, 3.0)
NP = st.sidebar.slider("Number of Pods (NP)", 50, 400, 200)

# -------------------------------
# 📊 Build input row for prediction
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
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# One-hot encode categorical G/S/C
input_df_encoded = pd.get_dummies(input_df, columns=["G", "S", "C"])

# Add any missing dummy columns with 0
for col in expected_features:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

# Ensure correct order
input_df_encoded = input_df_encoded[expected_features]

# -------------------------------
# 🚀 Predict
# -------------------------------
if st.button("🔍 Predict Composite Score"):
    prediction = model.predict(input_df_encoded)[0]
    st.success(f"🎯 Predicted Composite Score: **{prediction:.2f} / 100**")

    # Optional badge
    if prediction > 80:
        st.balloons()
        st.markdown("🌟 **Excellent Conditions!** Likely a top 20% producer.")
    elif prediction > 60:
        st.markdown("✅ **Good Conditions.** You can optimize further.")
    else:
        st.markdown("⚠️ Consider tweaking input features for better yield.")

# -------------------------------
# 📌 Footer
# -------------------------------
st.markdown("---")
st.caption("Built by Team 3 • Soybean Yield Optimizer 🌱")
