# soybean_predictor_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------
# 🎯 Load Trained Model and Scaler
# -----------------------------------

# Load your trained regression model and MinMaxScaler
model = joblib.load("regression_model_composite_yield.pkl")  # Save this from training
scaler = joblib.load("minmax_scaler_wyua_ppe.pkl")            # Same scaler used for WYUA and PPE

# -----------------------------------
# 🎛️ Streamlit UI
# -----------------------------------

st.set_page_config(page_title="Soybean Productivity Estimator", layout="centered")
st.title("🌿 Soybean Productivity Estimator")
st.markdown("Predict composite soybean yield metric using physiological & experimental inputs.")

# --- Categorical dropdowns
G = st.selectbox("Genotype (G)", options=[1, 2, 3, 4, 5, 6], index=0)
S = st.selectbox("Salicylic Acid (S)", options=[1, 2, 3], index=0)
C = st.selectbox("Water Stress (C)", options=[1, 2, 3], index=0)

# --- Continuous features
ph = st.slider("Plant Height (PH)", 20.0, 120.0, 80.0)
npods = st.slider("Number of Pods (NP)", 20, 200, 100)
bw = st.slider("Biological Weight (BW)", 30.0, 300.0, 150.0)
su = st.slider("Sugars (Su)", 1.0, 12.0, 6.0)
rwcl = st.slider("Relative Water Content in Leaves (RWCL)", 0.2, 0.9, 0.6)
chl_a = st.slider("ChlorophyllA663", 0.0, 10.0, 5.0)
chl_b = st.slider("Chlorophyllb649", 0.0, 10.0, 5.0)
ppe = st.slider("Protein Percentage (PPE)", 20.0, 50.0, 30.0)
w3s = st.slider("Weight of 300 Seeds (W3S)", 30.0, 120.0, 80.0)
lai = st.slider("Leaf Area Index (LAI)", 1.0, 10.0, 4.0)
syua = st.slider("Seed Yield per Unit Area (SYUA)", 30.0, 250.0, 100.0)
nsp = st.slider("Number of Seeds per Pod (NSP)", 1.0, 5.0, 2.5)
pco = st.slider("Protein Content (PCO)", 10.0, 30.0, 20.0)

# -----------------------------------
# 🧮 Feature Engineering
# -----------------------------------
wyua = (w3s / 300) * syua
wyua_scaled, ppe_scaled = scaler.transform([[wyua, ppe]])[0]
composite_metric = ((wyua_scaled + ppe_scaled) / 2) * 100

# -----------------------------------
# 🔢 Prepare Model Input
# -----------------------------------
input_data = pd.DataFrame([[
    ph, npods, bw, su, rwcl, chl_a, chl_b, ppe, w3s,
    lai, syua, nsp, pco, G, S, C
]], columns=[
    "Plant Height (PH)", "Number of Pods (NP)", "Biological Weight (BW)",
    "Sugars (Su)", "Relative Water Content in Leaves (RWCL)", "ChlorophyllA663",
    "Chlorophyllb649", "Protein Percentage (PPE)", "Weight of 300 Seeds (W3S)",
    "Leaf Area Index (LAI)", "Seed Yield per Unit Area (SYUA)", "Number of Seeds per Pod (NSP)",
    "Protein Content (PCO)", "G", "S", "C"
])

# -----------------------------------
# 📈 Predict
# -----------------------------------
pred = model.predict(input_data)[0]

st.markdown("---")
st.metric(label="📊 Composite Yield Metric (0–100)", value=f"{composite_metric:.2f}")
st.markdown(f"🧠 Model Prediction (Trained Output): `{pred:.2f}`")
st.caption("Note: Composite Yield Metric = Scaled (WYUA + PPE) / 2")

# Optional: Show feature summary
with st.expander("🔍 Show Feature Summary"):
    st.dataframe(input_data.T.rename(columns={0: "Your Input"}))

