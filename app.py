# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --------------------------
# Load Trained Model + Preprocessors
# --------------------------
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Sustainable Crop Recommendation", page_icon="🌱", layout="centered")

st.title("🌱 Sustainable Agriculture Assistant")
st.write("Enter your soil & climate details to get the best **crop recommendation**")

# Input fields
N = st.number_input("Nitrogen content (N)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus content (P)", min_value=0, max_value=200, value=50)
K = st.number_input("Potassium content (K)", min_value=0, max_value=200, value=50)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

# Predict button
if st.button("🌾 Recommend Crop"):
    # Prepare input
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)

    # Prediction with probabilities
    probs = model.predict_proba(input_scaled)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]  # Top 3 crops
    top3_crops = le.inverse_transform(top3_idx)
    top3_probs = probs[top3_idx] * 100

    # Main recommendation
    best_crop = top3_crops[0]
    st.success(f"✅ Recommended Sustainable Crop: **{best_crop.capitalize()}**")

    # Show top 3 with bar chart
    st.write("### 📊 Top 3 Crop Probabilities")
    prob_df = pd.DataFrame({"Crop": top3_crops, "Probability (%)": top3_probs})
    st.bar_chart(prob_df.set_index("Crop"))

    # Extra sustainability tip
    if best_crop in ["rice", "sugarcane"]:
        st.info("💧 Tip: These crops need lots of water. Ensure efficient irrigation!")
    elif best_crop in ["maize", "millet", "barley"]:
        st.info("🌾 Tip: These crops are climate-resilient and eco-friendly.")
    else:
        st.info("🌍 Tip: Rotate this crop with legumes for soil health improvement.")
