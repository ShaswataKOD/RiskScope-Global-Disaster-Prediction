# streamlit_app_simple.py
import streamlit as st
import joblib
import numpy as np

# Load model (ensure scikit-learn version matches the one used for training)
model = joblib.load('random_model.pkl')

st.title("Random Forest Predictor 🌳")
st.write("Enter values for all features to predict the target.")

# Input fields
exposure = st.number_input("Exposure", 0.0, 100.0, 50.0)
vulnerability = st.number_input("Vulnerability", 0.0, 100.0, 50.0)
susceptibility = st.number_input("Susceptibility", 0.0, 100.0, 50.0)
lack_coping = st.number_input("Lack of Coping Capabilities", 0.0, 100.0, 50.0)
lack_adaptive = st.number_input("Lack of Adaptive Capacities", 0.0, 100.0, 50.0)
year = st.number_input("Year", 1900, 2100, 2025)

# Predict button
if st.button("Predict"):
    features = np.array([[exposure, vulnerability, susceptibility, lack_coping, lack_adaptive, year]])
    prediction = model.predict(features)
    st.success(f"Predicted Target Value: {prediction[0]:.2f}")
st.write("Note: Ensure all inputs are provided for accurate prediction.")