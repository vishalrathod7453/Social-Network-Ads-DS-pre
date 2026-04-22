import streamlit as st
import pandas as pd
import pickle
import requests
from streamlit_lottie import st_lottie

# --- Page Config ---
st.set_page_config(page_title="Ads Prediction App", page_icon="📈", layout="centered")

# --- Function to load Lottie animations ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- Load Model ---
@st.cache_resource
def load_model():
    with open('ModelRa.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- UI Styling ---
st.title("📊 Social Network Ads Predictor")
st.markdown("Enter the customer details below to predict if they will purchase the ad.")

# Lottie Animation
lottie_url = "https://assets5.lottiefiles.com/packages/lf20_m60s5y.json" # Tech/Data animation
lottie_json = load_lottieurl(lottie_url)
if lottie_json:
    st_lottie(lottie_json, height=200, key="data")

# --- Sidebar Inputs ---
st.sidebar.header("Customer Information")

# Encoding mapping (CRITICAL: Match this to how you trained your model)
gender_map = {'Male': 1, 'Female': 0}

gender_input = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 30)
salary = st.sidebar.number_input("Estimated Salary ($)", min_value=15000, max_value=150000, value=50000)

# Process Input
gender_encoded = gender_map[gender_input]
user_id = 0 # Placeholder as it usually doesn't affect prediction

# Create DataFrame for prediction
input_df = pd.DataFrame({
    'User ID': [user_id],
    'Gender': [gender_encoded],
    'Age': [age],
    'EstimatedSalary': [salary]
})

# --- Prediction ---
if st.button("Predict Purchase"):
    prediction = model.predict(input_df)
    
    st.balloons() # Visual animation effect
    
    if prediction[0] == 1:
        st.success("🎉 This customer is likely to purchase!")
    else:
        st.error("📉 This customer is unlikely to purchase.")
