import streamlit as st
import pickle
import pandas as pd
import requests
from streamlit_lottie import st_lottie

# --- Page Config ---
st.set_page_config(page_title="Ad Predictor", page_icon="🎯", layout="centered")

# --- CSS for Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Function to Load Lottie Animation ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Animation
lottie_url = "https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json" # Example animation
lottie_data = load_lottieurl(lottie_url)

# --- Load Model ---
@st.cache_resource
def load_model():
    with open('ModelDD.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- Sidebar / Header ---
st.title("🎯 Social Network Ad Predictor")
if lottie_data:
    st_lottie(lottie_data, height=200, key="coding")

st.markdown("### Enter Customer Details")

# --- User Input Form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
    
    with col2:
        salary = st.number_input("Estimated Salary ($)", min_value=15000, max_value=150000, value=50000)
    
    submit_button = st.form_submit_button(label="Predict Purchase Likelihood")

# --- Prediction Logic ---
if submit_button:
    # Prepare data for model
    # Mapping Gender: Male=1, Female=0 (Ensure this matches your training data!)
    gender_val = 1 if gender == "Male" else 0
    
    # Input DataFrame (User ID is typically excluded for prediction)
    input_data = pd.DataFrame([[gender_val, age, salary]], 
                              columns=['Gender', 'Age', 'EstimatedSalary'])
    
    # Prediction
    prediction = model.predict(input_data)
    
    # Display Result
    if prediction[0] == 1:
        st.success("✅ The customer is likely to purchase!")
    else:
        st.warning("❌ The customer is unlikely to purchase.")
