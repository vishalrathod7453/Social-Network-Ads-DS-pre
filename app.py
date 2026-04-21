import streamlit as st
import pickle
import pandas as pd
import requests
from streamlit_lottie import st_lottie

# --- Page Config ---
st.set_page_config(page_title="Ad Predictor", page_icon="🚀", layout="centered")

# --- CSS for Modern Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #007BFF; color: white; font-weight: bold; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- Animation Helper ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Animation URL (Data Analysis theme)
lottie_url = "https://assets10.lottiefiles.com/packages/lf20_qp1556hp.json"
lottie_data = load_lottieurl(lottie_url)

# --- Load Model ---
@st.cache_resource
def load_model():
    with open('ModelRa.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# --- UI Layout ---
st.title("🚀 Ad Conversion Predictor")
st.markdown("Enter the customer profile below to predict purchase likelihood.")

if lottie_data:
    st_lottie(lottie_data, height=200, key="data_anim")

# --- Input Form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
    
    with col2:
        # User ID is hidden/defaulted as it's not a relevant feature for prediction
        user_id = 0 
        salary = st.number_input("Estimated Salary ($)", min_value=15000, max_value=150000, value=50000)
    
    submit_button = st.form_submit_button(label="Predict Likelihood")

# --- Prediction Logic ---
if submit_button:
    # Encode Gender (Ensure this matches your training encoding: Male=1, Female=0)
    gender_val = 1 if gender == "Male" else 0
    
    # Create Input DataFrame
    input_df = pd.DataFrame([[user_id, gender_val, age, salary]], 
                            columns=['User ID', 'Gender', 'Age', 'EstimatedSalary'])
    
    # Prediction
    prediction = model.predict(input_df)
    
    # Display Result
    if prediction[0] == 1:
        st.success("🎉 The customer is likely to purchase!")
    else:
        st.info("📉 The customer is unlikely to purchase.")
