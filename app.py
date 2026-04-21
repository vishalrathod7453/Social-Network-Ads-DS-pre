import streamlit as st
import pickle
import numpy as np
import json
from streamlit_lottie import st_lottie

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    with open("ModelDD.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ------------------ LOAD LOTTIE ANIMATION ------------------
def load_lottiefile(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Download any Lottie JSON and save as "animation.json"
lottie_animation = load_lottiefile("animation.json")

# ------------------ UI DESIGN ------------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to right, #1f4037, #99f2c8);
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: white;
    }
    .subtext {
        text-align: center;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">🚀 ML Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Predict using your trained model</p>', unsafe_allow_html=True)

# ------------------ ANIMATION ------------------
st_lottie(lottie_animation, height=250)

# ------------------ INPUT FIELDS ------------------
st.subheader("📊 Enter Input Features")

# ⚠️ MODIFY THESE INPUTS based on your model features
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

# ------------------ PREDICTION ------------------
if st.button("🔮 Predict"):
    try:
        features = np.array([[feature1, feature2, feature3]])
        prediction = model.predict(features)

        st.success(f"✅ Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
