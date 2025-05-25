# Streamlit demo app for RUL prediction
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
import tensorflow as tf

# Custom CSS for stylish gradient background, buttons, cards
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
        color: #eaeaea;
    }
    .stApp {
        background: linear-gradient(135deg, #232526 0%, #232526 40%, #3a6186 100%) !important;
        color: #eaeaea;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #43cea2, #185a9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        letter-spacing: 1.2px;
    }
    .css-1cpxqw2 edgvbvh3 { /* File uploader button */
        background: linear-gradient(90deg, #43cea2, #185a9d);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff9966, #ff5e62);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        transition: 0.2s;
    }
    .stButton>button:hover {
        filter: brightness(1.1);
        box-shadow: 0 4px 12px rgba(67,206,162,0.25);
    }
    .stAlert {
        background: linear-gradient(90deg,#43cea2,#185a9d)!important;
        color:#fff!important;
        font-weight:600;
        border-radius:8px!important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center;'>🛠️ RUL Prediction Demo <span style='font-size:0.7em;'>(NASA C-MAPSS, LSTM)</span></h1>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload normalized test window (shape: 1, 30, features)", type=["npy"])
if uploaded is not None:
    window = np.load(uploaded)
    model = tf.keras.models.load_model("models/best_lstm.h5", compile=False)
    pred = model.predict(window)
    st.markdown(f"""
        <div style='background:linear-gradient(90deg,#43cea2,#185a9d);padding:24px 0 24px 0;border-radius:18px;text-align:center;margin-top:32px;'>
            <span style='font-size:1.6em;font-weight:700;color:#fff;'>Predicted RUL:<br><span style='font-size:2.3em;'>{pred.flatten()[0]:.2f}</span> cycles</span>
        </div>""", unsafe_allow_html=True)
else:
    st.info("Upload a single LSTM input window (numpy array) to get RUL prediction.")
