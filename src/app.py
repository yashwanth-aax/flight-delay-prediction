import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================================
# Locate project root automatically
# ==========================================================

CURRENT = Path(__file__).resolve()

PROJECT_ROOT = CURRENT.parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "xgboost.pkl"
ENCODER_PATH = MODELS_DIR / "encoders.pkl"

# Debug information
st.write("Project Root:", PROJECT_ROOT)
st.write("Models Folder:", MODELS_DIR)

if not MODEL_PATH.exists():
    st.error(f"Model not found:\n{MODEL_PATH}")
    st.stop()

if not ENCODER_PATH.exists():
    st.error(f"Encoders not found:\n{ENCODER_PATH}")
    st.stop()

# ==========================================================
# Load model
# ==========================================================

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)

# ==========================================================
# Page
# ==========================================================

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈ Flight Delay Prediction System")

st.markdown(
"""
Predict whether a flight will arrive **more than 15 minutes late** using an
XGBoost model trained on historical flight data.
"""
)

st.sidebar.title("Model Information")

st.sidebar.success("Model : XGBoost")
st.sidebar.info("Accuracy : 93.94%")
st.sidebar.info("ROC-AUC : 96.96%")

col1, col2 = st.columns(2)

with col1:

    month = st.selectbox(
        "Month",
        list(range(1, 13))
    )

    airline = st.selectbox(
        "Airline",
        sorted(encoders["AIRLINE"].classes_)
    )

    origin = st.selectbox(
        "Origin Airport",
        sorted(encoders["ORIGIN_AIRPORT"].classes_)
    )

    destination = st.selectbox(
        "Destination Airport",
        sorted(encoders["DESTINATION_AIRPORT"].classes_)
    )

with col2:

    scheduled_departure = st.number_input(
        "Scheduled Departure (HHMM)",
        value=900
    )

    scheduled_time = st.number_input(
        "Scheduled Flight Time",
        value=120
    )

    distance = st.number_input(
        "Distance",
        value=500
    )

    departure_delay = st.number_input(
        "Departure Delay",
        value=0
    )

    taxi_out = st.number_input(
        "Taxi Out",
        value=15
    )

if st.button("Predict Flight Delay"):

    departure_hour = scheduled_departure // 100

    is_morning = int(
        5 <= departure_hour < 12
    )

    input_df = pd.DataFrame({

        "MONTH":[month],

        "AIRLINE":[
            encoders["AIRLINE"].transform([airline])[0]
        ],

        "ORIGIN_AIRPORT":[
            encoders["ORIGIN_AIRPORT"].transform([origin])[0]
        ],

        "DESTINATION_AIRPORT":[
            encoders["DESTINATION_AIRPORT"].transform([destination])[0]
        ],

        "SCHEDULED_DEPARTURE":[scheduled_departure],

        "SCHEDULED_TIME":[scheduled_time],

        "DISTANCE":[distance],

        "DEPARTURE_DELAY":[departure_delay],

        "TAXI_OUT":[taxi_out],

        "DEPARTURE_HOUR":[departure_hour],

        "IS_MORNING":[is_morning]

    })

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:

        st.error("⚠ Flight is likely to be delayed.")

    else:

        st.success("✅ Flight is likely to arrive on time.")

    st.metric(
        "Delay Probability",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    st.subheader("Flight Summary")

    st.write(f"**Airline:** {airline}")
    st.write(f"**Route:** {origin} ➜ {destination}")
    st.write(f"**Departure Delay:** {departure_delay} min")
    st.write(f"**Taxi Out:** {taxi_out} min")