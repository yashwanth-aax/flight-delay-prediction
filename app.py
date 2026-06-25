import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "xgboost.pkl"
ENCODER_PATH = ROOT / "models" / "encoders.pkl"

# ==========================================================
# Load Model
# ==========================================================

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)

# ==========================================================
# Streamlit Config
# ==========================================================

st.set_page_config(
    page_title="Flight Delay Prediction System",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Flight Delay Prediction System")

st.markdown(
"""
Predict whether a flight will arrive **more than 15 minutes late**
using a machine learning model trained on **300,000+ historical flights**.
"""
)

st.divider()

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("About")

st.sidebar.write("Model : XGBoost")

st.sidebar.write("Dataset Size : 300K Flights")

st.sidebar.write("Target : Arrival Delay > 15 min")

st.sidebar.write("ROC-AUC : 0.9696")

# ==========================================================
# Input Columns
# ==========================================================

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

    scheduled_departure = st.number_input(
        "Scheduled Departure Time (HHMM)",
        min_value=0,
        max_value=2359,
        value=900
    )

with col2:

    scheduled_time = st.number_input(
        "Scheduled Flight Time (minutes)",
        min_value=30,
        value=120
    )

    distance = st.number_input(
        "Distance (Miles)",
        min_value=50,
        value=500
    )

    departure_delay = st.number_input(
        "Departure Delay (minutes)",
        value=0
    )

    taxi_out = st.number_input(
        "Taxi Out Time (minutes)",
        min_value=0,
        value=15
    )

# ==========================================================
# Prediction
# ==========================================================

if st.button("Predict Flight Delay", use_container_width=True):

    departure_hour = scheduled_departure // 100

    is_morning = 1 if 5 <= departure_hour < 12 else 0

    airline_encoded = encoders["AIRLINE"].transform(
        [airline]
    )[0]

    origin_encoded = encoders["ORIGIN_AIRPORT"].transform(
        [origin]
    )[0]

    destination_encoded = encoders["DESTINATION_AIRPORT"].transform(
        [destination]
    )[0]

    input_df = pd.DataFrame({

        "MONTH":[month],

        "AIRLINE":[airline_encoded],

        "ORIGIN_AIRPORT":[origin_encoded],

        "DESTINATION_AIRPORT":[destination_encoded],

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

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠ Flight is likely to be Delayed.")

    else:

        st.success("✅ Flight is likely to arrive On Time.")

    st.metric(

        "Delay Probability",

        f"{probability*100:.2f}%"

    )

    st.progress(float(probability))

    st.subheader("Flight Summary")

    st.write(f"**Airline:** {airline}")

    st.write(f"**Route:** {origin} ➜ {destination}")

    st.write(f"**Departure Time:** {scheduled_departure}")

    st.write(f"**Distance:** {distance} miles")

    st.write(f"**Departure Delay:** {departure_delay} min")

    st.write(f"**Taxi Out:** {taxi_out} min")

    st.divider()

    if probability >= 0.80:

        st.error("Very High Risk of Delay")

    elif probability >= 0.50:

        st.warning("Moderate Risk of Delay")

    else:

        st.success("Low Risk of Delay")

st.divider()

st.caption(
"Built using Python, Scikit-learn, XGBoost and Streamlit"
)