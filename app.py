import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Traffic Volume Predictor",
    page_icon="🚦",
    layout="wide"
)

model = joblib.load("models/traffic_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

st.title("🚦 Traffic Volume Prediction System")

st.markdown("""
This application predicts **traffic volume** using a machine learning model.

**Model Used:** XGBoost  
**Deployment:** Streamlit  
**Purpose:** Estimate traffic congestion based on weather and time conditions.
""")

st.divider()

st.sidebar.header("Enter Traffic Parameters")

holiday = st.sidebar.selectbox("Holiday (0 = No, 1 = Yes)", [0,1])
temp = st.sidebar.number_input("Temperature (°C)")
rain_1h = st.sidebar.number_input("Rain in last hour (mm)")
snow_1h = st.sidebar.number_input("Snow in last hour (mm)")
clouds_all = st.sidebar.slider("Cloud Cover (%)",0,100)

hour = st.sidebar.slider("Hour (0-23)",0,23)
day = st.sidebar.slider("Day (1-31)",1,31)
month = st.sidebar.slider("Month (1-12)",1,12)
weekday = st.sidebar.slider("Weekday (0-6)",0,6)

weather_main = st.sidebar.selectbox(
    "Weather Main",
    ["Clouds","Clear","Rain","Snow","Mist","Fog","Haze","Thunderstorm"]
)

weather_description = st.sidebar.selectbox(
    "Weather Description",
    [
        "sky is clear",
        "broken clouds",
        "few clouds",
        "scattered clouds",
        "overcast clouds",
        "light rain",
        "moderate rain",
        "heavy intensity rain"
    ]
)


input_data = pd.DataFrame({
    "holiday":[holiday],
    "temp":[temp],
    "rain_1h":[rain_1h],
    "snow_1h":[snow_1h],
    "clouds_all":[clouds_all],
    "hour":[hour],
    "day":[day],
    "month":[month],
    "weekday":[weekday],
    "weather_main":[weather_main],
    "weather_description":[weather_description]
})


input_encoded = pd.get_dummies(input_data)


final_input = input_encoded.reindex(columns=model_columns, fill_value=0)

st.subheader("Prediction")

if st.button("Predict Traffic Volume 🚗"):

    prediction = model.predict(final_input)
    traffic_volume = int(prediction[0])

    st.success(f"🚗 Estimated Traffic Volume: **{traffic_volume} vehicles**")

    if traffic_volume < 2000:
        st.info("🟢 Traffic Level: LOW congestion")
    elif traffic_volume < 4000:
        st.warning("🟡 Traffic Level: MODERATE congestion")
    else:
        st.error("🔴 Traffic Level: HIGH congestion")

    st.subheader("Input Summary")
    st.dataframe(input_data)

st.divider()

st.caption("Machine Learning Traffic Prediction Project")
