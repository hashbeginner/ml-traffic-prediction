import streamlit as st
import pandas as pd
import joblib


model = joblib.load("models/traffic_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

st.title("Traffic Volume Prediction 🚦🚗")

holiday = st.selectbox("Holiday "
"(Select '0' for Non Holiday and '1' for Holiday)", [0,1])
temp = st.number_input("Temperature (°C)")
rain_1h = st.number_input("Rain in last hour (mm)")
snow_1h = st.number_input("Snow in last hour (mm)")
clouds_all = st.slider("Cloud Cover %",0,100)

hour = st.slider("Hour (0-23)",0,23)
day = st.slider("Day(1-31)",1,31)
month = st.slider("Month(1-12)",1,12)
weekday = st.slider("Weekday(0-6)",0,6)

weather_main = st.selectbox(
    "Weather Main",
    ["Clouds","Clear","Rain","Snow","Mist","Fog","Haze","Thunderstorm"]
)

weather_description = st.selectbox(
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


if st.button("Predict Traffic"):
    prediction = model.predict(final_input)
    st.success(f"Predicted Traffic Volume: {int(prediction[0])}")