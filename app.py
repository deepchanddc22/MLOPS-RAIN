import streamlit as st
import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from deeplearning_package.config import config
import subprocess

@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        return None

def preprocess_data(temperature, humidity, wind_speed, cloud_cover, rain_today):
    standard_scaler = StandardScaler()
    rain_today_binary = 1 if rain_today.lower() == "yes" else 0
    scaled_temperature = standard_scaler.fit_transform([[temperature]])[0][0]
    scaled_humidity = standard_scaler.fit_transform([[humidity]])[0][0]
    scaled_wind_speed = standard_scaler.fit_transform([[wind_speed]])[0][0]
    scaled_cloud_cover = standard_scaler.fit_transform([[cloud_cover]])[0][0]
    return np.array([[scaled_temperature, scaled_humidity, scaled_wind_speed, scaled_cloud_cover, rain_today_binary]])

# Make prediction
def make_prediction(model, input_data, threshold=0.7):
    predictions = model.predict(input_data)
    binary_prediction = (predictions > threshold).astype(int)[0][0]
    return binary_prediction

def main():
    # Load model
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME + ".h5")
    loaded_model = load_model(save_path)

    # Set background color to black
    st.markdown(
        """
        <style>
        body {
            background-color: #000000;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Streamlit UI
    st.title("🌧️ Rain Predictor")
    st.text("Made by: Deepchand O A")
    temperature = st.number_input("Temperature (F)", step=1, value=0, format="%d", min_value=0)
    humidity = st.number_input("Humidity (%)", step=1, value=0, format="%d", min_value=0)
    wind_speed = st.number_input("Wind Speed (mph)", step=1, value=0, format="%d", min_value=0)
    cloud_cover = st.number_input("Cloud Cover (%)", step=1, value=0, format="%d", min_value=0)
    rain_today = st.selectbox("Rain Today", ["Yes", "No"])

    if st.button("Predict"):
        if loaded_model:
            input_data = preprocess_data(temperature, humidity, wind_speed, cloud_cover, rain_today)
            prediction = make_prediction(loaded_model, input_data)
            if prediction == 1:
                st.success("Yes, it will rain")
            else:
                st.warning("No, it will not rain")
        else:
            st.error("Model not loaded.")

if __name__ == "__main__":
    # Run the Streamlit app using subprocess
    command = [
        "streamlit", "run", __file__,
        "--server.address", "0.0.0.0",
        "--server.port", "8080"
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode())
    print(stderr.decode())

    main()
