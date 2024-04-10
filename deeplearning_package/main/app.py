import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from deeplearning_package.data.data_handling import DataPipeline
from deeplearning_package.config import config
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import uvicorn

app = FastAPI()
standard_scaler = StandardScaler()
# Step 1: Load the saved model
save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME + ".h5")

# Check if the file exists before attempting to load the model
if os.path.exists(save_path):
    # Load the model
    loaded_model = tf.keras.models.load_model(save_path)
    print(f"Model loaded successfully from {save_path}")
else:
    loaded_model = None
    print("Model file not found at:", save_path)

# Step 2: Define a FastAPI endpoint to handle predictions
@app.post("/predict")
async def predict(request: Request, temperature: float = Form(...), humidity: float = Form(...), 
                  wind_speed: float = Form(...), cloud_cover: float = Form(...), rain_today: str = Form(...)):
    
    if loaded_model:
        # Convert "rain_today" string to binary representation
        rain_today_binary = 1 if rain_today.lower() == "yes" else 0
        
        # Preprocess the input data
        scaled_temperature = standard_scaler.fit_transform([[temperature]])[0][0]
        scaled_humidity = standard_scaler.fit_transform([[humidity]])[0][0]
        scaled_wind_speed = standard_scaler.fit_transform([[wind_speed]])[0][0]
        scaled_cloud_cover = standard_scaler.fit_transform([[cloud_cover]])[0][0]
        
        # Preprocess the input data
        input_data = np.array([[scaled_temperature, scaled_humidity, scaled_wind_speed, scaled_cloud_cover, rain_today_binary]])
        
        
        # Make prediction
        predictions = loaded_model.predict(input_data)
        threshold = 0.7
        binary_prediction = (predictions > threshold).astype(int)[0][0]
        predicted_label = "Yes, it will rain" if binary_prediction == 1 else "No, it will not rain"
        return {"prediction": predicted_label}
    else:
        return {"error": "Model not loaded."}

# Step 3: Serve the HTML website
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    html_content = """
    <html>
    <head>
        <title>Rain Prediction</title>
    </head>
    <body>
        <h1>Rain Prediction</h1>
        <form id="prediction-form">
            <label for="temperature">Temperature (F):</label><br>
            <input type="number" id="temperature" name="temperature" step="0.01" required><br><br>
            <label for="humidity">Humidity (%):</label><br>
            <input type="number" id="humidity" name="humidity" step="0.01" required><br><br>
            <label for="wind_speed">Wind Speed (mph):</label><br>
            <input type="number" id="wind_speed" name="wind_speed" step="0.01" required><br><br>
            <label for="cloud_cover">Cloud Cover (%):</label><br>
            <input type="number" id="cloud_cover" name="cloud_cover" step="0.01" required><br><br>
            <label for="rain_today">Rain Today:</label><br>
            <select id="rain_today" name="rain_today" required>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select><br><br>
            <button type="submit">Predict</button>
        </form>
        <div id="prediction-result"></div>
        <script>
            const form = document.getElementById('prediction-form');
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(form);
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                document.getElementById('prediction-result').innerText = result.prediction;
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
