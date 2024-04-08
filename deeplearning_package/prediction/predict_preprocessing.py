from deeplearning_package.data.data_handling import DataPipeline
from deeplearning_package.config import config
import os
import tensorflow as tf
# Step 1: Create an instance of DataPipeline for the test dataset
test_data_pipeline = DataPipeline(config.TEST_DATA_FILE)

# Step 2: Preprocess the test dataset
test_data_pipeline.preprocess_data()

# Step 3: Load the saved model
save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
if os.path.exists(save_path):
    loaded_model = tf.keras.models.load_model(save_path)
    X_test = test_data_pipeline.data[config.FEATURES]  # Assuming FEATURES contains the relevant features for prediction
    predictions = loaded_model.predict(X_test)
    print(predictions)
else:
    print("Model file not found at:", save_path)

# Step 4: Make predictions on the preprocessed test dataset using the loaded model

# Print or use the predictions as needed
