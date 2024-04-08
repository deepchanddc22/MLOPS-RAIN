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
    X_test = test_data_pipeline.data[config.FEATURES]  # Assuming this is the correct way to access your test data
    start = 4
    end = 5
    print(X_test[start:end:])  # Print a sample of your test data to verify it's in the correct format
    
    predictions = loaded_model.predict(X_test[start:end:])  # Making predictions for a single sample
    threshold = 0.5
    
    def map_binary_to_labels(binary_predictions):
        for pred in binary_predictions:
            if pred > threshold:
                print("Yes, it will rain")
            else:
                print("No, it will not rain")
    
    binary_predictions = (predictions > threshold).astype(int)
    map_binary_to_labels(binary_predictions)
else:
    print("Model file not found at:", save_path)

# Step 4: Make predictions on the preprocessed test dataset using the loaded model

# Print or use the predictions as needed
