import pathlib
import os
import deeplearning_package

        # Data configuration
PACKAGE_ROOT = pathlib.Path(deeplearning_package.__file__).resolve().parent
DATA_DIR = os.path.join(PACKAGE_ROOT,"datasets")
TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'
        
MODEL_NAME = 'rain_prediction.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'saved_model')

TARGET = 'RainTomorrow'

FEATURES =  [
        'Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 
        'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 
        'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday'
        ]

TEST_FEATURES =  [
        'Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 
        'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 
        'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
        ]

NUM_FEATURES = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
        'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
        ] 

CAT_FEATURES = [
        'Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow'
        ]

FEATURES_TO_ENCODE =  [
        'Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow'
        ]

        # Model configuration
INPUT_SIZE = len(FEATURES)
OUTPUT_SIZE = 1  # Binary classification (rain or no rain)
DROPOUT_RATE = 0.2

        # Training configuration
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5

        # Evaluation configuration
EVAL_METRICS = ['accuracy', 'precision', 'recall', 'f1_score']

        # Other configurations
RANDOM_SEED = 42
