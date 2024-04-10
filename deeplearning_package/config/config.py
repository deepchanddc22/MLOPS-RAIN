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

TARGET = 'Rain Tomorrow'

FEATURES =  [
        'Temperature (F)','Humidity (%)','Wind Speed (mph)','Cloud Cover (%)','Rain Today','Rain Tomorrow'
        ]

TEST_FEATURES =  [
        'Temperature (F)','Humidity (%)','Wind Speed (mph)','Cloud Cover (%)','Rain Today'
        ]
NUM_FEATURES = [
      'Temperature (F)','Humidity (%)','Wind Speed (mph)','Cloud Cover (%)'
        ] 

CAT_FEATURES = [
        'Rain Today','Rain Tomorrow'
        ]

FEATURES_TO_ENCODE =  [
        'Rain Today','Rain Tomorrow'
        ]

        # Model configuration
INPUT_SIZE = len(TEST_FEATURES)
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
