import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from deeplearning_package.config import config
import os
import mlflow

class DataPipeline:
    def __init__(self, file_name):
        self.file_name = file_name
    
    def load_dataset(self):
        filepath = os.path.join(config.DATA_DIR, self.file_name)
        self.data = pd.read_csv(filepath)
    
    def preprocess_data(self):
        with mlflow.start_run(nested=True):
            mlflow.log_param("File Name", self.file_name)
            
            self.load_dataset()
            self.drop_null_values()
            self.label_encode_categorical_data()
            self.standardize_features()
        
        # Log preprocessing steps
        mlflow.log_params({
            "Drop Null Values": True,
            "Label Encode Categorical Data": True,
            "Standardize Features": True
        })
    
    def drop_null_values(self):
        self.data.dropna(inplace=True)
    
    def label_encode_categorical_data(self):
        categorical_features = config.CAT_FEATURES
        label_encoders = {}
        for feature in categorical_features:
            label_encoders[feature] = LabelEncoder()
            self.data[feature] = label_encoders[feature].fit_transform(self.data[feature])
    
    def standardize_features(self):
        all_features = config.FEATURES
        scaler = StandardScaler()
        self.data[all_features] = scaler.fit_transform(self.data[all_features])
    
    def split_data(self):
        X = self.data[config.FEATURES]
        y = self.data[config.TARGET]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def print_data_head(self):
        print("DataFrame head after preprocessing:")
        print(self.data.head())

if __name__ == "__main__":
    with mlflow.start_run(run_name="Data Preprocessing"):
        data_pipeline = DataPipeline(config.TRAIN_DATA_FILE)
        data_pipeline.preprocess_data()
        X_train, X_test, y_train, y_test = data_pipeline.split_data()
        data_pipeline.print_data_head()
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)
