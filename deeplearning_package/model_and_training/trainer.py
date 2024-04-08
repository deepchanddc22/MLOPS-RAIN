import mlflow
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from deeplearning_package.data.data_handling import DataPipeline
from deeplearning_package.config import config
import joblib
import pickle
import os

def train_model_and_log_metrics():
    # Initialize data pipeline
    data_pipeline = DataPipeline(config.TRAIN_DATA_FILE)

    # Preprocess the data
    data_pipeline.preprocess_data()
    X_train, X_test, y_train, y_test = data_pipeline.split_data()

    # Start MLflow run
    with mlflow.start_run(run_name="Keras Model Training", nested=True):

        # Log parameters
        mlflow.log_param("Model Architecture", "Feedforward Neural Network")
        mlflow.log_param("Optimizer", "Adam")
        mlflow.log_param("Learning Rate", config.LEARNING_RATE)
        mlflow.log_param("Epochs", config.EPOCHS)
        mlflow.log_param("Batch Size", config.BATCH_SIZE)

        # Build Keras model
        model = Sequential([
            Dense(units=32, kernel_initializer='uniform', activation='relu', input_dim=config.INPUT_SIZE),
            Dense(units=32, kernel_initializer='uniform', activation='relu'),
            Dense(units=16, kernel_initializer='uniform', activation='relu'),
            Dropout(0.25),
            Dense(units=16, kernel_initializer='uniform', activation='relu'),
            Dropout(0.25),
            Dense(units=8, kernel_initializer='uniform', activation='relu'),
            Dropout(0.5),
            Dense(units=8, kernel_initializer='uniform', activation='relu'),
            Dropout(0.5),
            Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
        ])

        # Compile the model
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Define early stopping callback
        early_stopping = EarlyStopping(min_delta=config.LEARNING_RATE, patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True)

        # Train the model
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, callbacks=[early_stopping])

        # Log metrics
        mlflow.log_metric("Final Loss", history.history['loss'][-1])
        mlflow.log_metric("Final Accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("Final Validation Loss", history.history['val_loss'][-1])
        mlflow.log_metric("Final Validation Accuracy", history.history['val_accuracy'][-1])

        # Return history object

        model_save_path = config.SAVE_MODEL_PATH # Define the path where you want to save the model
        save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
        joblib.dump(model,save_path)
        print(f"Model has been saved under the name {config.MODEL_NAME}")
        # Log the saved model as an artifact
        mlflow.log_artifact(model_save_path, artifact_path="trained_models")
        return history
        