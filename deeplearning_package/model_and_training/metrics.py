import matplotlib.pyplot as plt
from deeplearning_package.model_and_training.trainer import train_model_and_log_metrics
import mlflow
import pandas as pd

# Get history without training
history = train_model_and_log_metrics()

# Convert history to DataFrame
history_df = pd.DataFrame(history.history)

# Plot training and validation loss
plt.figure(figsize=(10, 4), dpi=200)
plt.plot(history_df.loc[:, ['loss']], "Red", label='Training loss')
plt.plot(history_df.loc[:, ['val_loss']], "Green", label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")

# Save the plot as a PNG file
plt.savefig("training_validation_loss.png")

# Log the plot to MLflow as an artifact
with mlflow.start_run(nested=True):
    mlflow.log_artifact("training_validation_loss.png")

# Close the plot
plt.close()
