import subprocess
from pathlib import Path
from deeplearning_package.config import config
import os


# Define the paths to the Python scripts
data_folder = os.path.join(config.PACKAGE_ROOT,'data') # Assuming the scripts are in a folder named "scripts"
data_sub_folder = "data_handling.py"

subprocess.run(["python", str(os.path.join(data_folder, data_sub_folder))])



model_and_training_folder = os.path.join(config.PACKAGE_ROOT,'model_and_training')
model_and_training_folder_sub_folder_one = "trainer.py"
model_and_training_folder_sub_folder_two = "metrics.py"

subprocess.run(["python", str(os.path.join(model_and_training_folder,model_and_training_folder_sub_folder_one))])

subprocess.run(["python", str(os.path.join(model_and_training_folder,model_and_training_folder_sub_folder_two))])


