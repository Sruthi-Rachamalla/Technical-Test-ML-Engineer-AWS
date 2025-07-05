import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tarfile
from sklearn.metrics import mean_squared_error

# Paths
tar_path = "/opt/ml/processing/model/model.tar.gz"
extract_path = "/opt/ml/processing/model/extracted"
evaluation_output_path = "/opt/ml/processing/evaluation"
test_path = os.environ["SM_CHANNEL_TEST"]

# Untar the model
os.makedirs(extract_path, exist_ok=True)
with tarfile.open(tar_path) as tar:
    tar.extractall(path=extract_path)

# Load test data
test_df = pd.read_csv(os.path.join(test_path, "test.csv"))

def prepare_data(dataset, time_interval=25, return_3d=True):
    dataX, dataY = [], []
    for i in range(0, len(dataset) - time_interval - 1):
        dataX.append(dataset[i:i + time_interval])
        dataY.append(dataset[i + time_interval])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    if return_3d:
        return dataX.reshape(dataX.shape[0], dataX.shape[1], 1), dataY.reshape(dataY.shape[0], 1)
    else:
        return dataX, dataY

X_test, y_test = prepare_data(test_df.squeeze(), time_interval=25, return_3d=True)

# Load the model from extracted path
model = tf.keras.models.load_model(extract_path)

# Predict and compute metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse}")

# Save evaluation metrics
metrics = {
    "regression_metrics": {
        "mse": mse
    }
}

os.makedirs(evaluation_output_path, exist_ok=True)
with open(os.path.join(evaluation_output_path, "evaluation.json"), "w") as f:
    json.dump(metrics, f)
