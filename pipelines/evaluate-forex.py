#!/usr/bin/env python3
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
test_path = "/opt/ml/processing/test"

# Untar the model
os.makedirs(extract_path, exist_ok=True)
with tarfile.open(tar_path) as tar:
    tar.extractall(path=extract_path)

# Load test data
test_df = pd.read_csv(os.path.join(test_path, "test.csv"))

def prepare_data(dataset, time_interval=25, return_3d=True):
    dataX, dataY = [], []
    dataset = np.array(dataset)
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


try:
    model = tf.keras.models.load_model(extract_path)
    print("Model loaded successfully using keras.models.load_model().")
    y_pred = model(X_test)

    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()

except Exception as e:
    print(f"Error loading model using keras.models.load_model: {e}")

    print("Attempting tf.saved_model.load fallback...")
    loaded_model = tf.saved_model.load(extract_path)
    infer = loaded_model.signatures["serving_default"]
    output_dict = infer(tf.constant(X_test))
    y_pred = list(output_dict.values())[0].numpy()

# Flatten if needed
if y_pred.ndim > 1 and y_pred.shape[1] == 1:
    y_pred = y_pred.flatten()

mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse}")

# Save metrics
metrics = {
    "regression_metrics": {
        "mse": {
            "value": float(mse),
        }
    }
}

os.makedirs(evaluation_output_path, exist_ok=True)
evaluation_file_path = os.path.join(evaluation_output_path, "evaluation.json")
with open(evaluation_file_path, "w") as f:
    json.dump(metrics, f)

print(f"Evaluation metrics saved to {evaluation_file_path}")