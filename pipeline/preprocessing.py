import os
import math
import numpy as np
import pandas as pd

# Load and clean data
file_path = "/opt/ml/processing/input/02_07_2024_COP.csv"
raw_data = pd.read_csv(file_path)

# Drop NaN and skip first row (possibly header/metadata)
cleaned_data = raw_data.dropna().iloc[1:]

# Convert relevant columns to numeric
numeric_columns = ['High', 'Low', 'Open', 'Close']
cleaned_data[numeric_columns] = cleaned_data[numeric_columns].apply(pd.to_numeric, errors='coerce')


# Function to create volatility dataset
def create_volatility_estimate_dataset(dataset, time_step=25):
    if dataset.empty:
        return None

    # Vectorized volatility feature engineering
    hl_ratio = dataset['High'] / dataset['Low']
    co_ratio = dataset['Close'] / dataset['Open']
    high_by_low = np.log(np.square(hl_ratio))
    close_by_open = np.log(np.square(co_ratio))

    # GK estimator components
    gk_constant = (2 * math.log(2) - 1)
    diff = 0.5 * high_by_low - gk_constant * close_by_open

    # Rolling window volatility estimates
    volatility_data = [
        math.sqrt(np.mean(diff[i:i + time_step]))
        for i in range(len(diff) - time_step - 1)
    ]
    return pd.Series(volatility_data)


# Function to split dataset
def split_train_test_data(dataset, train_size_percentage=65):
    if not dataset or len(dataset) == 0:
        return None
    split_index = int(len(dataset) * train_size_percentage / 100)
    return dataset[:split_index], dataset[split_index:]


# Execute transformations
volatility_series = create_volatility_estimate_dataset(cleaned_data, time_step=25)
train_data, test_data = split_train_test_data(volatility_series, train_size_percentage=80)

# Save to CSV for ML pipelines
os.makedirs("/opt/ml/processing/train", exist_ok=True)
os.makedirs("/opt/ml/processing/test", exist_ok=True)
train_data.to_csv("/opt/ml/processing/train/train.csv", index=False)
test_data.to_csv("/opt/ml/processing/test/test.csv", index=False)
