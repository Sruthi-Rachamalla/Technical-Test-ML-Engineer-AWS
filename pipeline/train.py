import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

train_path = os.environ['SM_CHANNEL_TRAIN']
test_path = os.environ['SM_CHANNEL_TEST']
model_dir = "/opt/ml/processing/model"

train_df = pd.read_csv(os.path.join(train_path, 'train.csv'))
test_df = pd.read_csv(os.path.join(test_path, 'test.csv'))

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

X_train, y_train = prepare_data(train_df.squeeze(), time_interval=25, return_3d=True)
X_test, y_test = prepare_data(test_df.squeeze(), time_interval=25, return_3d=True)

simple_lstm_model = keras.models.Sequential([
    keras.layers.LSTM(30, return_sequences=True, input_shape=[X_train.shape[1], X_train.shape[2]]),
    keras.layers.LSTM(30),
    keras.layers.Dense(1)
])

lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10,
    decay_rate=0.96,
    staircase=True
)

simple_lstm_model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler)
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    restore_best_weights=True
)

history = simple_lstm_model.fit(
    X_train, y_train,
    epochs=10000,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Save model in SageMaker format
model_save_path = os.path.join(model_dir, "1")
simple_lstm_model.save(model_save_path)
