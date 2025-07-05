
import os
import pandas as pd
import xgboost as xgb
import joblib

# SageMaker environment variables
input_path = os.environ['SM_CHANNEL_TRAIN']
model_dir = os.environ['SM_MODEL_DIR']

# Load processed CSV
data = pd.read_csv(os.path.join(input_path, 'processed.csv'))


def prepare_data(dataset, time_interval=25, return_3d=True):
    dataX, dataY = [], []
    for i in range(0, len(dataset)-time_interval-1) :
        dataX.append(dataset[i:i+time_interval])
        dataY.append(dataset[i+time_interval])
    dataX=np.array(dataX)
    dataY=np.array(dataY)
    if (return_3d == True):
      return dataX.reshape(dataX.shape[0], dataX.shape[1],1), dataY.reshape(dataY.shape[0], 1,1)
    else:
      return dataX, dataY


# Split features and target
x, y = prepare_data(data, time_interval=25, return_3d=False)

# Define and train the model
model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(x, y)

# Save model to the model directory
joblib.dump(model, os.path.join(model_dir, 'model.joblib'))
