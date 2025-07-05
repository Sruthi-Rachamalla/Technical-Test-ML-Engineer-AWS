# Technical-Test-ML-Engineer-AWS

# SageMaker MLOps Project: Forex Price Prediction Pipeline

This project demonstrates the full ML model lifecycle using **Amazon SageMaker Pipelines**, including:
- Data ingestion from S3
- Data preprocessing using a Processing Job
- Model training using LSTM
- Model registration in SageMaker Model Registry
- Model deployment for real-time inference

---

## 📊 Dataset

- **Source**: A custom historical forex dataset (`cop=x_forex-data.csv`)
- **Goal**: Predict the volatility using garmin klass volatility measure of currency pair usd/cop based on historical data.
- **Preprocessing**:
  - Normalize/clean numeric columns
  - Prepare time series data based on time interval (25 steps)
  - Prepare train and test splits for evaluating
