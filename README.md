# Forex Volatility Prediction Pipeline (AWS CDK)

This repository deploys an AWS SageMaker Pipeline for predicting currency pair volatility using a TensorFlow LSTM model. The pipeline is fully defined via AWS CDK in Python and includes preprocessing, training, evaluation, and conditional model registration.

---

## 📂 Project Structure
```
├── cdk
	├── app.py
	├── cdk.json
	├── forex_pipeline_stack.py
	├── pipeline-definition.json
	├── requirements.txt
├── pipeline
	├── preprocessing-forex.py
	├── train-lstm.py
	├── evaluate-forex.py
	├── ml_pipeline.py
├── artifacts
	├── model.tar.gz
	├── aws_config.txt
	├── pipeline_execution.png
```
### Files Explained:
```
|------------------------------------------------------------------------------------------------------------------------------|
| File                        | Description                                                                                    |
|-----------------------------|------------------------------------------------------------------------------------------------|
| `app.py`                    | CDK app entrypoint. Instantiates the CDK Stack.                                                |
| `cdk.json`                  | Defines CDK command to run the app.                                                            |
| `forex_pipeline_stack.py`   | CDK stack defining the SageMaker pipeline and its resources.                                   |
| `pipeline-definition.json`  | JSON definition of the SageMaker pipeline (steps, parameters, and workflow logic).             |
| `requirements.txt`  	      | Required packages for the cdk library.							       |
| `preprocessing-forex.py`    | Script executed by SageMaker Processing step to generate volatility datasets.                  |
| `train-lstm.py`             | LSTM model training script run during the SageMaker Training step.                             |
| `evaluate-forex.py`         | Model evaluation script run during SageMaker Processing for computing MSE metrics.             |
| `ml_pipeline.py`            | Main pipeline file  preprocessing, evaluation, condition checking and registering the model.   |
| `model.tar.gz`              | Saved LSTM model file.								               |
| `aws_config.txt`            | AWS configuration required related to S#, SageMaker, Roles, Policies and VPC	               |
| `pipeline_execution.png`    | Screenshot of successful creation of pipeline in AWS CloudFormation using cdk.                 |
|------------------------------------------------------------------------------------------------------------------------------|
```
---

## ⚙️ Pipeline Overview

The pipeline includes:

✅ **PreprocessForexData**  
- Loads Forex CSV data  
- Computes volatility estimates via the Garman-Klass estimator
- Prepare time interval data.
- Splits the data into train/test sets

✅ **TrainForexModel**  
- Trains an LSTM model in TensorFlow using SageMaker Training  
- Outputs the trained model artifact to S3

✅ **EvaluateForexModel**  
- Loads trained model and test data  
- Computes MSE  
- Stores evaluation metrics to `evaluation.json`

✅ **CheckForexMSE**  
- Checks if MSE ≤ a defined threshold  
- Registers the model in SageMaker Model Registry if successful

---

## 🚀 Deployment Instructions

> **Prerequisites:**
> - Python 3.8+  
> - AWS CDK v2  
> - AWS credentials configured locally (e.g. via `aws configure`)  
> - Appropriate permissions to create SageMaker resources, IAM roles, and S3 buckets

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Deploying the pipeline

```bash
cdk bootstrap
cdk deploy
```

### ✅ Outputs
ForexPipelineStack.PipelineName = CurrencyPairVolatilityPredictionPipeline-1
