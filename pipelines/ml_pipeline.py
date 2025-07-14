import os
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import ParameterString
from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.parameters import ParameterFloat
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.tensorflow import TensorFlowProcessor
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.functions import Join
from sagemaker import image_uris
from sagemaker.processing import ScriptProcessor, ProcessingOutput, ProcessingInput
from sagemaker.workflow.steps import ProcessingStep


#Initializing Role, Region, PipelineSession, Bucket, Path variables
role = "arn:aws:iam::637423302756:role/datazone_usr_role_5gs1xk78tlvew9_diae53l2gtnvhl"
region = "us-east-2"
pipeline_session = PipelineSession()
bucket = "currencypair-ml-data"
input_data = ParameterString(name="InputDataUrl", default_value=f"s3://{bucket}/02_07_2025_COP=X.csv")
mse_threshold_param = ParameterFloat(name="MSEThreshold", default_value=0.05)
processing_output_path = f"s3://{bucket}/processed/"
model_output_path = f"s3://{bucket}/model/"


# Processing Stage
sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name="forex-preprocessing",
    sagemaker_session=pipeline_session
)

processing_step = ProcessingStep(
    name="PreprocessForexData",
    processor=sklearn_processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/train"
        ),
        sagemaker.processing.ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/test"
        )
    ],
    code="preprocessing-forex.py"
)


# Model Training Stage
tf_estimator = TensorFlow(
    entry_point="train-lstm.py",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    framework_version="2.12",
    py_version="py310",
    base_job_name="forex-lstm",
    output_path=model_output_path,
    sagemaker_session=pipeline_session
)

training_step = TrainingStep(
    name="TrainForexModel",
    estimator=tf_estimator,
    inputs={
        "train": processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        "test": processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
    }
)

#Model Evaluation Stage

evaluation_report = PropertyFile(
    name="evaluation",
    output_name="evaluation",
    path="evaluation.json"
)

image_uri = image_uris.retrieve(
    framework='tensorflow',
    region='us-east-2',
    version='2.12',
    instance_type='ml.m5.xlarge',
    image_scope='training'
)

processor = ScriptProcessor(
    image_uri=image_uri,                  
    command=["python3"],
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name="forex-eval"
)

evaluation_step = ProcessingStep(
    name="EvaluateForexModel",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation"
        )
    ],
    code="evaluate-forex.py",
    property_files=[evaluation_report]
)


# Threshold value (adjust as needed)

evaluation_json_uri = Join(
    on="/",
    values=[
        evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
        "evaluation.json"
    ]
)

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=evaluation_json_uri,
        content_type="application/json"
    )
)


# create model object
model = Model(
    image_uri=tf_estimator.training_image_uri(),
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=pipeline_session,
    role=role,
)

# prepare registration arguments
model_registration_args = model.register(
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name="ForexPredictionModelPackageGroup",
    approval_status="Approved",
    model_metrics=model_metrics,
)

register_step = ModelStep(
    name="RegisterForexModel",
    step_args=model_registration_args
)


#Check for condition
condition_step = ConditionStep(
    name="CheckForexMSE",
    conditions=[
        ConditionLessThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file="evaluation",
                json_path="regression_metrics.mse.value"
            ),
            right=mse_threshold_param
        )
    ],
    if_steps=[register_step],
    else_steps=[]
)


#Start the pipeline
pipeline = Pipeline(
    name="ForexPredictionPipeline-LSTM",
    parameters=[input_data, mse_threshold_param],
    steps=[processing_step, training_step, evaluation_step, condition_step],
    sagemaker_session=pipeline_session
)

pipeline.upsert(role_arn=role)

execution = pipeline.start()


response = client.list_model_packages(
    ModelPackageGroupName="ForexPredictionModelPackageGroup",
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1
)

model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]

sm_client = boto3.client("sagemaker", region_name="us-east-2")

pipeline_name = "ForexPredictionPipeline-LSTM"

# Retrieve the pipeline definition
response = sm_client.describe_pipeline(
    PipelineName=pipeline_name
)

pipeline_definition_json = response["PipelineDefinition"]

# Parse JSON
pipeline_dict = json.loads(pipeline_definition_json)

# Save full pipeline definition
with open("pipeline-definition.json", "w") as f:
    json.dump(pipeline_dict, f, indent=2)


#Deploying the model
model = ModelPackage(
    role=role,
    model_package_arn=model_package_arn,
    sagemaker_session=pipeline_session
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="forex-lstm-endpoint"
)

#Sample Inference Prediction: Predicting the volatility of USD/COP for next 25 days
def predict_n_values(model = predictor, dataset, n_predict=25, time_steps=25):
    predictX, predictY =[],[]
    dataset_copy = np.array(dataset)
    start_index=dataset_copy.shape[0]-time_steps
    for i in range(1,n_predict+1):
        print(start_index)
        predictX = dataset_copy[start_index:start_index+time_steps,]
        print("{} day input {}".format(i, predictX))
        next_predicted_value=model.predict(predictX.reshape(1, predictX.shape[0],1))
        print("day output : day_output_size {},  {}".format(next_predicted_value.shape, next_predicted_value))
        predictY.append(next_predicted_value)
        dataset_copy= np.append(dataset_copy,next_predicted_value)
        start_index= start_index+1
    return np.array(predictY)

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

file_path = "02_07_2025_COP=X.csv"
raw_data = pd.read_csv(file_path)
cleaned_data = raw_data.dropna().iloc[1:]
numeric_columns = ['High', 'Low', 'Open', 'Close']
cleaned_data[numeric_columns] = cleaned_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
volatility_series = create_volatility_estimate_dataset(cleaned_data, time_step=25)

predict_next_25_days = predict_n_values(dataset=volatility_series)
