from aws_cdk import (
    Stack,
    aws_sagemaker as sagemaker,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3_assets as s3_assets,
    CfnTag,
    RemovalPolicy,
    CfnOutput,
)
from constructs import Construct
import json

class ForexPipelineStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # Pipeline role
        pipeline_role = iam.Role(
            self,
            "PipelineRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
            ]
        )

        # Load the original JSON definition
        with open("pipeline-definition.json", "r") as f:
            pipeline_definition_body = json.load(f)

        # Convert the loaded dictionary to a JSON string and wrap it in the expected format
        formatted_pipeline_definition = {
            "PipelineDefinitionBody": json.dumps(pipeline_definition_body)
        }

        # Now deploy
        pipeline = sagemaker.CfnPipeline(
            self,
            "ForexPipeline",
            pipeline_name="CurrencyPairVolatilityPredictionPipeline-1",
            role_arn=pipeline_role.role_arn,
            pipeline_definition=formatted_pipeline_definition,
            tags=[
                CfnTag(
                    key="project",
                    value="ForexPrediction"
                )
            ]
        )

        CfnOutput(
            self,
            "PipelineName",
            value=pipeline.pipeline_name
        )
