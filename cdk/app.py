#!/usr/bin/env python3

import aws_cdk as cdk
from forex_pipeline_stack import ForexPipelineStack

app = cdk.App()
ForexPipelineStack(app, "ForexPipelineStack")

app.synth()