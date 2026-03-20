"""
scripts/deploy_sagemaker_endpoint.py
Deploys the trained model to a real-time SageMaker endpoint.
After this runs, you have a live REST endpoint on AWS.
"""
import os
from dotenv import load_dotenv
load_dotenv()

import boto3
import sagemaker
from sagemaker.sklearn import SKLearnModel
from config.settings import get_settings

settings = get_settings()

ROLE          = os.environ.get("AWS_SAGEMAKER_ROLE")
BUCKET        = os.environ.get("AWS_S3_BUCKET", "sentinel-ml-artifacts-devv")
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "sentinel-pulse-scorer")
REGION        = settings.aws_region

# Read model URI saved by training script
with open(".sagemaker_model_uri") as f:
    model_uri = f.read().strip()

print(f"Deploying model: {model_uri}")
print(f"Endpoint name:   {ENDPOINT_NAME}")

session = sagemaker.Session(
    boto_session=boto3.Session(
        region_name=REGION,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
)

model = SKLearnModel(
    model_data=model_uri,
    role=ROLE,
    entry_point="inference.py",
    source_dir="sagemaker",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=session,
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",    # cheap for dev — 2 vCPU, 4GB RAM
    endpoint_name=ENDPOINT_NAME,
    wait=True,
)

print(f"\n✓ Endpoint live: {ENDPOINT_NAME}")
print(f"  Region: {REGION}")
print(f"  URL: https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{ENDPOINT_NAME}/invocations")
print("\nUpdate your .env:")
print(f"  SAGEMAKER_ENDPOINT_NAME={ENDPOINT_NAME}")