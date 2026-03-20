"""
scripts/launch_sagemaker_training.py
Launches a SageMaker training job, waits for it to complete,
then downloads the trained model back to your local machine.
"""
import os
from dotenv import load_dotenv
load_dotenv()

import boto3
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.inputs import TrainingInput
from config.settings import get_settings

settings = get_settings()

ROLE       = os.environ.get("AWS_SAGEMAKER_ROLE")
BUCKET     = os.environ.get("AWS_S3_BUCKET", "sentinel-ml-artifacts-devv")
REGION     = settings.aws_region

session = sagemaker.Session(
    boto_session=boto3.Session(
        region_name=REGION,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
)

print("Launching SageMaker training job...")
print(f"  Role  : {ROLE}")
print(f"  Bucket: s3://{BUCKET}")
print(f"  Region: {REGION}")

# Use SageMaker's built-in SKLearn container
# This container has sklearn, lightgbm, pandas, numpy, shap pre-installed
estimator = SKLearn(
    entry_point="train.py",
    source_dir="sagemaker",             # folder containing train.py + inference.py
    role=ROLE,
    instance_type="ml.m5.large",       # 4 vCPU, 16GB RAM — good for 1500 customers
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=session,
    output_path=f"s3://{BUCKET}/model-artifacts/",
    code_location=f"s3://{BUCKET}/code/",
    hyperparameters={
        "n_estimators":  600,
        "num_leaves":    63,
        "learning_rate": 0.05,
        "lgd":           0.45,
    },
    base_job_name="sentinel-lgbm",
)

# Point to training data in S3
training_input = TrainingInput(
    s3_data=f"s3://{BUCKET}/training-data/",
    content_type="application/x-parquet",
)

# Launch the job (this returns immediately — job runs on AWS)
estimator.fit({"training": training_input}, wait=True, logs=True)
# wait=True means this script waits until training finishes
# logs=True streams CloudWatch logs to your terminal

print("\nTraining complete!")
print(f"Model artifacts at: {estimator.model_data}")

# Save the S3 model path so deploy script can use it
with open(".sagemaker_model_uri", "w") as f:
    f.write(estimator.model_data)
print(f"Model URI saved to .sagemaker_model_uri")