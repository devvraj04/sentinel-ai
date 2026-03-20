"""
scripts/upload_to_s3.py
Uploads training data and model artifacts to S3.
Run this before launching a SageMaker training job.
"""
import boto3
import os
from config.settings import get_settings

settings = get_settings()
BUCKET   = os.environ.get("AWS_S3_BUCKET", "sentinel-ml-artifacts-devv")

s3 = boto3.client(
    "s3",
    region_name=settings.aws_region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
)

def upload(local_path: str, s3_key: str):
    print(f"Uploading {local_path} → s3://{BUCKET}/{s3_key} ...", end="", flush=True)
    s3.upload_file(local_path, BUCKET, s3_key)
    print(" ✓")

if __name__ == "__main__":
    # Upload training dataset
    upload(
        "models/training_data/training_dataset.parquet",
        "training-data/training_dataset.parquet",
    )
    print(f"\nDone. Training data at:")
    print(f"  s3://{BUCKET}/training-data/training_dataset.parquet")