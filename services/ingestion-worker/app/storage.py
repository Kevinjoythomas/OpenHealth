"""S3 upload/download helpers using boto3."""
import logging
import os

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)


def _get_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("S3_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def download_bytes(s3_key: str) -> bytes:
    """Download an object from S3 and return its raw bytes."""
    bucket = os.getenv("S3_BUCKET", "openhealth-docs")
    log.info("Downloading s3://%s/%s", bucket, s3_key)
    client = _get_client()
    response = client.get_object(Bucket=bucket, Key=s3_key)
    return response["Body"].read()


def upload_bytes(content: bytes, s3_key: str, content_type: str = "application/pdf") -> str:
    """Upload bytes to S3 and return the full s3:// URI."""
    bucket = os.getenv("S3_BUCKET", "openhealth-docs")
    log.info("Uploading to s3://%s/%s", bucket, s3_key)
    client = _get_client()
    client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=content,
        ContentType=content_type,
    )
    return f"s3://{bucket}/{s3_key}"


def object_exists(s3_key: str) -> bool:
    bucket = os.getenv("S3_BUCKET", "openhealth-docs")
    try:
        _get_client().head_object(Bucket=bucket, Key=s3_key)
        return True
    except ClientError:
        return False
