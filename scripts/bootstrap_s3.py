from __future__ import annotations

import os

import boto3
from botocore.exceptions import ClientError


def main() -> None:
    buckets = [
        name.strip()
        for name in os.environ["BUCKET_NAMES"].split(",")
        if name.strip()
    ]
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.getenv("REGION", "us-east-1"),
    )
    for bucket in buckets:
        try:
            client.head_bucket(Bucket=bucket)
        except ClientError as error:
            status = error.response.get("ResponseMetadata", {}).get(
                "HTTPStatusCode"
            )
            if status != 404:
                raise
            client.create_bucket(Bucket=bucket)


if __name__ == "__main__":
    main()
