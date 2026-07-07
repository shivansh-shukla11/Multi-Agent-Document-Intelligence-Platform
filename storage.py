"""
Storage abstraction: dev uses local disk, prod uses S3, same interface.
This is the pattern you want to point to for the "AWS (EC2, S3)" resume
line — the app code never talks to boto3 directly, it talks to this
interface, so swapping backends is a one-line config change.
"""
import os
from abc import ABC, abstractmethod

from config import settings


class StorageBackend(ABC):
    @abstractmethod
    def save(self, filename: str, content: bytes) -> str:
        """Persist content, return a storage path/key."""

    @abstractmethod
    def load(self, path: str) -> bytes:
        """Retrieve content given a storage path/key."""


class LocalStorage(StorageBackend):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, filename: str, content: bytes) -> str:
        path = os.path.join(self.base_dir, filename)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def load(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()


class S3Storage(StorageBackend):
    def __init__(self, bucket: str, region: str):
        import boto3  # imported lazily so local dev doesn't need boto3 configured

        self.bucket = bucket
        self.client = boto3.client("s3", region_name=region)

    def save(self, filename: str, content: bytes) -> str:
        key = f"documents/{filename}"
        self.client.put_object(Bucket=self.bucket, Key=key, Body=content)
        return key

    def load(self, path: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=path)
        return obj["Body"].read()


def get_storage_backend() -> StorageBackend:
    if settings.STORAGE_BACKEND == "s3":
        if not settings.S3_BUCKET:
            raise ValueError("STORAGE_BACKEND=s3 requires S3_BUCKET to be set")
        return S3Storage(settings.S3_BUCKET, settings.AWS_REGION)
    return LocalStorage(settings.LOCAL_STORAGE_DIR)
