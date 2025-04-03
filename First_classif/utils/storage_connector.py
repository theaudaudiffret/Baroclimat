import io
import os

import pandas as pd
from azure.storage.blob import BlobServiceClient


class StorageConnector:
    """Class to connect to Azure Blob Storage and perform operations on it."""

    def __init__(self, connection_string: str, container_name: str):
        """
        Args:
            connection_string (str): Connection string to the Azure Blob Storage.
            container_name (str): Container name in the Azure Blob Storage.
        """
        if not connection_string:
            raise ValueError(
                """Connection string to the Azure Blob Storage is required,
                make sure to provide as env var AZURE_STORAGE_CONNECTION_STRING."""
            )
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = container_name

    @property
    def container_client(self):
        return self.blob_service_client.get_container_client(self.container_name)

    def list_containers(self):
        """List all the containers in the Azure Blob Storage."""
        return [container.name for container in self.blob_service_client.list_containers()]

    def upload_file(self, local_file_path: str, blob_name: str):
        """Upload a file to the Azure Blob Storage."""
        blob_client = self.container_client.get_blob_client(blob_name)
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data)

    def upload_directory(self, local_dir_path: str, blob_dir_name: str):
        """Upload a directory to the Azure Blob Storage."""
        for root, _, files in os.walk(local_dir_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                blob_name = os.path.join(blob_dir_name, local_file_path[len(local_dir_path) + 1 :])
                self.upload_file(local_file_path, blob_name)

    def delete_directory(self, blob_dir_name: str):
        """Delete a directory from the Azure Blob Storage."""
        blob_list = self.list_blobs(prefix=blob_dir_name)
        for blob in blob_list:
            self.remove_file(blob)

    def copy_blob(self, source_blob_name: str, destination_container_name: str, destination_blob_name: str):
        """Copy a blob from one container to another."""
        source_blob = self.container_client.get_blob_client(source_blob_name)
        destination_blob = self.blob_service_client.get_container_client(destination_container_name).get_blob_client(
            destination_blob_name
        )
        destination_blob.start_copy_from_url(source_blob.url)

    def download_file(self, blob_name: str, local_file_path: str):
        """Download a file from the Azure Blob Storage."""
        blob_client = self.container_client.get_blob_client(blob_name)
        with open(local_file_path, "wb") as data:
            data.write(blob_client.download_blob().readall())

    def remove_file(self, blob_name: str):
        """Remove a file from the Azure Blob Storage."""
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.delete_blob()

    def list_blobs(self, prefix: str = None):
        """List all the blobs in the container."""
        blob_list = self.container_client.list_blobs(name_starts_with=prefix)
        return [blob.name for blob in blob_list]

    def get_df_from_blob(self, blob_name: str) -> pd.DataFrame:
        """Get a pandas DataFrame from a CSV blob."""
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob()
        return pd.read_csv(io.BytesIO(blob_data.readall()))
