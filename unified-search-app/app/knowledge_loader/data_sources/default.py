# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Data sources default module."""

import os

container_name = "law-cases"
blob_container_name = os.getenv("BLOB_CONTAINER_NAME", container_name)
blob_account_name = os.getenv("BLOB_ACCOUNT_NAME")
blob_connection_string = os.getenv("BLOB_STORAGE_CONNECTION_STRING")

local_data_root = os.getenv("DATA_ROOT")

LISTING_FILE = "listing.json"

if local_data_root is None and blob_account_name is None and blob_connection_string is None:
    error_message = (
        "DATA_ROOT, BLOB_ACCOUNT_NAME, or BLOB_STORAGE_CONNECTION_STRING environment variable"
        " must be set to locate datasets."
    )
    raise ValueError(error_message)
