# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import socket

import pytest


def _check_connection(host: str, port: int) -> bool:
    """Return True if connection succeeds."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def require_blob_emulator() -> None:
    """Skip tests if the blob emulator is unreachable."""
    if not _check_connection("127.0.0.1", 10000):
        pytest.skip(
            "Blob Storage emulator not running on 127.0.0.1:10000",
            allow_module_level=True,
        )


def require_cosmos_emulator() -> None:
    """Skip tests if the Cosmos DB emulator is unreachable."""
    if not _check_connection("127.0.0.1", 8081):
        pytest.skip(
            "CosmosDB emulator not running on 127.0.0.1:8081",
            allow_module_level=True,
        )
