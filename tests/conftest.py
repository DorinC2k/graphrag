# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import logging
import sys
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run_slow", action="store_true", default=False, help="run slow tests"
    )

@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Ensure log messages appear in pytest output."""
    logging.basicConfig(
        level=logging.DEBUG,  # or INFO if you prefer
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )