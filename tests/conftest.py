# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License


def pytest_addoption(parser):
    parser.addoption(
        "--run_slow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--run_queries_only",
        action="store_true",
        default=False,
        help=(
            "Only execute the query validations for smoke fixtures. "
            "Requires pre-built index outputs."
        ),
    )
