# Development Guide

# Requirements

| Name                | Installation                                                 | Purpose                                                                             |
| ------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| Python 3.10-3.12    | [Download](https://www.python.org/downloads/)                | The library is Python-based.                                                        |
| Poetry              | [Instructions](https://python-poetry.org/docs/#installation) | Poetry is used for package management and virtualenv management in Python codebases |

# Getting Started

## Install Dependencies

```sh
# Install Python dependencies.
poetry install
```

## Execute the Indexing Engine

```sh
poetry run poe index <...args>
```

## Executing Queries

```sh
poetry run poe query <...args>
```

# Azurite

Some unit and smoke tests use Azurite to emulate Azure resources. This can be started by running:

```sh
./scripts/start-azurite.sh
```

or by simply running `azurite` in the terminal if already installed globally. See the [Azurite documentation](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azurite) for more information about how to install and use Azurite.

# Cosmos DB Emulator

Integration tests that use the `CosmosDBVectorStore` require the Azure Cosmos DB Emulator. Start it with:

```sh
./scripts/start-cosmos-emulator.sh
```

This launches the emulator with ports `8081` and `10250-10255` exposed and with data persistence enabled.

# Lifecycle Scripts

Our Python package utilizes Poetry to manage dependencies and [poethepoet](https://pypi.org/project/poethepoet/) to manage build scripts.

Available scripts are:

- `poetry run poe index` - Run the Indexing CLI
- `poetry run poe query` - Run the Query CLI
- `poetry build` - This invokes `poetry build`, which will build a wheel file and other distributable artifacts.
- `poetry run poe test` - This will execute all tests.
- `poetry run poe test_unit` - This will execute unit tests.
- `poetry run poe test_integration` - This will execute integration tests.
- `poetry run poe test_smoke` - This will execute smoke tests.
- `poetry run poe test_verbs` - This will execute tests of the basic workflows.
- `poetry run poe check` - This will perform a suite of static checks across the package, including:
  - formatting
  - documentation formatting
  - linting
  - security patterns
  - type-checking
- `poetry run poe fix` - This will apply any available auto-fixes to the package. Usually this is just formatting fixes.
- `poetry run poe fix_unsafe` - This will apply any available auto-fixes to the package, including those that may be unsafe.
- `poetry run poe format` - Explicitly run the formatter across the package.

## Troubleshooting

### "RuntimeError: llvm-config failed executing, please point LLVM_CONFIG to the path for llvm-config" when running poetry install

Make sure llvm-9 and llvm-9-dev are installed:

`sudo apt-get install llvm-9 llvm-9-dev`

and then in your bashrc, add

`export LLVM_CONFIG=/usr/bin/llvm-config-9`

### "numba/\_pymodule.h:6:10: fatal error: Python.h: No such file or directory" when running poetry install

Make sure you have python3.10-dev installed or more generally `python<version>-dev`

`sudo apt-get install python3.10-dev`

### LLM call constantly exceeds TPM, RPM or time limits

`GRAPHRAG_LLM_THREAD_COUNT` and `GRAPHRAG_EMBEDDING_THREAD_COUNT` are both set to 50 by default. You can modify these values
to reduce concurrency. Please refer to the [Configuration Documents](config/overview.md)
