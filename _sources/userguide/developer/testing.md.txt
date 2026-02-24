# Testing Guide

Testing is a core part of Earth2Studio development and we have strict requirements for
unit testing any new feature.
Presently we expect new functionality to have over 90% coverage within reason.
The core tool used for testing in Earth2Studio is [PyTest](https://docs.pytest.org/en/8.2.x/)
and all unit tests can be found inside the [test](https://github.com/NVIDIA/earth2studio/tree/main/test)
folder.
Ensure that developer dependencies are install as outline in the [developer overview](#developer_overview)
section.

## PyTest

Earth2Studio uses [tox](https://tox.wiki/) to manage test environments and dependencies.
Many unit tests, particularly ones for data sources, have [timeout limits](https://github.com/pytest-dev/pytest-timeout)
and are allowed to fail using the [xfail](https://docs.pytest.org/en/6.2.x/skipping.html#xfail-mark-test-functions-as-expected-to-fail)
decorator.
It is normal for some tests to fail, particularly for machines with slower internet
connections.
Users should always look at the test summary to understand which tests produced
an error but did not fail.

### Test Environments

Earth2Studio organizes tests into separate tox environments based on functionality:

- **`test`** - Core tests that need no extra dependencies (IO, utilities, auto-models, batch)
- **`test-data`** - Data source and lexicon tests (requires `data` and `cbottle` extras)
- **`test-perturb`** - Perturbation method tests (requires `perturbation` extra)
- **`test-stats`** - Statistical method tests (requires `statistics` and `utils` extras)
- **`test-px-models`** - Prognostic model tests (requires `all` extra)
- **`test-dx-models`** - Diagnostic model tests (requires `all` extra)
- **`test-serve`** - Serve/API tests (requires `serve` extra)

### Running Tests with Tox

The recommended way to run tests is using the Makefile with a specific `TOX_ENV`:

```bash
# Run core tests
make pytest TOX_ENV=test

# Run data source tests
make pytest TOX_ENV=test-data

# Run model tests
make pytest TOX_ENV=test-px-models
make pytest TOX_ENV=test-dx-models
```

You can also run tox directly:

```bash
# Run a specific test environment
uvx tox -c tox.ini run -e test-data

# Run with additional pytest arguments
uvx tox -c tox.ini run -e test-data -- -v
```

### Running Tests Directly with Pytest

For quick iteration during development, you can run pytest directly:

```bash
# Quick test suite (skips slow tests)
pytest test/

# Standard test suite (includes slow tests)
pytest --slow test/

# Run specific test module
pytest test/data/test_gfs.py
```

:::{note}
When running pytest directly, ensure you have the appropriate dependencies installed
for the tests you're running. The tox environments handle dependency management automatically.
:::

### CI Pipeline Structure

The CI pipeline runs tests in separate jobs for each test environment, which improves
debugging and allows parallel execution. Each job only runs when relevant files change:

- **`test`** - Always runs (core functionality)
- **`test-data`** - Runs when `earth2studio/data`, `earth2studio/lexicon`, or their tests change
- **`test-perturb`** - Runs when perturbation code or tests change
- **`test-stats`** - Runs when statistics code or tests change
- **`test-px-models`** - Runs when prognostic model code or tests change
- **`test-dx-models`** - Runs when diagnostic model code or tests change
- **`test-serve`** - Runs when serve/API code or tests change

To force all tests to run regardless of changes, set the `CI_PYTEST_ALL` environment variable
to `1`, `true`, `yes`, or `on`.

#### CI Model Package Cache

In the CI pipeline a NFS mounted folder is used as a pre-cached store of all
pre-trained model checkpoint files (automodels) so each pipeline does not need to rerun
the download.
The cache folder is managed internally by the developers.
This should reflect a local cache if a user was to fetch and load all possible models.
The point of truth for generating this cache of model checkpoints **should** always be
`test/models/test_auto_models.py`.
This test all listed auto-models and create a `./cache` folder with the following
command:

```bash
pytest --model-download -s test/models/test_auto_models.py

ls -l ./cache
```

## Coverage

Coverage is calculated using the [coverage.py](https://coverage.readthedocs.io/en/7.5.3/)
package configured in the [pyproject.toml](https://github.com/NVIDIA/earth2studio/blob/main/pyproject.toml).
Presently the CI pipeline will fail if unit test coverage drops below 90%.

The CI pipeline runs tests using tox environments which automatically collect coverage data.
After all test jobs complete, coverage is combined and reported:

```bash
# Run tests (coverage is collected automatically by tox)
make pytest TOX_ENV=test-data

# Combine coverage from all test runs
make coverage
```

The `make coverage` command combines coverage data from all test environments and generates
a report. The CI pipeline runs this automatically after all test jobs complete.

## Contributing Tests

We expect all contributors to also provide unit tests for their feature.
Tests are always evolving based on new errors discovered and features added.
Right now the best way to contribute unit tests for a new feature is to modify the tests
for a feature of a similar type.
For example, for a new data source, copy and modify the tests for say GFS or ARCO.
For additional information about the tests, reach out with an issue and the developers can
provide more information.

If you are contributing a model, be sure to make the developers aware of this to
properly integrate it into the CI pipeline.
