# Testing Guide

Testing is a core part of Earth2Studio development and we have strict requirements for
unit testing any new feature.
Presently we expect new functionality to have over 90% coverage within reason.
The core tool used for testing in Earth2Studio is [pytest](https://docs.pytest.org/en/8.2.x/)
and all unit tests can be found inside the [test](https://github.com/NVIDIA/earth2studio/tree/main/test)
folder.
Ensure that developer dependencies are installed as outlined in [Developer Overview](overview.md#developer_overview).
Run `make` targets below from the repository root (the directory that contains `Makefile`
and `pyproject.toml`).

## pytest

Earth2Studio uses [tox](https://tox.wiki/) to manage test environments and dependencies.
Many unit tests, particularly ones for data sources, have [timeout limits](https://github.com/pytest-dev/pytest-timeout)
and are allowed to fail using the [xfail](https://docs.pytest.org/en/6.2.x/skipping.html#xfail-mark-test-functions-as-expected-to-fail)
decorator.
It is normal for some tests to time out or be skipped, particularly on machines with slower
internet connections.
Always read the test summary to see which tests **failed**, which **errored**, and which
were **xfail** (expected failure).

### Test Environments

Earth2Studio organizes tests into separate tox environments based on functionality:

- **`test`** - Core tests that need no extra dependencies (IO, utilities, auto-models, batch)
- **`test-data`** - Data source and lexicon tests (requires `data` and `cbottle` extras)
- **`test-perturb`** - Perturbation method tests (requires `perturbation` extra)
- **`test-stats`** - Statistical method tests (requires `statistics` and `utils` extras)
- **`test-px-models`** - Prognostic model tests (requires `all` extra)
- **`test-dx-models`** - Diagnostic model tests (requires `all` extra)
- **`test-da-models`** - Data assimilation model tests (requires `all` extra)

### Automatic Test Skipping

Test files that require optional dependencies are automatically skipped during pytest
collection when those dependencies are not installed. This is handled by a
`pytest_ignore_collect` hook in `test/conftest.py` that checks dependency availability
before Python imports the test module.

The hook uses a `_TEST_DEPENDENCIES` mapping that associates test file paths with their
required dependency groups (from `pyproject.toml`) or individual packages. For example:

```python
_TEST_DEPENDENCIES = {
    "test/models/px/test_aifs.py": ["aifs"],
    "test/statistics/test_crps.py": ["statistics"],
    "test/data/test_nnja.py": ["earth2bufr"],
    # ...
}
```

This approach has several benefits:

1. **No import errors during collection** - Files are skipped before Python tries to
   import unavailable packages
2. **Version-aware checking** - Version constraints from `pyproject.toml` are respected
   (e.g., `anemoi-models==0.5.1` for AIFS vs `anemoi-models>=0.9.3,<0.9.5` for AIFS2)
3. **Centralized configuration** - All dependency mappings are in one place (`conftest.py`)
4. **Clean test output** - Skipped files don't appear as errors or warnings

When adding new tests that require optional dependencies, add an entry to the
`_TEST_DEPENDENCIES` dict in `test/conftest.py`:

```python
_TEST_DEPENDENCIES: dict[str, list[str]] = {
    # ...existing entries...
    "test/models/px/test_newmodel.py": ["newmodel-group"],
}
```

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
make pytest TOX_ENV=test-da-models
```

You can also run tox directly:

```bash
# Run a specific test environment
uvx tox -c tox.ini run -e test-data

# Run with additional pytest arguments
uvx tox -c tox.ini run -e test-data -- -v
```

### Test results

Tox writes JUnit reports to `.cache/test-results/<tox-env>/`. CI includes full console
logs and uploads `test-results-<tox-env>` artifacts even on failure, retained for **3 days**.

### Bounded data-source tests

`test-data` runs supervisor checks, lexicon and Icechunk tests, then isolated data-test processes
with **8 workers**, **300s per file**, and **1,200s total**, plus termination grace.
Override these with `DATA_TEST_WORKERS`, `DATA_TEST_FILE_TIMEOUT`, and `DATA_TEST_TIMEOUT`.
File budgets include startup, shutdown, and restarts; exceeding a budget fails the run.
Logs and partial results: `.cache/test-results/test-data/data/`.
Once pytest finalization finishes, lingering processes are terminated after the shutdown
grace period and the recorded test outcome is preserved. Forced cleanup is logged explicitly.

Mark network tests with `@pytest.mark.live(timeout=60)` (default: 60s). This also
marks them `slow`, so they require `--slow`. Live selection provides an additional filter:

- `--live=none`: skip live tests for offline checks.
- `--live=changed`: run affected sources (CI default); shared changes enable all sources.
  CI compares against the PR merge base or the pushed branch's previous tip.
  Manual dispatch defaults to `all`, with an explicit live-mode selector.
  Locally, `DATA_TEST_BASE` defaults to `HEAD`, including uncommitted changes.
- `--live=all`: select all live tests (default; still requires `--slow`).

Under supervision, live-call timeouts become XFAIL; the process is killed and remaining
tests resume in a fresh process. Other errors fail normally. File summaries show
`PASSED (XFAIL)` when expected failures occurred. Selected live files bypass testmon.
Use `--live-timeout` to override marker deadlines.

```bash
# Run changed sources.
DATA_TEST_BASE=upstream/1.0.0-rc uvx tox -e test-data -- --slow --live=changed --testmon

# Run one source with longer budgets.
DATA_TEST_PATHS=test/data/test_gfs.py \
DATA_TEST_FILE_TIMEOUT=3600 DATA_TEST_TIMEOUT=7200 \
    uvx tox -e test-data -- --slow --live=all --live-timeout=600 --package
```

Arguments after `--` apply only to data-source tests and replace their Tox defaults;
include `--testmon` and `--cov` as needed. Lexicon and Icechunk tests run with fixed defaults.

### Running Tests Directly with pytest

For local development, plain pytest skips slow and live tests. Add `--slow` to
enable both, or combine it with `--live=none` to run only non-live slow tests:

```bash
# Run the test suite (skips tests marked slow)
uv run pytest test/

# Standard test suite (includes slow tests)
uv run pytest --slow test/

# Include slow tests but skip live requests
uv run pytest --slow --live=none test/

# Run a specific test module, including live requests
uv run pytest --slow test/data/test_gfs.py

# Filter tests within a module
uv run pytest --slow test/data/test_gfs.py -k cache
```

Direct pytest uses `pytest-timeout` for live deadlines (including `--live-timeout`
overrides). These timeouts are failures, without process cleanup or restart;
the supervised Tox workflow records live-call timeouts as XFAIL and restarts.

!!! note
    When running pytest directly, ensure you have the appropriate dependencies installed
    for the tests you're running. The tox environments handle dependency management automatically.

### CI Pipeline Structure

The CI pipeline runs tests in separate jobs for each test environment, which improves
debugging and allows parallel execution. Each job only runs when relevant files change:

- **`test`** - Always runs (core functionality)
- **`test-data`** - Runs when `earth2studio/data`, `earth2studio/lexicon`, or their tests change
- **`test-perturb`** - Runs when perturbation code or tests change
- **`test-stats`** - Runs when statistics code or tests change
- **`test-px-models`** - Runs when prognostic model code or tests change
- **`test-dx-models`** - Runs when diagnostic model code or tests change
- **`test-da-models`** - Runs when data assimilation model code or tests change

To force all tests to run regardless of changes, set the `CI_PYTEST_ALL` environment variable
to `1`, `true`, `yes`, or `on`. In CI, also set `DATA_TEST_LIVE=all` to bypass
change-based live selection. Supervised runs retain finite process budgets.

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
For example, for a new data source, copy and modify the tests for say GFS or ARCO_ERA5.
For additional information about the tests, open an issue and the developers can
provide more information.

If you are contributing a model, be sure to make the developers aware of this to
properly integrate it into the CI pipeline.
