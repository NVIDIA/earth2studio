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

Earth2Studio has several configurations for testing listed below.
Many unit tests, particularly ones for data sources, have [timeout limits](https://github.com/pytest-dev/pytest-timeout)
and are allowed to fail using the [xfail](https://docs.pytest.org/en/6.2.x/skipping.html#xfail-mark-test-functions-as-expected-to-fail)
decorator.
It is normal for some tests to fail, particularly for machines with slower internet
connections.
Users should always look at the test summary to under to understand which tests produced
an error but did not fail.

### Quick Unit Test Suite

This will run the quickest test suite which skips many tests that are longer running.

```bash
pytest  test/
```

### Standard Unit Test Suite

To run the full unit test suite, add the `--slow` flag.
Compared to the quick test suite, this one will run much longer

```bash
pytest --slow test/
```

### Complete Unit Test Suite

The complete unit tests ran by the CI pipeline is ran with the `--ci-cache`.
This option enables tests that load and test full model archiectures with their model
package which can be quite expensive.
These tests will look for model files inside a specific folder unique to the CI pipeline
on Nvidia systems or whatever is set in the `EARTH2STUDIO_CACHE` environment variable.
See [configuration](#configuration_userguide) section for details on this environment
variable.

```bash
 pytest --slow --ci-cache test/
```

:::{warning}
The full model cache is rather large (order of 10Gb) which will be downloaded running
this command.
The CI pipeline will already have all model files downloaded on the runner machine to
skip this step.
Additional information on this can be found below.
:::

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
Running coverage with pytest just requires the following:

```bash
# Run tests
coverage pytest --slow test/

# Compile reports
coverage combine
```

The CI pipeline uses the commands defined in the [Makefile](https://github.com/NVIDIA/earth2studio/blob/main/Makefile):

```bash
make pytest

make coverage
```

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
