Run the test suite for a specific tox environment.

## Choosing the right environment

Each tox environment installs a specific set of uv extras before running. When the user asks
to test a file or feature, check the top of the relevant source file for
`OptionalDependencyFailure("group-name")` calls — the group name tells you which extra (and
therefore which tox environment) is required.

| TOX_ENV | uv extras | Test paths |
|---|---|---|
| `test` | _(none)_ | `test/io`, `test/run`, `test/utils/`, `test/models/test_auto_models.py` |
| `test-data` | `data`, `cbottle` | `test/data`, `test/lexicon` |
| `test-perturb` | `perturbation` | `test/perturbation` |
| `test-stats` | `statistics`, `utils` | `test/statistics`, `test/utils/test_interp.py` |
| `test-px-models` | `all` | `test/models/px` (except ACE2) |
| `test-dx-models` | `all` | `test/models/dx` |
| `test-da-models` | `all` | `test/models/da` |
| `test-px-models-ace2` | `ace2` | `test/models/px/test_ace2.py` |
| `test-serve` | `serve` | `test/serve` |

> **Note**: The `test` environment also covers `test/models/test_batch.py` and
> `test/utils/test_imports.py`. Check `tox.ini` for the full authoritative list.

If the user has not specified a `TOX_ENV`, infer it from the file or feature being tested using
the table above.

## Running via tox (recommended)

```bash
make pytest TOX_ENV=<env>
```

tox handles the `uv sync --extra <extras>` automatically before running pytest.

## Running a single test directly (faster, no tox overhead)

First sync the required extras manually:

```bash
uv sync --extra <extras>   # e.g. uv sync --extra data,cbottle
```

Then run pytest directly:

```bash
uv run pytest test/path/to/test_file.py -v
uv run pytest test/path/to/test_file.py -k "test_name" -v
```

## Slow tests

Many tests are marked `@pytest.mark.slow` and skipped by default (network calls, large
downloads). To include them, pass `--slow`:

```bash
uv run pytest test/path/to/test_file.py -v --slow
```

Slow tests are often also marked `@pytest.mark.xfail` — an `XPASS` result (unexpected
pass) is fine and means the test succeeded against live data.

Report the test results, including any failures with their tracebacks.
