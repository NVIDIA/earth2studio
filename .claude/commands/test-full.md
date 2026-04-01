Run the full test suite across all tox environments with coverage enabled.

This requires all optional extras to be available. tox handles syncing each environment's
extras automatically via `uv sync`.

```bash
make pytest-full
```

After the run completes, print a coverage summary:

```bash
make coverage
```

Report overall pass/fail counts and the final coverage percentage. Flag any modules below the
90% coverage threshold.
