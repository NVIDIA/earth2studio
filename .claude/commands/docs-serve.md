Build the HTML documentation and then serve it locally so it can be viewed in a browser.

Ask the user which build mode they want if not specified:

**Step 1 — build (choose one):**

Fast build (skips example gallery):

```bash
make docs
```

Full build (includes all examples, slow):

```bash
make docs-full
```

Targeted single-example build (ask the user for the filename stem if not provided):

```bash
make docs-dev FILENAME=<example_stem>
```

**Step 2 — serve:**

```bash
uv run python -m http.server 8000 --directory docs/_build/html
```

The docs will be available at <http://localhost:8000>. Inform the user of the URL and that they
can stop the server with Ctrl-C.
