Build the HTML documentation. Ask the user which build mode they want if not specified:

**Fast build** — skips the example gallery, quickest iteration:

```bash
make docs
```

**Full build** — includes the complete example gallery (slow, requires all extras):

```bash
make docs-full
```

**Targeted example build** — builds a single example page by filename stem, useful when
developing or debugging a specific example. Ask the user for the example filename if not
provided:

```bash
make docs-dev FILENAME=<example_stem>
```

For example, to build only `examples/run_inference.py`:

```bash
make docs-dev FILENAME=run_inference
```

All builds output to `docs/_build/html/`. Report any Sphinx warnings or errors encountered.
