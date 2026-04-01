# Build Process

This page describes how to set the package version and build a wheel for Earth2Studio
from a local clone. It is intended for contributors and maintainers.

**Prerequisites**: [uv](https://docs.astral.sh/uv/) installed, a clone of the repository,
and a synced development environment (for example, `uv sync` as described in the
{ref}`developer_overview`). Run all commands below from the repository root (the directory
that contains `pyproject.toml`).

## Setting the Version

To check the version of the current package:

```bash
uv run hatch version
```

To bump the version automatically, use the following [hatch commands](https://hatch.pypa.io/1.9/version/):

```bash
# To bump to next pre-release
uv run hatch version minor,a

# To bump to release candidate version
uv run hatch version rc

# To bump to final release version
uv run hatch version release
```

## Building a Wheel

To build the wheel, use the [uv build](https://docs.astral.sh/uv/guides/package/)
command:

```bash
uv build
```
