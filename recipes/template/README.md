# Earth2Studio Recipe Template

This is a template designed to be boiler plate.
It is expected all recipes follow the same template with best effort.
For more information on *what* a recipe is, refer to the [developer documentation](https://nvidia.github.io/earth2studio/userguide/developer/overview.html).

Recipe Description Here, keep it

## Prerequisites

List anything specific a user is expected to have installed / accessible here.
This could be things such as domain knowledge, data of certainer type, model checkpoints
etc.

### Software

Pyproject TOML / reqirements.txt handles any Python dependencies.

```bash
uv export --format requirements-txt --no-hashes > requirements.txt
```

If there are system dependencies needed list them here, e.g.

- Docker - minimum version: 23.0.1
- NVIDIA Container Toolkit minimum version: 1.13.5
- PyTorch Image 25.03

### Hardware

If applicable, e.g.

- GPUs: L40, L40S, RTX6000
- CPU: 16 cores
- RAM: ≥32GB
- Storage: 16Gb NVMe SSD

## Quick Start

Add a quick start to get the user running with the user up and running as fast as
possible.

## Documentation

Any additional documentation needed to explain to the user different options, APIs, etc.

## References

- Relevant papers
- Blog posts
- Repos
