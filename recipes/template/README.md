# Earth2Studio Recipe Template

This is a template designed to be boiler plate.
All recipes should follow this template as closely as possible
For more information on *what* a recipe is, refer to the [developer documentation](https://nvidia.github.io/earth2studio/userguide/developer/overview.html).

Recipe description here, keep it around a single paragraph.

## Prerequisites

List anything specific a user is expected to have installed or accessible here.
This could be things such as domain knowledge, data of certain type, model checkpoints.

### Software

Pyproject TOML handles any Python dependencies.

If there are system dependencies needed list them here, for example:

- Docker - minimum version: 23.0.1
- NVIDIA Container Toolkit minimum version: 1.13.5
- PyTorch Image 25.03

### Hardware

If applicable, for example:

- GPUs: L40, L40S, RTX6000
- CPU: 16 cores
- RAM: ≥32GB
- Storage: 16Gb NVMe SSD

## Quick Start

Add a quick start to get the user running with the user up and running as fast as
possible, for example:

Start by installing the required packages with pip:

```bash
pip install -r requirements.txt
```

Or set up a uv virtual environment:

```bash
uv sync
```

Run the template

```bash
uv run python main.py

>> Hello
>> 0.12.1

uv run python main.py print.hello False

>> 0.12.1
```

## Documentation

Any additional documentation needed to explain to the user different options, APIs.
This can be as extensive or brief as desired.
It is expected for users to interact or modify the source code.
For example:

Possible options for this template include:

- print.hello : print "Hello"
- print.version : print the version of Earth2Studio package

For more options see [template.yml](cfg/template.yml).

## References

- [Recipe Guide](https://nvidia.github.io/earth2studio/userguide/developer/recipes.html)
- Relevant papers
- Blog posts
- Repos
