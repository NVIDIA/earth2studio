# Development Examples

Runnable tutorials for proposed APIs and workflows.

- Use the example-gallery script format with concise narrative sections.
- Keep examples focused, reproducible, and inexpensive to run.
- Do not commit generated notebooks, outputs, or artifacts.
- Update examples when their demonstrated API changes.

## Coordinate and model tutorials

- `01_grid_tutorial.py`: grid definitions, registration, inference, and selection.
- `02_model_contract_tutorial.py`: legacy tensor execution, hooks, and conformance.
- `03_coordinate_signatures.py`: new allocation-free DataArray signatures,
  grid-backed regional geometry, output planning, and precipitation statistics.

Run the signature tutorial from the repository root with
`.venv/bin/python dev/examples/03_coordinate_signatures.py`. It uses CPU-only
synthetic coordinates and requires no model weights or downloads.
