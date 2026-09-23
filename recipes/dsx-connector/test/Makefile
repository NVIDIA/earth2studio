.PHONY: test
# Torch-free; run from inside the earth2studio venv (see docs/stormcast-environment.md) so
# numpy/scipy/jsonschema/yaml are importable. `uv run` is intentionally NOT used —
# the StormCast GPU stack it would resolve is not pip/uv-installable.
test:
	cd ..; python3 -m pytest test/ -v
