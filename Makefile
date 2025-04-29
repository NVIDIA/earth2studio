install:
	uv venv --python=3.12
	uv sync
	uv sync --extra all

.PHONY: setup-ci
setup-ci:
	uv venv --python=3.12
	uv sync
	uv run pre-commit install --install-hooks

.PHONY: format
format:
	uv run pre-commit run black -a --show-diff-on-failure

.PHONY: black
black:
	uv run pre-commit run black -a --show-diff-on-failure

.PHONY: interrogate
interrogate:
	uv run pre-commit run interrogate -a

.PHONY: lint
lint:
	uv run pre-commit run check-added-large-files -a
	uv run pre-commit run trailing-whitespace -a
	uv run pre-commit run end-of-file-fixer -a
	uv run pre-commit run debug-statements -a
	uv run pre-commit run markdownlint -a
	uv run pre-commit run name-tests-test -a
	uv run pre-commit run pyupgrade -a --show-diff-on-failure
	uv run pre-commit run ruff -a
	uv run pre-commit run mypy -a

.PHONY: license
license:
	uv run python test/_license/header_check.py

.PHONY: pytest
pytest:
	uv run coverage run -m pytest --ci-cache --slow test/

.PHONY: pytest-submodule
pytest-submodule:
	uv run coverage run --source=$(COVERAGE_SOURCE) -m pytest --ci-cache --slow $(PYTEST_SOURCE)

.PHONY: doctest
doctest:
	echo "Skipping doc test"

.PHONY: coverage
coverage:
	uv run coverage combine
	uv run coverage report --fail-under=90

.PHONY: docs
docs:
	uv sync --group docs
	uv run $(MAKE) -C docs clean
	uv run $(MAKE) -C docs html

.PHONY: docs-full
docs-full:
	uv sync --extra all --group docs
	$(MAKE) docs-build-examples

.PHONY: docs-build-examples
docs-build-examples:
	rm -rf docs/examples
	rm -rf docs/modules/generated
	rm -rf docs/modules/backreferences
	uv run $(MAKE) -C docs clean
	rm -rf examples/outputs
	uv run $(MAKE) -C docs html
	PLOT_GALLERY=True RUN_STALE_EXAMPLES=True uv run $(MAKE) -j 8 -C docs html


.PHONY: docs-dev
docs-dev:
	# rm -rf examples/outputs
	uv sync --extra all --group docs
	PLOT_GALLERY=True RUN_STALE_EXAMPLES=True FILENAME_PATTERN=$(FILENAME) uv run $(MAKE) -j 4 -C docs html
