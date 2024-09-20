install:
	pip install --upgrade pip
	pip install "makani[all] @ git+https://github.com/NVIDIA/modulus-makani.git@v0.1.0"
	pip install -e .[all]

.PHONY: setup-ci
setup-ci:
	pip install .[dev]
	pre-commit install

.PHONY: format
format:
	pre-commit run black -a --show-diff-on-failure

.PHONY: black
black:
	pre-commit run black -a --show-diff-on-failure

.PHONY: interrogate
interrogate:
	pre-commit run interrogate -a

.PHONY: lint
lint:
	pre-commit run check-added-large-files -a
	pre-commit run trailing-whitespace -a
	pre-commit run end-of-file-fixer -a
	pre-commit run debug-statements -a
	pre-commit run name-tests-test -a
	pre-commit run pyupgrade -a --show-diff-on-failure
	pre-commit run ruff -a
	pre-commit run mypy -a

.PHONY: license
license:
	python test/_license/header_check.py

.PHONY: pytest
pytest:
	coverage run -m pytest --slow --ci-cache test/

.PHONY: doctest
doctest:
	echo "Skipping doc test"

.PHONY: coverage
coverage:
	coverage combine
	coverage report --fail-under=90

.PHONY: docs
docs:
	pip install .[docs]
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

EXAMPLE_FILENAMES = \
	01_deterministic_workflow.py \
	02_diagnostic_workflow.py \
	03_ensemble_workflow.py \
	04_corrdiff_inference.py

.PHONY: docs-full
docs-full:
	pip install .[docs]
	rm -rf docs/examples
	rm -rf docs/modules/generated
	rm -rf docs/modules/backreferences
	$(MAKE) -C docs clean
	rm -rf examples/outputs
	# for FILENAME in $(EXAMPLE_FILENAMES) ; do \
	# 	PLOT_GALLERY=True RUN_STALE_EXAMPLES=True FILENAME_PATTERN=$$FILENAME $(MAKE) -C docs html \
	# done
	PLOT_GALLERY=True RUN_STALE_EXAMPLES=True $(MAKE) -C docs html

.PHONY: docs-dev
docs-dev:
	rm -rf examples/outputs
	PLOT_GALLERY=True RUN_STALE_EXAMPLES=True FILENAME_PATTERN=$(FILENAME) $(MAKE) -C docs html
