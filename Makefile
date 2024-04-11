install:
	pip install --upgrade pip
	pip install -e .[all]
	pip install "makani[all] @ git+https://github.com/NVIDIA/modulus-makani.git@v0.1.0"

.PHONY: setup-ci
setup-ci:
	pip install .[dev]
	pre-commit install

.PHONY: format
format:
	pre-commit run black -a

.PHONY: black
black:
	pre-commit run black -a

.PHONY: interrogate
interrogate:
	pre-commit run interrogate -a

.PHONY: lint
lint:
	echo "TODO: add interrogate"
	pre-commit run check-added-large-files -a
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
	coverage report --fail-under=89

.PHONY: report
report:
	coverage xml
	curl -Os https://uploader.codecov.io/latest/linux/codecov
	chmod +x codecov
	./codecov -v -f e2studio.coverage.xml $(COV_ARGS)

.PHONY: docs
docs:
	pip install .[docs]
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

.PHONY: docs-full
docs-full:
	pip install .[docs]
	$(MAKE) -C docs clean
	rm -rf examples/outputs
	PLOT_GALLERY=True RUN_STALE_EXAMPLES=True $(MAKE) -C docs html

.PHONY: docs-dev
docs-dev:
	rm -rf examples/outputs
	PLOT_GALLERY=True RUN_STALE_EXAMPLES=True FILENAME_PATTERN=$(FILENAME) $(MAKE) -C docs html