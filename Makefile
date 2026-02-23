.PHONY: install
install:
	uv sync
	uv sync --extra all --extra aifs

.PHONY: install-docker
install-docker:
	uv pip install --system --break-system-packages .
	uv pip install --system --break-system-packages .[all] --group dev

.PHONY: setup-ci
setup-ci:
	uv venv --python=3.12
	uv sync
	uv run pre-commit install --install-hooks
	uv tool install tox --with tox-uv
	uv sync --extra all

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
	uvx tox -c tox-min.ini run

.PHONY: pytest-full
pytest-full:
	uvx tox -c tox.ini run -- --cov --cov-append --slow --package --testmon-noselect

# Select which pytest target to run in CI based on environment
ifneq (,$(filter 1 true TRUE True yes YES on ON,$(CI_PYTEST_ALL)))
PYTEST_CI_TARGET := pytest-full
else
PYTEST_CI_TARGET := pytest
endif

.PHONY: pytest-ci
pytest-ci:
	$(MAKE) $(PYTEST_CI_TARGET)

.PHONY: coverage
coverage:
	uv run coverage combine || true
	uv run coverage report --fail-under=90 || true

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

.PHONY: container-service
# Example DOCKER_REPO?=nvcr.io/dycvht5ows21
E2S_RELEASE_TAG?=0.11.0
E2S_IMAGE_NAME=$(DOCKER_REPO)/earth2studio-scicomp
E2S_IMAGE_TAG=v$(E2S_RELEASE_TAG).20260220.0
container-service:
	@test -n "$(DOCKER_REPO)" || (echo "DOCKER_REPO is not set!" && exit 1)
	DOCKER_BUILDKIT=1 docker build -t $(E2S_IMAGE_NAME):$(E2S_IMAGE_TAG) -f serve/Dockerfile .
