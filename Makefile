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
	uv venv --python=3.13
	uv sync
	uv run pre-commit install --install-hooks
	uv tool install tox --with tox-uv
	uv sync --extra all
	uv sync --extra aifs
	uv sync --extra aifs2
	uv sync --extra aifs2ens
	uv sync --extra aifsens
	uv sync --extra stormcast-conus

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

.PHONY: zizmor
zizmor:
	uv run pre-commit run zizmor -a

.PHONY: license
license:
	uv run python test/_license/header_check.py

.PHONY: pytest
pytest:
	@test -n "$(TOX_ENV)" || (echo "TOX_ENV is required! Usage: make pytest TOX_ENV=<env>" && exit 1)
	uvx tox -c tox.ini run -e $(TOX_ENV)

.PHONY: pytest-full
pytest-full:
	uvx tox -c tox.ini run -- --cov --cov-append --slow --package --testmon-noselect

# Select which pytest target to run in CI based on environment
ifneq (,$(filter 1 true TRUE True yes YES on ON,$(CI_PYTEST_ALL)))
PYTEST_CI_TARGET := pytest-full
else
PYTEST_CI_TARGET := pytest TOX_ENV=$(TOX_ENV)
endif

.PHONY: pytest-ci
pytest-ci:
	uv run python test/_ci/check_gpu.py || exit $?
	$(MAKE) $(PYTEST_CI_TARGET)

.PHONY: coverage
coverage:
	uv run coverage combine || true
	uv run coverage report --fail-under=90 || true

UV_DOCS := uv run --locked --group docs

.PHONY: docs-generate
docs-generate:
	$(UV_DOCS) python docs/generate_api.py
	$(UV_DOCS) python docs/generate_catalog.py
	$(UV_DOCS) python docs/generate_install_options.py
	$(UV_DOCS) python docs/generate_scorecard.py
	$(UV_DOCS) python docs/generate_gallery.py

.PHONY: docs
docs:
	uv sync --locked --group docs
	$(MAKE) docs-generate
	E2S_GALLERY_EXECUTE=never $(UV_DOCS) mkdocs build --clean
	rm -rf site/__pycache__ site/_build/html
	find site -maxdepth 1 -type f -name "*.py" -delete

.PHONY: docs-full
docs-full:
	uv sync --locked --group docs
	$(MAKE) docs-generate
	$(MAKE) docs-build-examples
	$(UV_DOCS) python docs/generate_gallery.py
	E2S_GALLERY_EXECUTE=never $(UV_DOCS) mkdocs build --clean
	rm -rf site/__pycache__ site/_build/html
	find site -maxdepth 1 -type f -name "*.py" -delete

.PHONY: docs-build-examples
docs-build-examples:
	test/_ci/build_docs_examples.sh

DOCS_JOBS ?= 1
.PHONY: docs-dev
docs-dev:
	uv sync --locked --group docs
	$(MAKE) docs-generate
	@if [ -n "$(FILENAME)" ]; then \
		$(UV_DOCS) e2s-gallery build "$(FILENAME)" --execute stale --jobs $(DOCS_JOBS); \
		$(UV_DOCS) python docs/generate_gallery.py; \
	fi
	E2S_GALLERY_EXECUTE=never $(UV_DOCS) mkdocs serve -a 0.0.0.0:$(PORT)

DOC_VERSION ?= main
.PHONY: docs-build-version
docs-build-version:
	uv sync --locked --group docs
	$(MAKE) docs-generate
	DOC_VERSION=$(DOC_VERSION) E2S_GALLERY_EXECUTE=never $(UV_DOCS) mkdocs build --clean
	rm -rf site/__pycache__ site/_build/html
	find site -maxdepth 1 -type f -name "*.py" -delete

.PHONY: docs-deploy-version
docs-deploy-version:
	$(MAKE) docs-build-version

.PHONY: docs-version-serve
docs-version-serve:
	uv sync --locked --group docs
	$(UV_DOCS) mike serve

PORT ?= 8001
.PHONY: docs-serve
docs-serve:
	uv sync --locked --group docs
	$(MAKE) docs-generate
	E2S_GALLERY_EXECUTE=never $(UV_DOCS) mkdocs serve -a 0.0.0.0:$(PORT)

.PHONY: container-service
# Example DOCKER_REPO?=nvcr.io/dycvht5ows21
E2S_RELEASE_TAG?=0.15.0
E2S_IMAGE_NAME=$(DOCKER_REPO)/earth2studio-scicomp
E2S_IMAGE_TAG=v$(E2S_RELEASE_TAG).20260514.0
container-service:
	@test -n "$(DOCKER_REPO)" || (echo "DOCKER_REPO is not set!" && exit 1)
	DOCKER_BUILDKIT=1 docker build -t $(E2S_IMAGE_NAME):$(E2S_IMAGE_TAG) -f serve/Dockerfile .
