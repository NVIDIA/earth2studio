PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

help:
	@echo "Available targets: setup, install, demo, clean"

setup:
	$(PYTHON) -m venv $(VENV)

install:
	$(ACTIVATE) && python -m pip install --upgrade pip
	$(ACTIVATE) && python -m pip install -e .

demo:
	$(ACTIVATE) && python -m nyc_flash_flood_case_study --demo

clean:
	rm -rf $(VENV) data/processed data/raw figures outputs
