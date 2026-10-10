PYTHON ?= .venv-kev/bin/python
PYTHON_SOURCES = zils miner scripts tests

.PHONY: check lint format test

check: lint test

lint:
	$(PYTHON) -m ruff check $(PYTHON_SOURCES)
	$(PYTHON) -m ruff format --check $(PYTHON_SOURCES)

format:
	$(PYTHON) -m ruff check --select I --fix $(PYTHON_SOURCES)
	$(PYTHON) -m ruff format $(PYTHON_SOURCES)

test:
	$(PYTHON) -c "import bittensor, bittensor_wallet, kev, torch"
	$(PYTHON) -m unittest discover -s tests -t . -v
