PYTHON ?= .venv-kev/bin/python
PYTHON_SOURCES = fez miner scripts tests

.PHONY: check lint format test check-website check-queue-db

check: lint test $(if $(wildcard website/package.json),check-website)

lint:
	$(PYTHON) -m ruff check $(PYTHON_SOURCES)
	$(PYTHON) -m ruff format --check $(PYTHON_SOURCES)

format:
	$(PYTHON) -m ruff check --select I --fix $(PYTHON_SOURCES)
	$(PYTHON) -m ruff format $(PYTHON_SOURCES)

test:
	$(PYTHON) -c "import bittensor, bittensor_wallet, kev, torch"
	$(PYTHON) -m unittest discover -v

check-website:
	@for file in website/*.js website/*.mjs; do node --check "$$file" || exit 1; done
	npm --prefix website test
	npm --prefix website run build

check-queue-db:
	$(PYTHON) -m scripts.check_queue_db
