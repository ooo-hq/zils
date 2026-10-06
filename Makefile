PYTHON ?= .venv-kev/bin/python
PYTHON_SOURCES = zils miner scripts tests

.PHONY: check lint format test check-website check-queue-db check-api

check: lint test $(if $(wildcard website/package.json),check-website)

lint:
	$(PYTHON) -m ruff check $(PYTHON_SOURCES)
	$(PYTHON) -m ruff format --check $(PYTHON_SOURCES)

format:
	$(PYTHON) -m ruff check --select I --fix $(PYTHON_SOURCES)
	$(PYTHON) -m ruff format $(PYTHON_SOURCES)

test:
	$(PYTHON) -c "import bittensor, bittensor_wallet, kev, torch"
	$(PYTHON) -m unittest discover -s tests -t . -v

check-website:
	@for file in website/*.js website/*.mjs; do node --check "$$file" || exit 1; done
	npm --prefix website test
	npm --prefix website run build

check-queue-db:
	$(PYTHON) -m scripts.check_queue_db

check-api:
	$(PYTHON) -m unittest tests.test_decisions tests.test_decision_http tests.test_jev_server tests.test_api_store tests.test_api tests.test_batches -v
