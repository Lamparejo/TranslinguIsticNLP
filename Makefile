VENV_DIR ?= .venv
PYTHON := $(VENV_DIR)/bin/python
CONFIG ?= config/pipeline.yaml

ifeq (,$(wildcard $(PYTHON)))
$(error Virtual environment not found at $(PYTHON). Run `python3 -m venv $(VENV_DIR)` to create it.)
endif

.PHONY: install test pipeline train dashboard workflow full lint

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

pipeline:
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train_gnn.py --config $(CONFIG)

dashboard:
	$(PYTHON) -m streamlit run apps/dashboard.py

workflow:
	$(PYTHON) scripts/run_workflow.py --config $(CONFIG)

full:
	$(PYTHON) scripts/run_full_pipeline.py --config $(CONFIG)
