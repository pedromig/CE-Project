SHELL := /bin/bash
ENV := venv
PYTHON := python3
REQUIREMENTS := requirements.txt

.PHONY: env clean activate deactivate

env:
	@echo "Setting up virtual environment..."
	@$(PYTHON) -m venv $(ENV)
	@source $(ENV)/bin/activate && \
		$(PYTHON) -m pip install --upgrade pip && \
		$(PYTHON) -m pip install -r $(REQUIREMENTS)
	@echo "DONE!" 

clean:
	@echo -n "Removing environment... "
	@rm -rf $(ENV)
	@echo "DONE!"
