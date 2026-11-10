# CFB situational success residuals
# Usage: make venv && cp .env.example .env  # then add CFBD_API_KEY
#        make all

PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

.PHONY: venv all download clean-data model report defense sensitivity wins luck inference extension

venv:
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt

all:
	bash src/run_all.sh

download:
	$(PYTHON) src/download.py

clean-data:
	$(PYTHON) src/clean.py

model:
	$(PYTHON) src/model.py

sensitivity:
	$(PYTHON) src/sensitivity.py

report:
	$(PYTHON) src/report.py

defense:
	$(PYTHON) src/defense.py && $(PYTHON) src/defense_report.py

wins:
	$(PYTHON) src/wins.py

luck:
	$(PYTHON) src/luck.py

inference:
	$(PYTHON) src/inference.py

# Luck and inference run from packaged team-season / pair tables (no API key).
# `make wins` downloads CFBD /games (gitignored) and needs CFBD_API_KEY.
extension:
	$(PYTHON) src/luck.py && $(PYTHON) src/inference.py
