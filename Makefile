PYTHON ?= python3

.PHONY: all reproduce test download build calibrate eventstudy placebo softsharp clean

all: download build calibrate eventstudy placebo softsharp

# The single command that regenerates every table and figure in results/ and
# figures/ from scratch. About seven minutes, of which the 500-draw placebo is
# roughly six.
#
# The odds CSVs are NOT in the repo: football-data.co.uk is the source and
# data/raw/ is gitignored, so this starts by fetching twenty small CSVs. The
# fetch is idempotent, so re-running is cheap. The only committed input is
# src/sackings.csv, the manager-departure list.
reproduce: all
	@echo
	@echo "Regenerated: results/*.csv and figures/*.png"
	@echo "Compare against the committed copies with: git diff --stat results figures"

# Unit tests for the odds-to-probability normalisation. No data needed.
test:
	$(PYTHON) -m pytest tests/ -q

download:
	$(PYTHON) src/download.py

build:
	$(PYTHON) src/build.py

calibrate:
	$(PYTHON) src/calibration.py

eventstudy:
	$(PYTHON) src/eventstudy.py

placebo:
	$(PYTHON) src/placebo.py

softsharp:
	$(PYTHON) src/softsharp.py

clean:
	rm -rf data/raw/* data/processed/*
