PYTHON ?= python3

.PHONY: all download build calibrate eventstudy placebo softsharp clean

all: download build calibrate eventstudy placebo softsharp

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
