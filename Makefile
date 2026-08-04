PYTHON ?= python3

.PHONY: all download build calibrate eventstudy clean

all: download build calibrate eventstudy

download:
	$(PYTHON) src/download.py

build:
	$(PYTHON) src/build.py

calibrate:
	$(PYTHON) src/calibration.py

eventstudy:
	$(PYTHON) src/eventstudy.py

clean:
	rm -rf data/raw/* data/processed/*
