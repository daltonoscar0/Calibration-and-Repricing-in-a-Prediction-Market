# Decision log

- Processed match table written as CSV, not Parquet: Parquet requires `pyarrow`, outside the four permitted dependencies.
- Repo pinned to a local virtualenv for development; `Makefile` uses `PYTHON ?= python3` so it runs against any 3.11+ interpreter with `requirements.txt` installed.
