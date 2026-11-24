"""Put src/ on the path so the stage modules import as top-level names.

The project is a set of scripts run through the Makefile rather than an
installable package, so there is no `pip install -e .` step to rely on.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
