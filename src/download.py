"""Fetch English Premier League (E0) match-and-odds CSVs from football-data.co.uk.

Produces: data/raw/E0_{season}.csv for each season in the configured range, where
{season} is the four-digit football-data season code (e.g. 2324 for 2023/24).

Seasons that 404 or fail to decode are logged and skipped; the pipeline is
tolerant of missing seasons downstream.
"""

import os
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# 2005/06 through 2024/25 inclusive.
START_YEAR = 2005
END_YEAR = 2024

USER_AGENT = "Mozilla/5.0 (compatible; overround-research/1.0)"
RETRIES = 3
RETRY_SLEEP = 2.0


def season_code(start_year):
    """2005 -> '0506'; 2019 -> '1920'."""
    return "{:02d}{:02d}".format(start_year % 100, (start_year + 1) % 100)


def season_label(start_year):
    """2005 -> '2005-06'."""
    return "{}-{:02d}".format(start_year, (start_year + 1) % 100)


def fetch(url):
    """Return raw bytes for url, retrying on transient failures."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last_error = exc
        except Exception as exc:  # network flakiness
            last_error = exc
        time.sleep(RETRY_SLEEP * (attempt + 1))
    raise last_error


def decode(payload):
    """Decode football-data bytes, which are inconsistently utf-8 / latin-1."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("latin-1", errors="replace")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    present, skipped = 0, []

    for start_year in range(START_YEAR, END_YEAR + 1):
        code = season_code(start_year)
        target = os.path.join(RAW_DIR, "E0_{}.csv".format(code))

        if os.path.exists(target) and os.path.getsize(target) > 0:
            print("cached   {}".format(season_label(start_year)))
            present += 1
            continue

        url = BASE_URL.format(season=code)
        try:
            text = decode(fetch(url))
        except urllib.error.HTTPError as exc:
            print("skip     {}  HTTP {} for {}".format(season_label(start_year), exc.code, url))
            skipped.append(code)
            continue
        except Exception as exc:
            print("skip     {}  {} for {}".format(season_label(start_year), exc, url))
            skipped.append(code)
            continue

        with open(target, "w", encoding="utf-8") as handle:
            handle.write(text)

        n_lines = sum(1 for line in text.splitlines() if line.strip())
        print("download {}  {} lines".format(season_label(start_year), n_lines))
        present += 1

    print("\n{} season files present, {} skipped".format(present, len(skipped)))
    if skipped:
        print("skipped codes: {}".format(", ".join(skipped)))
    if present == 0:
        print("no data downloaded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
