"""Assemble the raw football-data season files into one tidy match-level table.

Produces: data/processed/matches.csv with one row per Premier League match and
exactly one set of decimal 1X2 odds per match, selected by a fixed bookmaker
priority (Pinnacle closing > Pinnacle > market average > Bet365). The column
`odds_source` records which quote was used.
"""

import glob
import os
import sys

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_PATH = os.path.join(PROCESSED_DIR, "matches.csv")

# Highest priority first. Each entry is (label, home_col, draw_col, away_col).
ODDS_PRIORITY = [
    ("pinnacle_close", "PSCH", "PSCD", "PSCA"),
    ("pinnacle", "PSH", "PSD", "PSA"),
    ("market_avg", "BbAvH", "BbAvD", "BbAvA"),
    ("market_avg", "AvgH", "AvgD", "AvgA"),
    ("bet365", "B365H", "B365D", "B365A"),
]

DATE_FORMATS = ["%d/%m/%Y", "%d/%m/%y"]

MIN_ODD = 1.0
EXPECTED_MATCHES = 380


def season_label(code):
    """'0506' -> '2005-06'. football-data codes are unambiguous within 2000-2099."""
    return "20{}-{}".format(code[:2], code[2:])


def read_season(path):
    """Read one raw season file, tolerating encoding quirks and trailing blanks."""
    for encoding in ("utf-8", "latin-1"):
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise IOError("could not decode {}".format(path))

    frame.columns = [str(col).strip() for col in frame.columns]
    # football-data files carry trailing empty columns and trailing blank rows.
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
    frame = frame.dropna(axis=1, how="all")
    frame = frame.dropna(subset=["Date"])
    return frame


def parse_dates(raw):
    """Parse a Date column that switches between %d/%m/%y and %d/%m/%Y."""
    raw = raw.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        missing = parsed.isna()
        if not missing.any():
            break
        attempt = pd.to_datetime(raw[missing], format=fmt, errors="coerce")
        parsed.loc[missing] = attempt
    return parsed


def select_odds(frame):
    """Return (home, draw, away, source) series using the bookmaker priority."""
    n = len(frame)
    home = pd.Series([float("nan")] * n, index=frame.index, dtype=float)
    draw = home.copy()
    away = home.copy()
    source = pd.Series([None] * n, index=frame.index, dtype=object)

    for label, h_col, d_col, a_col in ODDS_PRIORITY:
        if not all(col in frame.columns for col in (h_col, d_col, a_col)):
            continue
        candidate_h = pd.to_numeric(frame[h_col], errors="coerce")
        candidate_d = pd.to_numeric(frame[d_col], errors="coerce")
        candidate_a = pd.to_numeric(frame[a_col], errors="coerce")

        usable = (
            source.isna()
            & candidate_h.notna()
            & candidate_d.notna()
            & candidate_a.notna()
            & (candidate_h > MIN_ODD)
            & (candidate_d > MIN_ODD)
            & (candidate_a > MIN_ODD)
        )
        home.loc[usable] = candidate_h.loc[usable]
        draw.loc[usable] = candidate_d.loc[usable]
        away.loc[usable] = candidate_a.loc[usable]
        source.loc[usable] = label

    return home, draw, away, source


def build_season(path):
    """Return a tidy frame for one season file, or None if unusable."""
    code = os.path.basename(path).replace("E0_", "").replace(".csv", "")
    frame = read_season(path)

    if "FTR" not in frame.columns or "HomeTeam" not in frame.columns:
        print("skip {}: missing core columns".format(code))
        return None

    home_odds, draw_odds, away_odds, source = select_odds(frame)

    tidy = pd.DataFrame(
        {
            "date": parse_dates(frame["Date"]),
            "season": season_label(code),
            "home": frame["HomeTeam"].astype(str).str.strip(),
            "away": frame["AwayTeam"].astype(str).str.strip(),
            "FTR": frame["FTR"].astype(str).str.strip(),
            "odds_h": home_odds,
            "odds_d": draw_odds,
            "odds_a": away_odds,
            "odds_source": source,
        }
    )

    n_raw = len(tidy)
    tidy = tidy[tidy["FTR"].isin(["H", "D", "A"])]
    tidy = tidy.dropna(subset=["date", "odds_h", "odds_d", "odds_a"])
    n_kept = len(tidy)

    breakdown = tidy["odds_source"].value_counts().to_dict()
    flag = "" if n_kept == EXPECTED_MATCHES else "  <- not {}".format(EXPECTED_MATCHES)
    print(
        "{}  raw={:4d}  kept={:4d}  dropped={:3d}  sources={}{}".format(
            season_label(code), n_raw, n_kept, n_raw - n_kept, breakdown, flag
        )
    )
    return tidy


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "E0_*.csv")))
    if not paths:
        print("no raw season files; run src/download.py first", file=sys.stderr)
        return 1

    seasons = [season for season in (build_season(path) for path in paths) if season is not None]
    matches = pd.concat(seasons, ignore_index=True).sort_values("date").reset_index(drop=True)

    # A full PL season is 380 matches. 2019/20 finished after a suspension but
    # still played all 380, so any shortfall signals a data problem.
    counts = matches.groupby("season").size()
    short = counts[counts < EXPECTED_MATCHES]
    assert short.empty, "seasons below {} matches: {}".format(EXPECTED_MATCHES, short.to_dict())
    assert (counts <= EXPECTED_MATCHES).all(), "seasons above {} matches".format(EXPECTED_MATCHES)

    matches.to_csv(OUT_PATH, index=False)

    print("\nwrote {} matches to {}".format(len(matches), OUT_PATH))
    print("date range: {} to {}".format(matches["date"].min().date(), matches["date"].max().date()))
    print("\nodds source shares (%):")
    print((matches["odds_source"].value_counts(normalize=True) * 100).round(1).to_string())
    print("\nresult shares (%):")
    print((matches["FTR"].value_counts(normalize=True) * 100).round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
