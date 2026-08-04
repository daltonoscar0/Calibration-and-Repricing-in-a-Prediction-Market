"""Event study of manager departures in the Premier League match-odds market.

Produces:
  results/eventstudy.csv          abnormal implied win probability by event time
  results/eventstudy_summary.csv  pre- and post-announcement aggregates
  figures/eventstudy.png          the average abnormal path with bootstrap CIs

Design follows MacKinlay (1997), with calendar time replaced by match time. For
each event the club's matches within the same season are indexed k = -5..+5
relative to the announcement (k < 0 strictly before, k >= 0 on or after). The
outcome is the club's overround-stripped implied win probability. The benchmark
is a club-season regression of that probability on a home dummy and opponent
strength, estimated on the club's matches outside every event window in that
season. Abnormal probability is the residual from that benchmark.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
MATCHES_PATH = os.path.join(HERE, "..", "data", "processed", "matches.csv")
EVENTS_PATH = os.path.join(HERE, "sackings.csv")
RESULTS_DIR = os.path.join(HERE, "..", "results")
FIGURES_DIR = os.path.join(HERE, "..", "figures")

WINDOW = 5                 # event time runs -WINDOW .. +WINDOW
MIN_WINDOW_MATCHES = 8     # events with a thinner window are dropped
MIN_ESTIMATION_MATCHES = 12
N_BOOTSTRAP = 5000
SEED = 20240804


def load_panel():
    """One row per club per match, with the club's implied win probability."""
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])

    raw = 1.0 / matches[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    total = raw.sum(axis=1)
    p_home = raw[:, 0] / total
    p_away = raw[:, 2] / total

    home_side = pd.DataFrame(
        {
            "date": matches["date"],
            "season": matches["season"],
            "club": matches["home"],
            "opponent": matches["away"],
            "is_home": 1.0,
            "p_win": p_home,
        }
    )
    away_side = pd.DataFrame(
        {
            "date": matches["date"],
            "season": matches["season"],
            "club": matches["away"],
            "opponent": matches["home"],
            "is_home": 0.0,
            "p_win": p_away,
        }
    )

    panel = pd.concat([home_side, away_side], ignore_index=True)

    # Opponent strength: the opponent's own season-average implied win probability.
    strength = panel.groupby(["season", "club"])["p_win"].mean().rename("strength")
    panel = panel.merge(
        strength.rename_axis(["season", "opponent"]).reset_index(),
        on=["season", "opponent"],
        how="left",
    )
    panel = panel.sort_values(["season", "club", "date"]).reset_index(drop=True)
    return panel


def season_of(panel, club, announce_date):
    """Season label containing announce_date for this club, or None."""
    club_rows = panel[panel["club"] == club]
    bounds = club_rows.groupby("season")["date"].agg(["min", "max"])
    for season, row in bounds.iterrows():
        if row["min"] <= announce_date <= row["max"]:
            return season
    return None


def assign_event_time(club_season, announce_date):
    """Index a club-season's matches in event time relative to announce_date."""
    ordered = club_season.sort_values("date").reset_index(drop=True)
    after = ordered["date"] >= announce_date
    if not after.any() or after.all():
        return None  # announcement falls outside the club's playing season
    first_after = int(np.argmax(after.to_numpy()))
    ordered = ordered.copy()
    ordered["k"] = ordered.index - first_after
    return ordered


def fit_benchmark(estimation):
    """OLS of implied win probability on a home dummy and opponent strength."""
    design = np.column_stack(
        [
            np.ones(len(estimation)),
            estimation["is_home"].to_numpy(dtype=float),
            estimation["strength"].to_numpy(dtype=float),
        ]
    )
    target = estimation["p_win"].to_numpy(dtype=float)
    coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coef
    resid = target - fitted
    dof = max(len(estimation) - design.shape[1], 1)
    return coef, float(np.sqrt((resid ** 2).sum() / dof))


def predict(coef, frame):
    design = np.column_stack(
        [
            np.ones(len(frame)),
            frame["is_home"].to_numpy(dtype=float),
            frame["strength"].to_numpy(dtype=float),
        ]
    )
    return design @ coef


def build_event_windows(panel, events):
    """Return (long frame of abnormal probabilities, list of dropped events)."""
    # Every event window belonging to a club-season is excluded from that
    # club-season's estimation sample, so a second sacking cannot contaminate
    # the benchmark for the first.
    resolved = []
    dropped = []

    for _, event in events.iterrows():
        season = season_of(panel, event["club"], event["announce_date"])
        if season is None:
            dropped.append((event, "announcement falls outside the club's season"))
            continue
        resolved.append((event, season))

    windows_by_club_season = {}
    for event, season in resolved:
        club_season = panel[(panel["club"] == event["club"]) & (panel["season"] == season)]
        indexed = assign_event_time(club_season, event["announce_date"])
        if indexed is None:
            dropped.append((event, "no matches on either side of the announcement"))
            continue
        window = indexed[indexed["k"].abs() <= WINDOW]
        windows_by_club_season.setdefault((event["club"], season), []).append(
            set(window["date"])
        )

    records = []
    for event, season in resolved:
        club = event["club"]
        club_season = panel[(panel["club"] == club) & (panel["season"] == season)]
        indexed = assign_event_time(club_season, event["announce_date"])
        if indexed is None:
            continue

        window = indexed[indexed["k"].abs() <= WINDOW]
        if len(window) < MIN_WINDOW_MATCHES:
            dropped.append((event, "only {} matches in the event window".format(len(window))))
            continue

        excluded_dates = set()
        for dates in windows_by_club_season.get((club, season), []):
            excluded_dates |= dates
        estimation = indexed[~indexed["date"].isin(excluded_dates)]
        if len(estimation) < MIN_ESTIMATION_MATCHES:
            dropped.append((event, "only {} matches in the estimation sample".format(len(estimation))))
            continue

        coef, sigma = fit_benchmark(estimation)
        window = window.copy()
        window["expected"] = predict(coef, window)
        window["abnormal"] = window["p_win"] - window["expected"]
        window["event_id"] = "{} {} {}".format(
            club, event["manager"], event["announce_date"].date()
        )
        window["manager"] = event["manager"]
        window["announce_date"] = event["announce_date"]
        window["resid_sd"] = sigma
        records.append(
            window[
                ["event_id", "club", "manager", "announce_date", "season", "date", "k",
                 "is_home", "strength", "p_win", "expected", "abnormal", "resid_sd"]
            ]
        )

    if not records:
        return pd.DataFrame(), dropped
    return pd.concat(records, ignore_index=True), dropped


def bootstrap_by_event(values, event_ids, statistic, n_boot=N_BOOTSTRAP, seed=SEED):
    """Percentile bootstrap of `statistic`, resampling whole events."""
    rng = np.random.default_rng(seed)
    groups = {}
    for value, event_id in zip(values, event_ids):
        groups.setdefault(event_id, []).append(value)
    keys = list(groups)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(keys), size=len(keys))
        pooled = np.concatenate([groups[keys[i]] for i in picked])
        draws[b] = statistic(pooled)
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def event_time_table(windows):
    """Mean abnormal probability at each event time with clustered bootstrap CIs."""
    rows = []
    for k in range(-WINDOW, WINDOW + 1):
        slice_k = windows[windows["k"] == k]
        if slice_k.empty:
            continue
        values = slice_k["abnormal"].to_numpy()
        ids = slice_k["event_id"].to_numpy()
        lo, hi = bootstrap_by_event(values, ids, np.mean, seed=SEED + k)
        rows.append(
            {
                "k": k,
                "n_events": slice_k["event_id"].nunique(),
                "mean_abnormal": values.mean(),
                "sd_abnormal": values.std(ddof=1),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame(rows)


def summary_cut(windows):
    """Pre- versus post-announcement aggregates, plus a pre-window drift slope."""
    rows = []
    cuts = {
        "pre  (k=-5..-1)": windows["k"] < 0,
        "post (k=0..+5)": windows["k"] >= 0,
        "early pre (k=-5..-3)": windows["k"].between(-5, -3),
        "late pre  (k=-2..-1)": windows["k"].between(-2, -1),
        "immediate post (k=0..+2)": windows["k"].between(0, 2),
        "later post (k=+3..+5)": windows["k"].between(3, 5),
    }
    for label, mask in cuts.items():
        subset = windows[mask]
        values = subset["abnormal"].to_numpy()
        lo, hi = bootstrap_by_event(values, subset["event_id"].to_numpy(), np.mean)
        rows.append(
            {
                "cut": label,
                "n_obs": len(subset),
                "n_events": subset["event_id"].nunique(),
                "mean_abnormal": values.mean(),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )

    pre = windows[windows["k"] < 0]
    slope = np.polyfit(pre["k"].to_numpy(dtype=float), pre["abnormal"].to_numpy(), 1)[0]

    def slope_of(pooled):
        return np.polyfit(pooled[:, 0], pooled[:, 1], 1)[0]

    stacked = np.column_stack([pre["k"].to_numpy(dtype=float), pre["abnormal"].to_numpy()])
    lo, hi = bootstrap_by_event(list(stacked), pre["event_id"].to_numpy(), slope_of)
    rows.append(
        {
            "cut": "pre-window drift per match",
            "n_obs": len(pre),
            "n_events": pre["event_id"].nunique(),
            "mean_abnormal": slope,
            "ci_lo": lo,
            "ci_hi": hi,
        }
    )
    return pd.DataFrame(rows)


def plot_event_study(table, n_events, path):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, zorder=1)
    ax.axvline(-0.5, color="#b3541e", linestyle=":", linewidth=1.3, zorder=1)
    ax.fill_between(table["k"], table["ci_lo"], table["ci_hi"],
                    color="#1b3a6b", alpha=0.15, linewidth=0, zorder=2)
    ax.plot(table["k"], table["mean_abnormal"], "o-", color="#1b3a6b",
            markersize=5, linewidth=1.5, zorder=3)

    ax.annotate("announcement", xy=(-0.5, ax.get_ylim()[1]),
                xytext=(-0.35, ax.get_ylim()[1] * 0.92),
                fontsize=8.5, color="#b3541e")
    ax.set_xlabel("Match in event time (k < 0 before announcement, k >= 0 on or after)")
    ax.set_ylabel("Abnormal implied win probability")
    ax.set_title(
        "Repricing around Premier League manager departures ({} events)".format(n_events)
    )
    ax.set_xticks(range(-WINDOW, WINDOW + 1))
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if not os.path.exists(MATCHES_PATH):
        print("missing {}; run src/build.py first".format(MATCHES_PATH), file=sys.stderr)
        return 1

    panel = load_panel()
    events = pd.read_csv(EVENTS_PATH, parse_dates=["announce_date"])
    print("{} club-match observations, {} candidate events".format(len(panel), len(events)))

    windows, dropped = build_event_windows(panel, events)
    if windows.empty:
        print("no usable events", file=sys.stderr)
        return 1

    if dropped:
        print("\ndropped {} events:".format(len(dropped)))
        for event, reason in dropped:
            print("  {} {} {} — {}".format(
                event["club"], event["manager"], event["announce_date"].date(), reason))

    n_events = windows["event_id"].nunique()
    print("\n{} events retained, {} club-match observations in event windows".format(
        n_events, len(windows)))
    per_event = windows.groupby("event_id").size()
    print("window length: min {}, median {:.0f}, max {}".format(
        per_event.min(), per_event.median(), per_event.max()))
    assert per_event.min() >= MIN_WINDOW_MATCHES

    table = event_time_table(windows)
    print("\nAbnormal implied win probability by event time:")
    print(table.round(4).to_string(index=False))

    summary = summary_cut(windows)
    print("\nPre- versus post-announcement:")
    print(summary.round(4).to_string(index=False))

    table.round(6).to_csv(os.path.join(RESULTS_DIR, "eventstudy.csv"), index=False)
    summary.round(6).to_csv(os.path.join(RESULTS_DIR, "eventstudy_summary.csv"), index=False)
    windows.round({col: 6 for col in windows.select_dtypes("number").columns}).to_csv(
        os.path.join(RESULTS_DIR, "eventstudy_windows.csv"), index=False
    )
    plot_event_study(table, n_events, os.path.join(FIGURES_DIR, "eventstudy.png"))

    print("\nwrote results/eventstudy.csv, results/eventstudy_summary.csv, "
          "results/eventstudy_windows.csv, figures/eventstudy.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
