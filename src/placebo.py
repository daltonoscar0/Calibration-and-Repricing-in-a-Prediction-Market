"""Falsification test for the manager-departure event study.

Produces:
  results/placebo.csv   null distribution of the event-study statistics
  figures/placebo.png   null distribution against the observed estimates

The event study reports an abnormal implied win probability of roughly -0.8 pp
over the six matches from a departure onward. That estimate means something only
if the identical machinery returns nothing when pointed at departures that never
happened. This script draws pseudo-announcements from club-seasons carrying no
event, matched to the real events' position within the season so that any
artifact of where in a campaign the window sits is reproduced, runs the same
pipeline, and compares the observed statistics to the resulting null.

Matching on season position matters: real sackings cluster between October and
March, and a benchmark fitted on a club's whole season will leave systematically
signed residuals at particular points in the campaign. An unmatched placebo
would not reproduce that and would understate the null.

The placebo pool is conservative by construction. `src/sackings.csv` holds 70
prominent departures rather than every managerial change, so some club-seasons
treated here as event-free did in fact see one. That contamination pushes the
null toward the observed estimate, making this a harder test to pass, not an
easier one.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eventstudy as es

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
FIGURES_DIR = os.path.join(HERE, "..", "figures")

N_DRAWS = 500
SEED = 20240804


def event_season_positions(panel, events):
    """For each real event, the index of its k=0 match within the club-season."""
    positions = []
    for _, event in events.iterrows():
        season = es.season_of(panel, event["club"], event["announce_date"])
        if season is None:
            continue
        club_season = panel[(panel["club"] == event["club"]) & (panel["season"] == season)]
        ordered = club_season.sort_values("date").reset_index(drop=True)
        after = ordered["date"] >= event["announce_date"]
        if not after.any() or after.all():
            continue
        positions.append(int(np.argmax(after.to_numpy())))
    return positions


def clean_club_seasons(panel, events):
    """Club-seasons carrying no event in the hand-compiled list."""
    used = set()
    for _, event in events.iterrows():
        season = es.season_of(panel, event["club"], event["announce_date"])
        if season is not None:
            used.add((event["club"], season))
    everything = set(map(tuple, panel[["club", "season"]].drop_duplicates().to_numpy()))
    return sorted(everything - used)


def statistics_from(windows):
    """The three headline numbers, recomputed on any set of event windows."""
    if windows.empty:
        return None
    return {
        "post": windows.loc[windows["k"] >= 0, "abnormal"].mean(),
        "pre": windows.loc[windows["k"] < 0, "abnormal"].mean(),
        "later_post": windows.loc[windows["k"].between(3, 5), "abnormal"].mean(),
    }


def draw_placebo(panel, pool, positions, rng, n_events):
    """One synthetic event list: random event-free club-seasons, matched timing."""
    picked = rng.choice(len(pool), size=min(n_events, len(pool)), replace=False)
    rows = []
    for j, index in enumerate(picked):
        club, season = pool[index]
        club_season = panel[(panel["club"] == club) & (panel["season"] == season)]
        ordered = club_season.sort_values("date").reset_index(drop=True)
        position = positions[j % len(positions)]
        if position >= len(ordered):
            continue
        rows.append(
            {
                "club": club,
                "manager": "placebo",
                "announce_date": ordered.loc[position, "date"],
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    panel = es.load_panel()
    events = pd.read_csv(es.EVENTS_PATH, parse_dates=["announce_date"])

    observed_windows, _ = es.build_event_windows(panel, events)
    observed = statistics_from(observed_windows)
    n_events = observed_windows["event_id"].nunique()
    print("observed: post={:+.5f}  pre={:+.5f}  later_post={:+.5f}  ({} events)\n".format(
        observed["post"], observed["pre"], observed["later_post"], n_events))

    positions = event_season_positions(panel, events)
    pool = clean_club_seasons(panel, events)
    print("{} event-free club-seasons in the placebo pool".format(len(pool)))
    print("season positions of real events: min {}, median {:.0f}, max {}\n".format(
        min(positions), float(np.median(positions)), max(positions)))

    rng = np.random.default_rng(SEED)
    draws = []
    for draw in range(N_DRAWS):
        fake = draw_placebo(panel, pool, positions, rng, n_events)
        windows, _ = es.build_event_windows(panel, fake)
        stats = statistics_from(windows)
        if stats is None:
            continue
        stats["n_events"] = windows["event_id"].nunique()
        draws.append(stats)
        if (draw + 1) % 100 == 0:
            print("  {} draws".format(draw + 1))

    null = pd.DataFrame(draws)
    print("\nnull distribution over {} draws (mean events per draw {:.0f}):".format(
        len(null), null["n_events"].mean()))

    summary = []
    for key in ("post", "pre", "later_post"):
        values = null[key].to_numpy()
        # Two-sided: how often is the placebo at least as extreme as observed?
        p_two = float(np.mean(np.abs(values) >= abs(observed[key])))
        p_one = float(np.mean(values <= observed[key]))
        row = {
            "statistic": key,
            "observed": observed[key],
            "null_mean": values.mean(),
            "null_sd": values.std(ddof=1),
            "null_p2.5": np.percentile(values, 2.5),
            "null_p97.5": np.percentile(values, 97.5),
            "p_two_sided": p_two,
            "p_one_sided_lower": p_one,
        }
        summary.append(row)
        print("  {:11}  observed={:+.5f}  null={:+.5f} (sd {:.5f})  "
              "95% null [{:+.5f}, {:+.5f}]  p(two-sided)={:.3f}".format(
                  key, row["observed"], row["null_mean"], row["null_sd"],
                  row["null_p2.5"], row["null_p97.5"], p_two))

    summary = pd.DataFrame(summary)
    summary.round(6).to_csv(os.path.join(RESULTS_DIR, "placebo.csv"), index=False)
    null.round(6).to_csv(os.path.join(RESULTS_DIR, "placebo_draws.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, key, title in zip(axes, ("post", "later_post"),
                              ("Post-announcement (k = 0..+5)", "Later post (k = +3..+5)")):
        values = null[key].to_numpy()
        ax.hist(values, bins=36, color="#8fa5c4", edgecolor="white", linewidth=0.4)
        ax.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0)
        ax.axvline(observed[key], color="#b3541e", linewidth=1.8)
        ax.annotate("observed\n{:+.4f}".format(observed[key]),
                    xy=(observed[key], ax.get_ylim()[1] * 0.86),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color="#b3541e")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Mean abnormal implied win probability")
        ax.set_ylabel("Placebo draws")
        ax.grid(True, color="0.93", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Placebo test: {} draws of pseudo-announcements from event-free club-seasons".format(
            len(null)), fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "placebo.png"), dpi=160)
    plt.close(fig)

    print("\nwrote results/placebo.csv, results/placebo_draws.csv, figures/placebo.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
