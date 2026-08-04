"""Overround measurement and calibration of Premier League 1X2 odds.

Produces:
  results/overround.csv     overround by season and odds source
  results/calibration.csv   binned implied vs realized frequencies and returns
  figures/calibration.png   reliability diagram with Wilson binomial CIs
  figures/flb.png           expected return per unit staked by implied-probability bin

The market's gross margin (the overround) is stripped two ways: proportional
normalisation, and Shin's (1993) insider-trading adjustment. Proportional
probabilities are the headline series; Shin probabilities are reported alongside.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(__file__)
MATCHES_PATH = os.path.join(HERE, "..", "data", "processed", "matches.csv")
RESULTS_DIR = os.path.join(HERE, "..", "results")
FIGURES_DIR = os.path.join(HERE, "..", "figures")

OUTCOMES = ["H", "D", "A"]

# Deciles over the body of the distribution, finer bins in both tails where the
# favourite-longshot bias lives.
BIN_EDGES = np.array(
    [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40,
     0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]
)

Z = stats.norm.ppf(0.975)


def wilson_interval(successes, trials, z=Z):
    """Wilson score interval for a binomial proportion."""
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    halfwidth = z * np.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials ** 2)) / denom
    return centre - halfwidth, centre + halfwidth


def shin_probabilities(raw):
    """Shin (1993) probabilities for one book.

    `raw` is the vector of gross implied probabilities (1/odds), summing to
    1 + overround. Solves for the insider share z that makes the implied true
    probabilities sum to one.
    """
    raw = np.asarray(raw, dtype=float)
    total = raw.sum()

    def implied(z):
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * raw * raw / total) - z) / (2.0 * (1.0 - z))

    def gap(z):
        return implied(z).sum() - 1.0

    lo, hi = 1e-10, 0.5
    if gap(lo) * gap(hi) > 0:
        # No sign change (degenerate book); fall back to proportional.
        return raw / total, 0.0
    z_hat = optimize.brentq(gap, lo, hi, xtol=1e-12)
    probs = implied(z_hat)
    return probs / probs.sum(), z_hat


def add_probabilities(matches):
    """Attach raw, proportional and Shin probabilities plus the overround."""
    odds = matches[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    raw = 1.0 / odds
    total = raw.sum(axis=1)

    matches = matches.copy()
    matches["overround"] = total - 1.0
    for i, outcome in enumerate(OUTCOMES):
        matches["raw_" + outcome] = raw[:, i]
        matches["prop_" + outcome] = raw[:, i] / total

    shin = np.empty_like(raw)
    z_values = np.empty(len(raw))
    for i in range(len(raw)):
        shin[i, :], z_values[i] = shin_probabilities(raw[i, :])
    for i, outcome in enumerate(OUTCOMES):
        matches["shin_" + outcome] = shin[:, i]
    matches["shin_z"] = z_values
    return matches


def report_overround(matches):
    """Print and persist the overround distribution by season and odds source."""
    by_season = matches.groupby("season").agg(
        n=("overround", "size"),
        mean_overround=("overround", "mean"),
        median_overround=("overround", "median"),
        sd_overround=("overround", "std"),
        mean_shin_z=("shin_z", "mean"),
        source=("odds_source", lambda s: s.mode().iat[0]),
    )
    by_source = matches.groupby("odds_source").agg(
        n=("overround", "size"),
        mean_overround=("overround", "mean"),
        median_overround=("overround", "median"),
        sd_overround=("overround", "std"),
        mean_shin_z=("shin_z", "mean"),
    )

    print("Overround by season (fraction of stake):")
    print(by_season.round(4).to_string())
    print("\nOverround by odds source:")
    print(by_source.round(4).to_string())

    by_season.round(6).to_csv(os.path.join(RESULTS_DIR, "overround.csv"))
    return by_season, by_source


def stack_outcomes(matches, prefix):
    """Long frame of (implied probability, decimal odds, realised 0/1) triples."""
    parts = []
    odds_cols = {"H": "odds_h", "D": "odds_d", "A": "odds_a"}
    for outcome in OUTCOMES:
        parts.append(
            pd.DataFrame(
                {
                    "season": matches["season"].to_numpy(),
                    "odds_source": matches["odds_source"].to_numpy(),
                    "outcome": outcome,
                    "p": matches[prefix + "_" + outcome].to_numpy(),
                    "odds": matches[odds_cols[outcome]].to_numpy(),
                    "win": (matches["FTR"] == outcome).astype(int).to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def bin_table(pairs):
    """Aggregate (implied, realised) pairs into the fixed probability bins."""
    pairs = pairs.copy()
    pairs["bin"] = pd.cut(pairs["p"], BIN_EDGES, include_lowest=True)
    # Return on a 1-unit stake: odds - 1 if the outcome lands, -1 otherwise.
    pairs["ret"] = pairs["win"] * pairs["odds"] - 1.0

    grouped = pairs.groupby("bin", observed=True)
    table = grouped.agg(
        n=("win", "size"),
        wins=("win", "sum"),
        mean_implied=("p", "mean"),
        mean_odds=("odds", "mean"),
        mean_return=("ret", "mean"),
        sd_return=("ret", "std"),
    )
    table["realized"] = table["wins"] / table["n"]
    table["ci_lo"], table["ci_hi"] = wilson_interval(table["wins"], table["n"])
    table["se_return"] = table["sd_return"] / np.sqrt(table["n"])
    table["ret_ci_lo"] = table["mean_return"] - Z * table["se_return"]
    table["ret_ci_hi"] = table["mean_return"] + Z * table["se_return"]
    return table.reset_index()


def calibration_fit(pairs):
    """Observation-level linear probability fit of realisation on implied prob."""
    result = stats.linregress(pairs["p"], pairs["win"])
    return result.slope, result.intercept, result.stderr


def plot_calibration(prop_table, shin_table, path):
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    ax.plot([0, 1], [0, 1], color="0.35", linestyle="--", linewidth=1.0,
            label="perfect calibration", zorder=1)

    # Clip at zero: at a boundary proportion the Wilson endpoint coincides with
    # the point estimate and rounding can make the arm marginally negative.
    yerr = np.vstack([
        (prop_table["realized"] - prop_table["ci_lo"]).clip(lower=0.0),
        (prop_table["ci_hi"] - prop_table["realized"]).clip(lower=0.0),
    ])
    ax.errorbar(
        prop_table["mean_implied"], prop_table["realized"], yerr=yerr,
        fmt="o", color="#1b3a6b", ecolor="#1b3a6b", elinewidth=1.1,
        capsize=3, markersize=5, zorder=3, label="proportional (95% Wilson CI)",
    )
    ax.plot(shin_table["mean_implied"], shin_table["realized"], "s",
            color="#b3541e", markersize=4.5, alpha=0.85, zorder=2, label="Shin")

    ax.set_xlabel("Implied probability (overround stripped)")
    ax.set_ylabel("Realised frequency")
    ax.set_title("Reliability of Premier League 1X2 odds, 2005/06–2024/25")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_flb(table, path):
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, zorder=1)
    ax.errorbar(
        table["mean_implied"], table["mean_return"],
        yerr=Z * table["se_return"],
        fmt="o-", color="#1b3a6b", ecolor="#1b3a6b", elinewidth=1.1,
        capsize=3, markersize=5, linewidth=1.2, zorder=3,
    )
    # The tail bins are thin; label n so the wide intervals read as sampling error.
    for _, row in table.iterrows():
        ax.annotate(
            "n={:,}".format(int(row["n"])),
            (row["mean_implied"], row["mean_return"]),
            textcoords="offset points", xytext=(0, 9), ha="center",
            fontsize=6.5, color="0.35", rotation=45,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Implied probability (log scale)")
    ax.set_ylabel("Mean return per unit staked")
    ax.set_title("Favourite–longshot bias in Premier League 1X2 markets")
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

    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    matches = add_probabilities(matches)
    print("{} matches, {} to {}\n".format(
        len(matches), matches["date"].min().date(), matches["date"].max().date()))

    report_overround(matches)

    prop_sums = matches[["prop_H", "prop_D", "prop_A"]].sum(axis=1)
    shin_sums = matches[["shin_H", "shin_D", "shin_A"]].sum(axis=1)
    assert np.allclose(prop_sums, 1.0), "proportional probabilities do not sum to 1"
    assert np.allclose(shin_sums, 1.0), "Shin probabilities do not sum to 1"

    mean_overround = matches["overround"].mean()
    print("\nmean overround: {:.4f}   median: {:.4f}".format(
        mean_overround, matches["overround"].median()))
    assert 0.005 < mean_overround < 0.12, "mean overround outside plausible band"

    prop_pairs = stack_outcomes(matches, "prop")
    shin_pairs = stack_outcomes(matches, "shin")
    prop_table = bin_table(prop_pairs)
    shin_table = bin_table(shin_pairs)

    slope, intercept, slope_se = calibration_fit(prop_pairs)
    print("\nlinear probability fit (realised on implied): "
          "slope={:.4f} (se {:.4f}), intercept={:.4f}".format(slope, slope_se, intercept))

    mean_abs_dev = np.average(
        np.abs(prop_table["realized"] - prop_table["mean_implied"]), weights=prop_table["n"]
    )
    print("n-weighted mean absolute calibration deviation: {:.4f}".format(mean_abs_dev))

    print("\nCalibration table (proportional probabilities):")
    display = prop_table[["bin", "n", "mean_implied", "realized", "ci_lo", "ci_hi",
                          "mean_odds", "mean_return"]]
    print(display.round(4).to_string(index=False))

    combined = prop_table.copy()
    combined.insert(0, "method", "proportional")
    shin_out = shin_table.copy()
    shin_out.insert(0, "method", "shin")
    pd.concat([combined, shin_out], ignore_index=True).round(6).to_csv(
        os.path.join(RESULTS_DIR, "calibration.csv"), index=False
    )

    plot_calibration(prop_table, shin_table, os.path.join(FIGURES_DIR, "calibration.png"))
    plot_flb(prop_table, os.path.join(FIGURES_DIR, "flb.png"))
    print("\nwrote results/calibration.csv, results/overround.csv, "
          "figures/calibration.png, figures/flb.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
