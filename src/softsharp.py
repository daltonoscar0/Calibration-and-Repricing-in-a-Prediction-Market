"""Soft-versus-sharp backtest: does a soft bookmaker's price beat the sharp price?

Produces:
  results/softsharp.csv   realised return by comparison, threshold and era
  figures/softsharp.png   realised return versus staking threshold

Method. Pinnacle is treated as the sharp book: its overround-stripped implied
probabilities are the best available estimate of the true probability. A soft
book's decimal price o on the same outcome is a positive-expectation bet when
p_sharp * o - 1 exceeds a staking threshold. Every qualifying outcome is backed
for one unit and the realised return is win * o - 1.

Timing discipline. A soft price and the sharp price it is compared against must
be sampled at the same moment, or the comparison uses information that did not
exist when the soft price was posted. Two clean comparisons are therefore run:
closing against closing (all books carry closing quotes from 2019/20), and
early against early (Pinnacle and Bet365 carry pre-closing quotes from 2012/13).
A deliberately contaminated look-ahead variant is reported alongside to show the
size of the bias that timing sloppiness introduces.
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RAW_DIR = os.path.join(HERE, "..", "data", "raw")
RESULTS_DIR = os.path.join(HERE, "..", "results")
FIGURES_DIR = os.path.join(HERE, "..", "figures")

OUTCOMES = ["H", "D", "A"]
THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05]
N_BOOTSTRAP = 2000
SEED = 20240804

# (label, sharp prefix, soft prefix, timing, note)
COMPARISONS = [
    ("B365 close vs Pinnacle close", "PSC", "B365C", "matched", "clean"),
    ("Best close vs Pinnacle close", "PSC", "MaxC", "matched", "Max panel includes Pinnacle"),
    ("B365 early vs Pinnacle early", "PS", "B365", "matched", "clean"),
    ("Best early vs Pinnacle early", "PS", "BEST", "matched", "Max panel includes Pinnacle"),
    ("B365 early vs Pinnacle CLOSE", "PSC", "B365", "look-ahead", "contaminated on purpose"),
]


def read_raw():
    """Concatenate the raw season files, keeping every odds column."""
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "E0_*.csv"))):
        code = os.path.basename(path)[3:7]
        for encoding in ("utf-8", "latin-1"):
            try:
                frame = pd.read_csv(path, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        frame.columns = [str(c).strip() for c in frame.columns]
        frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
        frame = frame.dropna(subset=["Date"])
        # The best-available early price was renamed from BbMx* to Max* in
        # 2019/20; BEST* is one column spanning both naming eras.
        added = {"season": "20{}-{}".format(code[:2], code[2:])}
        for outcome in OUTCOMES:
            source = "Max" + outcome if "Max" + outcome in frame.columns else "BbMx" + outcome
            if source in frame.columns:
                added["BEST" + outcome] = frame[source].to_numpy()
        frames.append(pd.concat([frame, pd.DataFrame(added, index=frame.index)], axis=1))
    return pd.concat(frames, ignore_index=True)


def columns_for(prefix):
    return [prefix + outcome for outcome in OUTCOMES]


def build_bets(raw, sharp_prefix, soft_prefix):
    """One row per (match, outcome) where both books quote a usable price."""
    sharp_cols = columns_for(sharp_prefix)
    soft_cols = columns_for(soft_prefix)
    if not all(c in raw.columns for c in sharp_cols + soft_cols):
        return pd.DataFrame()

    sharp = raw[sharp_cols].apply(pd.to_numeric, errors="coerce")
    soft = raw[soft_cols].apply(pd.to_numeric, errors="coerce")

    usable = (
        sharp.notna().all(axis=1)
        & soft.notna().all(axis=1)
        & (sharp > 1.0).all(axis=1)
        & (soft > 1.0).all(axis=1)
        & raw["FTR"].isin(OUTCOMES)
    )
    sharp, soft, sub = sharp[usable], soft[usable], raw[usable]

    inverse = 1.0 / sharp.to_numpy()
    p_sharp = inverse / inverse.sum(axis=1, keepdims=True)
    soft_odds = soft.to_numpy()

    rows = []
    match_id = np.arange(len(sub))
    for i, outcome in enumerate(OUTCOMES):
        rows.append(
            pd.DataFrame(
                {
                    "match_id": match_id,
                    "season": sub["season"].to_numpy(),
                    "outcome": outcome,
                    "p_sharp": p_sharp[:, i],
                    "o_soft": soft_odds[:, i],
                    "o_sharp": sharp.to_numpy()[:, i],
                    "win": (sub["FTR"].to_numpy() == outcome).astype(int),
                }
            )
        )
    bets = pd.concat(rows, ignore_index=True)
    bets["edge"] = bets["p_sharp"] * bets["o_soft"] - 1.0
    bets["ret"] = bets["win"] * bets["o_soft"] - 1.0
    return bets


def bootstrap_return(bets, n_boot=N_BOOTSTRAP, seed=SEED):
    """Percentile bootstrap of mean return, resampling whole matches."""
    if bets.empty:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    by_match = {}
    for match_id, ret in zip(bets["match_id"].to_numpy(), bets["ret"].to_numpy()):
        by_match.setdefault(match_id, []).append(ret)
    keys = list(by_match)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(keys), size=len(keys))
        pooled = np.concatenate([by_match[keys[i]] for i in picked])
        draws[b] = pooled.mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def evaluate(label, bets, note):
    """Threshold sweep for one comparison."""
    rows = []
    for threshold in THRESHOLDS:
        selected = bets[bets["edge"] > threshold]
        if len(selected) < 30:
            rows.append({"comparison": label, "threshold": threshold, "n_bets": len(selected),
                         "note": note})
            continue
        lo, hi = bootstrap_return(selected)
        rows.append(
            {
                "comparison": label,
                "threshold": threshold,
                "n_bets": len(selected),
                "n_matches": selected["match_id"].nunique(),
                "mean_edge": selected["edge"].mean(),
                "mean_return": selected["ret"].mean(),
                "ci_lo": lo,
                "ci_hi": hi,
                "hit_rate": selected["win"].mean(),
                "note": note,
            }
        )
    return rows


def plot_curves(table, path):
    clean = table[table["note"] == "clean"].dropna(subset=["mean_return"])
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, zorder=1)
    colors = {"B365 close vs Pinnacle close": "#1b3a6b", "B365 early vs Pinnacle early": "#b3541e"}
    for label, group in clean.groupby("comparison"):
        ax.errorbar(
            group["threshold"], group["mean_return"],
            yerr=[group["mean_return"] - group["ci_lo"], group["ci_hi"] - group["mean_return"]],
            fmt="o-", capsize=3, markersize=5, linewidth=1.3,
            color=colors.get(label, "0.4"), ecolor=colors.get(label, "0.4"),
            elinewidth=1.0, label=label,
        )
        for _, row in group.iterrows():
            ax.annotate("n={:,}".format(int(row["n_bets"])),
                        (row["threshold"], row["mean_return"]),
                        textcoords="offset points", xytext=(0, 10), ha="center",
                        fontsize=6.5, color="0.35")
    ax.set_xlabel("Minimum edge required to stake")
    ax.set_ylabel("Realised return per unit staked")
    ax.set_title("Backing soft prices against the sharp price (timing-matched)")
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    raw = read_raw()
    print("{} raw match rows across {} seasons\n".format(len(raw), raw["season"].nunique()))

    all_rows = []
    for label, sharp_prefix, soft_prefix, timing, note in COMPARISONS:
        bets = build_bets(raw, sharp_prefix, soft_prefix)
        if bets.empty:
            print("{}: no overlapping quotes, skipped".format(label))
            continue

        seasons = sorted(bets["season"].unique())
        print("{}\n  {} bets over {} matches, {} to {}".format(
            label, len(bets), bets["match_id"].nunique(), seasons[0], seasons[-1]))
        soft_better = (bets["o_soft"] > bets["o_sharp"]).mean()
        print("  soft price beats sharp price on {:.1%} of outcomes".format(soft_better))
        print("  positive-edge outcomes: {:.1%}".format((bets["edge"] > 0).mean()))

        rows = evaluate(label, bets, note)
        for row in rows:
            if "mean_return" in row:
                print("    edge>{:.2f}  n={:6d}  expected={:+.4f}  realised={:+.4f}  "
                      "[{:+.4f}, {:+.4f}]".format(
                          row["threshold"], row["n_bets"], row["mean_edge"],
                          row["mean_return"], row["ci_lo"], row["ci_hi"]))
        all_rows.extend(rows)
        print()

    table = pd.DataFrame(all_rows)
    table.to_csv(os.path.join(RESULTS_DIR, "softsharp.csv"), index=False)
    plot_curves(table, os.path.join(FIGURES_DIR, "softsharp.png"))
    print("wrote results/softsharp.csv, figures/softsharp.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
