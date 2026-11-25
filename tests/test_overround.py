"""Odds-to-probability normalisation: stripping the bookmaker's overround.

Every expected value here is computed by hand from the odds, as an exact
fraction wherever one exists, and written down before the code was run. The
point is for these cases to be able to find the implementation wrong rather
than agree with whatever it currently returns.

Worked arithmetic for the asymmetric case, odds 1.50 / 4.00 / 7.00:

    raw        = 1/1.5, 1/4, 1/7      = 2/3, 1/4, 1/7
    total      = 2/3 + 1/4 + 1/7      = (56 + 21 + 12)/84 = 89/84
    overround  = 89/84 - 1            = 5/84      = 0.0595238095...
    prop_H     = (2/3) / (89/84)      = 56/89     = 0.6292134831...
    prop_D     = (1/4) / (89/84)      = 21/89     = 0.2359550562...
    prop_A     = (1/7) / (89/84)     = 12/89     = 0.1348314607...
"""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from calibration import OUTCOMES, add_probabilities, shin_probabilities

TOL = 1e-12


def frame(*books):
    """A matches frame carrying one row per (home, draw, away) odds triple."""
    return pd.DataFrame(
        [{"odds_h": h, "odds_d": d, "odds_a": a} for h, d, a in books]
    )


# --------------------------------------------------------------------------
# Proportional normalisation, against hand-computed fractions
# --------------------------------------------------------------------------


def test_asymmetric_book_matches_hand_computed_fractions():
    """The worked example in the module docstring, to the last digit."""
    out = add_probabilities(frame((1.5, 4.0, 7.0)))
    row = out.iloc[0]

    assert row["overround"] == pytest.approx(float(Fraction(5, 84)), abs=TOL)
    assert row["prop_H"] == pytest.approx(float(Fraction(56, 89)), abs=TOL)
    assert row["prop_D"] == pytest.approx(float(Fraction(21, 89)), abs=TOL)
    assert row["prop_A"] == pytest.approx(float(Fraction(12, 89)), abs=TOL)


def test_symmetric_book_strips_to_one_half_one_quarter_one_quarter():
    """1.90 / 3.80 / 3.80.

    1/1.9 = 2/3.8, so the gross probabilities are in the ratio 2 : 1 : 1 and
    proportional normalisation must give exactly 0.5 / 0.25 / 0.25. The
    overround is (4/3.8) - 1 = 1/19.
    """
    out = add_probabilities(frame((1.9, 3.8, 3.8)))
    row = out.iloc[0]

    assert row["overround"] == pytest.approx(float(Fraction(1, 19)), abs=TOL)
    assert row["prop_H"] == pytest.approx(0.50, abs=TOL)
    assert row["prop_D"] == pytest.approx(0.25, abs=TOL)
    assert row["prop_A"] == pytest.approx(0.25, abs=TOL)


@pytest.mark.parametrize(
    "book",
    [
        (2.0, 4.0, 4.0),   # 1/2 + 1/4 + 1/4 = 1
        (2.0, 3.0, 6.0),   # 1/2 + 1/3 + 1/6 = 1
        (1.5, 4.5, 9.0),   # 2/3 + 2/9 + 1/9 = 1
        (3.0, 3.0, 3.0),   # 1/3 x 3        = 1
    ],
)
def test_a_fair_book_has_zero_overround_and_is_left_alone(book):
    """When the gross probabilities already sum to one there is no margin to
    strip, so normalisation must be the identity."""
    out = add_probabilities(frame(book))
    row = out.iloc[0]

    assert row["overround"] == pytest.approx(0.0, abs=TOL)
    for outcome, odds in zip(OUTCOMES, book):
        assert row["prop_" + outcome] == pytest.approx(1.0 / odds, abs=TOL)


def test_overround_is_the_gross_sum_less_one():
    books = [(1.5, 4.0, 7.0), (1.9, 3.8, 3.8), (1.2, 7.0, 15.0), (2.5, 3.2, 2.9)]
    out = add_probabilities(frame(*books))

    for i, (h, d, a) in enumerate(books):
        expected = (1.0 / h) + (1.0 / d) + (1.0 / a) - 1.0
        assert out.iloc[i]["overround"] == pytest.approx(expected, abs=TOL)


def test_stripped_probabilities_sum_to_one():
    """The property every stage downstream asserts."""
    books = [(1.5, 4.0, 7.0), (1.9, 3.8, 3.8), (1.2, 7.0, 15.0), (6.0, 4.1, 1.6)]
    out = add_probabilities(frame(*books))

    for i in range(len(books)):
        row = out.iloc[i]
        prop = sum(row["prop_" + o] for o in OUTCOMES)
        shin = sum(row["shin_" + o] for o in OUTCOMES)
        assert prop == pytest.approx(1.0, abs=1e-12)
        assert shin == pytest.approx(1.0, abs=1e-10)


def test_proportional_normalisation_preserves_odds_ratios():
    """Dividing through by a constant cannot reorder anything, and the ratio
    between any two outcomes must survive untouched."""
    out = add_probabilities(frame((1.5, 4.0, 7.0)))
    row = out.iloc[0]

    assert row["prop_H"] / row["prop_D"] == pytest.approx(4.0 / 1.5, abs=1e-12)
    assert row["prop_D"] / row["prop_A"] == pytest.approx(7.0 / 4.0, abs=1e-12)
    assert row["prop_H"] > row["prop_D"] > row["prop_A"]


def test_scaling_every_price_by_the_same_factor_leaves_probabilities_unchanged():
    """A book with 1.5/4.0/7.0 and one with all three prices cut 10% carry the
    same information; only the margin differs."""
    base = add_probabilities(frame((1.5, 4.0, 7.0))).iloc[0]
    tight = add_probabilities(frame((1.35, 3.6, 6.3))).iloc[0]

    for outcome in OUTCOMES:
        assert tight["prop_" + outcome] == pytest.approx(
            base["prop_" + outcome], abs=1e-12
        )
    assert tight["overround"] > base["overround"]


# --------------------------------------------------------------------------
# Shin (1993)
# --------------------------------------------------------------------------


def test_shin_falls_back_to_proportional_on_a_fair_book():
    """With no margin there is no insider share to solve for, so z = 0 and
    Shin coincides with proportional."""
    out = add_probabilities(frame((2.0, 4.0, 4.0)))
    row = out.iloc[0]

    assert row["shin_z"] == pytest.approx(0.0, abs=1e-12)
    for outcome in OUTCOMES:
        assert row["shin_" + outcome] == pytest.approx(
            row["prop_" + outcome], abs=1e-12
        )


def test_shin_z_stays_in_its_bracket():
    books = [(1.5, 4.0, 7.0), (1.9, 3.8, 3.8), (1.2, 7.0, 15.0), (6.0, 4.1, 1.6)]
    out = add_probabilities(frame(*books))

    for i in range(len(books)):
        assert 0.0 <= out.iloc[i]["shin_z"] <= 0.5


def test_shin_shades_the_favourite_up_and_the_longshot_down():
    """The whole point of the adjustment: proportional normalisation spreads
    the margin evenly, Shin takes relatively more of it out of the longshots,
    which is the favourite-longshot bias it exists to correct."""
    out = add_probabilities(frame((1.5, 4.0, 7.0)))
    row = out.iloc[0]

    assert row["shin_H"] > row["prop_H"]
    assert row["shin_D"] < row["prop_D"]
    assert row["shin_A"] < row["prop_A"]


def test_shin_probabilities_returns_a_normalised_vector_and_a_z():
    raw = np.array([1.0 / 1.5, 1.0 / 4.0, 1.0 / 7.0])
    probs, z = shin_probabilities(raw)

    assert probs.sum() == pytest.approx(1.0, abs=1e-10)
    assert 0.0 < z < 0.5
    assert np.all(probs > 0.0)


def test_shin_is_degenerate_safe_on_an_even_three_way():
    """A perfectly even book has nothing for the solver to find; it must come
    back normalised rather than raising."""
    probs, z = shin_probabilities(np.array([1 / 3, 1 / 3, 1 / 3]))

    assert probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert probs == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-12)
    assert z == pytest.approx(0.0, abs=1e-12)
