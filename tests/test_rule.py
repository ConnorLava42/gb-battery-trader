"""The rule: yesterday's cheapest and dearest times, and never better than foresight."""

import numpy as np
import pandas as pd
import pytest

from batterytrader.battery import simulate
from batterytrader.strategies.optimiser import perfect_foresight
from batterytrader.strategies.rule import periods_to_fill, rule_schedule


def test_rule_uses_yesterdays_cheapest_and_dearest_times():
    t = pd.date_range("2025-01-01", periods=96, freq="30min", tz="UTC")
    day = np.full(48, 50.0)
    day[[4, 5, 6, 7]], day[[36, 37, 38, 39]] = 10.0, 150.0
    plan = rule_schedule(pd.Series(np.tile(day, 2), index=t))
    assert (plan.iloc[:48] == 0).all()  # no yesterday on day one
    today = plan.iloc[48:].to_numpy()
    assert periods_to_fill() == 4
    assert set(np.flatnonzero(today < 0)) == {4, 5, 6, 7}
    assert set(np.flatnonzero(today > 0)) == {36, 37, 38, 39}


def test_rule_skips_flat_days_and_needs_whole_days():
    t = pd.date_range("2025-01-01", periods=96, freq="30min", tz="UTC")
    assert (rule_schedule(pd.Series(50.0, index=t)) == 0).all()
    with pytest.raises(ValueError):
        rule_schedule(pd.Series(50.0, index=t[:-1]))


def test_rule_makes_money_but_never_beats_foresight(market):
    prices = market.loc["2025-03-01":"2025-03-31", "mid_price"]
    rule = simulate(prices, rule_schedule(prices))["profit"].sum()
    best = simulate(prices, perfect_foresight(prices))["profit"].sum()
    assert 0 < rule < best
