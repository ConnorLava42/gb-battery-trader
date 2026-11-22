"""The perfect-foresight optimiser: right answers on cases we can check by hand."""

import numpy as np
import pandas as pd
import pytest

from batterytrader.battery import simulate
from batterytrader.config import BATTERY, DT_HOURS
from batterytrader.strategies.optimiser import leftover_value, optimise, perfect_foresight


def test_optimiser_buys_low_sells_high():
    prices = np.r_[np.full(8, 10.0), np.full(8, 100.0)]
    plan = optimise(prices, energy0=BATTERY.e_min)  # start empty
    assert (plan[:8] <= 1e-6).all() and plan[:8].sum() < -3.5  # fill up while cheap
    assert (plan[8:] >= -1e-6).all() and plan[8:].sum() > 3  # sell it all while dear


def test_optimiser_profit_matches_the_simulator(market):
    prices = market.loc["2025-03-03":"2025-03-09", "mid_price"]
    plan = perfect_foresight(prices)
    # Recompute the LP's own profit from its plan and compare with the physics
    lp_profit = (prices * plan * DT_HOURS).sum() - (
        plan.clip(lower=0) * DT_HOURS * BATTERY.degradation_gbp_per_mwh
    ).sum()
    assert simulate(prices, plan)["profit"].sum() == pytest.approx(lp_profit, rel=1e-3)


def test_perfect_foresight_beats_random_schedules(market):
    prices = market.loc["2025-03-03":"2025-03-09", "system_price"]
    best = simulate(prices, perfect_foresight(prices))["profit"].sum()
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert simulate(prices, rng.uniform(-1, 1, len(prices)))["profit"].sum() <= best


def test_leftover_value_stops_the_end_of_plan_dump():
    prices = np.full(10, 50.0)
    dump = optimise(prices, energy0=BATTERY.e_max)  # leftovers worth nothing: sell it all
    keep = optimise(prices, energy0=BATTERY.e_max, terminal_value=80.0)
    assert dump.sum() > 1.5 and abs(keep.sum()) < 1e-6
    assert leftover_value(pd.Series([40.0, 60.0])) == pytest.approx((50 - 10) * BATTERY.eta)
