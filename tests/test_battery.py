"""The physics every strategy is scored with must be right, so test it hardest."""

import numpy as np
import pandas as pd
import pytest

from batterytrader.battery import simulate, start_energy, step
from batterytrader.config import BATTERY
from batterytrader.metrics import capture_by_regime, capture_rate, summarise

B = BATTERY


def test_cannot_charge_a_full_battery_or_empty_an_empty_one():
    power, energy, revenue, _ = step(B.e_max, -1.0, 50.0)
    assert power == 0 and energy == B.e_max and revenue == 0
    power, energy, revenue, wear = step(B.e_min, 1.0, 50.0)
    assert power == 0 and energy == B.e_min and wear == 0


def test_round_trip_loses_ten_percent():
    energy, bought = B.e_min, 0.0
    while energy < B.e_max - 1e-9:
        power, energy, _, _ = step(energy, -1.0, 0.0)
        bought -= power * 0.5
    sold = 0.0
    while energy > B.e_min + 1e-9:
        power, energy, _, _ = step(energy, 1.0, 0.0)
        sold += power * 0.5
    assert sold / bought == pytest.approx(0.90)
    assert bought == pytest.approx(B.usable_mwh / B.eta)  # more than 1.8 MWh to fill 1.8 MWh


def test_selling_earns_buying_costs():
    _, _, revenue, wear = step(start_energy(), 1.0, 80.0)
    assert revenue == pytest.approx(40.0) and wear == pytest.approx(5.0)  # 0.5 MWh sold
    _, _, revenue, wear = step(start_energy(), -1.0, 80.0)
    assert revenue == pytest.approx(-40.0) and wear == 0


def test_random_actions_never_break_the_limits():
    rng = np.random.default_rng(1)
    t = pd.date_range("2025-01-01", periods=5000, freq="30min", tz="UTC")
    prices = pd.Series(rng.normal(60, 40, len(t)), index=t)
    sim = simulate(prices, rng.uniform(-3, 3, len(t)))  # asks for impossible power too
    assert sim["power_mw"].abs().max() <= B.power_mw + 1e-9
    assert sim["energy_mwh"].between(B.e_min - 1e-9, B.e_max + 1e-9).all()
    assert sim["profit"].sum() == pytest.approx((sim["revenue"] - sim["degradation"]).sum())


def test_simulate_needs_one_action_per_price():
    t = pd.date_range("2025-01-01", periods=10, freq="30min", tz="UTC")
    with pytest.raises(ValueError):
        simulate(pd.Series(50.0, index=t), np.zeros(9))


def test_summary_counts_cycles():
    t = pd.date_range("2025-01-01", periods=48, freq="30min", tz="UTC")
    plan = np.zeros(48)
    plan[:4], plan[24:28] = -1.0, 1.0  # empty-ish to full-ish and back, once
    sim = simulate(pd.Series(50.0, index=t), plan, energy0=B.e_min)
    s = summarise(sim)
    assert 0.9 < s["cycles_per_day"] <= 1.0
    assert capture_rate(50, 200) == 25


def test_capture_by_regime():
    t = pd.date_range("2025-01-01", periods=8 * 48, freq="30min", tz="UTC")
    day = np.r_[np.full(24, 20.0), np.full(24, 80.0)]
    prices = pd.Series(np.concatenate([day * k for k in range(1, 9)]), index=t)  # wilder each day
    best = simulate(prices, np.tile(np.r_[np.full(24, -1.0), np.full(24, 1.0)], 8))
    regimes = capture_by_regime({"best": best}, best)
    assert regimes.loc["best", "calm"] == pytest.approx(100)
    assert regimes.loc["best", "volatile"] == pytest.approx(100)
