"""The scorecard: names, capture rates and the mean over RL seeds."""

import pytest

from batterytrader.battery import simulate
from batterytrader.evaluate import label, summary_table
from batterytrader.strategies.optimiser import perfect_foresight
from batterytrader.strategies.rule import rule_schedule


def test_labels():
    assert label("rl_dqn_seed2") == "RL: DQN (seed 2)"
    assert label("forecast_lp") == "Forecast + optimiser"


def test_summary_table(market):
    prices = market.loc["2025-03-01":"2025-03-31", "mid_price"]
    best = simulate(prices, perfect_foresight(prices))
    rule = simulate(prices, rule_schedule(prices))
    sims = {"perfect_foresight": best, "rule": rule, "rl_dqn_seed0": rule, "rl_dqn_seed1": rule}
    table = summary_table(sims)
    assert table.loc["Perfect foresight (upper bound)", "capture rate (%)"] == 100
    assert 0 < table.loc["Rule: yesterday's pattern", "capture rate (%)"] < 100
    mean_row = table.loc[table.index.str.startswith("RL: DQN, mean of 2 seeds")]
    assert len(mean_row) == 1
    assert mean_row["capture rate (%)"].iloc[0] == pytest.approx(
        table.loc["Rule: yesterday's pattern", "capture rate (%)"]
    )
