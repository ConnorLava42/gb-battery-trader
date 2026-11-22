"""Forecast + optimiser: with a perfect forecast it should nearly match perfect foresight."""

import pandas as pd
import pytest

from batterytrader.battery import simulate
from batterytrader.strategies.mpc import rolling_optimise
from batterytrader.strategies.optimiser import perfect_foresight


def _week(market: pd.DataFrame) -> pd.Series:
    return market.loc["2025-03-03":"2025-03-09", "mid_price"]


def test_with_perfect_forecasts_it_nearly_matches_foresight(market):
    prices, future = _week(market), market["mid_price"]
    rows = [future.iloc[future.index.get_loc(t) :][:48].to_numpy() for t in prices.index]
    forecasts = pd.DataFrame(rows, index=prices.index)
    best = simulate(prices, perfect_foresight(prices))["profit"].sum()
    mpc = simulate(prices, rolling_optimise(prices, forecasts))["profit"].sum()
    assert mpc / best > 0.95  # only the 24-hour horizon and the end effects separate them


def test_needs_a_forecast_for_every_period(market):
    prices = _week(market)
    forecasts = pd.DataFrame(50.0, index=prices.index[:-1], columns=range(48))
    with pytest.raises(ValueError):
        rolling_optimise(prices, forecasts)
