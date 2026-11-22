"""The price forecaster: no peeking at the future, and better than a naive guess."""

import numpy as np
import pandas as pd
import pytest

from batterytrader.forecasting import (
    FEATURES,
    HORIZON,
    WEEK,
    accuracy_by_horizon,
    feature_table,
    issue_positions,
    to_matrix,
)


def test_no_leakage(market):
    """Scramble every price from the issue time onwards: no feature may change."""
    t = market.index.get_loc(pd.Timestamp("2025-06-01 12:00", tz="UTC"))
    honest = feature_table(market, "mid_price", np.array([t]))
    scrambled = market.copy()
    rng = np.random.default_rng(0)
    scrambled.iloc[t:, 0] = rng.normal(500, 300, len(market) - t)  # column 0 = mid_price
    cheat = feature_table(scrambled, "mid_price", np.array([t]))
    pd.testing.assert_frame_equal(honest[FEATURES], cheat[FEATURES])
    assert not np.allclose(honest["target"], cheat["target"])  # the targets did change


def test_feature_table_shape(market):
    pos = issue_positions(market, "2025-01-01", "2025-01-02", targets_inside=False)
    table = feature_table(market, "mid_price", pos)
    assert len(table) == 48 * HORIZON
    assert table[FEATURES].notna().all().all()
    assert to_matrix(table, table["target"].to_numpy()).shape == (48, HORIZON)


def test_training_targets_stay_inside_the_window(market):
    pos = issue_positions(market, "2021-01-01", "2024-01-01")
    end = market.index.get_loc(pd.Timestamp("2024-01-01", tz="UTC"))
    assert pos.max() + HORIZON <= end
    assert pos.min() >= WEEK
    with pytest.raises(ValueError):
        feature_table(market, "mid_price", np.array([10]))  # no week of history yet


def test_model_beats_same_time_yesterday(market):
    from batterytrader.forecasting import forecast_test, train_model

    model = train_model(market, "mid_price", every=48)  # one issue time a day: quick
    forecasts = forecast_test(model, market, "mid_price")
    assert len(forecasts) == 365 * 48
    acc = accuracy_by_horizon(market, "mid_price", forecasts)
    assert (acc["model"] < acc["same_time_yesterday"]).all()
