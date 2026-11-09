"""Downloading and cleaning, all against the fake APIs."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from fake_apis import fake_get_json

from batterytrader import pipeline
from batterytrader.data import elexon, weather
from batterytrader.data.clean import build_market_table, fill_gaps, half_hour_grid


@pytest.fixture
def fake_network(monkeypatch):
    for module in (elexon, weather):
        monkeypatch.setattr(module, "get_json", fake_get_json)
    monkeypatch.setattr(weather.time, "sleep", lambda _: None)


def test_mid_chunks_join_without_duplicates(fake_network):
    mid = elexon.fetch_mid(date(2025, 1, 1), date(2025, 1, 20), chunk_days=7)
    assert len(mid) == 19 * 48  # chunk edges overlap, but each half-hour appears once
    assert mid["start_time"].is_monotonic_increasing
    assert mid["mid_price"].isna().sum() == 3  # one no-trade period per chunk becomes missing


def test_system_prices_one_request_per_day(fake_network):
    sp = elexon.fetch_system_prices(date(2025, 3, 1), date(2025, 3, 4))
    assert len(sp) == 3 * 48
    assert list(sp.columns) == ["start_time", "system_price", "niv"]


def test_weather_summary(fake_network):
    long = weather.fetch_weather(date(2025, 1, 1), date(2025, 1, 2))
    summary = weather.summarise(long)
    assert len(summary) == 48  # two days, hourly
    assert list(summary.columns) == ["time", "wind_speed", "temperature", "solar_radiation"]


def test_fill_gaps_short_by_line_long_by_last_week():
    t = half_hour_grid("2025-01-01", "2025-01-15")
    s = pd.Series(np.arange(len(t), dtype=float), index=t)
    s.iloc[400:402] = np.nan  # short gap: straight line
    s.iloc[500:520] = np.nan  # long gap: same time last week
    filled = fill_gaps(s)
    assert filled.iloc[400] == 400 and filled.iloc[401] == 401
    assert filled.iloc[510] == 510 - 336
    assert filled.notna().all()


def test_market_table_is_a_complete_grid():
    t = half_hour_grid("2025-01-01", "2025-01-03")
    mid = pd.DataFrame({"start_time": t[:-5], "mid_price": 50.0, "mid_volume": 1.0})
    system = pd.DataFrame({"start_time": t, "system_price": 60.0, "niv": 0.0})
    hourly = pd.date_range(t[0], t[-1], freq="h")
    w = pd.DataFrame(
        {"time": hourly, "wind_speed": 5.0, "temperature": 8.0, "solar_radiation": 0.0}
    )
    table, report = build_market_table(mid, system, w, "2025-01-01", "2025-01-03")
    assert len(table) == 96 and table.notna().all().all()
    assert report["mid_price"] == 5


def test_ingest_then_build(fake_network, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(pipeline, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(pipeline, "MARKET", tmp_path / "processed" / "market.parquet")
    pipeline.ingest("2020-12-25", "2021-01-05")
    names = sorted(p.name for p in (tmp_path / "raw").iterdir())
    # No weather file for 2020: Open-Meteo's archive of past forecasts starts in 2021
    assert names == [
        "mid_2020.parquet",
        "mid_2021.parquet",
        "system_prices_2020.parquet",
        "system_prices_2021.parquet",
        "weather_2021.parquet",
    ]
    table = pipeline.build("2020-12-25", "2021-01-05")
    assert len(table) == 11 * 48 and table.notna().all().all()
