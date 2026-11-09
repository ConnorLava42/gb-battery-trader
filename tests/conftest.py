"""Shared test data: a synthetic market that behaves roughly like GB prices."""

import numpy as np
import pandas as pd
import pytest

from batterytrader.config import DATA_END, HISTORY_START, UK


def synthetic_weather(t: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    local = t.tz_convert(UK)
    hour = (local.hour + local.minute / 60).to_numpy()
    doy = local.day_of_year.to_numpy()
    # Wind: a slow random walk, kept between 0 and 25 m/s
    wind = np.clip(8 + np.cumsum(rng.normal(0, 0.15, len(t))) % 20 - 4, 0, 25)
    temp = 10 + 7 * np.sin(2 * np.pi * (doy - 110) / 365) + 3 * np.sin(2 * np.pi * (hour - 9) / 24)
    sun = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * (
        300 + 250 * np.sin(2 * np.pi * (doy - 80) / 365)
    )
    return pd.DataFrame({"wind_speed": wind, "temperature": temp, "solar_radiation": sun}, index=t)


def synthetic_prices(t: pd.DatetimeIndex, weather: pd.DataFrame, seed: int = 0):
    """Two daily peaks, cheaper at weekends and when windy, plus noise and spikes."""
    rng = np.random.default_rng(seed + 1)
    local = t.tz_convert(UK)
    hour = (local.hour + local.minute / 60).to_numpy()
    shape = 15 * np.exp(-((hour - 8) ** 2) / 4) + 30 * np.exp(-((hour - 18) ** 2) / 3)
    weekend = np.isin(local.day_of_week, [5, 6]) * -10
    mid = 70 + shape + weekend - 2.5 * weather["wind_speed"].to_numpy() + rng.normal(0, 6, len(t))
    spikes = rng.random(len(t)) < 0.01
    system = mid + rng.normal(0, 15, len(t)) + spikes * rng.normal(0, 150, len(t))
    return mid, system


def make_market(start: str = HISTORY_START, end: str = DATA_END, seed: int = 0) -> pd.DataFrame:
    t = pd.date_range(
        pd.Timestamp(start, tz="UTC"),
        pd.Timestamp(end, tz="UTC"),
        freq="30min",
        inclusive="left",
        name="start_time",
    )
    weather = synthetic_weather(t, seed)
    mid, system = synthetic_prices(t, weather, seed)
    table = pd.DataFrame({"mid_price": mid, "system_price": system, "niv": 0.0}, index=t)
    return table.join(weather)


@pytest.fixture(scope="session")
def market() -> pd.DataFrame:
    """The whole history, 2020-12 to 2026-01: about 89,000 half-hours."""
    return make_market()
