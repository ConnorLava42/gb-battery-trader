"""Turn the raw downloads into one tidy half-hourly market table."""

import pandas as pd

from batterytrader.config import PERIODS_PER_DAY


def half_hour_grid(start: str, end: str) -> pd.DatetimeIndex:
    """Every half-hour start time in UTC, start inclusive, end exclusive.

    UTC days always have exactly 48 periods, which keeps the maths simple
    (UK settlement days have 46 or 50 on clock-change days).
    """
    return pd.date_range(
        pd.Timestamp(start, tz="UTC"),
        pd.Timestamp(end, tz="UTC"),
        freq="30min",
        inclusive="left",
        name="start_time",
    )


def fill_gaps(s: pd.Series, max_gap: int = 4) -> pd.Series:
    """Fill missing prices so every period can be traded and scored.

    Gaps of up to `max_gap` half-hours are filled by a straight line between
    their neighbours. Longer gaps take the price from the same time a week
    earlier, which keeps the daily shape instead of drawing a flat line.
    """
    missing = s.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("size")
    short = missing & (run_len <= max_gap)
    s = s.where(~short, s.interpolate(limit_area="inside"))
    for _ in range(4):  # a gap longer than a week needs more than one pass
        s = s.fillna(s.shift(7 * PERIODS_PER_DAY))
    return s.ffill().bfill()


def build_market_table(
    mid: pd.DataFrame, system: pd.DataFrame, weather: pd.DataFrame, start: str, end: str
) -> tuple[pd.DataFrame, dict]:
    """One row per half-hour: both prices, imbalance volume and the weather.

    Returns the table and a report of how many values were filled.
    """
    grid = half_hour_grid(start, end)
    table = pd.DataFrame(index=grid)
    table["mid_price"] = mid.set_index("start_time")["mid_price"].reindex(grid)
    table["system_price"] = system.set_index("start_time")["system_price"].reindex(grid)
    table["niv"] = system.set_index("start_time")["niv"].reindex(grid)

    # Weather is hourly: put it on the half-hour grid and draw a line between hours
    hourly = weather.set_index("time").sort_index()
    both = hourly.reindex(hourly.index.union(grid)).interpolate(method="time", limit=2)
    for col in ["wind_speed", "temperature", "solar_radiation"]:
        table[col] = both[col].reindex(grid)

    report = {col: int(table[col].isna().sum()) for col in table.columns}
    for col in ["mid_price", "system_price"]:
        table[col] = fill_gaps(table[col])
    for col in ["wind_speed", "temperature", "solar_radiation"]:
        table[col] = table[col].ffill().bfill()
    return table, report
