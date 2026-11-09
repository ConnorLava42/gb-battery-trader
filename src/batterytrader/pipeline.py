"""Download raw data year by year, then build the market table everything else reads."""

from datetime import date, timedelta

import pandas as pd

from batterytrader.config import (
    DATA_END,
    HISTORY_START,
    PRICE_COLUMN,
    PROCESSED_DIR,
    RAW_DIR,
    TEST_END,
    TEST_START,
)
from batterytrader.data import elexon, weather
from batterytrader.data.clean import build_market_table

MARKET = PROCESSED_DIR / "market.parquet"
WEATHER_START = date(2021, 1, 1)  # Open-Meteo's archive of past forecasts starts in 2021
SOURCES = {
    "mid": elexon.fetch_mid,
    "system_prices": elexon.fetch_system_prices,
    "weather": weather.fetch_weather,
}


def raw_path(source: str, year: int):
    return RAW_DIR / f"{source}_{year}.parquet"


def _year_range(year: int, start: date, end: date) -> tuple[date, date]:
    """The part of [start, end) that falls in `year`."""
    return max(start, date(year, 1, 1)), min(end, date(year + 1, 1, 1))


def ingest(start: str = HISTORY_START, end: str = DATA_END, force: bool = False) -> None:
    """Download every source, one file per source per year.

    A year already on disk is skipped, so if a download fails halfway you can
    run the command again and it carries on where it stopped.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    start_d, end_d = date.fromisoformat(start), date.fromisoformat(end)
    for year in range(start_d.year, end_d.year + 1):
        lo, hi = _year_range(year, start_d, end_d)
        if lo >= hi:
            continue
        for name, fetch in SOURCES.items():
            path = raw_path(name, year)
            if (path.exists() and not force) or (name == "weather" and hi <= WEATHER_START):
                continue
            print(f"Downloading {name} for {year}...")
            if name == "weather":
                # Open-Meteo's end date is inclusive, Elexon's is exclusive
                df = fetch(max(lo, WEATHER_START), hi - timedelta(days=1))
            else:
                df = fetch(lo, hi)
            df.to_parquet(path, index=False)


def _load_raw(source: str) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob(f"{source}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {source} files in {RAW_DIR}: run `batterytrader ingest`")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def build(start: str = HISTORY_START, end: str = DATA_END) -> pd.DataFrame:
    """Combine the raw files into data/processed/market.parquet."""
    weather_long = _load_raw("weather").drop_duplicates(["time", "site"])
    table, report = build_market_table(
        _load_raw("mid"), _load_raw("system_prices"), weather.summarise(weather_long), start, end
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(MARKET)
    print(f"Saved {len(table):,} half-hours to {MARKET}")
    print("Missing values filled per column:", report)
    return table


def load_market() -> pd.DataFrame:
    return pd.read_parquet(MARKET)


def prices_for(
    table: pd.DataFrame, market: str, start: str = TEST_START, end: str = TEST_END
) -> pd.Series:
    """One market's prices between start and end (by default the test year)."""
    t = table.index
    inside = (t >= pd.Timestamp(start, tz="UTC")) & (t < pd.Timestamp(end, tz="UTC"))
    return table.loc[inside, PRICE_COLUMN[market]]
