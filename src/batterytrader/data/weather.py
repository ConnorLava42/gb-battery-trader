"""Archived weather forecasts from Open-Meteo, as the forecaster would have seen them.

The historical-forecast API stores past forecasts, so the price model trains on
the same kind of input it will have when trading. Free, no key, CC BY 4.0.
"""

import time
from datetime import date, timedelta

import pandas as pd

from batterytrader.config import DEMAND_SITES, WEATHER_VARS, WIND_SITES
from batterytrader.data.http import get_json

URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
SITES = WIND_SITES | DEMAND_SITES


def parse_response(payload) -> pd.DataFrame:
    """One Open-Meteo response -> long format: one row per site per hour.

    With several locations Open-Meteo returns a list in request order, so the
    position in the list tells us which site each block belongs to.
    """
    blocks = payload if isinstance(payload, list) else [payload]
    frames = []
    for site, block in zip(SITES, blocks, strict=True):
        hourly = pd.DataFrame(block["hourly"])
        hourly["time"] = pd.to_datetime(hourly["time"]).dt.tz_localize("UTC")
        hourly["site"] = site
        frames.append(hourly)
    return pd.concat(frames, ignore_index=True)


def fetch_weather(start: date, end: date, chunk_days: int = 90) -> pd.DataFrame:
    """Hourly weather at every site, start to end inclusive."""
    lats, lons = zip(*SITES.values(), strict=True)
    base = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "GMT",  # timestamps in UTC, matching Elexon
        "wind_speed_unit": "ms",
    }
    frames, cur = [], start
    while cur <= end:
        stop = min(cur + timedelta(days=chunk_days - 1), end)
        params = base | {"start_date": cur.isoformat(), "end_date": stop.isoformat()}
        frames.append(parse_response(get_json(URL, params)))
        cur = stop + timedelta(days=1)
        time.sleep(1)  # a multi-site request counts as several calls against the free limit
    return pd.concat(frames, ignore_index=True)


def summarise(long: pd.DataFrame) -> pd.DataFrame:
    """Three national numbers per hour: mean wind at the wind farms, and mean
    temperature and solar radiation where people live."""
    wind = long[long["site"].isin(WIND_SITES)].groupby("time")["wind_speed_100m"].mean()
    people = long[long["site"].isin(DEMAND_SITES)].groupby("time")
    out = pd.DataFrame(
        {
            "wind_speed": wind,
            "temperature": people["temperature_2m"].mean(),
            "solar_radiation": people["shortwave_radiation"].mean(),
        }
    )
    out.index.name = "time"
    return out.reset_index()
