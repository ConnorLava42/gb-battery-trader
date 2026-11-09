"""GB power prices from Elexon's Insights Solution API (free, no key needed).

- Market Index Data (MID): the half-hourly wholesale price from short-term trading.
- System prices: the imbalance price, paid or charged for being out of balance.
  It is spikier and harder to predict.
"""

from datetime import date, timedelta

import pandas as pd

from batterytrader.data.http import get_json

BASE = "https://data.elexon.co.uk/bmrs/api/v1"


def parse_mid(rows: list[dict]) -> pd.DataFrame:
    """MID records -> one row per half-hour: start_time (UTC), mid_price, mid_volume."""
    df = pd.DataFrame(rows, columns=["startTime", "price", "volume"])
    out = pd.DataFrame(
        {
            "start_time": pd.to_datetime(df["startTime"], utc=True),
            "mid_price": df["price"].astype(float),
            "mid_volume": df["volume"].astype(float),
        }
    )
    # A period with no trades is published as price 0, volume 0: that is missing, not free power
    out.loc[out["mid_volume"] <= 0, "mid_price"] = float("nan")
    return out.drop_duplicates("start_time", keep="last").sort_values("start_time")


def fetch_mid(start: date, end: date, chunk_days: int = 7) -> pd.DataFrame:
    """Half-hourly APX Market Index prices from start (inclusive) to end (exclusive)."""
    rows, cur = [], start
    while cur < end:
        stop = min(cur + timedelta(days=chunk_days), end)
        params = {
            "from": f"{cur.isoformat()}T00:00Z",
            "to": f"{stop.isoformat()}T00:00Z",
            "dataProviders": "APXMIDP",
            "format": "json",
        }
        rows += get_json(f"{BASE}/balancing/pricing/market-index", params)["data"]
        cur = stop
    out = parse_mid(rows)
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    return out[(out["start_time"] >= t0) & (out["start_time"] < t1)]


def parse_system_prices(rows: list[dict]) -> pd.DataFrame:
    """System price records -> start_time (UTC), system_price, niv (net imbalance volume)."""
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "start_time": pd.to_datetime(df["startTime"], utc=True),
            # Since 2015 GB has a single imbalance price, so sell and buy prices are equal
            "system_price": df["systemSellPrice"].astype(float),
            "niv": df.get("netImbalanceVolume", pd.Series(float("nan"), index=df.index)),
        }
    )
    return out.drop_duplicates("start_time", keep="last").sort_values("start_time")


def fetch_system_prices(start: date, end: date) -> pd.DataFrame:
    """Imbalance prices, one request per settlement day, start inclusive, end exclusive."""
    rows, days = [], pd.date_range(start, end, freq="D", inclusive="left")
    for i, day in enumerate(days):
        url = f"{BASE}/balancing/settlement/system-prices/{day:%Y-%m-%d}"
        rows += get_json(url, {"format": "json"})["data"]
        if i % 100 == 99:
            print(f"  system prices: {i + 1}/{len(days)} days")
    return parse_system_prices(rows)
