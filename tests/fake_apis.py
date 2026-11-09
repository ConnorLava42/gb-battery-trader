"""A stand-in for every API, so tests never touch the network.

`fake_get_json` answers the same URLs as the real services with responses of
the same shape, built from the synthetic market in conftest.
"""

import pandas as pd
from conftest import synthetic_prices, synthetic_weather

from batterytrader.data.weather import SITES

PAGE = 500  # Octopus page size in the fake, small so pagination gets tested


def _half_hours(start, end, inclusive="both"):
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="30min", inclusive=inclusive)


def _mid(params):
    t = _half_hours(params["from"], params["to"])  # the real API includes the end time too
    mid, _ = synthetic_prices(t, synthetic_weather(t))
    rows = [
        {"startTime": s.isoformat(), "dataProvider": "APXMIDP", "price": float(p), "volume": 100.0}
        for s, p in zip(t, mid, strict=True)
    ]
    rows[3]["price"], rows[3]["volume"] = 0.0, 0.0  # a period with no trades
    return {"data": rows}


def _system(day: str):
    t = _half_hours(
        pd.Timestamp(day, tz="UTC"), pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1), "left"
    )
    _, system = synthetic_prices(t, synthetic_weather(t))
    return {
        "data": [
            {
                "startTime": s.isoformat(),
                "systemSellPrice": float(p),
                "systemBuyPrice": float(p),
                "netImbalanceVolume": -50.0,
            }
            for s, p in zip(t, system, strict=True)
        ]
    }


def _open_meteo(params):
    t = pd.date_range(
        params["start_date"],
        pd.Timestamp(params["end_date"]) + pd.Timedelta(hours=23),
        freq="h",
        tz="UTC",
    )
    w = synthetic_weather(t)
    hourly = {
        "time": [s.strftime("%Y-%m-%dT%H:%M") for s in t],
        "wind_speed_100m": w["wind_speed"].tolist(),
        "temperature_2m": w["temperature"].tolist(),
        "shortwave_radiation": w["solar_radiation"].tolist(),
    }
    return [{"hourly": hourly} for _ in SITES]


def _octopus(url, params):
    if params:  # first page: build every rate, newest first like the real API
        t = _half_hours(params["period_from"], params["period_to"], "left")
        mid, _ = synthetic_prices(t, synthetic_weather(t))
        _octopus.rows = [
            {
                "value_exc_vat": p / 10 / 1.05,
                "value_inc_vat": p / 10,
                "valid_from": s.isoformat(),
                "valid_to": (s + pd.Timedelta(minutes=30)).isoformat(),
            }
            for s, p in zip(t[::-1], mid[::-1], strict=True)
        ]
        page = 0
    else:
        page = int(url.split("page=")[1])
    chunk = _octopus.rows[page * PAGE : (page + 1) * PAGE]
    more = (page + 1) * PAGE < len(_octopus.rows)
    return {"results": chunk, "next": f"{url.split('?')[0]}?page={page + 1}" if more else None}


def fake_get_json(url: str, params: dict | None = None, **_):
    if url.endswith("/balancing/pricing/market-index"):
        return _mid(params)
    if "/balancing/settlement/system-prices/" in url:
        return _system(url.rstrip("/").split("/")[-1])
    if "open-meteo.com" in url:
        return _open_meteo(params)
    if "api.octopus.energy" in url:
        return _octopus(url, params)
    raise AssertionError(f"Unexpected request: {url}")
