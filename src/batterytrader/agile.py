"""A home battery on Octopus Agile, where the optimiser simply wins.

Agile prices for tomorrow are published around 4pm today. When you know the
prices, an optimiser finds the best plan exactly, so there is nothing left for
RL to learn.
"""

import numpy as np
import pandas as pd

from batterytrader.battery import simulate, start_energy, step
from batterytrader.config import UK, Battery
from batterytrader.data.http import get_json
from batterytrader.metrics import capture_rate
from batterytrader.strategies.optimiser import leftover_value, optimise, perfect_foresight
from batterytrader.strategies.rule import rule_schedule

PRODUCT = "AGILE-24-10-01"  # the Agile tariff available since October 2024
URL = "https://api.octopus.energy/v1/products/{p}/electricity-tariffs/E-1R-{p}-{r}/standard-unit-rates/"

# A typical home battery, in MW and MWh so the same code works: 3 kW, 10 kWh
HOME_BATTERY = Battery(
    power_mw=0.003,
    capacity_mwh=0.010,
    round_trip_efficiency=0.90,
    soc_min=0.10,
    soc_max=1.00,
    degradation_gbp_per_mwh=50.0,  # 5p per kWh discharged
)


def parse_agile(results: list[dict]) -> pd.Series:
    """Octopus unit rates (p/kWh including VAT) -> £/MWh, one value per half-hour, UTC."""
    df = pd.DataFrame(results)
    prices = pd.Series(
        df["value_inc_vat"].astype(float).to_numpy() * 10,  # 1 p/kWh = £10/MWh
        index=pd.to_datetime(df["valid_from"], utc=True),
        name="agile_price",
    )
    return prices[~prices.index.duplicated()].sort_index()


def fetch_agile(start: str, end: str, region: str = "C") -> pd.Series:
    """Every Agile price between start and end. Region C is London; A to P cover GB."""
    url = URL.format(p=PRODUCT, r=region)
    params = {"period_from": f"{start}T00:00Z", "period_to": f"{end}T00:00Z", "page_size": 1500}
    results = []
    while url:  # the API returns pages, each pointing to the next
        page = get_json(url, params)
        results += page["results"]
        url, params = page.get("next"), None  # the 'next' link already holds the query
    return parse_agile(results)


def plan_when_published(prices: pd.Series, battery: Battery = HOME_BATTERY) -> pd.Series:
    """Re-plan at 16:00 UK time each day, over every price known by then.

    At 16:00 the prices to the end of tomorrow are known, so the plan covers
    them all; it is carried out until the next day's 16:00 update.
    """
    p = prices.to_numpy(float)
    local = prices.index.tz_convert(UK)
    replans = [0] + list(np.flatnonzero((local.hour == 16) & (local.minute == 0)))
    energy, power = start_energy(battery), np.zeros(len(p))
    for i, s in enumerate(replans):
        # The first plan only knows today's prices; every later one knows tomorrow's too
        known_until = local[s].normalize() + pd.DateOffset(days=2 if i else 1)
        horizon_end = int(local.searchsorted(known_until))
        next_plan = replans[i + 1] if i + 1 < len(replans) else len(p)
        window = p[s:horizon_end]
        plan = optimise(
            window, battery, energy0=energy, terminal_value=leftover_value(window, battery)
        )
        for t in range(s, next_plan):
            power[t], energy, _, _ = step(energy, plan[t - s], p[t], battery)
    return pd.Series(power, index=prices.index, name="power_mw")


def compare(prices: pd.Series, battery: Battery = HOME_BATTERY) -> pd.DataFrame:
    """£ saved per year by each approach, assuming the house always uses at least
    what the battery discharges (so every kWh discharged is a kWh not bought)."""
    sims = {
        "Perfect foresight": simulate(prices, perfect_foresight(prices, battery), battery),
        "Optimiser at 4pm (realistic)": simulate(
            prices, plan_when_published(prices, battery), battery
        ),
        "Rule: yesterday's pattern": simulate(prices, rule_schedule(prices, battery), battery),
    }
    best = sims["Perfect foresight"]["profit"].sum()
    years = len(prices) / (48 * 365)
    return pd.DataFrame(
        {
            "£ saved per year": {k: s["profit"].sum() / years for k, s in sims.items()},
            "capture rate (%)": {k: capture_rate(s["profit"].sum(), best) for k, s in sims.items()},
        }
    ).round(1)
