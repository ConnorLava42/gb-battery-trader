"""Rung 1: a simple rule. Charge in yesterday's cheapest half-hours, sell in its dearest."""

import math

import numpy as np
import pandas as pd

from batterytrader.config import BATTERY, DT_HOURS, PERIODS_PER_DAY, Battery


def periods_to_fill(battery: Battery = BATTERY) -> int:
    """How many full-power half-hours it takes to charge from empty to full."""
    return math.ceil(battery.usable_mwh / (battery.power_mw * battery.eta * DT_HOURS))


def rule_schedule(prices: pd.Series, battery: Battery = BATTERY) -> pd.Series:
    """Today's plan from yesterday's prices.

    For each day, find yesterday's k cheapest and k most expensive half-hours
    (k = enough to fill the battery) and charge and discharge at those times of
    day today. Skip the day if yesterday's spread wouldn't cover the losses and wear.
    `prices` must start at midnight UTC and cover whole days.
    """
    if len(prices) % PERIODS_PER_DAY or prices.index[0] != prices.index[0].floor("D"):
        raise ValueError("prices must cover whole UTC days, starting at midnight")
    k = periods_to_fill(battery)
    days = prices.to_numpy(float).reshape(-1, PERIODS_PER_DAY)
    plan = np.zeros_like(days)
    for d in range(1, len(days)):
        yesterday = days[d - 1]
        order = np.argsort(yesterday)
        cheap, dear = order[:k], order[-k:]
        # Selling 1 MWh needs 1/eta^2 MWh bought, and wear is charged on what is sold
        margin = yesterday[dear].mean() - yesterday[cheap].mean() / battery.eta**2
        if margin > battery.degradation_gbp_per_mwh:
            plan[d, cheap] = -battery.power_mw
            plan[d, dear] = battery.power_mw
    return pd.Series(plan.ravel(), index=prices.index, name="power_mw")
