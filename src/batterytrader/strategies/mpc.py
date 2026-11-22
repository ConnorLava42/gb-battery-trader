"""Rung 2b: forecast + optimiser, re-solved every half-hour.

At each half-hour: take the latest 24-hour price forecast, find the best plan
for it starting from the battery's real state of charge, carry out only the
first step of that plan, then do it all again next half-hour with a fresh
forecast. This is called model predictive control, and it's close to how
battery optimisers in industry work.
"""

import numpy as np
import pandas as pd

from batterytrader.battery import start_energy, step
from batterytrader.config import BATTERY, Battery
from batterytrader.strategies.optimiser import leftover_value, optimise


def rolling_optimise(
    prices: pd.Series, forecasts: pd.DataFrame, battery: Battery = BATTERY
) -> pd.Series:
    """Trade `prices` using the forecast issued at the start of each period.

    `forecasts` has one row per issue time (the same index as `prices`) and one
    column per half-hour ahead. The real prices are used only to move the
    battery forward and never reach the optimiser.
    """
    f = forecasts.reindex(prices.index).to_numpy(float)
    if np.isnan(f).any():
        raise ValueError("a forecast is missing for some trading periods")
    p = prices.to_numpy(float)
    energy, delivered = start_energy(battery), np.empty(len(p))
    for t in range(len(p)):
        plan = optimise(f[t], battery, energy0=energy, terminal_value=leftover_value(f[t], battery))
        delivered[t], energy, _, _ = step(energy, plan[0], p[t], battery)
        if t % 2000 == 1999:
            print(f"  {t + 1:,}/{len(p):,} half-hours")
    return pd.Series(delivered, index=prices.index, name="power_mw")
