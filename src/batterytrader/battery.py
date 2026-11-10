"""The battery's physics and the money. Every strategy is scored by this one file.

Sign convention: power > 0 means discharging (selling to the grid),
power < 0 means charging (buying from the grid). Energy is what is stored
in the cells, in MWh.
"""

import numpy as np
import pandas as pd

from batterytrader.config import BATTERY, DT_HOURS, Battery


def step(energy: float, power: float, price: float, battery: Battery = BATTERY):
    """Apply one half-hour of requested power, clipped to what is physically possible.

    Returns (power actually delivered, new energy, revenue in £, degradation cost in £).
    Clipping instead of penalising means an impossible request (charging a full
    battery) simply does nothing, which is also how a real battery controller behaves.
    """
    power = min(max(power, -battery.power_mw), battery.power_mw)
    if power > 0:  # discharging: the cells lose more than the grid receives
        power = min(power, (energy - battery.e_min) * battery.eta / DT_HOURS)
        energy -= power * DT_HOURS / battery.eta
    else:  # charging: the cells gain less than the grid supplies
        power = max(power, -(battery.e_max - energy) / (battery.eta * DT_HOURS))
        energy -= power * DT_HOURS * battery.eta
    energy = min(max(energy, battery.e_min), battery.e_max)  # tidy floating-point dust
    revenue = price * power * DT_HOURS
    degradation = battery.degradation_gbp_per_mwh * max(power, 0.0) * DT_HOURS
    return power, energy, revenue, degradation


def start_energy(battery: Battery = BATTERY) -> float:
    """Every evaluation starts half full."""
    return battery.e_min + 0.5 * battery.usable_mwh


def simulate(
    prices: pd.Series, power: pd.Series | np.ndarray, battery: Battery = BATTERY, energy0=None
) -> pd.DataFrame:
    """Run a schedule of requested power through the battery, one period at a time."""
    p = prices.to_numpy(float)
    req = np.asarray(power, float)
    if len(req) != len(p):
        raise ValueError(f"{len(req)} actions for {len(p)} prices")
    energy = start_energy(battery) if energy0 is None else energy0
    out = np.empty((len(p), 4))
    for t in range(len(p)):
        delivered, energy, revenue, degradation = step(energy, req[t], p[t], battery)
        out[t] = delivered, energy, revenue, degradation
    sim = pd.DataFrame(
        out, index=prices.index, columns=["power_mw", "energy_mwh", "revenue", "degradation"]
    )
    sim.insert(0, "price", p)
    sim["profit"] = sim["revenue"] - sim["degradation"]
    sim["soc"] = sim["energy_mwh"] / battery.capacity_mwh
    return sim
