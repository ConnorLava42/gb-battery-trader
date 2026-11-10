"""How good is a trading strategy? Profit, capture rate, wear, and calm vs volatile days."""

import pandas as pd

from batterytrader.config import BATTERY, DT_HOURS, PERIODS_PER_DAY, Battery


def summarise(sim: pd.DataFrame, battery: Battery = BATTERY) -> dict:
    """Headline numbers for one simulated run (see battery.simulate)."""
    days = len(sim) / PERIODS_PER_DAY
    discharged = sim["power_mw"].clip(lower=0).sum() * DT_HOURS  # MWh sold to the grid
    full_cycle = battery.usable_mwh * battery.eta  # MWh sold by emptying a full battery once
    return {
        "profit_gbp": float(sim["profit"].sum()),
        "gbp_per_mw_year": float(sim["profit"].sum() / battery.power_mw / (days / 365)),
        "cycles_per_day": float(discharged / full_cycle / days),
        "degradation_gbp": float(sim["degradation"].sum()),
    }


def capture_rate(profit: float, best_profit: float) -> float:
    """Share of the perfect-foresight profit a strategy achieved, in %."""
    return 100 * profit / best_profit


def day_volatility(prices: pd.Series) -> pd.Series:
    """Each day's price range (max - min), the spread a battery can trade."""
    daily = prices.groupby(prices.index.floor("D"))
    return daily.max() - daily.min()


def capture_by_regime(
    sims: dict[str, pd.DataFrame], foresight: pd.DataFrame, quantile: float = 0.75
) -> pd.DataFrame:
    """Capture rate on calm days vs volatile days.

    A day is 'volatile' if its price range is in the top quarter of the test year.
    """
    vol = day_volatility(foresight["price"])
    volatile_days = vol.index[vol >= vol.quantile(quantile)]
    is_volatile = foresight.index.floor("D").isin(volatile_days)
    rows = {}
    for name, sim in sims.items():
        rows[name] = {
            "calm": capture_rate(
                sim.loc[~is_volatile, "profit"].sum(), foresight.loc[~is_volatile, "profit"].sum()
            ),
            "volatile": capture_rate(
                sim.loc[is_volatile, "profit"].sum(), foresight.loc[is_volatile, "profit"].sum()
            ),
        }
    return pd.DataFrame(rows).T
