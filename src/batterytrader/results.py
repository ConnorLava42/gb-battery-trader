"""Save and load each strategy's schedule: one parquet of power per method per market."""

import pandas as pd

from batterytrader.config import RESULTS_DIR

ORDER = ["perfect_foresight", "rule", "forecast_lp"]  # RL runs are listed after these


def results_path(market: str, method: str):
    return RESULTS_DIR / market / f"{method}.parquet"


def save_actions(power: pd.Series, market: str, method: str) -> None:
    path = results_path(market, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    power.rename("power_mw").to_frame().to_parquet(path)


def load_actions(market: str) -> dict[str, pd.Series]:
    """Every saved schedule for a market, in a sensible order."""
    found = {p.stem: p for p in (RESULTS_DIR / market).glob("*.parquet")}
    names = [m for m in ORDER if m in found] + sorted(m for m in found if m not in ORDER)
    return {m: pd.read_parquet(found[m])["power_mw"] for m in names}
