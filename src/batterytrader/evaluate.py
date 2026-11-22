"""Score every saved strategy on the test year with the same simulator."""

import re

import pandas as pd

from batterytrader.battery import simulate
from batterytrader.config import REPORT_DIR
from batterytrader.metrics import capture_by_regime, capture_rate, summarise
from batterytrader.pipeline import prices_for
from batterytrader.results import load_actions

NAMES = {
    "perfect_foresight": "Perfect foresight (upper bound)",
    "rule": "Rule: yesterday's pattern",
    "forecast_lp": "Forecast + optimiser",
}


def simulate_all(market_table: pd.DataFrame, market: str) -> dict[str, pd.DataFrame]:
    prices = prices_for(market_table, market)
    actions = load_actions(market)
    if "perfect_foresight" not in actions:
        raise FileNotFoundError(f"Run `batterytrader foresight --market {market}` first")
    return {m: simulate(prices, a.reindex(prices.index).fillna(0)) for m, a in actions.items()}


def label(method: str) -> str:
    """rl_dqn_seed2 -> 'RL: DQN (seed 2)'."""
    m = re.fullmatch(r"rl_(\w+)_seed(\d+)", method)
    return f"RL: {m[1].upper()} (seed {m[2]})" if m else NAMES.get(method, method)


def summary_table(sims: dict[str, pd.DataFrame]) -> pd.DataFrame:
    best = sims["perfect_foresight"]["profit"].sum()
    rows = {}
    for method, sim in sims.items():
        s = summarise(sim)
        rows[label(method)] = {
            "capture rate (%)": capture_rate(s["profit_gbp"], best),
            "£ per MW per year": s["gbp_per_mw_year"],
            "cycles per day": s["cycles_per_day"],
        }
    table = pd.DataFrame(rows).T

    # One extra row per RL algorithm: the mean and spread over its seeds
    rl = table[table.index.str.startswith("RL:")]
    for algo, group in rl.groupby(rl.index.str.extract(r"RL: (\w+)", expand=False).to_numpy()):
        if len(group) > 1:
            cap = group["capture rate (%)"]
            table.loc[f"RL: {algo}, mean of {len(group)} seeds (± sd {cap.std():.1f})"] = (
                group.mean()
            )
    return table.round({"capture rate (%)": 1, "£ per MW per year": 0, "cycles per day": 2})


def evaluate(market_table: pd.DataFrame, market: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Headline table and calm-vs-volatile table, saved as markdown for the README."""
    sims = simulate_all(market_table, market)
    table = summary_table(sims)
    regimes = capture_by_regime(sims, sims["perfect_foresight"]).round(1)
    regimes.index = [label(m) for m in regimes.index]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"summary_{market}.md").write_text(
        table.to_markdown()
        + "\n\nCapture rate (%) on calm and volatile days\n\n"
        + regimes.to_markdown()
    )
    return table, regimes
