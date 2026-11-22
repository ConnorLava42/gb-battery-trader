"""The three README charts: one day of trading, profit over the year, capture rates."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, never open windows
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from batterytrader.config import FIG_DIR, UK
from batterytrader.evaluate import label

COLOURS = {
    "perfect_foresight": "black",
    "rule": "tab:grey",
    "forecast_lp": "tab:blue",
    "dqn": "tab:orange",
    "ppo": "tab:green",
    "sac": "tab:purple",
}


def _colour(method: str) -> str:
    key = method.split("_")[1] if method.startswith("rl_") else method
    return COLOURS.get(key, "tab:red")


def day_plot(sims: dict[str, pd.DataFrame], day: str, market: str) -> Path:
    """Top: the price. Bottom: each strategy's state of charge. Buy low, sell high?"""
    start = pd.Timestamp(day, tz=UK).tz_convert("UTC")
    window = slice(start, start + pd.Timedelta(hours=23, minutes=30))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    first = next(iter(sims.values())).loc[window]
    local = first.index.tz_convert(UK).tz_localize(None)
    top.step(local, first["price"], where="post", color="black")
    top.set(ylabel="Price (£/MWh)", title=f"One day of trading, {day} ({market})")
    for method, sim in sims.items():
        if method.startswith("rl_") and not method.endswith("seed0"):
            continue  # one seed per algorithm keeps the chart readable
        bottom.step(
            local,
            100 * sim.loc[window, "soc"],
            where="post",
            color=_colour(method),
            label=label(method),
        )
    bottom.set(ylabel="State of charge (%)", xlabel="UK time", ylim=(0, 100))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    bottom.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    return _save(fig, f"day_{market}.png")


def cumulative_profit(sims: dict[str, pd.DataFrame], market: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, sim in sims.items():
        ax.plot(
            sim.index.tz_convert(UK).tz_localize(None),
            sim["profit"].cumsum() / 1000,
            color=_colour(method),
            lw=1 if method.startswith("rl_") else 2,
            alpha=0.6 if method.startswith("rl_") else 1,
            label=label(method),
        )
    ax.set(
        ylabel="Cumulative profit (£ thousand, per MW)",
        title=f"Profit through the test year ({market})",
    )
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, f"cumulative_profit_{market}.png")


def capture_bars(table: pd.DataFrame, market: str) -> Path:
    """Capture rate per strategy (the summary table from evaluate)."""
    rows = table[~table.index.str.contains("seed ")]  # RL as mean-of-seeds rows only
    if not rows.index.str.startswith("RL:").any():
        rows = table  # a single seed: show it
    rows = rows.drop(index=[i for i in rows.index if i.startswith("Perfect")])
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.5))
    ax.barh(rows.index, rows["capture rate (%)"], color="tab:blue")
    for y, v in enumerate(rows["capture rate (%)"]):
        ax.text(v + 1, y, f"{v:.0f}%", va="center")
    ax.invert_yaxis()
    ax.set(
        xlabel="Share of perfect-foresight profit captured (%)",
        xlim=(min(0, rows["capture rate (%)"].min() - 10), 100),  # room for a loss-maker
        title=f"Capture rate on the test year ({market})",
    )
    return _save(fig, f"capture_{market}.png")


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
