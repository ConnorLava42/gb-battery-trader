"""Command-line entry point: `uv run batterytrader --help` lists every command.

Each command imports what it needs inside itself, so the app starts fast and
commands for later phases can be added without touching the ones above them.
"""

import typer

from batterytrader import config

app = typer.Typer(
    help="GB Battery Trader: four ways to trade a grid battery.", no_args_is_help=True
)
MARKET = typer.Option("mid", help="mid (wholesale) or imbalance (system price)")


def _check(market: str) -> str:
    if market not in config.PRICE_COLUMN:
        raise typer.BadParameter(f"market must be one of {list(config.PRICE_COLUMN)}")
    return market


def _test_prices(market: str):
    from batterytrader.pipeline import load_market, prices_for

    return prices_for(load_market(), market)


@app.command()
def ingest(force: bool = False):
    """Download prices and weather, one file per source per year (resumable)."""
    from batterytrader.pipeline import ingest as run

    run(force=force)


@app.command()
def build():
    """Combine the raw files into data/processed/market.parquet."""
    from batterytrader.pipeline import build as run

    run()


@app.command()
def foresight(market: str = MARKET):
    """The perfect-foresight upper bound on the test year."""
    from batterytrader.results import save_actions
    from batterytrader.strategies.optimiser import perfect_foresight

    market = _check(market)
    save_actions(perfect_foresight(_test_prices(market)), market, "perfect_foresight")
    print("Saved the perfect-foresight schedule")


@app.command()
def rule(market: str = MARKET):
    """Charge at yesterday's cheapest times, sell at its dearest."""
    from batterytrader.pipeline import load_market, prices_for
    from batterytrader.results import save_actions
    from batterytrader.strategies.rule import rule_schedule

    market = _check(market)
    table = load_market()
    # The rule needs the day before the test year, so plan over all the data, then cut
    plan = rule_schedule(table[config.PRICE_COLUMN[market]])
    test = prices_for(table, market)
    save_actions(plan.reindex(test.index), market, "rule")
    print("Saved the rule schedule")


@app.command()
def forecast(
    market: str = MARKET,
    every: int = typer.Option(4, help="train on one issue time in this many (higher = faster)"),
):
    """Train the price forecaster, then forecast every half-hour of the test year."""
    from batterytrader.forecasting import accuracy_by_horizon, forecast_test, train_model
    from batterytrader.pipeline import load_market

    market = _check(market)
    table, col = load_market(), config.PRICE_COLUMN[market]
    model = train_model(table, col, every=every)
    print(f"Trained with {model.best_iteration_} trees")
    forecasts = forecast_test(model, table, col)
    forecasts.to_parquet(config.PROCESSED_DIR / f"forecasts_{market}.parquet")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(config.MODEL_DIR / f"price_model_{market}.txt")
    print("Mean absolute error (£/MWh), model vs 'same time yesterday':")
    print(accuracy_by_horizon(table, col, forecasts))


@app.command()
def mpc(market: str = MARKET):
    """Re-optimise against the latest forecast every half-hour of the test year."""
    import pandas as pd

    from batterytrader.results import save_actions
    from batterytrader.strategies.mpc import rolling_optimise

    market = _check(market)
    forecasts = pd.read_parquet(config.PROCESSED_DIR / f"forecasts_{market}.parquet")
    save_actions(rolling_optimise(_test_prices(market), forecasts), market, "forecast_lp")
    print("Saved the forecast + optimiser schedule")
