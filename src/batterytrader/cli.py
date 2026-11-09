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
