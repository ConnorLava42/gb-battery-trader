"""Every command, end to end, on the synthetic market (about two minutes)."""

import importlib.util

import pytest
from conftest import make_market
from fake_apis import fake_get_json
from typer.testing import CliRunner

from batterytrader import agile, cli, config, evaluate, pipeline, plots, results

HAS_RL = importlib.util.find_spec("stable_baselines3") is not None


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Point every output folder at a temporary directory and drop in a market table."""
    monkeypatch.setattr(pipeline, "MARKET", tmp_path / "market.parquet")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)  # cli.py reads these from config
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(results, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(evaluate, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(plots, "FIG_DIR", tmp_path / "figures")
    monkeypatch.setattr(agile, "get_json", fake_get_json)
    if HAS_RL:
        from batterytrader.rl import train

        monkeypatch.setattr(train, "MODEL_DIR", tmp_path / "models")
        monkeypatch.setitem(train.HYPER, "dqn", train.HYPER["dqn"] | {"learning_starts": 500})
    make_market().to_parquet(tmp_path / "market.parquet")
    return tmp_path


def run(*args):
    result = CliRunner().invoke(cli.app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def test_every_command(project):
    run("foresight", "--market", "mid")
    run("rule", "--market", "mid")
    out = run("forecast", "--market", "mid", "--every", "48")
    assert "same_time_yesterday" in out
    run("mpc", "--market", "mid")
    if HAS_RL:
        run("rl-train", "--market", "mid", "--algo", "dqn", "--steps", "2000")
    out = run("evaluate", "--market", "mid")
    assert "Perfect foresight (upper bound)" in out
    summary = (project / "reports" / "summary_mid.md").read_text()
    assert "Forecast + optimiser" in summary and "calm" in summary
    assert len(list((project / "figures").glob("*.png"))) == 3
    out = run("agile")
    assert "Optimiser at 4pm" in out


def test_unknown_market_is_refused(project):
    result = CliRunner().invoke(cli.app, ["foresight", "--market", "gas"])
    assert result.exit_code != 0
