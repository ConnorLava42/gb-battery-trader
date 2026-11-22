"""A very short training run end to end. Skipped unless the rl extra is installed."""

import pytest

pytest.importorskip("stable_baselines3")

from batterytrader.rl import train as rl  # noqa: E402


@pytest.mark.rl
@pytest.mark.parametrize("algo", ["dqn", "ppo"])
def test_train_keep_best_and_trade_the_test_year(market, tmp_path, monkeypatch, algo):
    monkeypatch.setattr(rl, "MODEL_DIR", tmp_path)
    monkeypatch.setitem(rl.HYPER, "dqn", rl.HYPER["dqn"] | {"learning_starts": 500})
    monkeypatch.setitem(rl.HYPER, "ppo", rl.HYPER["ppo"] | {"n_steps": 512, "batch_size": 128})
    folder = rl.train(market, "mid_price", "mid", algo, seed=0, steps=2_048, eval_every=1_024)
    assert (folder / "best_model.zip").exists()
    assert list(folder.glob("day_*.png"))  # the learning-progress plots
    power = rl.trade_test_year(market, "mid_price", "mid", algo, seed=0)
    assert len(power) == 365 * 48
    assert power.abs().max() <= 1.0
