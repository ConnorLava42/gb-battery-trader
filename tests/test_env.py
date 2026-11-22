"""The RL environment: valid for Gymnasium, honest about time, same physics as everyone."""

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from batterytrader.battery import simulate
from batterytrader.config import DT_HOURS
from batterytrader.rl.env import BatteryEnv, observation_table, price_scale


@pytest.mark.parametrize("discrete", [True, False])
def test_passes_the_gymnasium_checker(market, discrete):
    check_env(BatteryEnv(market, "mid_price", discrete=discrete), skip_render_check=True)


def test_observation_never_sees_the_current_price(market):
    t = market.index.get_loc(pd.Timestamp("2022-05-01 17:00", tz="UTC"))
    scale = price_scale(market, "system_price")
    honest = observation_table(market, "system_price", scale)[t]
    changed = market.copy()
    changed.iloc[t:, changed.columns.get_loc("system_price")] = 9999.0
    assert np.array_equal(honest, observation_table(changed, "system_price", scale)[t])


def test_training_episodes_are_a_week_from_a_random_day(market):
    env = BatteryEnv(market, "mid_price", episode_days=7)
    starts = set()
    for seed in range(5):
        env.reset(seed=seed)
        starts.add(env.t)
        assert (env.t - env.first) % 48 == 0
        steps, done = 0, False
        while not done:
            _, _, _, done, _ = env.step(1)
            steps += 1
        assert steps == 7 * 48
    assert len(starts) > 1


def test_rewards_are_the_simulators_profit(market):
    env = BatteryEnv(market, "mid_price", "2024-01-01", "2024-01-08", random_start=False)
    env.reset()
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 3, 7 * 48)
    rewards = [env.step(a)[1] for a in actions]
    prices = market.loc["2024-01-01":"2024-01-07", "mid_price"]
    sim = simulate(prices, (actions - 1).astype(float))
    scale = env.scale * env.battery.power_mw * DT_HOURS
    assert np.sum(rewards) * scale == pytest.approx(sim["profit"].sum())
