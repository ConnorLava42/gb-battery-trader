"""Train and run RL agents with Stable-Baselines3 (needs `uv sync --extra rl`)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, never open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from batterytrader.config import MODEL_DIR, TEST_END, TEST_START, TRAIN_START, VALID_START
from batterytrader.rl.env import BatteryEnv

ALGOS = {"dqn": DQN, "ppo": PPO, "sac": SAC}
DISCRETE = {"dqn": True, "ppo": False, "sac": False}
NET = {"net_arch": [128, 128]}
HYPER = {
    "dqn": {
        "learning_rate": 1e-4,
        "buffer_size": 200_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "gamma": 0.99,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.02,
        "target_update_interval": 2_000,
        "policy_kwargs": NET,
    },
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "gamma": 0.99,
        "ent_coef": 0.001,
        "policy_kwargs": NET,
    },
    "sac": {
        "learning_rate": 3e-4,
        "buffer_size": 200_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "gamma": 0.99,
        "policy_kwargs": NET,
    },
}


def model_dir(market: str, algo: str, seed: int) -> Path:
    return MODEL_DIR / f"rl_{market}" / f"{algo}_seed{seed}"


def run_policy(model, env: BatteryEnv) -> pd.DataFrame:
    """Play one whole evaluation episode with no exploration. Returns power, price, profit."""
    obs, _ = env.reset()
    rows, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, done, info = env.step(action)
        rows.append(info)
    return pd.DataFrame(rows)


def plot_day(day: pd.DataFrame, path: Path, title: str) -> None:
    """Price and the agent's power for one day: is it buying low and selling high?"""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(day["price"].to_numpy(), color="black", lw=1.5, label="Price (£/MWh)")
    ax2 = ax.twinx()
    ax2.bar(np.arange(len(day)), day["power_mw"], color="tab:blue", alpha=0.4, label="Power (MW)")
    ax.set(xlabel="Half-hour of the day", ylabel="£/MWh", title=title)
    ax2.set_ylabel("MW (+ sell, - buy)")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


class ValidationCallback(BaseCallback):
    """Every `every` steps, trade the whole validation year and keep the best model.

    This is early stopping for RL: training rewards are noisy, so the model we
    keep is the one that made the most money on 2024, a year it never trains on.
    It also saves a plot of one validation day, to watch the agent learn.
    """

    def __init__(self, valid_env: BatteryEnv, folder: Path, every: int):
        super().__init__()
        self.valid_env, self.folder, self.every = valid_env, folder, every
        self.best, self.history = -np.inf, []

    def _on_step(self) -> bool:
        if self.n_calls % self.every:
            return True
        run = run_policy(self.model, self.valid_env)
        profit = float(run["profit"].sum())
        self.history.append({"steps": self.num_timesteps, "valid_profit_gbp": round(profit)})
        print(f"  {self.num_timesteps:,} steps: validation profit £{profit:,.0f}")
        if profit > self.best:
            self.best = profit
            self.model.save(self.folder / "best_model")
        day = run.iloc[30 * 48 : 31 * 48]  # 31 January of the validation year
        plot_day(
            day, self.folder / f"day_{self.num_timesteps:08d}.png", f"{self.num_timesteps:,} steps"
        )
        return True


def train(
    market: pd.DataFrame,
    price_col: str,
    market_name: str,
    algo: str = "dqn",
    seed: int = 0,
    steps: int = 300_000,
    eval_every: int = 25_000,
) -> Path:
    """Train one agent on 2021-2023, keeping the version that did best on 2024."""
    folder = model_dir(market_name, algo, seed)
    folder.mkdir(parents=True, exist_ok=True)
    discrete = DISCRETE[algo]
    env = Monitor(BatteryEnv(market, price_col, TRAIN_START, VALID_START, discrete=discrete))
    valid = BatteryEnv(
        market, price_col, VALID_START, TEST_START, discrete=discrete, random_start=False
    )
    model = ALGOS[algo]("MlpPolicy", env, seed=seed, verbose=0, **HYPER[algo])
    callback = ValidationCallback(valid, folder, eval_every)
    model.learn(total_timesteps=steps, callback=callback)
    if callback.best == -np.inf:  # training was shorter than one evaluation
        model.save(folder / "best_model")
    (folder / "history.json").write_text(json.dumps(callback.history, indent=2))
    return folder


def trade_test_year(market: pd.DataFrame, price_col: str, market_name: str, algo: str, seed: int):
    """The best saved agent's power schedule over the test year."""
    model = ALGOS[algo].load(model_dir(market_name, algo, seed) / "best_model")
    env = BatteryEnv(
        market, price_col, TEST_START, TEST_END, discrete=DISCRETE[algo], random_start=False
    )
    run = run_policy(model, env)
    index = market.index[env.first : env.last]
    return pd.Series(run["power_mw"].to_numpy(), index=index, name="power_mw")
