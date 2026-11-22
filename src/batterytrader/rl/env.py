"""Rung 3: the battery as a Gymnasium environment for reinforcement learning.

Each step is one half-hour. The agent sees only the past (recent prices, the
time, its state of charge and the weather forecast for now), picks a power
level, and is paid the real price of that half-hour, which it hasn't seen.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from batterytrader.battery import start_energy
from batterytrader.battery import step as battery_step  # the env has its own step()
from batterytrader.config import (
    BATTERY,
    DT_HOURS,
    PERIODS_PER_DAY,
    TRAIN_START,
    UK,
    VALID_START,
    Battery,
)

N_LAGS = 8  # the last 4 hours of prices


def price_scale(market: pd.DataFrame, price_col: str) -> float:
    """A typical big price from the training years, used to scale prices to about -1..1.

    It is always measured on the training period, so the test year never leaks in.
    """
    t = market.index
    train = (t >= pd.Timestamp(TRAIN_START, tz="UTC")) & (t < pd.Timestamp(VALID_START, tz="UTC"))
    return float(market.loc[train, price_col].abs().quantile(0.95))


def observation_table(market: pd.DataFrame, price_col: str, scale: float) -> np.ndarray:
    """Everything the agent can see at each period except its state of charge.

    Row t only uses prices up to t - 1. Values are scaled to roughly -1..1 and
    clipped, so one crazy price spike can't swamp the network.
    """
    p = pd.Series(market[price_col].to_numpy(float)) / scale
    cols = [p.shift(lag) for lag in range(1, N_LAGS + 1)]
    day = p.shift(1).rolling(PERIODS_PER_DAY)
    cols += [day.mean(), day.std(), day.min(), day.max(), p.shift(PERIODS_PER_DAY)]

    local = market.index.tz_convert(UK)
    tod = (local.hour * 2 + local.minute // 30).to_numpy() / PERIODS_PER_DAY
    dow = local.day_of_week.to_numpy() / 7
    for x in (tod, dow):
        cols += [pd.Series(np.sin(2 * np.pi * x)), pd.Series(np.cos(2 * np.pi * x))]
    cols += [
        pd.Series(market["wind_speed"].to_numpy() / 15),
        pd.Series(market["temperature"].to_numpy() / 20),
        pd.Series(market["solar_radiation"].to_numpy() / 500),
    ]
    obs = pd.concat(cols, axis=1).to_numpy(np.float32)
    return np.clip(np.nan_to_num(obs), -5, 5)


class BatteryEnv(gym.Env):
    """A battery trading one market over [start, end).

    discrete=True: actions 0 = charge, 1 = hold, 2 = discharge (for DQN).
    discrete=False: one number in [-1, 1] = share of full power (for PPO, SAC).
    random_start=True (training): each episode is `episode_days` long, starting
    on a random day with a random state of charge, so the agent can't memorise
    one stretch of prices. False (evaluation): one episode over the whole window.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        market: pd.DataFrame,
        price_col: str,
        start: str = TRAIN_START,
        end: str = VALID_START,
        battery: Battery = BATTERY,
        discrete: bool = True,
        random_start: bool = True,
        episode_days: int = 7,
    ):
        super().__init__()
        self.battery, self.discrete, self.random_start = battery, discrete, random_start
        self.scale = price_scale(market, price_col)
        self.features = observation_table(market, price_col, self.scale)
        self.prices = market[price_col].to_numpy(float)
        self.first = int(market.index.searchsorted(pd.Timestamp(start, tz="UTC")))
        self.last = int(market.index.searchsorted(pd.Timestamp(end, tz="UTC")))  # exclusive
        self.episode_len = (
            episode_days * PERIODS_PER_DAY if random_start else self.last - self.first
        )
        if self.last - self.first < self.episode_len:
            raise ValueError("the window is shorter than one episode")

        self.action_space = (
            spaces.Discrete(3) if discrete else spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        )
        n_obs = self.features.shape[1] + 1
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(n_obs,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        soc = (self.energy - self.battery.e_min) / self.battery.usable_mwh  # 0 = empty, 1 = full
        row = min(self.t, len(self.features) - 1)  # the final observation of the data
        return np.append(self.features[row], np.float32(soc)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            n_days = (self.last - self.first - self.episode_len) // PERIODS_PER_DAY
            self.t = self.first + int(self.np_random.integers(0, n_days + 1)) * PERIODS_PER_DAY
            fraction = self.np_random.uniform(0, 1)
            self.energy = self.battery.e_min + fraction * self.battery.usable_mwh
        else:
            self.t, self.energy = self.first, start_energy(self.battery)
        self.end = self.t + self.episode_len
        return self._obs(), {}

    def action_to_power(self, action) -> float:
        if self.discrete:
            return (int(action) - 1) * self.battery.power_mw
        return float(np.asarray(action).reshape(-1)[0]) * self.battery.power_mw

    def step(self, action):
        price = self.prices[self.t]
        power, self.energy, revenue, wear = battery_step(
            self.energy, self.action_to_power(action), price, self.battery
        )
        # Reward = profit, scaled so a full-power half-hour at a typical big price is about 1
        reward = (revenue - wear) / (self.scale * self.battery.power_mw * DT_HOURS)
        info = {"power_mw": power, "profit": revenue - wear, "price": price}
        self.t += 1
        truncated = self.t >= self.end  # time's up; nothing "terminates" a battery
        return self._obs(), float(reward), False, truncated, info
