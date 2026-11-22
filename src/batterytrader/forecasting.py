"""Rung 2a: forecast the next 24 hours of prices with LightGBM.

Every half-hour t we issue a forecast for the 48 periods t, t+1, ..., t+47,
using only prices published before t. One model covers every horizon h:
each training row is one (issue time, horizon) pair with h as a feature.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from batterytrader.config import PERIODS_PER_DAY, TEST_END, TEST_START, TRAIN_START, UK, VALID_START

HORIZON = PERIODS_PER_DAY  # forecast 24 hours ahead
WEEK = 7 * PERIODS_PER_DAY
FEATURES = [
    "h",
    "tod_sin",
    "tod_cos",
    "dow",
    "last_price",
    "price_same_time_yesterday",
    "price_same_time_last_week",
    "mean_24h",
    "std_24h",
    "min_24h",
    "max_24h",
    "wind_speed",
    "wind_change",
    "temperature",
    "solar_radiation",
]


def feature_table(market: pd.DataFrame, price_col: str, issue_idx: np.ndarray) -> pd.DataFrame:
    """Features and target for every (issue time, horizon) pair.

    `issue_idx` are row positions in `market`. For issue time t and horizon h the
    target is the price at t + h. Every price feature looks at t - 1 or earlier:
    'same time yesterday' is t + h - 48, which is before t because h < 48.
    """
    p = market[price_col].to_numpy(float)
    issue_idx = np.asarray(issue_idx)
    if issue_idx.min() < WEEK or issue_idx.max() + HORIZON > len(p):
        raise ValueError("issue times need a week of history before and a day of data after")

    # Rolling stats of the day before the issue time: index t-1 summarises t-48 .. t-1
    roll = pd.Series(p).rolling(PERIODS_PER_DAY)
    stats = {name: getattr(roll, name)().to_numpy() for name in ("mean", "std", "min", "max")}

    local = market.index.tz_convert(UK)
    tod = (local.hour * 2 + local.minute // 30).to_numpy()
    dow = local.day_of_week.to_numpy()
    wind = market["wind_speed"].to_numpy(float)

    blocks = []
    for h in range(HORIZON):
        t, target = issue_idx, issue_idx + h
        blocks.append(
            pd.DataFrame(
                {
                    "issue": t,
                    "h": h,
                    "tod_sin": np.sin(2 * np.pi * tod[target] / 48),
                    "tod_cos": np.cos(2 * np.pi * tod[target] / 48),
                    "dow": dow[target],
                    "last_price": p[t - 1],
                    "price_same_time_yesterday": p[target - PERIODS_PER_DAY],
                    "price_same_time_last_week": p[target - WEEK],
                    "mean_24h": stats["mean"][t - 1],
                    "std_24h": stats["std"][t - 1],
                    "min_24h": stats["min"][t - 1],
                    "max_24h": stats["max"][t - 1],
                    # Weather at the target time comes from a forecast, so it is known at t
                    "wind_speed": wind[target],
                    "wind_change": wind[target] - wind[t],
                    "temperature": market["temperature"].to_numpy(float)[target],
                    "solar_radiation": market["solar_radiation"].to_numpy(float)[target],
                    "target": p[target],
                }
            )
        )
    return pd.concat(blocks, ignore_index=True).astype({"h": "int16", "dow": "int8"})


def issue_positions(
    market: pd.DataFrame, start: str, end: str, every: int = 1, targets_inside: bool = True
) -> np.ndarray:
    """Row positions of issue times in [start, end), one every `every` periods.

    With targets_inside=True every target also falls before `end`, so training
    rows never peek at prices from the validation or test period.
    """
    t = market.index
    lo, hi = (
        t.searchsorted(pd.Timestamp(start, tz="UTC")),
        t.searchsorted(pd.Timestamp(end, tz="UTC")),
    )
    pos = np.arange(max(lo, WEEK), hi)
    limit = hi if targets_inside else len(t)
    return pos[pos + HORIZON <= limit][::every]


def to_matrix(table: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """Long predictions -> a matrix with one row per issue time and one column per horizon."""
    wide = pd.DataFrame({"issue": table["issue"], "h": table["h"], "v": values})
    return wide.pivot(index="issue", columns="h", values="v").to_numpy()


PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 200,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "verbose": -1,
}


def train_model(market: pd.DataFrame, price_col: str, every: int = 4, seed: int = 42):
    """Fit on 2021-2023 and stop adding trees when 2024 stops improving.

    `every=4` uses one issue time in four (every 2 hours): neighbouring issue
    times share almost all their information, so this loses little and is 4x faster.
    The loss is squared error, so the model predicts the expected price, which
    is what a profit-maximising optimiser needs.
    """
    train = feature_table(
        market, price_col, issue_positions(market, TRAIN_START, VALID_START, every)
    )
    valid = feature_table(
        market, price_col, issue_positions(market, VALID_START, TEST_START, every)
    )
    model = lgb.LGBMRegressor(n_estimators=3000, random_state=seed, **PARAMS)
    model.fit(
        train[FEATURES],
        train["target"],
        eval_X=valid[FEATURES],
        eval_y=valid["target"],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def forecast_test(model, market: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """A 48-period forecast at every half-hour of the test year (rows = issue times)."""
    pos = issue_positions(market, TEST_START, TEST_END, targets_inside=False)
    table = feature_table(market, price_col, pos)
    matrix = to_matrix(table, model.predict(table[FEATURES]))
    return pd.DataFrame(matrix, index=market.index[pos], columns=[f"h{h}" for h in range(HORIZON)])


def accuracy_by_horizon(market: pd.DataFrame, price_col: str, forecasts: pd.DataFrame):
    """MAE of the model vs the naive 'same time yesterday' forecast, by hours ahead."""
    pos = market.index.get_indexer(forecasts.index)
    table = feature_table(market, price_col, pos)
    table["model"] = forecasts.to_numpy().ravel(order="F")  # column by column, like the table
    table["hours_ahead"] = (table["h"] // 2 + 1).clip(upper=24)
    err = pd.DataFrame(
        {
            "model": (table["model"] - table["target"]).abs(),
            "same_time_yesterday": (table["price_same_time_yesterday"] - table["target"]).abs(),
            "bucket": pd.cut(
                table["hours_ahead"], [0, 1, 4, 12, 24], labels=["0-1h", "1-4h", "4-12h", "12-24h"]
            ),
        }
    )
    return err.groupby("bucket", observed=True)[["model", "same_time_yesterday"]].mean().round(1)
