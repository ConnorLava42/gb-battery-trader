# %% Setup: load the market table
import matplotlib.pyplot as plt
import pandas as pd

from batterytrader.config import UK
from batterytrader.pipeline import load_market

market = load_market()
local = market.index.tz_convert(UK)
prices = market[["mid_price", "system_price"]]
prices.describe().round(1)

# %% Year by year: level, volatility, and the daily spread a battery can trade
daily = prices.groupby(local.date)
spread = (daily.max() - daily.min()).add_suffix("_daily_spread")
yearly = pd.concat(
    [
        prices.groupby(local.year).mean().add_suffix("_mean"),
        prices.groupby(local.year).std().add_suffix("_std"),
        spread.groupby(pd.to_datetime(spread.index).year).mean(),
        (prices < 0).groupby(local.year).mean().mul(100).add_suffix("_negative_%"),
    ],
    axis=1,
).loc[2021:2025]  # whole years only
yearly.round(1).T

# %% The 2022 gas crisis: monthly average price and daily spread
months = local.tz_localize(None).to_period("M")
monthly = prices["mid_price"].groupby(months).mean()
monthly_spread = (
    spread["mid_price_daily_spread"].groupby(pd.to_datetime(spread.index).to_period("M")).mean()
)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(monthly.index.to_timestamp(), monthly, label="Average price")
ax.plot(
    monthly_spread.index.to_timestamp(), monthly_spread, label="Average daily spread (max - min)"
)
for start, end, name in [
    ("2021-01-01", "2024-01-01", "train"),
    ("2024-01-01", "2025-01-01", "tune"),
    ("2025-01-01", "2026-01-01", "test"),
]:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.06, color="grey")
    ax.text(pd.Timestamp(start), ax.get_ylim()[1] * 0.95, f" {name}", va="top")
ax.set(ylabel="£/MWh", title="Wholesale (MID) price and daily spread by month")
ax.legend()

# %% [markdown]
# **The 2022 gas crisis is a different world.** From autumn 2021 to the end of 2022 the
# monthly average MID price rose from about £55–80/MWh to £150–350/MWh, peaking near
# £350/MWh in summer 2022. The average daily spread rose with it, to £150–250/MWh.
# From mid-2023 prices settled back to about £50–115/MWh, with daily spreads of
# £50–125/MWh, and the tuning year (2024) and test year (2025) look alike.
#
# So the whole crisis sits in the training data. A model that learns "an evening
# spread is £200" from 2022 will trade badly in 2025. Phase 8 tests this by also
# training on 2023 alone.
#
# The spread moves with the price level: when power is expensive, the gap between the
# cheapest and dearest half-hours widens too. January 2021 stands out as a spread spike
# at normal prices, in a month when GB day-ahead prices hit their £1,500/MWh cap.

# %% The daily shape: when is power cheap and dear? One line per year
tod = local.hour + local.minute / 60
shape = prices["mid_price"].groupby([local.year, tod]).mean().unstack(0)
ax = shape.plot(figsize=(9, 4), title="Average MID price by time of day (UK time)")
ax.set(xlabel="Hour of day", ylabel="£/MWh")

# %% [markdown]
# **The daily shape is the battery's income.** Every year has the same pattern: cheapest
# overnight (around 03:00–05:00), a morning rise to about 07:30–08:30, and the dearest
# half-hours at the evening peak around 17:00–18:30, falling away after 20:00.
#
# Charging overnight and selling at the evening peak earns the gap between them. On
# average that gap was about £40/MWh in 2024 and 2025 (roughly £55 to £95 and £65 to
# £105), against about £100/MWh in 2022 (£150 to £255).
#
# Year by year, the early afternoon (13:00–15:00) has crept down towards the overnight
# level; by 2025 it is almost as cheap. That is most likely solar. It opens a second
# cheap window, so on some days the battery could cycle twice: once for the morning
# peak and again for the evening one. The optimiser can find that; the rule can't.

# %% Imbalance vs wholesale: same level, far spikier
fig, ax = plt.subplots(figsize=(6, 6))
sample = prices.loc["2025-01-01":"2025-12-31"].sample(5000, random_state=0)
ax.scatter(sample["mid_price"], sample["system_price"], s=3, alpha=0.3)
ax.set(
    xlabel="MID price (£/MWh)",
    ylabel="System (imbalance) price (£/MWh)",
    title="2025, 5,000 random half-hours",
)
print("Correlation:", round(prices["mid_price"].corr(prices["system_price"]), 2))
print("Biggest imbalance prices:")
prices["system_price"].nlargest(5)

# %% [markdown]
# **Imbalance follows wholesale, until it doesn't.** In 2025 almost every half-hour sits
# on a tight diagonal: the imbalance price tracks the MID price between about -£100 and
# £300/MWh, and both markets sometimes go negative (bottom left). A handful break away
# completely: three points in this sample have an imbalance price near £2,900/MWh while
# MID was already £850–1,350/MWh. The biggest came on [date from the nlargest output].
#
# These few half-hours matter enormously. Selling 0.5 MWh at £2,900/MWh earns about
# £1,450, roughly two weeks of normal daily arbitrage. They are also nearly impossible to
# forecast, so the imbalance market tests how a strategy copes with spikes, not just how
# well it follows the daily pattern. The plot shows only 5,000 of the year's 17,520
# half-hours, so the full year probably has more of them.

# %% Back of the envelope: roughly what could a 1 MW / 2 MWh battery make in 2025?
# One full cycle a day: buy in the 4 cheapest half-hours, sell in the 4 dearest.
# A full cycle sells about 1.7 MWh, and each MWh sold needs 1/0.9 MWh bought.
test = market.loc["2025-01-01":"2025-12-31", "mid_price"]
by_day = test.groupby(test.index.tz_convert(UK).date)
margin = by_day.apply(lambda d: d.nlargest(4).mean() - d.nsmallest(4).mean() / 0.9)
estimate = margin.clip(lower=0).sum() * 1.7
print(f"One cycle a day at the best times: about £{estimate:,.0f} per MW in 2025")
# %%
