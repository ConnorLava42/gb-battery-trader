"""Every setting in one place: folders, the battery, the data split and the weather sites."""

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
RESULTS_DIR = REPORT_DIR / "results"  # one parquet of actions per method
FIG_DIR = REPORT_DIR / "figures"
MODEL_DIR = ROOT / "models"

UK = ZoneInfo("Europe/London")
DT_HOURS = 0.5  # one settlement period
PERIODS_PER_DAY = 48

# The time split. Train 2021-2023 (including the 2022 gas crisis), tune on 2024,
# report on 2025. Data starts a month early so the first lags exist.
HISTORY_START = "2020-12-01"
TRAIN_START = "2021-01-01"
VALID_START = "2024-01-01"
TEST_START = "2025-01-01"
TEST_END = "2026-01-01"  # exclusive
DATA_END = "2026-01-02"  # one extra day, so the last test forecasts have prices to look at

# The two markets the battery can trade, and the price column each one uses
PRICE_COLUMN = {"mid": "mid_price", "imbalance": "system_price"}


@dataclass(frozen=True)
class Battery:
    """A grid battery. The defaults are a typical GB '2-hour' unit."""

    power_mw: float = 1.0
    capacity_mwh: float = 2.0
    round_trip_efficiency: float = 0.90
    soc_min: float = 0.05  # fractions of capacity
    soc_max: float = 0.95
    degradation_gbp_per_mwh: float = 10.0  # cost of wear, per MWh discharged

    @property
    def eta(self) -> float:
        """One-way efficiency: the losses are split evenly between charging and discharging."""
        return self.round_trip_efficiency**0.5

    @property
    def e_min(self) -> float:
        return self.soc_min * self.capacity_mwh

    @property
    def e_max(self) -> float:
        return self.soc_max * self.capacity_mwh

    @property
    def usable_mwh(self) -> float:
        return self.e_max - self.e_min


BATTERY = Battery()

# Weather that moves GB prices: wind at big wind farms, and temperature and
# sunshine where people live (demand and rooftop solar).
WIND_SITES = {
    "dogger_bank": (54.75, 1.90),
    "hornsea": (53.88, 1.79),
    "moray_firth": (58.10, -2.80),
    "whitelee": (55.68, -4.27),
}
DEMAND_SITES = {
    "london": (51.51, -0.13),
    "birmingham": (52.49, -1.89),
    "manchester": (53.48, -2.24),
}
WEATHER_VARS = ["wind_speed_100m", "temperature_2m", "shortwave_radiation"]
