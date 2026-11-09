"""The package installs and the battery settings add up."""

import pytest

from batterytrader.config import BATTERY


def test_battery_settings():
    assert BATTERY.usable_mwh == pytest.approx(1.8)
    assert BATTERY.eta**2 == pytest.approx(BATTERY.round_trip_efficiency)
