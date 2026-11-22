"""The home battery chapter, against the fake Octopus API."""

import pytest
from fake_apis import fake_get_json

from batterytrader import agile
from batterytrader.data.clean import fill_gaps, half_hour_grid


def test_fetch_follows_every_page_and_converts_units(monkeypatch):
    monkeypatch.setattr(agile, "get_json", fake_get_json)
    prices = agile.fetch_agile("2025-02-01", "2025-02-15")
    assert len(prices) == 14 * 48  # 672 rates arrive over two pages
    assert prices.index.is_monotonic_increasing  # the API sends newest first
    assert prices.between(-200, 600).all()  # £/MWh, not p/kWh


def test_planning_at_4pm_is_nearly_perfect(monkeypatch):
    monkeypatch.setattr(agile, "get_json", fake_get_json)
    grid = half_hour_grid("2025-02-01", "2025-03-01")
    prices = fill_gaps(agile.fetch_agile("2025-02-01", "2025-03-01").reindex(grid))
    result = agile.compare(prices * 5)  # retail prices run several times the wholesale level
    capture = result["capture rate (%)"]
    assert capture["Perfect foresight"] == pytest.approx(100)
    assert capture["Optimiser at 4pm (realistic)"] > 95  # knowing the prices is enough
    assert capture["Rule: yesterday's pattern"] < capture["Optimiser at 4pm (realistic)"]
