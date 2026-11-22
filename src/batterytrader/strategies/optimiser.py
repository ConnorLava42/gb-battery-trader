"""The best possible schedule for a known set of prices, as a linear programme.

Variables for each period t: charge c_t and discharge d_t (MW, grid side, both >= 0)
and e_t, the energy in the cells at the end of the period (MWh).

    maximise   sum_t  price_t * (d_t - c_t) * dt  -  degradation * d_t * dt  +  value * e_T
    subject to e_t = e_(t-1) + eta * c_t * dt - d_t * dt / eta
               0 <= c_t, d_t <= power,   e_min <= e_t <= e_max

It is solved with HiGHS, the open-source solver that ships inside SciPy.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from batterytrader.battery import start_energy
from batterytrader.config import BATTERY, DT_HOURS, Battery


def optimise(
    prices: np.ndarray,
    battery: Battery = BATTERY,
    energy0: float | None = None,
    terminal_value: float = 0.0,
) -> np.ndarray:
    """Optimal net power (+ discharge, - charge) for each period of `prices`.

    `terminal_value` is what each MWh left in the cells at the end is worth.
    Without it the optimiser empties the battery at the end of every plan,
    because it thinks the world ends there.
    """
    p = np.asarray(prices, float)
    n, dt, eta = len(p), DT_HOURS, battery.eta
    e0 = start_energy(battery) if energy0 is None else energy0

    # linprog minimises, so the objective is minus the profit; x = [c, d, e]
    cost = np.concatenate([p * dt, (battery.degradation_gbp_per_mwh - p) * dt, np.zeros(n)])
    cost[-1] -= terminal_value

    # Energy balance: e_t - e_(t-1) - eta*dt*c_t + dt/eta*d_t = 0, with e_(-1) = e0
    eye = sparse.identity(n, format="csr")
    previous = sparse.eye(n, k=-1, format="csr")
    a_eq = sparse.hstack([-eta * dt * eye, dt / eta * eye, eye - previous], format="csr")
    b_eq = np.zeros(n)
    b_eq[0] = e0

    bounds = [(0, battery.power_mw)] * (2 * n) + [(battery.e_min, battery.e_max)] * n
    res = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Optimiser failed: {res.message}")
    charge, discharge = res.x[:n], res.x[n : 2 * n]
    return discharge - charge


def leftover_value(prices: np.ndarray, battery: Battery = BATTERY) -> float:
    """What 1 MWh still in the cells at the end of a plan is worth: sold later at about
    the typical (median) price, minus wear, after the discharge losses."""
    return float(np.median(prices) - battery.degradation_gbp_per_mwh) * battery.eta


def perfect_foresight(prices: pd.Series, battery: Battery = BATTERY) -> pd.Series:
    """The upper bound: one plan for the whole period, knowing every price in advance."""
    return pd.Series(optimise(prices.to_numpy(), battery), index=prices.index, name="power_mw")
