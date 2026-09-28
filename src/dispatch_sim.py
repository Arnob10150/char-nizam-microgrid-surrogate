"""
Physics-based off-grid microgrid dispatch simulator (the ground-truth oracle).

Simulates one year (8760 h) of a PV + wind + battery + diesel hybrid serving the
Char Nizam island load (poster P0050: 165.59 kWh/day, 23.32 kW peak), using a
standard load-following dispatch. For a component-sizing configuration it returns
the levelized cost of energy (LCOE), net present cost (NPC), renewable fraction,
and unmet-load fraction. This is the expensive evaluator that HOMER Pro performs
by exhaustive search; our ML surrogate learns to approximate it.

Grounded in the poster's real parameters:
  load 165.59 kWh/day, peak 23.32 kW; PV 1300 $/kW; wind 2300 $/kW (10 kW turb.);
  battery 300 $/kWh (lead-acid 1 kWh unit); converter 300 $/kW; diesel 220 $/kW,
  fuel 0.875 $/L; site 4.86 kWh/m^2/day irradiance, coastal wind.
Deterministic given a weather seed.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_NASA = _ROOT / "data" / "nasa_power.json"

# ---- site & economic constants (from poster P0050) ----------------------- #
LAT = 21.8725
DAILY_LOAD_KWH = 165.59
PEAK_KW = 23.32
PROJECT_YEARS = 25
DISCOUNT = 0.08                 # nominal discount rate
INFLATION = 0.02

C_PV = 1300.0                   # $/kW capital
C_WIND = 2300.0                 # $/kW capital (10 kW turbines)
C_BATT = 300.0                  # $/kWh capital (lead-acid)
C_CONV = 300.0                  # $/kW converter
C_DIESEL = 220.0               # $/kW diesel genset
OM_PV = 20.0; OM_WIND = 20.0; OM_BATT = 10.0; OM_DIESEL = 262.3   # $/yr/unit-ish
FUEL_PRICE = 0.875            # $/L
LHV_DIESEL = 10.0             # kWh/L effective
DERATE_PV = 0.80
WIND_CI, WIND_R, WIND_CO = 3.0, 12.0, 25.0   # cut-in/rated/cut-out m/s


def _crf(rate, n):
    return rate * (1 + rate) ** n / ((1 + rate) ** n - 1)


_CACHE = {}


def _load_nasa():
    """Real hourly GHI (W/m2), wind speed (m/s), temp (C) for Char Nizam from
    the NASA POWER archive (the same source HOMER Pro uses)."""
    if "raw" in _CACHE:
        return _CACHE["raw"]
    d = json.load(open(_NASA))
    p = d["properties"]["parameter"]
    keys = sorted(p["ALLSKY_SFC_SW_DWN"].keys())          # YYYYMMDDHH
    ghi = np.array([p["ALLSKY_SFC_SW_DWN"][k] for k in keys], float)
    ws = np.array([p["WS10M"][k] for k in keys], float)
    t2m = np.array([p["T2M"][k] for k in keys], float)
    ghi[ghi < 0] = 0.0; ws[ws < 0] = 0.0                  # fill missing
    ws10 = ws * (50.0 / 10.0) ** 0.14                     # log-law to ~50 m hub
    _CACHE["raw"] = (ghi[:8760], ws10[:8760], t2m[:8760])
    return _CACHE["raw"]


def synthetic_year(seed=0):
    """Return (solar 0..1 normalized GHI, wind m/s at hub, load kW). Weather is
    the REAL NASA POWER year; `seed`>0 applies a small bootstrap perturbation for
    the uncertainty analysis, seed=0 is the measured year."""
    ghi, ws, t2m = _load_nasa()
    solar = ghi / 1000.0                                  # normalize to STC 1 kW/m2
    wind = ws.copy()
    if seed:                                              # perturb for UQ draws
        rng = np.random.default_rng(seed)
        solar = np.clip(solar * rng.normal(1.0, 0.08, 8760), 0, 1.2)
        wind = np.clip(wind * rng.normal(1.0, 0.10, 8760), 0, 30)
    # load: 250 households, evening + midday peaks, scaled to 165.59 kWh/day
    hod = np.arange(8760) % 24
    base = 0.45 + 0.55 * (np.exp(-((hod - 19) ** 2) / 8) +
                          0.4 * np.exp(-((hod - 12) ** 2) / 10))
    load = base / base.sum() * DAILY_LOAD_KWH * 365
    load = np.clip(load, 0, PEAK_KW)
    return solar, wind, load


def wind_power(v, rated_kw):
    """Per-turbine power (kW) from wind speed with cut-in/rated/cut-out."""
    p = np.zeros_like(v)
    ramp = (v >= WIND_CI) & (v < WIND_R)
    p[ramp] = rated_kw * ((v[ramp] ** 3 - WIND_CI ** 3) / (WIND_R ** 3 - WIND_CI ** 3))
    p[(v >= WIND_R) & (v < WIND_CO)] = rated_kw
    return p


def simulate(pv_kw, wind_units, batt_kwh, diesel_kw, seed=0, conv_kw=None):
    """Load-following dispatch. Returns dict of KPIs."""
    solar, wind, load = synthetic_year(seed)
    if conv_kw is None:
        conv_kw = PEAK_KW * 1.2
    pv_gen = pv_kw * DERATE_PV * solar                       # kW each hour
    wind_gen = wind_units * wind_power(wind, 10.0)           # 10 kW turbines
    renew = pv_gen + wind_gen

    soc = 0.5 * batt_kwh
    soc_min, soc_max = 0.2 * batt_kwh, batt_kwh
    eta_batt = 0.90
    fuel_l = 0.0; unmet = 0.0; renew_served = 0.0; diesel_kwh = 0.0
    throughput = 0.0
    for t in range(8760):
        net = renew[t] - load[t]
        if net >= 0:                                         # surplus -> charge
            room = soc_max - soc
            store = min(net, room) * eta_batt
            soc += store; throughput += store
            renew_served += load[t]
        else:                                                # deficit
            deficit = -net
            renew_served += renew[t]
            avail = (soc - soc_min) * eta_batt
            from_batt = min(deficit, avail)
            soc -= from_batt / eta_batt; throughput += from_batt
            deficit -= from_batt
            if deficit > 1e-9:                               # diesel covers rest
                d = min(deficit, diesel_kw)
                diesel_kwh += d; fuel_l += d / LHV_DIESEL
                deficit -= d
            if deficit > 1e-9:
                unmet += deficit

    total_load = load.sum()
    # economics
    cap = (pv_kw * C_PV + wind_units * 10 * C_WIND + batt_kwh * C_BATT +
           diesel_kw * C_DIESEL + conv_kw * C_CONV)
    om_yr = (pv_kw * 0.02 * C_PV + wind_units * OM_WIND + batt_kwh * OM_BATT +
             diesel_kw * OM_DIESEL / 10)
    fuel_yr = fuel_l * FUEL_PRICE
    # battery + diesel replacements (mid-life)
    repl = (batt_kwh * C_BATT + diesel_kw * C_DIESEL) * 0.6 / (1 + DISCOUNT) ** 10
    # NPC: capital + PV of annual (o&m + fuel) + replacement
    pv_factor = (1 - (1 + DISCOUNT) ** -PROJECT_YEARS) / DISCOUNT
    npc = cap + (om_yr + fuel_yr) * pv_factor + repl
    crf = _crf(DISCOUNT, PROJECT_YEARS)
    served = max(total_load - unmet, 1.0)
    lcoe = npc * crf / served
    return {
        "pv_kw": pv_kw, "wind_units": wind_units, "batt_kwh": batt_kwh,
        "diesel_kw": diesel_kw,
        "lcoe": float(lcoe), "npc": float(npc),
        "renew_frac": float(renew_served / total_load),
        "unmet_frac": float(unmet / total_load),
        "diesel_kwh": float(diesel_kwh), "capital": float(cap),
    }


if __name__ == "__main__":
    # sanity: a PV-wind-battery config near the poster's optimum
    r = simulate(pv_kw=38.6, wind_units=21, batt_kwh=202, diesel_kw=15)
    for k, v in r.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
