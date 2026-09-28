"""
Camera-ready robustness of the selected design to the two modelling choices a
reader is most likely to question (Review-3, items 2-3):

  * load shape: the synthetic profile carries the surveyed 165.59 kWh/day but
    peaks at 11.2 kW (load factor 0.61); a peak-matched variant keeps the same
    energy and reaches the surveyed 23.32 kW peak (load factor 0.30);
  * genset fuel model: the simulator's linear 0.1 L/kWh (no idle burn) vs the
    HOMER default curve 0.08145 L/h per rated kW + 0.246 L/kWh.

For each variant the published design is re-simulated, and the problem is
re-optimised with differential evolution run DIRECTLY on the simulator (hard
feasibility and >=50% renewable constraints, so no surrogate error enters).
Appends to outputs/tables/camera_ready.json.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_ready_experiments as C
from pipeline import BOUNDS, MAX_UNMET, MIN_RENEW, SEED

HOMER_FUEL = (0.08145, 0.246)
TAB = C.TAB


def peak_matched(peak=C.D.PEAK_KW, daily=C.D.DAILY_LOAD_KWH):
    """Same daily energy, evening peak raised to the surveyed peak: shrink the
    night baseline until max(load) == peak (bisection on the baseline)."""
    h = np.arange(24)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        a = (lo + hi) / 2
        shape = a + np.exp(-((h - 19) ** 2) / 3) + 0.30 * np.exp(-((h - 12) ** 2) / 8)
        prof = shape / shape.sum() * daily
        lo, hi = (a, hi) if prof.max() > peak else (lo, a)
    return np.tile(prof, 365)


def sim_opt(load=None, fuel_curve=None):
    def f(x):
        r = C.simulate(x[0], int(round(x[1])), x[2], x[3], load=load, fuel_curve=fuel_curve)
        return (r["lcoe"] + 10 * max(0, r["unmet_frac"] - MAX_UNMET)
                + 2 * max(0, MIN_RENEW - r["renew_frac"]))
    t0 = time.time()
    res = differential_evolution(f, BOUNDS, seed=SEED, maxiter=60, popsize=20, tol=1e-6, polish=False)
    c = res.x
    out = C.simulate(c[0], int(round(c[1])), c[2], c[3], load=load, fuel_curve=fuel_curve)
    out.update(pv_kw=float(c[0]), wind_units=int(round(c[1])), batt_kwh=float(c[2]),
               diesel_kw=float(c[3]), search_s=time.time() - t0)
    return out


def main():
    R = json.load(open(TAB / "camera_ready.json"))
    Lp = peak_matched()
    Lg = C.base_profile() * 1.20
    cases = {"baseline": (None, None), "peak_matched": (Lp, None), "homer_fuel": (None, HOMER_FUEL),
             "peak_matched+homer_fuel": (Lp, HOMER_FUEL), "growth_20": (Lg, None)}
    E = {"peak_matched_profile_kw": [round(x, 3) for x in Lp[:24]],
         "peak_matched_load_factor": float(Lp.mean() / Lp.max())}
    for k, (L, fc) in cases.items():
        fixed = C.simulate(**C.OPT, load=L, fuel_curve=fc)
        dieselonly = C.simulate(0, 0, 0, 30.0 if L is None else float(np.ceil((L.max()) * 1.05)), load=L, fuel_curve=fc)
        reopt = sim_opt(L, fc)
        E[k] = {"fixed_design": fixed, "diesel_only": dieselonly, "reopt_sim": reopt}
        print(f"{k:24s} fixed LCOE {fixed['lcoe']:.4f} unmet {fixed['unmet_frac']*100:.2f}% RF {fixed['renew_frac']*100:.1f}% | "
              f"diesel-only {dieselonly['lcoe']:.4f} | reopt LCOE {reopt['lcoe']:.4f} unmet {reopt['unmet_frac']*100:.2f}% "
              f"RF {reopt['renew_frac']*100:.1f}% cfg PV {reopt['pv_kw']:.1f} W {reopt['wind_units']} B {reopt['batt_kwh']:.1f} D {reopt['diesel_kw']:.1f} "
              f"CO2 {reopt['co2_t']:.2f} t ({reopt['search_s']:.0f}s)", flush=True)
    R["extra"] = E
    json.dump(R, open(TAB / "camera_ready.json", "w"), indent=2)
    print("wrote", TAB / "camera_ready.json")


if __name__ == "__main__":
    main()
