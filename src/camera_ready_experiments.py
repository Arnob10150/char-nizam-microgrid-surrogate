"""
Camera-ready experiments (IEEE i-COSTE 2026, Review-3).

  C1 assumptions audit  -> every techno-economic constant as implemented, CO2,
                           fuel use; sensitivity of the selected design to the
                           two simplifications the simulator makes (converter
                           losses, component lifetimes beyond battery/genset)
  C2 load profile       -> how the hourly profile is built; seasonal,
                           day-to-day behavioural and growth variants of the
                           165.59 kWh/day load, re-simulated on the design and
                           re-optimised end-to-end under the seasonal profile
  C3 sampling           -> distribution of the 1,500 training configurations,
                           density near the feasibility / renewable boundaries,
                           uniform vs Latin-hypercube accuracy near the boundary
  C4 leakage-free CV    -> nested 5-fold CV: hyper-parameters tuned by an inner
                           3-fold search inside every outer training fold,
                           scaling fitted in-fold (Pipeline), outer fold unseen

Writes outputs/tables/camera_ready.json.  Deterministic (seed 42).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import dispatch_sim as D
from pipeline import BOUNDS, FEATURES, MAX_UNMET, MIN_RENEW, SEED, sample_configs

TAB = ROOT / "outputs" / "tables"
OPT = dict(pv_kw=35.84450690682372, wind_units=2, batt_kwh=7.644850012513643,
           diesel_kw=10.825760703098632)          # surrogate optimum (results.json)
CO2_KG_PER_L = 2.68                               # diesel combustion factor
N_HOUSEHOLDS = 250


# --------------------------------------------------------------------------
# generalised simulator: identical to dispatch_sim.simulate at defaults
# --------------------------------------------------------------------------
def simulate(pv_kw, wind_units, batt_kwh, diesel_kw, load=None, eta_conv=1.0,
             strict_life=False, fuel_price=None, fuel_curve=None):
    """fuel_curve=(a, b): genset burns a*rated_kW + b*output_kWh litres in every
    running hour (HOMER default a=0.08145, b=0.246). None = original linear
    0.1 L/kWh model (LHV_eff 10 kWh/L, no idle term)."""
    solar, wind, base_load = D.synthetic_year(0)
    load = base_load if load is None else load
    conv_kw = D.PEAK_KW * 1.2
    pv_gen = pv_kw * D.DERATE_PV * solar * eta_conv          # PV reaches AC bus via converter
    renew = pv_gen + wind_units * D.wind_power(wind, 10.0)
    soc = 0.5 * batt_kwh; soc_min, soc_max = 0.2 * batt_kwh, batt_kwh
    eta_b = 0.90 * (eta_conv if eta_conv < 1 else 1.0)       # battery path also crosses converter
    fuel_l = unmet = renew_served = diesel_kwh = 0.0
    for t in range(8760):
        net = renew[t] - load[t]
        if net >= 0:
            soc += min(net, soc_max - soc) * eta_b; renew_served += load[t]
        else:
            deficit = -net; renew_served += renew[t]
            fb = min(deficit, (soc - soc_min) * eta_b); soc -= fb / eta_b; deficit -= fb
            if deficit > 1e-9:
                d = min(deficit, diesel_kw); diesel_kwh += d; deficit -= d
                fuel_l += d / D.LHV_DIESEL if fuel_curve is None else fuel_curve[0] * diesel_kw + fuel_curve[1] * d
            if deficit > 1e-9:
                unmet += deficit
    total = load.sum()
    cap = (pv_kw * D.C_PV + wind_units * 10 * D.C_WIND + batt_kwh * D.C_BATT +
           diesel_kw * D.C_DIESEL + conv_kw * D.C_CONV)
    om = pv_kw * 0.02 * D.C_PV + wind_units * D.OM_WIND + batt_kwh * D.OM_BATT + diesel_kw * D.OM_DIESEL / 10
    fp = D.FUEL_PRICE if fuel_price is None else fuel_price
    r, N = D.DISCOUNT, D.PROJECT_YEARS
    if not strict_life:
        repl = (batt_kwh * D.C_BATT + diesel_kw * D.C_DIESEL) * 0.6 / (1 + r) ** 10
    else:   # full replacement at end of every rated life, linear salvage at year N
        repl = 0.0
        for c, life in ((pv_kw * D.C_PV, 25), (wind_units * 10 * D.C_WIND, 20),
                        (batt_kwh * D.C_BATT, 10), (diesel_kw * D.C_DIESEL, 15), (conv_kw * D.C_CONV, 15)):
            y = life
            while y < N:
                repl += c / (1 + r) ** y; y += life
            last = (N // life) * life if N % life else N
            remaining = (last + life - N) if N % life else 0
            repl -= c * remaining / life / (1 + r) ** N
    npc = cap + (om + fuel_l * fp) * (1 - (1 + r) ** -N) / r + repl
    served = max(total - unmet, 1.0)
    return {"lcoe": float(npc * D._crf(r, N) / served), "npc": float(npc),
            "renew_frac": float(renew_served / total), "unmet_frac": float(unmet / total),
            "fuel_l": float(fuel_l), "co2_t": float(fuel_l * CO2_KG_PER_L / 1000),
            "diesel_kwh": float(diesel_kwh), "capital": float(cap), "peak_kw": float(load.max()),
            "annual_kwh": float(total)}


# --------------------------------------------------------------------------
# load variants (all preserve the 165.59 kWh/day annual mean unless growth)
# --------------------------------------------------------------------------
def base_profile():
    return D.synthetic_year(0)[2]


def seasonal(amp, peak_day=150):
    L = base_profile(); day = np.arange(8760) // 24
    f = 1 + amp * np.cos(2 * np.pi * (day - peak_day) / 365)
    L2 = L * f
    return L2 / L2.sum() * L.sum()


def behavioural(sd_day=0.10, sd_hour=0.05, seed=SEED):
    rng = np.random.default_rng(seed); L = base_profile()
    fd = np.repeat(rng.lognormal(0, sd_day, 365), 24)
    fh = rng.lognormal(0, sd_hour, 8760)
    L2 = L * fd * fh
    return L2 / L2.sum() * L.sum()


def realistic(amp=0.15, seed=SEED):
    """seasonal x behavioural x weekly (Friday +5%), mean preserved."""
    L = seasonal(amp) * behavioural(seed=seed) / base_profile()
    day = np.arange(8760) // 24
    L = L * np.where(day % 7 == 4, 1.05, 1.0)
    return L / L.sum() * base_profile().sum()


# --------------------------------------------------------------------------
def dataset(X, load=None):
    rows = [simulate(c[0], int(c[1]), c[2], c[3], load=load) for c in X]
    df = pd.DataFrame(rows); df[FEATURES] = X; df["wind_units"] = df["wind_units"].astype(int)
    return df


def gbr():
    return GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=SEED)


def reoptimise(load):
    """Full surrogate pipeline under an alternative load: sample, simulate,
    train, search, verify on the simulator."""
    X = sample_configs(1500, seed=SEED); df = dataset(X, load)
    Xf = df[FEATURES].values
    m_l = gbr().fit(Xf, df.lcoe); m_r = gbr().fit(Xf, df.renew_frac)
    m_f = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     random_state=SEED).fit(Xf, (df.unmet_frac <= MAX_UNMET).astype(int))

    def f(x):
        x = np.array(x).reshape(1, -1)
        return (m_l.predict(x)[0] + 5 * max(0, .5 - m_f.predict_proba(x)[0, 1])
                + 2 * max(0, MIN_RENEW - m_r.predict(x)[0]))
    res = differential_evolution(f, BOUNDS, seed=SEED, maxiter=60, popsize=20, tol=1e-6, polish=True)
    c = res.x; out = simulate(c[0], int(round(c[1])), c[2], c[3], load=load)
    out.update(pv_kw=float(c[0]), wind_units=int(round(c[1])), batt_kwh=float(c[2]), diesel_kw=float(c[3]))
    return out


def main():
    R = {}
    # ---- sanity: generalised simulator == original ----------------------
    a = D.simulate(**OPT); b = simulate(**OPT)
    assert abs(a["lcoe"] - b["lcoe"]) < 1e-12 and abs(a["unmet_frac"] - b["unmet_frac"]) < 1e-12
    R["baseline_design"] = b

    # ---- C1 assumption audit -----------------------------------------------
    R["assumptions"] = {
        "load_kwh_day": D.DAILY_LOAD_KWH, "peak_kw": D.PEAK_KW, "households": N_HOUSEHOLDS,
        "pv_capex": D.C_PV, "pv_om_per_kw_yr": 0.02 * D.C_PV, "pv_derate": D.DERATE_PV, "pv_life": 25,
        "wind_capex_per_kw": D.C_WIND, "wind_om_per_turbine_yr": D.OM_WIND, "wind_life": 20,
        "wind_ci_r_co": [D.WIND_CI, D.WIND_R, D.WIND_CO], "hub_height_m": 50, "shear_exp": 0.14,
        "batt_capex": D.C_BATT, "batt_om": D.OM_BATT, "batt_eta_oneway": 0.9, "soc_min": 0.2,
        "soc_init": 0.5, "batt_life": 10,
        "diesel_capex": D.C_DIESEL, "diesel_om_per_kw_yr": D.OM_DIESEL / 10, "diesel_life": 15,
        "fuel_kwh_per_l": D.LHV_DIESEL, "fuel_price": D.FUEL_PRICE, "co2_kg_per_l": CO2_KG_PER_L,
        "conv_capex": D.C_CONV, "conv_kw": D.PEAK_KW * 1.2, "conv_life": 15,
        "replacement": "battery+genset once at yr10 at 60% of capital (discounted)",
        "discount": D.DISCOUNT, "years": D.PROJECT_YEARS, "max_unmet": MAX_UNMET, "min_renew": MIN_RENEW}
    R["audit"] = {"converter_eta_0.95": simulate(**OPT, eta_conv=0.95),
                  "strict_lifetimes": simulate(**OPT, strict_life=True),
                  "both": simulate(**OPT, eta_conv=0.95, strict_life=True)}
    # all-diesel reference for emissions avoided
    R["diesel_only"] = simulate(0, 0, 0, 30.0)
    print("C1", json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in R["audit"].items()}, indent=0))

    # ---- C2 load profile ---------------------------------------------------
    L = base_profile(); hod = L[:24]
    R["load_profile"] = {"hourly_kw": [round(x, 3) for x in hod], "peak_kw": float(L.max()),
                         "min_kw": float(L.min()), "load_factor": float(L.mean() / L.max()),
                         "daily_kwh": float(L[:24].sum()), "peak_hour": int(hod.argmax()),
                         "per_household_kwh_day": D.DAILY_LOAD_KWH / N_HOUSEHOLDS}
    variants = {"seasonal_15": seasonal(0.15), "seasonal_25": seasonal(0.25),
                "behavioural": behavioural(), "realistic_combined": realistic(),
                "growth_10": base_profile() * 1.10, "growth_20": base_profile() * 1.20}
    R["load_sensitivity"] = {k: simulate(**OPT, load=v) for k, v in variants.items()}
    for k, v in R["load_sensitivity"].items():
        print(f"C2 {k:20s} LCOE {v['lcoe']:.4f} unmet {v['unmet_frac']*100:.2f}% RF {v['renew_frac']*100:.1f}% peak {v['peak_kw']:.1f}")
    t0 = time.time()
    R["reopt_realistic"] = reoptimise(realistic())
    R["reopt_growth_20"] = reoptimise(base_profile() * 1.20)
    print("C2 reopt", R["reopt_realistic"], R["reopt_growth_20"], f"{time.time()-t0:.0f}s")

    # ---- C3 sampling -------------------------------------------------------
    df = pd.read_csv(TAB / "sim_dataset.csv")
    X = df[FEATURES].values
    u = df.unmet_frac.values; rf = df.renew_frac.values
    R["sampling"] = {
        "scheme": "independent uniform (continuous PV/battery/diesel, integer turbines)",
        "n": int(len(df)), "feasible_rate": float((u <= MAX_UNMET).mean()),
        "near_feas_boundary_n": int(((u > 0.002) & (u < 0.05)).sum()),
        "near_renew_boundary_n": int((np.abs(rf - MIN_RENEW) <= 0.05).sum()),
        "both_constraints_active_n": int(((u <= MAX_UNMET) & (np.abs(rf - MIN_RENEW) <= 0.05)).sum()),
        "infeasible_n": int((u > MAX_UNMET).sum()),
        "infeasible_by_diesel_lt_peak": float(((u > MAX_UNMET) & (X[:, 3] < D.PEAK_KW)).sum() / max((u > MAX_UNMET).sum(), 1)),
        "per_dim_quartile_counts": {f: np.histogram(X[:, i], bins=4, range=BOUNDS[i])[0].tolist()
                                    for i, f in enumerate(FEATURES)},
        "min_pairwise_dist_norm": None}
    # uniform vs Latin hypercube on the same budget, scored on a fresh test set
    lo = np.array([b[0] for b in BOUNDS], float); hi = np.array([b[1] for b in BOUNDS], float)
    lhs = qmc.LatinHypercube(d=4, seed=SEED).random(1500) * (hi - lo) + lo; lhs[:, 1] = np.round(lhs[:, 1])
    Xtest = sample_configs(1000, seed=777); dte = dataset(Xtest)
    near = (np.abs(dte.renew_frac.values - MIN_RENEW) <= 0.05) & (dte.unmet_frac.values <= 0.05)
    comp = {}
    for nm, Xs in (("uniform", X), ("lhs", lhs)):
        dtr = df if nm == "uniform" else dataset(lhs)
        m = gbr().fit(dtr[FEATURES].values, dtr.lcoe.values)
        p = m.predict(dte[FEATURES].values)
        comp[nm] = {"R2_all": float(r2_score(dte.lcoe, p)), "R2_near": float(r2_score(dte.lcoe[near], p[near])),
                    "MAE_near": float(mean_absolute_error(dte.lcoe[near], p[near])), "n_near": int(near.sum())}
    R["sampling"]["uniform_vs_lhs"] = comp
    print("C3", json.dumps(R["sampling"], indent=0, default=str))

    # ---- C4 nested CV ------------------------------------------------------
    grid = {"n_estimators": [200, 400, 800], "max_depth": [2, 3, 4], "learning_rate": [0.03, 0.05, 0.1]}
    outer = KFold(5, shuffle=True, random_state=SEED)
    nested = {}
    for tgt in ("lcoe", "renew_frac"):
        y = df[tgt].values; r2s, maes, chosen = [], [], []
        for tr, te in outer.split(X):
            gs = GridSearchCV(GradientBoostingRegressor(subsample=0.8, random_state=SEED), grid,
                              cv=KFold(3, shuffle=True, random_state=SEED), scoring="r2", n_jobs=-1)
            gs.fit(X[tr], y[tr])                               # tuning sees the training fold only
            p = gs.predict(X[te]); r2s.append(r2_score(y[te], p)); maes.append(mean_absolute_error(y[te], p))
            chosen.append(gs.best_params_)
        nested[tgt] = {"R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)), "R2_min": float(np.min(r2s)),
                       "MAE_mean": float(np.mean(maes)), "chosen": chosen}
    R["nested_cv"] = nested
    print("C4", json.dumps(nested, indent=0))
    json.dump(R, open(TAB / "camera_ready.json", "w"), indent=2)
    print("wrote", TAB / "camera_ready.json")


if __name__ == "__main__":
    main()
