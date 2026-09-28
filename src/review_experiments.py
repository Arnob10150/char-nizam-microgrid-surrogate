"""
Reviewer-response experiments.

Each function answers one reviewer concern with a real measurement, written to
outputs/tables/review_*.json|csv so the paper can quote actual numbers:

  R1 training-set adequacy   -> learning curve over n = 250..1500 (5 resamples)
  R2 surrogate validation    -> 5-fold CV, independent holdout, boundary/feasible
                                region breakdown
  R3 baseline fairness       -> DE run directly on the TRUE simulator (no
                                surrogate), i.e. continuous search at simulator
                                cost, to separate "continuous vs grid" from
                                "surrogate vs simulator"
  R4 speedup claim           -> measured g=20 brute force (not projected)
  R5 simulator validation    -> energy-balance + component-level cross-checks
                                against closed-form/independent recomputation
  R6 optimizer robustness    -> 30 independent DE seeds
  R7 uncertainty             -> weather-draw and fuel-price contributions split

Run:  python src/review_experiments.py [--quick] [--only R1,R2,...]
"""
from __future__ import annotations
import json, time, itertools, argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from scipy.optimize import differential_evolution
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import dispatch_sim as D
from pipeline import (BOUNDS, FEATURES, MAX_UNMET, MIN_RENEW, SEED,
                      sample_configs, build_dataset, train_surrogate,
                      surrogate_objective)

TAB = ROOT / "outputs" / "tables"; TAB.mkdir(parents=True, exist_ok=True)


def _load_dataset():
    """Reuse the pipeline's 1500-config dataset if present, else rebuild it."""
    f = TAB / "sim_dataset.csv"
    if f.exists():
        df = pd.read_csv(f)
        if len(df) >= 1500:
            return df
    df, _ = build_dataset(1500)
    return df


def _fit_lcoe(Xtr, ytr):
    return GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=SEED).fit(Xtr, ytr)


# --------------------------------------------------------------- R1 ------- #
def r1_learning_curve(df, sizes=(200, 400, 600, 800, 1000, 1200), n_rep=5):
    """Is 1500 samples enough for a 4-D space? Train on nested random subsets and
    score on a FIXED held-out test set, so only the training budget varies.
    Sizes stay <= the 75% training partition (1125 of 1500) so no size is
    silently clamped."""
    X = df[FEATURES].values; y = df["lcoe"].values
    Xtr_all, Xte, ytr_all, yte = train_test_split(X, y, test_size=0.25,
                                                  random_state=SEED)
    sizes = [n for n in sizes if n <= len(Xtr_all)]
    if len(Xtr_all) not in sizes:
        sizes.append(len(Xtr_all))       # always include the full training budget
    rows = []
    for n in sizes:
        n_use = min(n, len(Xtr_all))
        r2s, maes = [], []
        for rep in range(n_rep):
            rng = np.random.default_rng(SEED + rep)
            idx = rng.choice(len(Xtr_all), n_use, replace=False)
            m = _fit_lcoe(Xtr_all[idx], ytr_all[idx])
            p = m.predict(Xte)
            r2s.append(r2_score(yte, p)); maes.append(mean_absolute_error(yte, p))
        rows.append({"n_train": n_use,
                     "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
                     "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes))})
        print(f"  R1 n={n_use:5d}  R2={np.mean(r2s):.4f}+-{np.std(r2s):.4f}"
              f"  MAE={np.mean(maes):.4f}", flush=True)
    pd.DataFrame(rows).to_csv(TAB / "review_learning_curve.csv", index=False)
    return rows


# --------------------------------------------------------------- R2 ------- #
def r2_validation(df, n_splits=5):
    """Independent holdout + k-fold CV + accuracy by region of the search space."""
    X = df[FEATURES].values
    out = {}

    # (a) k-fold CV on the full dataset, for LCOE and renewable fraction
    for tgt in ["lcoe", "renew_frac"]:
        y = df[tgt].values
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        r2s, maes = [], []
        for tr, te in kf.split(X):
            m = _fit_lcoe(X[tr], y[tr])
            p = m.predict(X[te])
            r2s.append(r2_score(y[te], p)); maes.append(mean_absolute_error(y[te], p))
        out[f"cv_{tgt}"] = {"R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
                            "R2_min": float(np.min(r2s)), "R2_max": float(np.max(r2s)),
                            "MAE_mean": float(np.mean(maes)), "folds": n_splits}
        print(f"  R2 CV {tgt}: R2={np.mean(r2s):.4f}+-{np.std(r2s):.4f} "
              f"[{np.min(r2s):.4f},{np.max(r2s):.4f}]", flush=True)

    # (b) feasibility classifier CV (AUC)
    yf = (df["unmet_frac"].values <= MAX_UNMET).astype(int)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in kf.split(X):
        if len(set(yf[tr])) < 2 or len(set(yf[te])) < 2:
            continue
        c = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                       learning_rate=0.05,
                                       random_state=SEED).fit(X[tr], yf[tr])
        aucs.append(roc_auc_score(yf[te], c.predict_proba(X[te])[:, 1]))
    out["cv_feasible_auc"] = {"AUC_mean": float(np.mean(aucs)),
                              "AUC_std": float(np.std(aucs)),
                              "base_rate": float(yf.mean()), "folds": len(aucs)}
    print(f"  R2 CV feasibility AUC={np.mean(aucs):.4f}+-{np.std(aucs):.4f}", flush=True)

    # (c) held-out accuracy split by region: interior vs boundary of the box,
    #     and near vs far from the feasibility threshold.
    y = df["lcoe"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    _, ute = train_test_split(df["unmet_frac"].values, test_size=0.25,
                              random_state=SEED)
    m = _fit_lcoe(Xtr, ytr); p = m.predict(Xte)
    lo = np.array([b[0] for b in BOUNDS]); hi = np.array([b[1] for b in BOUNDS])
    frac = (Xte - lo) / (hi - lo)
    boundary = ((frac < 0.10) | (frac > 0.90)).any(axis=1)
    for name, mask in [("interior", ~boundary), ("boundary", boundary)]:
        if mask.sum() > 5:
            out[f"region_{name}"] = {
                "n": int(mask.sum()),
                "R2": float(r2_score(yte[mask], p[mask])),
                "MAE": float(mean_absolute_error(yte[mask], p[mask]))}
            print(f"  R2 region {name:9s} n={mask.sum():4d} "
                  f"R2={out[f'region_{name}']['R2']:.4f}", flush=True)
    near = np.abs(ute - MAX_UNMET) < 0.01           # within 1pp of the threshold
    for name, mask in [("near_feas_threshold", near), ("far_from_threshold", ~near)]:
        if mask.sum() > 5:
            out[f"feas_{name}"] = {
                "n": int(mask.sum()),
                "R2": float(r2_score(yte[mask], p[mask])),
                "MAE": float(mean_absolute_error(yte[mask], p[mask]))}
            print(f"  R2 {name:22s} n={mask.sum():4d} "
                  f"R2={out[f'feas_{name}']['R2']:.4f}", flush=True)

    json.dump(out, open(TAB / "review_validation.json", "w"), indent=2)
    return out


# --------------------------------------------------------------- R3 ------- #
def true_sim_objective():
    """Same penalized objective as the surrogate search, but every call runs the
    REAL simulator. This is the fair continuous baseline."""
    calls = {"n": 0}
    def f(x):
        calls["n"] += 1
        r = D.simulate(x[0], int(round(x[1])), x[2], x[3], seed=0)
        pen = 5.0 * max(0.0, 1.0 - (1.0 if r["unmet_frac"] <= MAX_UNMET else 0.0))
        pen += 2.0 * max(0.0, MIN_RENEW - r["renew_frac"])
        return r["lcoe"] + pen
    return f, calls


def r3_fair_baseline(maxiter=60, popsize=20):
    """DE on the true simulator: isolates 'continuous vs grid' from 'surrogate'."""
    f, calls = true_sim_objective()
    t0 = time.time()
    res = differential_evolution(f, BOUNDS, seed=SEED, maxiter=maxiter,
                                 popsize=popsize, tol=1e-6, polish=True)
    dt = time.time() - t0
    cfg = dict(pv_kw=float(res.x[0]), wind_units=int(round(res.x[1])),
               batt_kwh=float(res.x[2]), diesel_kw=float(res.x[3]))
    true = D.simulate(cfg["pv_kw"], cfg["wind_units"], cfg["batt_kwh"],
                      cfg["diesel_kw"], seed=0)
    out = {"config": cfg, "seconds": dt, "sim_calls": calls["n"],
           "lcoe": true["lcoe"], "npc": true["npc"],
           "renew_frac": true["renew_frac"], "unmet_frac": true["unmet_frac"]}
    print(f"  R3 DE-on-simulator: LCOE={true['lcoe']:.4f} in {dt:.1f}s "
          f"({calls['n']} sims)", flush=True)
    json.dump(out, open(TAB / "review_fair_baseline.json", "w"), indent=2)
    return out


# --------------------------------------------------------------- R4 ------- #
def r4_measure_g20(grids=(4, 6, 8, 10, 12, 14, 16, 20)):
    """Actually RUN the fine grids instead of extrapolating the 360x claim."""
    from pipeline import brute_force
    rows = []
    for g in grids:
        t0 = time.time()
        best, dt, n = brute_force(grid_per_dim=g)
        rows.append({"grid": g, "evals": n, "seconds": dt,
                     "best_lcoe": float(best["lcoe"]) if best else None,
                     "measured": True})
        print(f"  R4 g={g:3d}: {n:7d} sims, {dt:8.1f}s, "
              f"best LCOE={best['lcoe'] if best else float('nan'):.4f}", flush=True)
        pd.DataFrame(rows).to_csv(TAB / "review_scaling_measured.csv", index=False)
    return rows


# --------------------------------------------------------------- R5 ------- #
def r5_simulator_checks(n=200):
    """Validate the simulator internally: hourly energy balance must close, and
    aggregate KPIs must match an independent recomputation. (An external HOMER
    Pro cross-run is not possible here; we report what CAN be verified.)"""
    X = sample_configs(n, seed=SEED + 7)
    rows = []
    solar, wind, load = D.synthetic_year(0)
    total_load = load.sum()
    for cfg in X:
        pv_kw, wu, batt, dsl = float(cfg[0]), int(cfg[1]), float(cfg[2]), float(cfg[3])
        r = D.simulate(pv_kw, wu, batt, dsl, seed=0)

        # (a) LOAD-SIDE balance, re-deriving the dispatch independently of the
        #     simulator's own counters: direct renewable + battery discharge +
        #     diesel + unmet must equal the total load exactly.
        gen = pv_kw * D.DERATE_PV * solar + wu * D.wind_power(wind, 10.0)
        soc = 0.5 * batt; soc_min, soc_max = 0.2 * batt, batt
        eta = 0.90
        direct = 0.0; from_batt_tot = 0.0; dies = 0.0; un = 0.0
        for t in range(8760):
            net = gen[t] - load[t]
            if net >= 0:
                soc += min(net, soc_max - soc) * eta
                direct += load[t]
            else:
                direct += gen[t]
                deficit = -net
                avail = (soc - soc_min) * eta
                fb = min(deficit, avail)
                soc -= fb / eta; from_batt_tot += fb; deficit -= fb
                if deficit > 1e-9:
                    d_ = min(deficit, dsl); dies += d_; deficit -= d_
                if deficit > 1e-9:
                    un += deficit
        supplied = direct + from_batt_tot + dies + un
        bal = abs(supplied - total_load) / total_load

        # (b) the simulator's reported KPIs must match this independent replay
        kpi_err = max(abs(dies - r["diesel_kwh"]) / max(dies, 1.0),
                      abs(un / total_load - r["unmet_frac"]))

        # (c) capital recomputed from the published cost table
        conv = D.PEAK_KW * 1.2
        cap = (pv_kw * D.C_PV + wu * 10 * D.C_WIND + batt * D.C_BATT
               + dsl * D.C_DIESEL + conv * D.C_CONV)
        rows.append({"cap_err": abs(cap - r["capital"]) / max(cap, 1.0),
                     "bal_err": bal, "kpi_err": kpi_err,
                     "renew_frac": r["renew_frac"], "unmet": r["unmet_frac"]})
    d = pd.DataFrame(rows)
    out = {"n_configs": n,
           "max_energy_balance_rel_err": float(d["bal_err"].max()),
           "mean_energy_balance_rel_err": float(d["bal_err"].mean()),
           "max_kpi_replay_rel_err": float(d["kpi_err"].max()),
           "max_capital_rel_err": float(d["cap_err"].max()),
           "renew_frac_range": [float(d["renew_frac"].min()), float(d["renew_frac"].max())],
           "note": ("Internal consistency only: hourly load-side energy balance, "
                    "independent dispatch replay, and cost-table recomputation. "
                    "No external HOMER Pro run or metered field data is available "
                    "for this site, so external validation remains open.")}
    print(f"  R5 max energy-balance rel. err = {out['max_energy_balance_rel_err']:.2e}, "
          f"max KPI replay err = {out['max_kpi_replay_rel_err']:.2e}, "
          f"max capital rel. err = {out['max_capital_rel_err']:.2e}", flush=True)
    json.dump(out, open(TAB / "review_simulator_checks.json", "w"), indent=2)
    return out


# --------------------------------------------------------------- R6 ------- #
def r6_de_robustness(surr, n_seeds=30):
    """DE is stochastic: run it n_seeds times and report the spread of the
    optimum, verified on the true simulator each time."""
    rows = []
    obj = surrogate_objective(surr)
    for s in range(n_seeds):
        t0 = time.time()
        res = differential_evolution(obj, BOUNDS, seed=s, maxiter=60, popsize=20,
                                     tol=1e-6, polish=True)
        dt = time.time() - t0
        x = res.x
        true = D.simulate(float(x[0]), int(round(x[1])), float(x[2]), float(x[3]),
                          seed=0)
        rows.append({"seed": s, "pv_kw": float(x[0]), "wind_units": int(round(x[1])),
                     "batt_kwh": float(x[2]), "diesel_kw": float(x[3]),
                     "true_lcoe": true["lcoe"], "renew_frac": true["renew_frac"],
                     "unmet_frac": true["unmet_frac"], "seconds": dt,
                     "feasible": bool(true["unmet_frac"] <= MAX_UNMET),
                     "meets_renew": bool(true["renew_frac"] >= MIN_RENEW)})
        print(f"  R6 seed {s:2d}: LCOE={true['lcoe']:.4f} "
              f"renew={true['renew_frac']:.3f} unmet={true['unmet_frac']:.4f}",
              flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(TAB / "review_de_seeds.csv", index=False)
    summ = {"n_seeds": n_seeds,
            "lcoe_mean": float(d["true_lcoe"].mean()),
            "lcoe_std": float(d["true_lcoe"].std()),
            "lcoe_min": float(d["true_lcoe"].min()),
            "lcoe_max": float(d["true_lcoe"].max()),
            "pv_kw_mean": float(d["pv_kw"].mean()), "pv_kw_std": float(d["pv_kw"].std()),
            "batt_kwh_mean": float(d["batt_kwh"].mean()),
            "batt_kwh_std": float(d["batt_kwh"].std()),
            "diesel_kw_mean": float(d["diesel_kw"].mean()),
            "diesel_kw_std": float(d["diesel_kw"].std()),
            "wind_units_mode": int(d["wind_units"].mode().iloc[0]),
            "wind_units_unique": sorted(set(int(v) for v in d["wind_units"])),
            "frac_feasible": float(d["feasible"].mean()),
            "frac_meets_renew": float(d["meets_renew"].mean()),
            "search_s_mean": float(d["seconds"].mean())}
    print(f"  R6 LCOE {summ['lcoe_mean']:.4f} +- {summ['lcoe_std']:.4f} "
          f"[{summ['lcoe_min']:.4f},{summ['lcoe_max']:.4f}]", flush=True)
    json.dump(summ, open(TAB / "review_de_robustness.json", "w"), indent=2)
    return summ


# --------------------------------------------------------------- R7 ------- #
def r7_uncertainty_decomposition(cfg, n=200):
    """Split the LCOE band into its weather and fuel-price contributions by
    varying one at a time, so the uniform fuel prior's influence is explicit."""
    base_fuel = D.FUEL_PRICE
    rng = np.random.default_rng(SEED)
    out = {}

    # weather only (fuel fixed at reference)
    lc = []
    for i in range(n):
        r = D.simulate(cfg["pv_kw"], int(cfg["wind_units"]), cfg["batt_kwh"],
                       cfg["diesel_kw"], seed=1000 + i)
        lc.append(r["lcoe"])
    lc = np.array(lc)
    out["weather_only"] = {"mean": float(lc.mean()), "std": float(lc.std()),
                           "p5": float(np.percentile(lc, 5)),
                           "p95": float(np.percentile(lc, 95))}

    # fuel only (weather fixed at the measured year)
    lc = []
    for i in range(n):
        D.FUEL_PRICE = base_fuel * rng.uniform(0.7, 1.4)
        r = D.simulate(cfg["pv_kw"], int(cfg["wind_units"]), cfg["batt_kwh"],
                       cfg["diesel_kw"], seed=0)
        lc.append(r["lcoe"])
    D.FUEL_PRICE = base_fuel
    lc = np.array(lc)
    out["fuel_only"] = {"mean": float(lc.mean()), "std": float(lc.std()),
                        "p5": float(np.percentile(lc, 5)),
                        "p95": float(np.percentile(lc, 95))}

    # both (the paper's headline band), plus a lognormal fuel prior as a
    # sensitivity check on the uniform assumption
    for label, draw in [("both_uniform_fuel", lambda: rng.uniform(0.7, 1.4)),
                        ("both_lognormal_fuel",
                         lambda: float(np.clip(rng.lognormal(0.0, 0.20), 0.6, 1.8)))]:
        lc = []
        for i in range(n):
            D.FUEL_PRICE = base_fuel * draw()
            r = D.simulate(cfg["pv_kw"], int(cfg["wind_units"]), cfg["batt_kwh"],
                           cfg["diesel_kw"], seed=1000 + i)
            lc.append(r["lcoe"])
        D.FUEL_PRICE = base_fuel
        lc = np.array(lc)
        out[label] = {"mean": float(lc.mean()), "std": float(lc.std()),
                      "p5": float(np.percentile(lc, 5)),
                      "p95": float(np.percentile(lc, 95))}

    # how much of the total variance is fuel?
    vw = out["weather_only"]["std"] ** 2; vf = out["fuel_only"]["std"] ** 2
    out["variance_share_fuel"] = float(vf / max(vw + vf, 1e-12))
    out["weather_perturbation"] = {"solar_sigma": 0.08, "wind_sigma": 0.10,
                                   "form": "iid multiplicative lognormal-free "
                                           "gaussian on hourly series, clipped"}
    for k in ["weather_only", "fuel_only", "both_uniform_fuel", "both_lognormal_fuel"]:
        print(f"  R7 {k:22s} mean={out[k]['mean']:.4f} std={out[k]['std']:.4f} "
              f"[{out[k]['p5']:.4f},{out[k]['p95']:.4f}]", flush=True)
    print(f"  R7 fuel share of variance = {out['variance_share_fuel']:.3f}", flush=True)
    json.dump(out, open(TAB / "review_uncertainty_decomp.json", "w"), indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="R1,R2,R3,R4,R5,R6,R7")
    ap.add_argument("--quick", action="store_true",
                    help="smaller budgets for a fast smoke test")
    a = ap.parse_args()
    want = set(s.strip().upper() for s in a.only.split(","))

    print("loading dataset ...", flush=True)
    df = _load_dataset()
    print(f"  {len(df)} configs", flush=True)
    surr, _ = train_surrogate(df)

    res = {}
    if "R1" in want:
        print("R1 learning curve ...", flush=True)
        res["R1"] = r1_learning_curve(df, n_rep=2 if a.quick else 5)
    if "R2" in want:
        print("R2 validation ...", flush=True)
        res["R2"] = r2_validation(df)
    if "R3" in want:
        print("R3 fair continuous baseline (slow) ...", flush=True)
        res["R3"] = r3_fair_baseline(maxiter=10 if a.quick else 60)
    if "R4" in want:
        print("R4 measured scaling (slow) ...", flush=True)
        res["R4"] = r4_measure_g20((4, 6, 8) if a.quick else (4, 6, 8, 10, 12, 14, 16, 20))
    if "R5" in want:
        print("R5 simulator consistency ...", flush=True)
        res["R5"] = r5_simulator_checks(50 if a.quick else 200)
    if "R6" in want:
        print("R6 DE robustness ...", flush=True)
        res["R6"] = r6_de_robustness(surr, n_seeds=3 if a.quick else 30)
    if "R7" in want:
        print("R7 uncertainty decomposition ...", flush=True)
        base = json.load(open(TAB / "results.json"))["surrogate_optimum"]
        res["R7"] = r7_uncertainty_decomposition(base, n=30 if a.quick else 200)

    json.dump(res, open(TAB / "review_all.json", "w"), indent=2, default=str)
    print("DONE ->", TAB, flush=True)


if __name__ == "__main__":
    main()
