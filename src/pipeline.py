"""
Surrogate-and-search microgrid sizing (the CS/ML contribution).

Replaces HOMER Pro's exhaustive brute-force sizing search with:
  1. a physics-based dispatch simulator (dispatch_sim, the ground-truth oracle);
  2. a learned SURROGATE (gradient-boosted trees) that predicts LCOE / renewable
     fraction / unmet fraction from a sizing configuration in microseconds;
  3. an EVOLUTIONARY SEARCH (differential evolution) over the surrogate to find a
     near-optimal, feasible configuration far faster than brute force;
  4. UNCERTAINTY QUANTIFICATION: the optimum is re-simulated under many weather /
     fuel-price draws to report an LCOE distribution, which a single HOMER point
     estimate cannot give.

Outputs (CSV/JSON) drive the paper's tables and figures. Deterministic (seed 42).
"""
from __future__ import annotations
import json, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from scipy.optimize import differential_evolution
import warnings; warnings.filterwarnings("ignore")
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import dispatch_sim as D

SEED = 42; np.random.seed(SEED)
TAB = ROOT / "outputs" / "tables"; TAB.mkdir(parents=True, exist_ok=True)

# search space (component sizing bounds)
BOUNDS = [(0.0, 120.0),    # pv_kw
          (0, 40),         # wind_units (10 kW each)
          (0.0, 500.0),    # batt_kwh
          (0.0, 40.0)]     # diesel_kw
FEATURES = ["pv_kw", "wind_units", "batt_kwh", "diesel_kw"]
MAX_UNMET = 0.01           # feasibility: <=1% unmet load
MIN_RENEW = 0.50           # environmental target: >=50% renewable (poster goal)


def sample_configs(n, seed=0):
    rng = np.random.default_rng(seed)
    pv = rng.uniform(*BOUNDS[0], n)
    wind = rng.integers(BOUNDS[1][0], BOUNDS[1][1] + 1, n)
    batt = rng.uniform(*BOUNDS[2], n)
    diesel = rng.uniform(*BOUNDS[3], n)
    return np.column_stack([pv, wind, batt, diesel])


def build_dataset(n=1500):
    """Run the true simulator on n random configs -> training data."""
    X = sample_configs(n, seed=SEED)
    rows = []
    t0 = time.time()
    for cfg in X:
        r = D.simulate(cfg[0], int(cfg[1]), cfg[2], cfg[3], seed=0)
        rows.append(r)
    sim_time = time.time() - t0
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "sim_dataset.csv", index=False)
    return df, sim_time / n


def train_surrogate(df):
    X = df[FEATURES].values
    surrogates, metrics = {}, {}
    for tgt in ["lcoe", "renew_frac"]:                        # regression targets
        y = df[tgt].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
        m = GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=SEED)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        metrics[tgt] = {"R2": float(r2_score(yte, pred)),
                        "MAE": float(mean_absolute_error(yte, pred))}
        surrogates[tgt] = m
    # feasibility as a CLASSIFIER (unmet <= MAX_UNMET), which is the right frame
    yf = (df["unmet_frac"].values <= MAX_UNMET).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, yf, test_size=0.25, random_state=SEED)
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                     learning_rate=0.05, random_state=SEED).fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])) if len(set(yte)) > 1 else 1.0
    metrics["feasible"] = {"AUC": auc, "base_rate": float(yf.mean())}
    surrogates["feasible"] = clf
    json.dump(metrics, open(TAB / "surrogate_metrics.json", "w"), indent=2)
    return surrogates, metrics


def benchmark_models(df):
    """Train several ML model families as the LCOE surrogate and compare accuracy
    and inference speed. This justifies the gradient-boosted choice: it must win
    the accuracy/latency trade-off that the search loop actually depends on."""
    X = df[FEATURES].values
    y = df["lcoe"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    models = {
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale")),
        "MLP (64,64)": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000,
                         random_state=SEED)),
        "Random forest": RandomForestRegressor(n_estimators=400, max_depth=12,
                                               random_state=SEED, n_jobs=-1),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=SEED),
    }
    rows = []
    x1 = Xte[:1]
    for name, m in models.items():
        t0 = time.time(); m.fit(Xtr, ytr); fit_s = time.time() - t0
        pred = m.predict(Xte)
        # per-query inference latency (what the DE search pays each evaluation)
        t0 = time.time()
        for _ in range(2000):
            m.predict(x1)
        infer_ms = 1000 * (time.time() - t0) / 2000
        rows.append({
            "model": name,
            "R2": float(r2_score(yte, pred)),
            "MAE": float(mean_absolute_error(yte, pred)),
            "fit_s": float(fit_s),
            "infer_ms": float(infer_ms),
        })
    bench = pd.DataFrame(rows)
    bench.to_csv(TAB / "model_benchmark.csv", index=False)
    return bench


def surrogate_objective(surr):
    """Penalized LCOE: the feasibility classifier gates configs likely to leave
    load unmet (P(feasible) below 0.5 pays a penalty proportional to the gap)."""
    def f(x):
        x = np.array(x).reshape(1, -1)
        lcoe = surr["lcoe"].predict(x)[0]
        pfeas = surr["feasible"].predict_proba(x)[0, 1]
        renew = surr["renew_frac"].predict(x)[0]
        pen = 5.0 * max(0.0, 0.5 - pfeas)              # feasibility gate
        pen += 2.0 * max(0.0, MIN_RENEW - renew)       # environmental constraint
        return lcoe + pen
    return f


def search_surrogate(surr):
    t0 = time.time()
    res = differential_evolution(surrogate_objective(surr), BOUNDS, seed=SEED,
                                 maxiter=60, popsize=20, tol=1e-6, polish=True)
    dt = time.time() - t0
    x = res.x
    return dict(pv_kw=float(x[0]), wind_units=int(round(x[1])),
                batt_kwh=float(x[2]), diesel_kw=float(x[3])), dt


def brute_force(grid_per_dim=8):
    """HOMER-style exhaustive search on the TRUE simulator (the expensive path)."""
    pv = np.linspace(0, 120, grid_per_dim)
    wind = np.linspace(0, 40, grid_per_dim).round().astype(int)
    batt = np.linspace(0, 500, grid_per_dim)
    diesel = np.linspace(0, 40, grid_per_dim)
    best = None; n = 0; t0 = time.time()
    for p, w, b, d in itertools.product(pv, wind, batt, diesel):
        r = D.simulate(p, int(w), b, d, seed=0); n += 1
        if r["unmet_frac"] <= MAX_UNMET and r["renew_frac"] >= MIN_RENEW:
            if best is None or r["lcoe"] < best["lcoe"]:
                best = r
    dt = time.time() - t0
    return best, dt, n


def uncertainty(cfg, n_draws=200):
    """Re-simulate the optimum under many weather + fuel-price draws."""
    rng = np.random.default_rng(SEED)
    lcoes = []
    base_fuel = D.FUEL_PRICE
    for i in range(n_draws):
        D.FUEL_PRICE = base_fuel * rng.uniform(0.7, 1.4)      # fuel volatility
        r = D.simulate(cfg["pv_kw"], int(cfg["wind_units"]), cfg["batt_kwh"],
                       cfg["diesel_kw"], seed=1000 + i)
        lcoes.append(r["lcoe"])
    D.FUEL_PRICE = base_fuel
    lcoes = np.array(lcoes)
    return {"mean": float(lcoes.mean()), "std": float(lcoes.std()),
            "p5": float(np.percentile(lcoes, 5)),
            "p95": float(np.percentile(lcoes, 95)),
            "samples": lcoes.tolist()}


def main():
    print("building simulator dataset ...", flush=True)
    df, per_sim_s = build_dataset(1500)
    print(f"  {len(df)} configs, {per_sim_s*1000:.1f} ms per true simulation", flush=True)

    surr, metrics = train_surrogate(df)
    print("surrogate:", {k: (round(v.get('R2', v.get('AUC')), 4)) for k, v in metrics.items()}, flush=True)

    print("benchmarking ML model families ...", flush=True)
    bench = benchmark_models(df)
    print(bench.to_string(index=False), flush=True)

    # timing: single surrogate eval vs single true simulation
    x0 = np.array([[38.6, 21, 202, 15]])
    t0 = time.time()
    for _ in range(1000):
        surr["lcoe"].predict(x0)
    surr_eval_ms = 1000 * (time.time() - t0) / 1000

    print("surrogate search ...", flush=True)
    opt_cfg, search_s = search_surrogate(surr)
    # verify the surrogate optimum on the TRUE simulator
    opt_true = D.simulate(opt_cfg["pv_kw"], int(opt_cfg["wind_units"]),
                          opt_cfg["batt_kwh"], opt_cfg["diesel_kw"], seed=0)

    print("brute-force baseline (HOMER-style) ...", flush=True)
    bf_best, bf_s, bf_n = brute_force(grid_per_dim=8)

    # scaling: brute-force cost grows as g^4; surrogate search stays ~constant.
    # We time brute force at increasing grid resolution and project the crossover.
    scaling = []
    for g in (4, 6, 8, 10):
        b, s, n = brute_force(grid_per_dim=g)
        scaling.append({"grid": g, "evals": n, "seconds": s,
                        "best_lcoe": round(b["lcoe"], 4) if b else None})
        print(f"  brute g={g}: {n} sims, {s:.1f}s", flush=True)
    # projected time for a HOMER-like fine grid (g=20 -> 160k configs)
    per_sim = scaling[-1]["seconds"] / scaling[-1]["evals"]
    proj_20 = per_sim * (20 ** 4)

    unc = uncertainty(opt_cfg, 200)

    result = {
        "per_true_sim_ms": per_sim_s * 1000,
        "surrogate_eval_ms": surr_eval_ms,
        "surrogate_metrics": metrics,
        "surrogate_optimum": opt_cfg,
        "surrogate_optimum_true": {k: opt_true[k] for k in
                                   ["lcoe", "npc", "renew_frac", "unmet_frac"]},
        "surrogate_search_s": search_s,
        "bruteforce_optimum": {k: bf_best[k] for k in FEATURES} if bf_best else None,
        "bruteforce_kpi": {k: bf_best[k] for k in
                           ["lcoe", "npc", "renew_frac", "unmet_frac"]} if bf_best else None,
        "bruteforce_s": bf_s, "bruteforce_evals": bf_n,
        "speedup_vs_bruteforce": bf_s / max(search_s, 1e-9),
        "scaling": scaling,
        "projected_bruteforce_g20_s": proj_20,
        "projected_speedup_g20": proj_20 / max(search_s, 1e-9),
        "lcoe_uncertainty": {k: unc[k] for k in ["mean", "std", "p5", "p95"]},
        "model_benchmark": bench.to_dict(orient="records"),
    }
    # explicit "beats HOMER-style search" deltas (cost + speed), from real numbers
    if bf_best:
        result["cost_reduction_vs_bruteforce_pct"] = float(
            100.0 * (bf_best["lcoe"] - opt_true["lcoe"]) / bf_best["lcoe"])
    pd.DataFrame(scaling).to_csv(TAB / "scaling.csv", index=False)
    json.dump({**result, "unc_samples": unc["samples"]},
              open(TAB / "results.json", "w"), indent=2)
    # readable summary
    print("\n=== RESULTS ===")
    print("surrogate optimum:", opt_cfg)
    print("  true LCOE at optimum:", round(opt_true["lcoe"], 4),
          "renew:", round(opt_true["renew_frac"], 3),
          "unmet:", round(opt_true["unmet_frac"], 4))
    if bf_best:
        print("brute-force optimum LCOE:", round(bf_best["lcoe"], 4),
              f"({bf_n} sims, {bf_s:.1f}s)")
    print(f"surrogate search {search_s:.3f}s  -> speedup "
          f"{result['speedup_vs_bruteforce']:.0f}x")
    print("LCOE uncertainty: mean {:.3f} +/- {:.3f}  [{:.3f}, {:.3f}]".format(
        unc["mean"], unc["std"], unc["p5"], unc["p95"]))
    print("DONE ->", TAB)


if __name__ == "__main__":
    main()
