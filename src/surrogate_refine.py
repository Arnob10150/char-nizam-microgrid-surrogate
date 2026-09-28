"""
R3 follow-up: is the surrogate's ~7% solution-quality gap intrinsic, or an
artifact of a one-shot surrogate trained on a uniform sample?

A one-shot surrogate is trained on points spread over the whole box, so it is
least accurate exactly where the optimizer ends up (near the constrained
optimum) -- the region breakdown in the paper already shows this. The standard
fix in surrogate-based optimization is ADAPTIVE REFINEMENT: search the surrogate,
simulate the candidate it returns, add that point to the training set, retrain,
and repeat. Each round costs a handful of simulations, not thousands.

This script measures whether a few refinement rounds close the gap to the
continuous simulator-based search (LCOE 0.2128), and at what simulation cost.

Writes outputs/tables/review_refinement.json.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import dispatch_sim as D
from pipeline import (BOUNDS, FEATURES, MAX_UNMET, MIN_RENEW, SEED,
                      build_dataset, train_surrogate, surrogate_objective)

TAB = ROOT / "outputs" / "tables"; TAB.mkdir(parents=True, exist_ok=True)


def _simulate_row(x):
    r = D.simulate(float(x[0]), int(round(x[1])), float(x[2]), float(x[3]), seed=0)
    return r


def _local_samples(x, n, rng, frac=0.15):
    """Sample n points in a shrinking box around the incumbent x."""
    lo = np.array([b[0] for b in BOUNDS], float)
    hi = np.array([b[1] for b in BOUNDS], float)
    span = (hi - lo) * frac
    pts = rng.normal(np.asarray(x, float), span, size=(n, len(BOUNDS)))
    return np.clip(pts, lo, hi)


def main(rounds=6, per_round=25):
    print("building base dataset ...", flush=True)
    f = TAB / "sim_dataset.csv"
    df = pd.read_csv(f) if f.exists() else build_dataset(1500)[0]
    print(f"  {len(df)} configs", flush=True)

    rng = np.random.default_rng(SEED)
    history = []
    total_extra_sims = 0
    t_start = time.time()

    surr, _ = train_surrogate(df)
    res = differential_evolution(surrogate_objective(surr), BOUNDS, seed=SEED,
                                 maxiter=60, popsize=20, tol=1e-6, polish=True)
    true = _simulate_row(res.x)
    history.append({"round": 0, "extra_sims": 0,
                    "cum_seconds": time.time() - t_start,
                    "lcoe": true["lcoe"], "npc": true["npc"],
                    "pv_kw": float(res.x[0]), "wind_units": int(round(res.x[1])),
                    "batt_kwh": float(res.x[2]), "diesel_kw": float(res.x[3]),
                    "renew_frac": true["renew_frac"], "unmet_frac": true["unmet_frac"],
                    "feasible": bool(true["unmet_frac"] <= MAX_UNMET and
                                     true["renew_frac"] >= MIN_RENEW)})
    print(f"  round 0 (one-shot): LCOE={true['lcoe']:.4f} "
          f"renew={true['renew_frac']:.3f} unmet={true['unmet_frac']:.4f}", flush=True)

    x_inc = res.x
    for k in range(1, rounds + 1):
        # simulate the incumbent plus a local cloud around it, append, retrain
        pts = np.vstack([np.asarray(x_inc, float)[None, :],
                         _local_samples(x_inc, per_round - 1, rng,
                                        frac=0.15 * (0.7 ** (k - 1)))])
        new_rows = [_simulate_row(p) for p in pts]
        total_extra_sims += len(new_rows)
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        surr, _ = train_surrogate(df)
        res = differential_evolution(surrogate_objective(surr), BOUNDS, seed=SEED,
                                     maxiter=60, popsize=20, tol=1e-6, polish=True)
        true = _simulate_row(res.x)
        feas = bool(true["unmet_frac"] <= MAX_UNMET and true["renew_frac"] >= MIN_RENEW)
        history.append({"round": k, "extra_sims": total_extra_sims,
                        "cum_seconds": time.time() - t_start,
                        "lcoe": true["lcoe"], "npc": true["npc"],
                        "pv_kw": float(res.x[0]), "wind_units": int(round(res.x[1])),
                        "batt_kwh": float(res.x[2]), "diesel_kw": float(res.x[3]),
                        "renew_frac": true["renew_frac"],
                        "unmet_frac": true["unmet_frac"], "feasible": feas})
        print(f"  round {k}: +{total_extra_sims:4d} sims  {time.time()-t_start:6.1f}s  "
              f"LCOE={true['lcoe']:.4f}  renew={true['renew_frac']:.3f}  "
              f"unmet={true['unmet_frac']:.4f}  feasible={feas}", flush=True)
        if feas:
            x_inc = res.x

    elapsed = time.time() - t_start

    # reference points measured elsewhere in the review experiments
    ref = {}
    p = TAB / "review_fair_baseline.json"
    if p.exists():
        b = json.load(open(p))
        ref["de_on_simulator"] = {"lcoe": b["lcoe"], "seconds": b["seconds"],
                                  "sim_calls": b["sim_calls"]}
    feas_hist = [h for h in history if h["feasible"]]
    best = min(feas_hist, key=lambda h: h["lcoe"]) if feas_hist else None
    out = {"history": history, "reference": ref, "best_feasible": best,
           "total_extra_sims": total_extra_sims, "wall_clock_s": elapsed,
           "one_shot_lcoe": history[0]["lcoe"]}
    if best and "de_on_simulator" in ref:
        out["gap_one_shot_pct"] = 100.0 * (history[0]["lcoe"] - ref["de_on_simulator"]["lcoe"]) / ref["de_on_simulator"]["lcoe"]
        out["gap_refined_pct"] = 100.0 * (best["lcoe"] - ref["de_on_simulator"]["lcoe"]) / ref["de_on_simulator"]["lcoe"]
        out["sim_budget_ratio"] = (1500 + best["extra_sims"]) / ref["de_on_simulator"]["sim_calls"]
        print(f"\n  one-shot gap vs DE-on-simulator : {out['gap_one_shot_pct']:+.1f}%")
        print(f"  refined  gap vs DE-on-simulator : {out['gap_refined_pct']:+.1f}% "
              f"(after +{best['extra_sims']} sims)")
        print(f"  total sim budget vs DE-on-sim   : "
              f"{1500 + best['extra_sims']} vs {ref['de_on_simulator']['sim_calls']} "
              f"({out['sim_budget_ratio']:.2f}x)")

    json.dump(out, open(TAB / "review_refinement.json", "w"), indent=2)
    print("\nDONE ->", TAB / "review_refinement.json")


if __name__ == "__main__":
    main()
