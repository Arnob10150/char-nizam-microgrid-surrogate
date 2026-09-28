"""
R5: external cross-check of the dispatch simulator against a published HOMER Pro
study of the SAME site (poster P0050, Char Nizam, Bhola, Bangladesh).

That study reports, for the same load (165.59 kWh/day, 23.32 kW peak) and the
same component cost table (PV 1300 $/kW, wind 2300 $/kW on 10 kW turbines,
battery 300 $/kWh, converter 300 $/kW, fuel ~0.87 $/L, 25-year horizon):

    configuration : 38.6 kW PV, 21 x 10 kW turbines, 202 kWh battery
    LCOE          : 0.426 $/kWh
    NPC           : 332,688 $
    initial capital: 180,135 $

This is not a matched-input replication (HOMER's dispatch, degradation and
replacement schedules differ from ours, and the poster does not publish its
diesel rating or hourly series), but it IS an independent, published evaluation
of the same design problem, so running OUR simulator on THEIR configuration
gives a real external reference point instead of internal consistency only.

Writes outputs/tables/review_homer_crosscheck.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import dispatch_sim as D

TAB = ROOT / "outputs" / "tables"; TAB.mkdir(parents=True, exist_ok=True)

# --- published HOMER Pro result for this site (poster P0050) --------------- #
HOMER = {
    "pv_kw": 38.6,
    "wind_units": 21,          # 21 x 10 kW
    "batt_kwh": 202.0,
    "lcoe": 0.426,
    "npc": 332688.0,
    "capital": 180135.0,
}


def main():
    out = {"reference": "HOMER Pro study of Char Nizam (poster P0050)",
           "homer_reported": HOMER}

    # Our simulator on THEIR configuration. The poster does not state a diesel
    # rating (it reports a 100% renewable design), so we sweep a small range and
    # report the 0 kW case as the like-for-like comparison.
    rows = []
    for dsl in (0.0, 5.0, 10.0, 15.0, 23.32):
        r = D.simulate(HOMER["pv_kw"], HOMER["wind_units"], HOMER["batt_kwh"],
                       dsl, seed=0)
        rows.append({"diesel_kw": dsl, "lcoe": r["lcoe"], "npc": r["npc"],
                     "capital": r["capital"], "renew_frac": r["renew_frac"],
                     "unmet_frac": r["unmet_frac"]})
        print(f"  diesel={dsl:5.1f} kW  LCOE={r['lcoe']:.4f}  NPC={r['npc']:,.0f}  "
              f"renew={r['renew_frac']:.3f}  unmet={r['unmet_frac']:.4f}", flush=True)
    out["ours_on_homer_config"] = rows

    base = rows[0]                      # 0 kW diesel: their 100%-renewable case
    out["comparison_zero_diesel"] = {
        "lcoe_ours": base["lcoe"], "lcoe_homer": HOMER["lcoe"],
        "lcoe_rel_diff_pct": 100.0 * (base["lcoe"] - HOMER["lcoe"]) / HOMER["lcoe"],
        "npc_ours": base["npc"], "npc_homer": HOMER["npc"],
        "npc_rel_diff_pct": 100.0 * (base["npc"] - HOMER["npc"]) / HOMER["npc"],
        "capital_ours": base["capital"], "capital_homer": HOMER["capital"],
        "capital_rel_diff_pct": 100.0 * (base["capital"] - HOMER["capital"]) / HOMER["capital"],
    }
    c = out["comparison_zero_diesel"]
    print(f"\n  LCOE   ours {c['lcoe_ours']:.4f} vs HOMER {c['lcoe_homer']:.4f} "
          f"({c['lcoe_rel_diff_pct']:+.1f}%)")
    print(f"  NPC    ours {c['npc_ours']:,.0f} vs HOMER {c['npc_homer']:,.0f} "
          f"({c['npc_rel_diff_pct']:+.1f}%)")
    print(f"  capital ours {c['capital_ours']:,.0f} vs HOMER {c['capital_homer']:,.0f} "
          f"({c['capital_rel_diff_pct']:+.1f}%)")

    # Capital is the component most nearly comparable across tools: a pure
    # function of the published cost table and the sizing, with no dispatch or
    # replacement modelling in it. It therefore isolates WHERE the tools diverge.
    conv_kw = D.PEAK_KW * 1.2
    cap_per_kw = (HOMER["pv_kw"] * 1300 + HOMER["wind_units"] * 10 * 2300
                  + HOMER["batt_kwh"] * 300 + conv_kw * 300)
    # Alternative reading of the same table: 2300 $ per TURBINE, not per kW.
    cap_per_turbine = (HOMER["pv_kw"] * 1300 + HOMER["wind_units"] * 2300
                       + HOMER["batt_kwh"] * 300 + conv_kw * 300)
    out["capital_reconciliation"] = {
        "homer_reported_capital": HOMER["capital"],
        "wind_at_2300_per_kW": {
            "value": cap_per_kw,
            "rel_diff_pct": 100.0 * (cap_per_kw - HOMER["capital"]) / HOMER["capital"]},
        "wind_at_2300_per_turbine": {
            "value": cap_per_turbine,
            "rel_diff_pct": 100.0 * (cap_per_turbine - HOMER["capital"]) / HOMER["capital"]},
        "wind_line_item_per_kW": HOMER["wind_units"] * 10 * 2300,
        "finding": ("Under the per-kW reading the wind line item alone "
                    "(483,000 $) exceeds the poster's entire reported capital "
                    "(180,135 $), so the published figures cannot be reconciled "
                    "with a per-kW turbine cost. The per-turbine reading "
                    "reproduces the reported capital to within ~7%."),
    }
    print(f"\n  capital @2300/kW      : {cap_per_kw:,.0f} "
          f"({out['capital_reconciliation']['wind_at_2300_per_kW']['rel_diff_pct']:+.1f}%)")
    print(f"  capital @2300/turbine : {cap_per_turbine:,.0f} "
          f"({out['capital_reconciliation']['wind_at_2300_per_turbine']['rel_diff_pct']:+.1f}%)")
    print(f"  wind line item @/kW   : {HOMER['wind_units']*10*2300:,.0f} "
          f"(exceeds poster's total capital {HOMER['capital']:,.0f})")

    # Re-run OUR simulator under the per-turbine reading, which is the only one
    # consistent with the poster's own totals, to get a like-for-like LCOE.
    saved = D.C_WIND
    try:
        D.C_WIND = 230.0        # 2300 $/turbine == 230 $/kW on a 10 kW machine
        alt = [dict(diesel_kw=d,
                    **{k: v for k, v in D.simulate(HOMER["pv_kw"], HOMER["wind_units"],
                                                   HOMER["batt_kwh"], d, seed=0).items()
                       if k in ("lcoe", "npc", "capital", "renew_frac", "unmet_frac")})
               for d in (0.0, 5.0)]
    finally:
        D.C_WIND = saved
    out["ours_under_per_turbine_reading"] = alt
    a0 = alt[0]
    out["comparison_per_turbine_reading"] = {
        "lcoe_ours": a0["lcoe"], "lcoe_homer": HOMER["lcoe"],
        "lcoe_rel_diff_pct": 100.0 * (a0["lcoe"] - HOMER["lcoe"]) / HOMER["lcoe"],
        "npc_ours": a0["npc"], "npc_homer": HOMER["npc"],
        "npc_rel_diff_pct": 100.0 * (a0["npc"] - HOMER["npc"]) / HOMER["npc"],
    }
    print(f"\n  under per-turbine reading, ours on HOMER's config:")
    print(f"    LCOE ours {a0['lcoe']:.4f} vs HOMER {HOMER['lcoe']:.4f} "
          f"({out['comparison_per_turbine_reading']['lcoe_rel_diff_pct']:+.1f}%)")
    print(f"    NPC  ours {a0['npc']:,.0f} vs HOMER {HOMER['npc']:,.0f} "
          f"({out['comparison_per_turbine_reading']['npc_rel_diff_pct']:+.1f}%)")

    # Also: what does OUR optimizer's design cost under our simulator, versus
    # the HOMER design under our simulator? Same evaluator, both designs -> the
    # comparison that is actually apples-to-apples.
    ours = D.simulate(35.84, 2, 7.64, 10.83, seed=0)
    out["same_evaluator_comparison"] = {
        "our_design": {k: ours[k] for k in ["lcoe", "npc", "renew_frac", "unmet_frac"]},
        "homer_design_zero_diesel": {k: base[k] for k in
                                     ["lcoe", "npc", "renew_frac", "unmet_frac"]},
        "our_design_lcoe_lower_by_pct":
            100.0 * (base["lcoe"] - ours["lcoe"]) / base["lcoe"],
    }
    print(f"\n  same evaluator: our design LCOE {ours['lcoe']:.4f} vs HOMER design "
          f"{base['lcoe']:.4f} "
          f"({out['same_evaluator_comparison']['our_design_lcoe_lower_by_pct']:.1f}% lower)")

    json.dump(out, open(TAB / "review_homer_crosscheck.json", "w"), indent=2)
    print("\nDONE ->", TAB / "review_homer_crosscheck.json")


if __name__ == "__main__":
    main()
