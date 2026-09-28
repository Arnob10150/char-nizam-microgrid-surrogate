# ML Surrogate Framework for Off-Grid Hybrid Microgrid Sizing (Char Nizam, Bangladesh)

Code for *A Machine-Learning Surrogate Framework for Designing and Optimizing the Char Nizam
Off-Grid Hybrid Power System in Bangladesh* (IEEE iCOSTE 2026).

Arnob Aich Anurag, Shamiul Islam, MD.Nasif Rafidi, MD. Sadman Hossain, Faria Sultana, Shohana Khondoker, Shandipan Paul

## What is here

| Script | Purpose |
|---|---|
| `src/dispatch_sim.py` | hourly (8,760 h) PV–wind–battery–diesel load-following dispatch simulator with LCOE/NPC economics (the ground truth) |
| `src/pipeline.py` | 1,500-sample dataset, gradient-boosted surrogate, model benchmark, differential-evolution search, grid-search baseline, uncertainty |
| `src/review_experiments.py` | 5-fold CV, learning curve, DE run directly on the simulator, measured g=20 grid, 30-seed robustness, uncertainty decomposition |
| `src/homer_crosscheck.py` | cross-check against a published HOMER Pro study of the island |
| `src/surrogate_refine.py` | adaptive surrogate refinement |
| `src/camera_ready_experiments.py` | assumption audit and CO₂, load-profile variants, sampling density (uniform vs. LHS), nested cross-validation |
| `src/camera_ready_extra.py` | peak-matched load and HOMER fuel curve, re-optimized directly on the simulator |
| `data/nasa_power.json` | one year of hourly NASA POWER irradiance, wind and temperature at 21.87°N, 90.75°E |
| `outputs/tables/` | measured results reported in the paper |

## Key results

- The surrogate reaches R² 0.998 on held-out data and 0.9967 ± 0.0031 under nested CV. It is 36× faster than a full simulation.
- The selected design (35.8 kW PV, 2×10 kW wind, 7.6 kWh battery, 10.8 kW diesel) costs $0.228/kWh at a 50.6% renewable fraction. Search is 322× faster than a measured g=20 grid.
- Robustness (Table VII of the paper): the design is feasible under seasonal and behavioral load variation. It needs re-optimizing for a 23.32 kW peak or for +20% load growth.

## Reproduce

```bash
pip install numpy pandas scipy scikit-learn
python src/pipeline.py
python src/review_experiments.py
python src/surrogate_refine.py
python src/camera_ready_experiments.py
python src/camera_ready_extra.py
```

Everything runs on a CPU in minutes. Seed 42 throughout.

## License

Code is released for academic use. Third-party datasets remain under their original licenses.
