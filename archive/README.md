# Archive — the first (full) pipeline, 2026-09-30 → 2026-10-01

Everything in this directory is the earlier, larger version of the project,
kept unchanged for the record. **The live pipeline in `src/` does not read,
import or cite anything in here.** It re-derives every choice from the raw
data with its own rules (see `notes/METHOD.md`).

To re-run the archived pipeline, check out the tag where it last ran:

```bash
git checkout v0-full-pipeline
```

Its outputs as of that run are preserved on disk under `output/archive/`
(gitignored). The archived scripts expect `output/` paths, so they will not
run in place from `archive/src/`; use the tag.

| archived script | what it did | note with the results |
|---|---|---|
| `src/00_triage.py` | Day-1 gate: past/future placebo, spline turnover, variance budget, geocoding atoms, routing test, specification curve | `notes/day1_triage.md` |
| `src/01_build_panel.py` | CBG × month panel plus point tables | `notes/STATUS.md` |
| `src/02_reporting.py` | NumPyro refit of the predecessor's CBG two-equation model (β₀ = 0.55–0.58) and four kernel w_R variants over five bandwidths | `notes/phase3_reporting.md` |
| `src/03_litter_panel.py` | survey model table with 31 covariates and 8 w_R columns | `notes/phase4_field.md` |
| `src/04_field.py` | 6 × 4 (h, τ) sweep, decoupled bandwidths, alternative denominators, polygon benchmarks, dedup and outage fields | `notes/phase4_field.md` |
| `src/05_calibrate.py` | 8-spec ladder, profile likelihood, 100-rep bootstrap, LR recalibration, ten controls | `notes/phase5_calibration.md` |
| `src/06_cv.py` | nested blocked CV, temporal holdout, buffer and α sensitivity, covariate spec | `notes/phase5_calibration.md` |
| `src/07_surface.py`, `src/08_figures.py` | rasters at two bandwidths; figure set | `notes/phase5_calibration.md` |
| `plans/project-propose-plan-*.md` | the plan that pipeline implemented | — |

Why it was retired: the pipeline had grown to 4,300 lines, nine scripts and
2.5 hours per run, with more variants and controls than the research question
needs. The owner could no longer follow it. The replacement keeps one w_R, one
field, one estimator and one validation design, each justified inside the
pipeline, and is documented in `notes/METHOD.md`.
