# Process FMEA - simulated CMP-like run-to-run process

This FMEA covers the **simulated** process in `src/fabpc/simulator.py` and the
run-to-run controller in `src/fabpc/r2r.py`. It is an illustration of the
method: the ratings are the author's engineering judgement for that toy
process, not data from a real tool. RPN = Severity x Occurrence x Detection
(docs/sources.md, Q1). The table below is generated from `results/fmea.json`
by `python -m fabpc.cli report`; edit `src/fabpc/fmea.py`, not the table.
"Wafer" and "lot" below refer to the simulated CMP process; they say nothing
about SECOM, whose rows are anonymous "production entities".

## Rating scales used here (this repo's own definitions)

| Rating | Severity (effect on the wafer) | Occurrence (how often the cause appears) | Detection (chance the current control catches it before the next lot) |
|---|---|---|---|
| 1-2 | no measurable effect on thickness or uniformity | less than once per year of operation | almost certain: automatic interlock |
| 3-4 | small offset, still inside spec | a few times per year | high: SPC chart on a direct measurement |
| 5-6 | out-of-spec excursion that can be reworked | monthly | moderate: indirect measurement or sampled metrology |
| 7-8 | scrap of a lot | weekly, or every consumable life | low: found only at end-of-line test |
| 9-10 | scrap of many lots or safety issue | every run | none |

## Table (sorted by RPN; S', O', D', RPN' = after the recommended action)

<!-- BEGIN:fmea_full -->
| # | Item | Failure mode | Cause | S | O | D | RPN | Action | S' | O' | D' | RPN' | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Polish step | Removal rate drifts down over pad life | Pad glazing / wear (modelled as intercept drift) | 7 | 8 | 4 | **224** | EWMA run-to-run on polish time; consider d-EWMA when drift bias matters | 7 | 8 | 2 | 112 | results/r2r_results.json (no control vs EWMA vs d-EWMA) |
| 2 | Metrology | Measurement missing or late | Metrology tool queue / sampling skip | 5 | 5 | 7 | **175** | Log measurement delay per run; hold recipe update when data are stale | 5 | 5 | 4 | 100 | not simulated in this repo |
| 3 | R2R controller | Controller oscillates or diverges | Process gain larger than model gain so lambda*beta/b >= 2 | 8 | 3 | 6 | **144** | Cap lambda so lambda*beta/b stays well below 2 for the worst credible gain; re-identify gain by DOE | 8 | 2 | 3 | 48 | results/r2r_results.json (stability sweep) |
| 4 | R2R controller | Controller removes drift too slowly | Small lambda with a steady drift (bias = b*delta/(beta*lambda)) | 5 | 6 | 4 | **120** | Choose lambda from the bias/variance trade-off or add drift term (d-EWMA) | 5 | 3 | 4 | 60 | results/r2r_results.json (mean offset per lambda) |
| 5 | Slurry delivery | Slurry flow drops | Pump wear, line clog | 6 | 4 | 5 | **120** | EWMA chart on flow to catch small sustained drops | 6 | 4 | 3 | 72 | results/doe_results.json (slurry flow acts mainly through curvature) |
| 6 | Down-force | Pressure set-point error | Regulator calibration drift | 6 | 3 | 6 | **108** | I-MR chart on measured down-force; tighter calibration interval | 6 | 3 | 3 | 54 | results/doe_results.json (down-force is the largest screened effect) |
| 7 | Pad conditioner | Abrupt change in removal rate | Conditioner disc wear-out or failure | 7 | 3 | 5 | **105** | Tabular CUSUM on controller residuals to flag steps | 7 | 3 | 3 | 63 | not simulated in this repo |
<!-- END:fmea_full -->

## How the simulation informs the ratings

* "Controller oscillates or diverges": the stability check in
  `results/r2r_results.json` reproduces the published condition
  0 < lambda*beta/b < 2 on every grid point, so the action caps lambda against the
  worst credible gain ratio instead of relying on detection.
* "Controller removes drift too slowly": the EWMA steady-state offset under drift
  equals b*delta/(beta*lambda) in simulation. With untuned weights d-EWMA removes
  it at the cost of a larger run-to-run SD; once both controllers are tuned on
  separate replications, d-EWMA has the lower RMSE on this simulated drift
  (README, R2R section). The comparison depends on the drift rate, noise and
  tuning, so it would have to be repeated on real tool data.
* Rows marked "not simulated in this repo" are included because a real FMEA
  would need them; their ratings have no quantitative backing here.
