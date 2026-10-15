# Sources

Every external constant, rule, formula or model form used in the code is listed
here with the place it was checked. All URLs were retrieved and read on
**2026-10-02**. Where a statement could only be checked through a secondary source,
that is said explicitly.

## Data

| # | What it supports | Source |
|---|---|---|
| D1 | SECOM dataset, citation, licence (CC BY 4.0), 1,567 instances, "591 features" on the UCI page and in `secom.names` ("a 1567 x 591 matrix"), while every row of `secom.data` has 590 fields - the discrepancy is in the source's own documentation (see `results/data_profile.json`); each row is "a single production entity" (`secom.names`), not explicitly a wafer, label coding (-1 pass / 1 fail) and per-row timestamps | UCI Machine Learning Repository, dataset 179: <https://archive.ics.uci.edu/dataset/179/secom>; download <https://archive.ics.uci.edu/static/public/179/secom.zip>; `secom.names` inside the zip. Citation: McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C54305> |

## Statistical process control (NIST/SEMATECH e-Handbook of Statistical Methods)

| # | Used in | What was checked | URL |
|---|---|---|---|
| N1 | `spc.weco_rules` | Western Electric (WECO) rules: any point beyond 3 sigma; 2 of last 3 beyond 2 sigma; 4 of last 5 beyond 1 sigma; 8 consecutive on one side; trend rules "6 in a row trending up or down" and "14 in a row alternating up and down"; note that WECO rules raise the false-alarm rate (in-control ARL about 91.75 vs 371) | <https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc32.htm> |
| N2 | `spc.XBAR_R_FACTORS`, `spc.xbar_r_limits` | X-bar/R limits `xbarbar +/- A2 Rbar`, `D3 Rbar`, `D4 Rbar`; table of A2, D3, D4 for n = 2..10 (copied exactly) | <https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc321.htm> |
| N3 | `spc.individuals_limits` | Individuals chart `xbar +/- 3 MRbar / 1.128`, "1.128 is the value of d2 for n = 2"; worked flow-rate example (UCL 55.8041, LCL 45.8159) used as a unit test | <https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc322.htm> |
| N4 | `spc.tabular_cusum` | Tabular CUSUM `S_hi(i) = max(0, S_hi(i-1) + x_i - mu0 - k)`, `S_lo(i) = max(0, S_lo(i-1) + mu0 - k - x_i)`; rule of thumb k = half the shift to detect (0.5 sigma) and h "around 4 or 5"; worked example (h = 4.1959, k = 0.3175, first signal at group 14) used as a unit test | <https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm> |
| N5 | README (CUSUM design), `spc_arl.py`, `tests/test_spc.py` | ARL table for k = 0.5: in-control ARL 336 (h = 4) and 930 (h = 5); 26.6 / 30.0 at a 0.5-sigma shift and 8.38 / 10.4 at 1 sigma. These are the values for **one one-sided** chart ("if one has to control both positive and negative deviations ... two one-sided charts are used"). The two-sided scheme used here has in-control ARL about half (simulated: `results/cusum_arl.json`). The simulation reproduces every entry except h = 5 at a 0.5-sigma shift, where it gives about 37.6 rather than 30.0 | <https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc3231.htm> |
| N6 | `spc.ewma`, `spc.ewma_limits` | `EWMA_t = lambda Y_t + (1 - lambda) EWMA_{t-1}`; lambda "usually set between 0.2 and 0.3"; asymptotic variance `lambda/(2 - lambda) s^2`; limits `EWMA_0 +/- k s_ewma` with k = 3; worked example (lambda = 0.3, UCL 52.5884) used as a unit test | <https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm> |
| N7 | `spc.capability` | `Cp = (USL - LSL)/(6 s)`, `Cpk = min[(USL - xbar)/(3 s), (xbar - LSL)/(3 s)]`; Cp = 1.33 corresponds to a spec width of 8 sigma | <https://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm> |
| N8 | background only | Hotelling T^2 definition. NIST says the F result for testing a known mean "does not apply directly to multivariate Shewhart-type charts (for which there is no mu_0)", and that each Phase gets its own limit form. This repo alarms on empirical limits from calibration-period passes and reports the Phase II F limit (N9) next to them | <https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc543.htm> |
| N9 | `mspc.t2_limit_phase2` | T^2 chart for individual observations, Phase II: UCL = p (m + 1)(m - 1) / (m^2 - m p) F_alpha(p, m - p), m historical observations, p characteristics. Applied here to the A retained PC scores (p = A) with m = number of reference rows; the same expression, `(n^2 - 1) A / (n (n - A)) F`, is in SCOUTer's `pcamb_classic` (M3) | <https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc5434.htm> |

## Multivariate monitoring

| # | Used in | What was checked | URL |
|---|---|---|---|
| M1 | `mspc.PCAMonitor` | PCA monitoring convention: scores `T = X P`, residuals `E = X - T P'`, `SPE = rowSums(E^2)`, `T2 = rowSums(T^2 / lambda)`, and per-PC T^2 contributions `T^2/lambda`; residual matrix E used as the per-variable SPE contribution | R package SCOUTer (Gonzalez et al., CRAN, GPL-3), function `pcame`: source <https://raw.githubusercontent.com/cran/SCOUTer/master/R/pcame.R>; reference manual <https://cran.r-project.org/web/packages/SCOUTer/refman/SCOUTer.html>. No code was copied. |
| M2 | `mspc.PCAMonitor.t2_contrib` | Per-variable T^2 contribution `c_j = z_j * sum_a (t_a/lambda_a) p_ja` = `z_j (M z)_j` with `M = P diag(1/lambda) P'`: the standard **partial decomposition contribution** (PDC) of a quadratic index, not something new here. Primary source: C. F. Alcala and S. J. Qin, "Analysis and generalization of fault diagnosis methods for process monitoring", J. Process Control 21(3), 2011, 322-330, doi:10.1016/j.jprocont.2010.10.005 (bibliographic record and abstract checked; full text paywalled, **not retrieved**). The PDC form was checked in a secondary source that restates Alcala & Qin's general decomposition contribution `GDC = t' Phi^(1-beta) xi xi' Phi^beta t`, which "reduces to partial decomposition contribution when beta = 0 or beta = 1": W. Yu, M. Wu, B. Huang, C. Lu, arXiv:2206.13437, eq. (87). The row-sum identity is tested in `tests/test_mspc.py`. | <https://scholars.ln.edu.hk/en/publications/analysis-and-generalization-of-fault-diagnosis-methods-for-proces/>; <https://arxiv.org/abs/2206.13437> |
| M3 | `mspc.spe_limit_jm` | Jackson-Mudholkar SPE limit `Q = theta1 [z sqrt(2 theta2 h0^2)/theta1 + 1 + theta2 h0 (h0 - 1)/theta1^2]^(1/h0)`, `theta_i` = sum of the i-th powers of the discarded eigenvalues, `h0 = 1 - 2 theta1 theta3 / (3 theta2^2)`, checked line by line against SCOUTer's `pcamb_classic` (no code copied). Original: J. E. Jackson and G. S. Mudholkar, "Control procedures for residuals associated with principal component analysis", Technometrics 21(3), 1979, 341-349, doi:10.1080/00401706.1979.10489779 (Crossref record checked; **not retrieved**). Accuracy checked by Monte Carlo in `tests/test_mspc.py` | <https://raw.githubusercontent.com/cran/SCOUTer/master/R/pcamb_classic.R> |

## Design of experiments (NIST/SEMATECH e-Handbook)

| # | Used in | What was checked | URL |
|---|---|---|---|
| E1 | `doe.full_factorial`, `doe.effects` | Two-level full factorial designs | <https://www.itl.nist.gov/div898/handbook/pri/section3/pri3331.htm> |
| E2 | `doe.fractional_2_4_1` | 2^(4-1) resolution IV, generator 4 = 123, defining relation I = 1234, aliases 12+34, 13+24, 14+23, exact run list, "effect = average of high - average of low" (unit test compares the run list) | <https://www.itl.nist.gov/div898/handbook/pri/section3/pri3347.htm>, <https://www.itl.nist.gov/div898/handbook/pri/section3/eqns/2to4m1.txt> |
| E3 | design notes | Confounding / aliasing algebra | <https://www.itl.nist.gov/div898/handbook/pri/section3/pri3343.htm> |
| E4 | `doe.ccd`, `doe.rotatable_alpha` | Central composite designs; rotatable alpha = [number of factorial runs]^(1/4) (1.682 for 2^3) | <https://www.itl.nist.gov/div898/handbook/pri/section3/pri3361.htm> |
| E5 | `doe.quad_model_matrix`, `doe.quad_to_bB`, `doe.stationary_point` | Second-order model; `B` with `b_ii` on the diagonal and `b_ij/2` off-diagonal; stationary point `x* = -1/2 B^-1 b`; nature from the eigenvalues of B; worked example x* = (-0.9285, 0.3472) used as a unit test | <https://www.itl.nist.gov/div898/handbook/pri/section5/pri5314.htm> |

## Run-to-run control

| # | Used in | What was checked | URL |
|---|---|---|---|
| R1 | `r2r.simulate` (EWMA-I), `r2r.ewma_stable`, `r2r.ewma_drift_bias`, `simulator.LinearDriftProcess` | J. Wang and Q. P. He, "EWMA Run-to-Run Controllers with Gain Updating: Stability and Sensitivity Analysis", Proc. 2008 American Control Conference, Seattle, paper ThB13.3, p. 2872 ff. Checked: EWMA-I control law eqs. (3)-(4); Table I (EWMA-I unstable when `(beta/b) w < 0` or `> 2`, i.e. stable iff `0 < w beta/b < 2`, attributed to Ingolfsson & Sachs); drifting-intercept model `y[n] = alpha + beta u[n] + n delta` (eq. 35); EWMA-I steady-state bias `y - T = b delta / (beta w)` (Sec. III-B) | <https://skoge.folk.ntnu.no/prost/proceedings/acc08/data/papers/1144.pdf> |
| R2 | (cited through R1 only) | A. Ingolfsson and E. Sachs, "Stability and sensitivity of an EWMA controller", J. Quality Technology 25 (1993) 271-287. **Not retrieved** (paywalled); the stability condition and drift bias are used as restated in R1 and then checked numerically in `results/r2r_results.json`. | - |
| R3 | `r2r.simulate` (d-EWMA) | A. Chen and R.-S. Guo, "Age-Based Double EWMA Controller and Its Application to CMP Processes", IEEE Trans. Semiconductor Manufacturing 14(1), 2001, pp. 11-19, doi:10.1109/66.909650. Checked (rendered equation images): PCC eqs. (5)-(6), drifting process eq. (7), PCC converges to target eq. (10), d-EWMA eq. (12): `a_i = w1 (Y_i - beta X_i) + (1 - w1)(a_{i-1} + p_{i-1})`, `p_i = w2 (Y_i - beta X_i - a_{i-1}) + (1 - w2) p_{i-1}`, recipe `X_{i+1} = (T - (a_i + p_i))/beta`. Eq. (12) is the authors' **modified** double EWMA: the text says it differs from "the original PCC formula" (eqs. 5-6, the predictor-corrector control scheme "termed by Butler and Stefani") by adding `p_{i-1}` inside the first EWMA. The code implements eq. (12), not the classic PCC. The paper assumes the gain is known; this repo uses the model gain b in its place. | <https://scholars.lib.ntu.edu.tw/bitstreams/d26328ad-6e6a-4d2a-a078-cc4e63ee62e9/download> (record: <https://scholars.lib.ntu.edu.tw/handle/123456789/82259>) |
| R4 | README (d-EWMA background) | Classic double-EWMA / predictor-corrector control (PCC): S. W. Butler and J. A. Stefani, "Supervisory run-to-run control of polysilicon gate etch using in situ ellipsometry", IEEE Trans. Semiconductor Manufacturing 7(2), 1994, 193-201, doi:10.1109/66.286855. Bibliographic data checked in the Crossref record; the paper itself was **not retrieved** - that it is the origin of PCC is taken from Chen & Guo's text (R3, their ref. [6]) | <https://api.crossref.org/works/10.1109/66.286855> |

## Quality tools

| # | Used in | What was checked | URL |
|---|---|---|---|
| Q1 | `fmea.py` | "Risk Priority Number (RPN) is the product of severity x occurrence x detection." The 1-10 rating tables in `docs/fmea.md` are this repo's own and are not claimed from a standard. | <https://help.reliasoft.com/xfmea_rcm/content/risk_priority_numbers_rpns.htm> |
| Q2 | `docs/8D_example.md` | 8D discipline names D0-D8 (Ford TOPS / Global 8D), incl. "escape point" in D4. The section headings of the 8D report use this revision's names (D0 Preparation and Emergency Response Actions, D1 Use a Team, D2 Describe the Problem, D3 Develop Interim Containment Plan, D4 Determine and Verify Root Causes and Escape Points, D5 Verify Permanent Corrections, D6 Define and Implement Corrective Actions, D7 Prevent Recurrence / System Problems, D8 Congratulate the Main Contributors to your Team) | Wikipedia, "Eight disciplines problem solving", revision 1356602541 (2026-05-28): <https://en.wikipedia.org/w/index.php?title=Eight_disciplines_problem_solving&oldid=1356602541> |

## Statistics references

| # | Used in | What was checked | URL |
|---|---|---|---|
| S1 | `metrics.random_ap_expected`, FDC no-skill line | The expected average precision of a random ranking is **not** the prevalence R/N; exact value `(1/R) sum_i sum_n P(i relevant in top n) (i/n)^2` (hypergeometric), R function `RandomAPExact`; examples N = 5, R = 2 -> 0.593 and `RandomAPExact(10, 4)` = 0.5285979 used as unit tests. Y. Bestgen, "Exact Expected Average Precision of the Random Baseline for System Evaluation", Prague Bulletin of Mathematical Linguistics 103 (2015) 131-138, doi:10.1515/pralin-2015-0007 | <https://ufal.mff.cuni.cz/pbml/103/art-bestgen.pdf> |

## Standard methods used without an external constant

Ordinary least squares with t- and extra-sum-of-squares F-tests, Fisher's exact
test (`scipy.stats.fisher_exact`) with the Bonferroni bound, the percentile bootstrap,
one-sided label-permutation tests of PR-AUC / ROC-AUC (p = (1 + #null >= observed) / (1 + n)),
a scan statistic for the densest alarm window (null by random placement of the
same number of alarms), Monte Carlo run lengths of the CUSUM, rolling-origin
(time-series) evaluation, average precision
(checked against `sklearn.metrics.average_precision_score` in the tests) and the
Mann-Whitney form of ROC-AUC (checked against `sklearn.metrics.roc_auc_score`).

## Choices that are assumptions, not sourced facts

* Spec limits for SECOM sensors (Phase-I mean +/- 4 Phase-I sigma) - SECOM has no spec limits.
* Fractions for the time split (50 / 15 / 35 %) and for the three rolling-origin folds, the hyperparameter grids (logistic-regression C, gradient-boosting learning rate / leaves / iterations, EWMA and d-EWMA weights), the 5 % false-alarm target, the 99 % empirical control limit, the 20 % missing / 95 % near-constant sensor filters, the EWMA lambda = 0.2 and CUSUM h = 5 choices within the NIST ranges.
* All simulator parameters (`fabpc/simulator.py`), the toy DOE response surface and the FMEA ratings.
