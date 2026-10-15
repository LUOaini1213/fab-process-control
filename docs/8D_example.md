# Worked 8D report: PCA-FDC alarm cluster on SECOM, 9-11 Oct 2008

**Scope and honesty statement.** SECOM sensors are anonymised (columns are
named `S000`-`S589` here by position). Nothing is known about what each sensor
measures, which tool or step it belongs to, or how the fab responded at the time.
Each SECOM row is "a single production entity" (`secom.names`); whether that is
a wafer, a lot or a tool run is not stated, so this report says **row**. This
is therefore a *data-level* root-cause analysis: it identifies *which sensors
and which rows* drive the alarms and what the signals look like. It does
**not** name a physical cause, and the containment / corrective actions are
written as what an engineer *would* do, not as things that happened.
Section headings follow the D0-D8 names of the cited Wikipedia revision
(docs/sources.md, Q2).

All numbers come from `results/excursion.json` unless another file is named.
Positions `pos 0-19` index the 20 rows of the window in time order
(`window_rows`, and `window_pos_*` under `signatures`).

---

## D0 - Preparation and emergency response actions

The PCA monitor (80 % variance model, 73 PCs, limits = 99th percentile of
calibration-period pass rows; `results/fdc_results.json`) produced its densest
alarm cluster of the held-out period between **2008-10-09 14:15** and
**2008-10-11 08:27**: **7 alarms in 20 consecutive rows** (T² 6, SPE 7), against
**0.99** expected at the test-period alarm rate of 4.9 %.

Is a cluster of 7 surprising, given that it was picked as the densest of 529
overlapping windows? Placing the same 27 test-period alarms at random positions
20,000 times gives a densest window of **3.90 alarms on average** (95th
percentile 5); a maximum of 7 or more occurred with probability **0.002**
(`scan_statistic`). So the cluster is real, not a selection artefact.

Emergency response (would be): put the rows produced in the window on hold for
disposition until D3 is done; no evidence was available that product was
affected (see D2).

## D1 - Use a team

Would be: FDC/APC engineer (owner), process engineer of the step that owns the
contributing sensors, equipment/sensor technician, data-acquisition (EI/FDC
system) owner, yield engineer. In this repo, one person played all roles.

## D2 - Describe the problem (is / is not)

| | IS | IS NOT |
|---|---|---|
| What | 7 multivariate alarms, SPE above limit on all 7 | a yield event: 1 fail in the window (5.0 %) vs 5.1 % for the whole test period |
| Where (sensors) | one group of five sensors (S007 and S004 / S140 / S275 / S413), see below | spread evenly over many sensors |
| When | 2008-10-09 14:15 to 2008-10-11 08:27 | the rest of the test period at the same density |
| How much | SPE up to about 4.19 million times its limit on two rows | a small, gradual shift |

Across the whole test period the alarmed rows failed at **0.0 %** (27 rows)
versus **5.4 %** for non-alarmed rows. The alarm is not a yield predictor here
(consistent with the PCA scores' PR-AUC and ROC-AUC being indistinguishable
from a random ranking in `results/fdc_results.json`).

The window contains one fault class with two faces, plus two isolated alarms:

1. **Sensor-group jump (pos 5, 2008-10-10 10:46, and pos 15, 2008-10-11 06:56).**
   Four sensors jump by three to four orders of magnitude together:
   S275 to 3331.6752 / 3332.4477 (reference 1st-99th percentile
   0.019985-0.370985), S140 to 9997.9106 / 9997.9792, S004 to 1114.5366 /
   1112.4728, S413 to 897.0464 / 898.7167. SPE is about 4.19 million times and
   T² about 2.29 million times its limit. On the **same two rows S007 reads
   exactly 0.0**.
2. **Exact zero with a missing block (pos 7, 18, 19).** Top contributor S007
   (77-80 % of SPE) reads exactly **0.0**, a value never seen in the reference
   period (reference zero fraction 0.0, range 0.1184-0.1269); 16 of the kept
   sensors are missing on each of these rows (median-imputed by the monitor),
   and **the same four sensors S004 / S140 / S275 / S413 are among the missing
   ones** (they are also missing on pos 9, which did not alarm).
3. **pos 0 (top contributor S053) and pos 17 (S338, 40 sensors beyond 3 sigma)**:
   isolated alarms without a shared pattern.

Signatures 1 and 2 involve the same five sensors - S007 is zero on all five of
these rows, and the four-sensor group is either far out of range or missing on
them - so they are treated as **one data-record fault class**, not two
independent problems.

How rare is it outside the window? Over **all 1,567 rows**
(`signatures[*].dataset_far_outside_ref`, `dataset_exact_zero_never_zero_in_ref`):

* the four-sensor jump (more than 10 reference ranges from the reference
  median) occurs on **4 rows**, all in the test block: the two in the window and
  two on 2008-10-05;
* S007 = 0.0 occurs on **8 rows**, all in the test block (2008-10-10 to
  2008-10-12), five of them in the window;
* **every one of these rows is a pass.** None occurs in the fit or calibration
  blocks, so the monitor had never seen the pattern.

**What is *not* evidence.** S004 / S140 / S275 / S413 repeat identical values
inside the window (e.g. S275 = 0.134 on four rows), which could be read as
stale values being re-sent. Compared with normal periods it is not unusual:
50 % of each sensor's non-missing values in the window repeat another value in
the same window (`window_repeated_share`),
while the average over every 20-row window of the fit period is 61-62 %
(`fit_windows_repeated_share_mean`; 57 % over the test period). Repeated values
are routine for these sensors - plausibly because they are recorded per lot or
carried forward between measurements - and say nothing about this event.

## D3 - Develop interim containment plan

Would be: (a) exclude rows whose readings are exact zeros on sensors that are
never zero, or more than 10x the reference range, from automatic FDC
disposition and route them to manual review; (b) keep product on hold only if a
process-side signal (not the data signatures above) is confirmed. The yield
data in D2 do not support scrapping.

## D4 - Determine and verify root causes and escape points (data level)

**Most likely cause class: data acquisition / record integrity, not a process
excursion.** Evidence:

* The four-sensor group jumps to values thousands of reference ranges away and
  is back in range on the very next row (pos 6) - the pattern of a shared
  acquisition fault or a sentinel/overflow value, not of a physical process that
  changes for one row.
* The same rows, and three more, carry an exact 0.0 on a sensor that was never
  zero before, and on three of them the four-sensor group is missing along with
  12 other sensors - typical of a dropped or partially written data record.
* All 10 affected rows in the whole data set (4 jump rows and 8 zero rows, 2 of
  them with both) fall between 2008-10-05 and 2008-10-12, inside the test
  block, and are all passes (D2).

What cannot be concluded: which equipment, which physical quantity, or whether
the tool itself misbehaved. Confirming the cause would need the sensor
dictionary and the FDC system logs, which SECOM does not provide.

**Escape point.** The monitor's preprocessing had no data-validity gate: missing
values are silently median-imputed and out-of-physical-range values enter the
z-scores unchanged. A simple univariate count of sensors beyond 3 sigma (the
"Count of sensors beyond 3 sigma" baseline, alarm threshold
`sigma_count_threshold` = 15 sensors) flags pos 5 (17), pos 15 (20) and pos 17
(40) but not the zero-and-missing rows (pos 7: 4, pos 18: 10, pos 19: 7), so the
univariate layer would have caught the jumps only.

## D5 - Verify permanent corrections (verification plan)

1. Add a **data-validity gate** before the FDC model: per-sensor physical /
   reference range check, exact-zero check for sensors that are never zero, and
   a missing-block check. Rows that fail the gate are labelled "data invalid",
   not "process alarm". (A repeated-value check is *not* proposed: D2 shows that
   repetition is normal for these sensors.)
2. Report T²/SPE only on rows that pass the gate; keep the gate's own alarm
   rate as a separate KPI for the data-acquisition owner.

Verification (would be): replay the test period through the gate and confirm
that the 10 affected rows are re-labelled while the false-alarm rate on
calibration passes is unchanged. Not done in this repo.

## D6 - Define and implement corrective actions

Not implemented. The current code keeps the plain monitor so the published
numbers stay reproducible; the gate is listed as future work in the README.

## D7 - Prevent recurrence / system problems

* Add the data-validity checks to the FDC model's acceptance criteria.
* Add "acquisition fault" as a failure mode in the FMEA of any monitored tool
  (cf. `docs/fmea.md`, "Measurement missing or late").
* When a multivariate alarm fires, show the contribution plot together with
  raw values and reference ranges (`results/figures/fdc_excursion_contributions.png`)
  so that data faults are recognised in minutes rather than chased as process
  problems.

## D8 - Congratulate the main contributors to your team

In a real 8D this step recognises the team; in this one-person exercise it is
used for the close-out lessons instead: (1) a multivariate alarm is a statement
about the *data*, and the first question is whether the data are valid;
(2) on SECOM the FDC alarms and the pass/fail label are largely unrelated, so
FDC performance must be judged on detection of abnormal operation, not only on
yield prediction; (3) "evidence" such as repeated values has to be compared with
the same statistic in normal periods before it is used.
