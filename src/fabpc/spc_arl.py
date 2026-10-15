"""Average run length (ARL) of the tabular CUSUM used in the SPC section, by
simulation, compared with the NIST/SEMATECH e-Handbook table (6.3.2.3.1).

The NIST table (k = 0.5) lists in-control ARL 336 (h = 4) and 930 (h = 5) - the
values for one one-sided chart. The SPC section runs both sides (S_hi and S_lo)
and signals on either, so its in-control ARL is about half of that. This module
simulates both versions so the README can quote the right one.
"""

from __future__ import annotations

import numpy as np

from .spc import cusum_run_lengths

K = 0.5
H_VALUES = (4.0, 5.0)
SHIFTS = (0.0, 0.5, 1.0)
N_REP = 20000
SEED = 31
# NIST e-Handbook 6.3.2.3.1, ARL table for k = 0.5 (one-sided chart): shift -> {h: ARL}
NIST_ONE_SIDED = {0.0: {4.0: 336.0, 5.0: 930.0}, 0.5: {4.0: 26.6, 5.0: 30.0}, 1.0: {4.0: 8.38, 5.0: 10.4}}


def run(n_rep: int = N_REP) -> dict:
    rows = []
    for h in H_VALUES:
        for shift in SHIFTS:
            row = {"k": K, "h": h, "shift_sigma": shift, "nist_one_sided": NIST_ONE_SIDED[shift][h]}
            for two, tag in ((False, "one_sided"), (True, "two_sided")):
                seed = SEED + int(10 * h) + int(100 * shift) + (1000 if two else 0)
                rl = cusum_run_lengths(K, h, n_rep, seed, shift=shift, two_sided=two)
                row[tag] = {"arl": float(rl.mean()), "se": float(rl.std(ddof=1) / np.sqrt(n_rep)),
                            "median_rl": float(np.median(rl)), "seed": seed}
            rows.append(row)
    return {"n_rep": n_rep, "k": K,
            "source": "NIST/SEMATECH e-Handbook 6.3.2.3.1 (pmc3231), ARL table for k = 0.5",
            "rows": rows}
