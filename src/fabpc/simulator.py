"""SIMULATED processes (not plant data).

1. ``LinearDriftProcess`` - single-input run-to-run process with a linear drift
   in the intercept:  y_n = alpha + beta * u_n + n * delta + eps_n,
   eps_n ~ N(0, sigma^2). This is the model form analysed for EWMA run-to-run
   controllers (Wang & He 2008, eq. 35; Chen & Guo 2001, eq. 7 - see
   docs/sources.md). The numbers below are illustrative, CMP-flavoured
   (u = polish time in s, y = removed thickness in nm); they are not measured.

2. ``toy_cmp_nonuniformity`` - a hand-made, smooth, NON-quadratic response
   surface used to exercise the DOE workflow. It is a toy, not a physical model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearDriftProcess:
    alpha: float = 35.0    # intercept (nm)
    beta: float = 4.0      # true gain (nm per s of polish time)
    delta: float = -0.5    # drift of the intercept per run (nm/run)
    sigma: float = 6.0     # run-to-run noise std (nm)

    def output(self, n: int, u: float, eps: float) -> float:
        """Output of run n (n = 1, 2, ...) for recipe u and noise eps."""
        return self.alpha + self.beta * u + n * self.delta + eps


# ---------------------------------------------------------------------------
# DOE toy: within-wafer non-uniformity (%) vs 4 coded factors in [-2, 2].
# x1 = down-force, x2 = platen speed, x3 = slurry flow, x4 = conditioner sweep
# (x4 is inert by construction - the screening design should find it inactive).
# ---------------------------------------------------------------------------
TOY_FACTORS = ("downforce", "platen_speed", "slurry_flow", "conditioner_sweep")


def toy_cmp_nonuniformity(x: np.ndarray) -> np.ndarray:
    """Noise-free toy response. ``x`` has shape (n, 4) or (4,), coded units."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    y = (4.0
         + 0.9 * (x1 - 0.35) ** 2
         + 0.6 * (x2 + 0.25) ** 2
         + 0.35 * (x3 - 0.15) ** 2
         + 0.45 * (x1 - 0.35) * (x2 + 0.25)
         + 0.08 * (x1 - 0.35) ** 4          # non-quadratic term
         + 0.3 * np.exp(0.5 * x3) - 0.3     # non-quadratic term
         )
    return y


def toy_observe(x: np.ndarray, rng: np.random.Generator, noise_sd: float = 0.15) -> np.ndarray:
    """Noisy observation of the toy response."""
    y = toy_cmp_nonuniformity(x)
    return y + rng.normal(0.0, noise_sd, size=y.shape)
