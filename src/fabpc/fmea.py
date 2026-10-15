"""Process FMEA for the SIMULATED CMP-like run-to-run process.

RPN = Severity x Occurrence x Detection (each rated 1-10). The ratings are the
author's engineering judgement for an illustrative, simulated process - they are
not derived from plant data. Where a row cites evidence, it points to a results
file produced by this repository.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FmeaRow:
    item: str
    failure_mode: str
    effect: str
    cause: str
    current_control: str
    s: int
    o: int
    d: int
    action: str
    s_after: int
    o_after: int
    d_after: int
    evidence: str

    def __post_init__(self) -> None:
        for name in ("s", "o", "d", "s_after", "o_after", "d_after"):
            v = getattr(self, name)
            if not isinstance(v, int) or not 1 <= v <= 10:
                raise ValueError(f"{name} must be an integer in 1..10, got {v!r}")

    @property
    def rpn(self) -> int:
        return self.s * self.o * self.d

    @property
    def rpn_after(self) -> int:
        return self.s_after * self.o_after * self.d_after


ROWS: tuple[FmeaRow, ...] = (
    FmeaRow("Polish step", "Removal rate drifts down over pad life",
            "Under-polish; remaining film above target", "Pad glazing / wear (modelled as intercept drift)",
            "Post-polish thickness metrology each lot", 7, 8, 4,
            "EWMA run-to-run on polish time; consider d-EWMA when drift bias matters", 7, 8, 2,
            "results/r2r_results.json (no control vs EWMA vs d-EWMA)"),
    FmeaRow("R2R controller", "Controller oscillates or diverges",
            "Thickness swings run to run; scrap", "Process gain larger than model gain so lambda*beta/b >= 2",
            "None beyond engineer review", 8, 3, 6,
            "Cap lambda so lambda*beta/b stays well below 2 for the worst credible gain; re-identify gain by DOE",
            8, 2, 3, "results/r2r_results.json (stability sweep)"),
    FmeaRow("R2R controller", "Controller removes drift too slowly",
            "Persistent offset from target", "Small lambda with a steady drift (bias = b*delta/(beta*lambda))",
            "Thickness SPC chart", 5, 6, 4,
            "Choose lambda from the bias/variance trade-off or add drift term (d-EWMA)", 5, 3, 4,
            "results/r2r_results.json (mean offset per lambda)"),
    FmeaRow("Down-force", "Pressure set-point error",
            "Within-wafer non-uniformity rises", "Regulator calibration drift",
            "Periodic calibration", 6, 3, 6,
            "I-MR chart on measured down-force; tighter calibration interval", 6, 3, 3,
            "results/doe_results.json (down-force is the largest screened effect)"),
    FmeaRow("Slurry delivery", "Slurry flow drops",
            "Non-uniformity rises; rate falls", "Pump wear, line clog",
            "Flow alarm at fixed threshold", 6, 4, 5,
            "EWMA chart on flow to catch small sustained drops", 6, 4, 3,
            "results/doe_results.json (slurry flow acts mainly through curvature)"),
    FmeaRow("Pad conditioner", "Abrupt change in removal rate",
            "Step offset on following lots", "Conditioner disc wear-out or failure",
            "Lot-level metrology", 7, 3, 5,
            "Tabular CUSUM on controller residuals to flag steps", 7, 3, 3,
            "not simulated in this repo"),
    FmeaRow("Metrology", "Measurement missing or late",
            "Controller updates on stale data", "Metrology tool queue / sampling skip",
            "None", 5, 5, 7,
            "Log measurement delay per run; hold recipe update when data are stale", 5, 5, 4,
            "not simulated in this repo"),
)


def table() -> list[dict]:
    rows = []
    for r in ROWS:
        d = asdict(r)
        d["rpn"] = r.rpn
        d["rpn_after"] = r.rpn_after
        rows.append(d)
    rows.sort(key=lambda z: (-z["rpn"], z["failure_mode"]))
    return rows


def to_markdown(rows: list[dict]) -> str:
    head = ("| # | Item | Failure mode | Cause | S | O | D | RPN | Action | S' | O' | D' | RPN' | Evidence |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['item']} | {r['failure_mode']} | {r['cause']} | {r['s']} | {r['o']} | {r['d']} | "
                     f"**{r['rpn']}** | {r['action']} | {r['s_after']} | {r['o_after']} | {r['d_after']} | "
                     f"{r['rpn_after']} | {r['evidence']} |")
    return head + "\n".join(lines) + "\n"
