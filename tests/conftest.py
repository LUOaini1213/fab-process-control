"""Shared pytest hooks.

Without the SECOM files the real-data tests skip, so a plain `pytest` on a
fresh clone still runs. CI sets FABPC_REQUIRE_SECOM=1 after fetching the
SHA-256-pinned data; then any skipped test fails the run, so the headline SECOM
numbers and the leakage tests cannot silently drop out of CI.
"""

import os

REQUIRE_ENV = "FABPC_REQUIRE_SECOM"


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get(REQUIRE_ENV, "") not in ("1", "true", "yes"):
        return
    rep = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = len(rep.stats.get("skipped", [])) if rep is not None else 0
    if skipped:
        if rep is not None:
            rep.write_line(f"{REQUIRE_ENV} is set but {skipped} test(s) were skipped; failing the run.", red=True)
        session.exitstatus = 1
