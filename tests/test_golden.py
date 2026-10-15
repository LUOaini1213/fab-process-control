"""Golden outputs: the committed results files must be reproduced byte-for-byte,
and the README / summary tables must be the ones generated from them."""

from pathlib import Path

import pytest

from fabpc import cli, report, secom

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@pytest.mark.parametrize("name,func", [("r2r_results.json", cli.cmd_r2r), ("doe_results.json", cli.cmd_doe),
                                       ("cusum_arl.json", cli.cmd_arl)])
def test_simulation_results_reproduce_byte_for_byte(tmp_path, name, func):
    assert func(results=tmp_path, figures=False) == 0
    assert (tmp_path / name).read_bytes() == (RESULTS / name).read_bytes()


def test_fmea_results_reproduce(tmp_path):
    cli.cmd_fmea(results=tmp_path)
    for name in ("fmea.json", "fmea_table.md"):
        assert (tmp_path / name).read_bytes() == (RESULTS / name).read_bytes()


def test_readme_blocks_in_sync_with_results(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert report.main(check=True, results=RESULTS, readme=ROOT / "README.md",
                       docs=(ROOT / "docs" / "fmea.md",)) == 0


def test_summary_in_sync_with_results(tmp_path):
    out = tmp_path / "r"
    out.mkdir()
    for p in RESULTS.glob("*.json"):
        (out / p.name).write_bytes(p.read_bytes())
    report.main(check=False, results=out, readme=tmp_path / "missing.md", docs=())
    assert (out / "summary.md").read_bytes() == (RESULTS / "summary.md").read_bytes()


def test_check_mode_detects_stale_block(tmp_path):
    readme = tmp_path / "R.md"
    readme.write_text("x\n<!-- BEGIN:r2r -->\nstale\n<!-- END:r2r -->\n", encoding="utf-8")
    assert report.main(check=True, results=RESULTS, readme=readme, docs=()) == 1
    assert "stale" in readme.read_text(encoding="utf-8")
    out = tmp_path / "res"
    out.mkdir()
    for p in RESULTS.glob("*.json"):
        (out / p.name).write_bytes(p.read_bytes())
    assert report.main(check=False, results=out, readme=readme, docs=()) == 0
    assert report.main(check=True, results=out, readme=readme, docs=()) == 0
    assert "stale" not in readme.read_text(encoding="utf-8")


def test_replace_blocks_only_touches_marked_regions():
    text = "a\n<!-- BEGIN:x -->\nold\n<!-- END:x -->\nb\n<!-- BEGIN:y -->\nkeep\n<!-- END:y -->\n"
    new = report.replace_blocks(text, {"x": "NEW\n"})
    assert new == "a\n<!-- BEGIN:x -->\nNEW\n<!-- END:x -->\nb\n<!-- BEGIN:y -->\nkeep\n<!-- END:y -->\n"


@pytest.mark.skipif(not secom.data_available(), reason="SECOM data not available (set SECOM_DIR)")
def test_secom_results_reproduce_byte_for_byte(tmp_path):
    assert cli.cmd_secom(results=tmp_path, figures=False) == 0
    for name in ("data_profile.json", "spc_summary.json", "fdc_results.json", "excursion.json", "fdc_rolling.json"):
        assert (tmp_path / name).read_bytes() == (RESULTS / name).read_bytes(), name
