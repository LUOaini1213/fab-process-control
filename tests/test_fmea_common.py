import json
import math

import numpy as np
import pytest

from fabpc import common, fmea


def test_round_sig():
    assert common.round_sig(123456789012.345) == 123456789000.0
    assert common.round_sig(0.000123456789012345) == 0.0001234567890
    assert common.round_sig(-2.0 / 3.0) == -0.6666666667
    assert common.round_sig(-0.0) == 0.0 and math.copysign(1.0, common.round_sig(-0.0)) == 1.0
    assert common.round_sig(float("nan")) is None
    assert common.round_sig(float("inf")) is None
    assert common.round_sig(1.23456789, sig=3) == 1.23


def test_clean_handles_numpy_and_nesting():
    obj = {"a": np.float64(1.0 / 3.0), 2: [np.int64(4), np.bool_(True), (np.float32(0.5),)],
           "arr": np.array([1.5, 2.25]), "s": "x", "n": None}
    out = common.clean(obj)
    assert out == {"a": 0.3333333333, "2": [4, True, [0.5]], "arr": [1.5, 2.25], "s": "x", "n": None}
    assert isinstance(out["2"][1], bool)
    with pytest.raises(TypeError):
        common.clean({"bad": object()})


def test_write_json_is_sorted_lf_and_rounded(tmp_path):
    p = common.write_json(tmp_path / "o.json", {"b": 1.0 / 7.0, "a": [1, 2]})
    raw = p.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    assert raw.index(b'"a"') < raw.index(b'"b"')
    assert json.loads(raw)["b"] == 0.1428571429


def test_fmt():
    assert common.fmt(2.0 / 3.0) == "0.667"
    assert common.fmt(None) == "n/a"


def test_rpn_and_sorting():
    rows = fmea.table()
    for r in rows:
        assert r["rpn"] == r["s"] * r["o"] * r["d"]
        assert r["rpn_after"] == r["s_after"] * r["o_after"] * r["d_after"]
    rpns = [r["rpn"] for r in rows]
    assert rpns == sorted(rpns, reverse=True)
    assert rows[0]["rpn"] == max(x.rpn for x in fmea.ROWS)
    assert all(r["rpn_after"] <= r["rpn"] for r in rows)


def test_rating_validation():
    base = dict(item="i", failure_mode="m", effect="e", cause="c", current_control="cc",
                s=5, o=5, d=5, action="a", s_after=5, o_after=4, d_after=3, evidence="-")
    assert fmea.FmeaRow(**base).rpn == 125
    assert fmea.FmeaRow(**base).rpn_after == 60
    for bad in (0, 11, 2.5):
        with pytest.raises(ValueError):
            fmea.FmeaRow(**{**base, "d": bad})


def test_markdown_table():
    md = fmea.to_markdown(fmea.table())
    lines = md.strip().splitlines()
    assert len(lines) == 2 + len(fmea.ROWS)
    assert lines[2].startswith("| 1 |") and f"**{fmea.table()[0]['rpn']}**" in lines[2]
