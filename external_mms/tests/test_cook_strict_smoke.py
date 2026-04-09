"""Smoke test for strict-preservation CLI pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from mms_agent.cli import run


def test_cook_strict_smoke(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "approved_scientist.json"
    inp = tmp_path / "approved_scientist.json"
    out = tmp_path / "report.json"
    inp.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    report = run(str(inp), "strict_preservation", str(out))

    assert report.status in {
        "success",
        "infeasible_under_strict_preservation",
        "unsupported_ir",
        "parse_error",
    }
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert "status" in loaded
