"""CLI entrypoint for external_mms strict-preservation symbolic verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from .approved_scientist_adapter import ApprovedScientistAdapter
from .ir_builder import IRBuilder
from .report_builder import ReportBuilder
from .schemas import VerificationReport


def run(input_path: str, mode: str, output_path: str) -> VerificationReport:
    """Execute full pipeline and return structured verification report."""
    try:
        payload = ApprovedScientistAdapter.load(input_path)
        extracted = ApprovedScientistAdapter.extract(payload)
        ir = IRBuilder.build(extracted)
        report = ReportBuilder.build(ir, mode=mode)
    except (ValidationError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        report = VerificationReport(
            status="parse_error",
            ir_valid=False,
            operator_coverage_ok=False,
            strict_preservation=mode == "strict_preservation",
            exact_candidate_found=False,
            strong_form_residual_zero=False,
            boundary_conditions_satisfied=False,
            weak_form_structurally_consistent=False,
            declared_ambiguities=[],
            diagnostics=[f"Parse/build error: {exc}"],
            candidate_solution=None,
        )

    Path(output_path).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def main() -> None:
    """CLI main function."""
    parser = argparse.ArgumentParser(description="external_mms symbolic strict-preservation verifier")
    parser.add_argument("--input", required=True, help="Path to approved_scientist.json")
    parser.add_argument("--mode", required=True, help="Verification mode (expected: strict_preservation)")
    parser.add_argument("--output", required=True, help="Path to write report.json")
    args = parser.parse_args()

    run(args.input, args.mode, args.output)


if __name__ == "__main__":
    main()
