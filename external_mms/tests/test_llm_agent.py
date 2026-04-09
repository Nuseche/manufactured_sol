"""Tests for mandatory Azure OpenAI LLM diagnostics integration."""

from __future__ import annotations

from mms_agent.llm_agent import enrich_report_with_llm
from mms_agent.schemas import CanonicalProblemIR, VerificationReport


def _minimal_ir() -> CanonicalProblemIR:
    return CanonicalProblemIR(
        problem_id="pid",
        mode="strict_preservation",
        problem_statement="ps",
        coordinates=["X", "Y"],
        unknowns=[],
        definitions=[],
        equations=[],
        geometry={"domain": {}, "boundaries": {}},
        boundary_conditions=[],
        initial_conditions=[],
        parameters={},
        assumptions=[],
        ambiguities=[],
        weak_form={},
    )


def _minimal_report() -> VerificationReport:
    return VerificationReport(
        status="infeasible_under_strict_preservation",
        ir_valid=True,
        operator_coverage_ok=True,
        strict_preservation=True,
        exact_candidate_found=False,
        strong_form_residual_zero=False,
        boundary_conditions_satisfied=False,
        weak_form_structurally_consistent=True,
        diagnostics=["base"],
    )


def test_llm_without_env_raises(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    ir = _minimal_ir()
    report = _minimal_report()
    try:
        enrich_report_with_llm(report, ir)
        assert False, "Expected RuntimeError when Azure env vars are missing."
    except RuntimeError as exc:
        assert "Missing Azure OpenAI configuration" in str(exc)


def test_llm_with_mock_requester(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt4o")
    ir = _minimal_ir()
    report = _minimal_report()

    def fake_requester(_settings, _payload):
        return {"choices": [{"message": {"content": "next steps ok"}}]}

    out = enrich_report_with_llm(report, ir, requester=fake_requester)
    assert any("LLM(AzureOpenAI) audit note: next steps ok" in d for d in out.diagnostics)
