"""Tests for mandatory Azure OpenAI LLM diagnostics integration."""

from __future__ import annotations

from pathlib import Path

import mms_agent.llm_agent as llm_agent
from mms_agent.llm_agent import (
    AzureOpenAISettings,
    _build_request_components,
    enrich_report_with_llm,
)
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
    monkeypatch.delenv("AZURE_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    monkeypatch.setattr(llm_agent, "_find_dotenv", lambda: None)
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
    monkeypatch.delenv("AZURE_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    monkeypatch.setattr(llm_agent, "_find_dotenv", lambda: None)
    ir = _minimal_ir()
    report = _minimal_report()

    def fake_requester(_settings, _payload):
        return {"choices": [{"message": {"content": "next steps ok"}}]}

    out = enrich_report_with_llm(report, ir, requester=fake_requester)
    assert any("LLM(AzureOpenAI) audit note: next steps ok" in d for d in out.diagnostics)


def test_llm_loads_parent_dotenv(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    pkg_dir = repo_root / "external_mms"
    pkg_dir.mkdir(parents=True)
    (repo_root / ".env").write_text(
        "\n".join(
            [
                'AZURE_OPENAI_API_KEY="k"',
                "AZURE_OPENAI_ENDPOINT=https://example.openai.azure.com/",
                'AZURE_OPENAI_DEPLOYMENT="gpt-5.4"',
                'AZURE_OPENAI_MODEL="gpt-5.4"',
                "AZURE_OPENAI_API_VERSION=2025-04-01-preview",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(pkg_dir)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

    settings = AzureOpenAISettings.from_env()

    assert settings is not None
    assert settings.endpoint == "https://example.openai.azure.com"
    assert settings.api_mode == "responses"
    assert settings.model == "gpt-5.4"


def test_responses_api_preview_url_and_response_shape(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-5.4")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

    settings = AzureOpenAISettings.from_env()
    assert settings is not None

    url, _, _ = _build_request_components(
        settings,
        {"model": "gpt-5.4", "input": "hello"},
    )
    assert url == "https://example.openai.azure.com/openai/responses?api-version=2025-04-01-preview"

    ir = _minimal_ir()
    report = _minimal_report()

    def fake_requester(_settings, payload):
        assert payload["model"] == "gpt-5.4"
        assert payload["input"]
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "responses ok",
                        }
                    ],
                }
            ]
        }

    out = enrich_report_with_llm(report, ir, requester=fake_requester)
    assert any("LLM(AzureOpenAI) audit note: responses ok" in d for d in out.diagnostics)
