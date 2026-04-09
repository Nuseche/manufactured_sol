"""Smoke test for strict-preservation CLI pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import mms_agent.llm_agent as llm_agent
from mms_agent.cli import run


def test_cook_strict_smoke(tmp_path: Path, monkeypatch) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "approved_scientist.json"
    inp = tmp_path / "approved_scientist.json"
    out = tmp_path / "report.json"
    inp.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt4o")
    monkeypatch.delenv("AZURE_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    monkeypatch.setattr(llm_agent, "_find_dotenv", lambda: None)

    def fake_requester(_settings, _payload):
        return {"choices": [{"message": {"content": "smoke ok"}}]}

    report = run(str(inp), "strict_preservation", str(out), llm_requester=fake_requester)

    assert report.status in {
        "success",
        "infeasible_under_strict_preservation",
        "unsupported_ir",
        "parse_error",
    }
    assert any("LLM(AzureOpenAI) audit note: smoke ok" in d for d in report.diagnostics)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert "status" in loaded


def test_cook_strict_smoke_missing_azure_env_is_parse_error(tmp_path: Path, monkeypatch) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "approved_scientist.json"
    inp = tmp_path / "approved_scientist.json"
    out = tmp_path / "report.json"
    inp.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    monkeypatch.setattr(llm_agent, "_find_dotenv", lambda: None)

    report = run(str(inp), "strict_preservation", str(out))
    assert report.status == "parse_error"
    assert any("Missing Azure OpenAI configuration" in d for d in report.diagnostics)
