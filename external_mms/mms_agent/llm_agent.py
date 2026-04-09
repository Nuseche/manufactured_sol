"""Optional Azure OpenAI LLM integration for additional diagnostics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from urllib import request

from .schemas import CanonicalProblemIR, VerificationReport


@dataclass(frozen=True)
class AzureOpenAISettings:
    """Runtime settings for Azure OpenAI access."""

    api_key: str
    endpoint: str
    deployment: str
    api_version: str = "2024-10-21"
    timeout_seconds: int = 20

    @classmethod
    def from_env(cls) -> "AzureOpenAISettings | None":
        """Load settings from environment variables."""
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21").strip()
        timeout = int(os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "20"))

        if not api_key or not endpoint or not deployment:
            return None
        return cls(
            api_key=api_key,
            endpoint=endpoint,
            deployment=deployment,
            api_version=api_version,
            timeout_seconds=timeout,
        )


def _default_requester(
    settings: AzureOpenAISettings,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Call Azure OpenAI chat completions endpoint with stdlib HTTP client."""
    url = (
        f"{settings.endpoint}/openai/deployments/{settings.deployment}/chat/completions"
        f"?api-version={settings.api_version}"
    )
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "api-key": settings.api_key,
        },
        method="POST",
    )
    with request.urlopen(req, timeout=settings.timeout_seconds) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _build_prompt(ir: CanonicalProblemIR, report: VerificationReport) -> str:
    """Build a compact deterministic prompt for post-check diagnostics."""
    diag = "\n".join(f"- {d}" for d in report.diagnostics[-8:])
    return (
        "Provide a short audit note (max 5 bullets) for a strict-preservation MMS symbolic verifier.\n"
        f"Problem ID: {ir.problem_id}\n"
        f"Mode: {ir.mode}\n"
        f"Status: {report.status}\n"
        f"Weak form structurally consistent: {report.weak_form_structurally_consistent}\n"
        f"Exact candidate found: {report.exact_candidate_found}\n"
        "Recent diagnostics:\n"
        f"{diag}\n"
        "Rules: do not suggest relaxing physics or boundary conditions; focus on next symbolic steps only."
    )


def enrich_report_with_llm(
    report: VerificationReport,
    ir: CanonicalProblemIR,
    enabled: bool,
    settings: Optional[AzureOpenAISettings] = None,
    requester: Optional[Callable[[AzureOpenAISettings, Dict[str, Any]], Dict[str, Any]]] = None,
) -> VerificationReport:
    """Optionally append Azure OpenAI diagnostic notes into the report."""
    if not enabled:
        return report

    resolved_settings = settings or AzureOpenAISettings.from_env()
    if resolved_settings is None:
        report.diagnostics.append(
            "LLM diagnostics skipped: missing AZURE_OPENAI_API_KEY/AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_DEPLOYMENT."
        )
        return report

    call = requester or _default_requester
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an auditing assistant for symbolic verification reports in computational mechanics."
                ),
            },
            {"role": "user", "content": _build_prompt(ir, report)},
        ],
        "temperature": 0.0,
        "max_tokens": 240,
    }

    try:
        response = call(resolved_settings, payload)
        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if content:
            report.diagnostics.append(f"LLM(AzureOpenAI) audit note: {content}")
        else:
            report.diagnostics.append("LLM diagnostics requested but Azure response contained no content.")
    except Exception as exc:
        report.diagnostics.append(f"LLM diagnostics failed (Azure OpenAI): {exc}")

    return report
