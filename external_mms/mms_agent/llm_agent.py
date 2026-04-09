"""Azure OpenAI LLM integration for mandatory post-check diagnostics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib import error, parse, request

from .schemas import CanonicalProblemIR, VerificationReport


@dataclass(frozen=True)
class AzureOpenAISettings:
    """Runtime settings for Azure OpenAI access."""

    api_key: str
    endpoint: str
    deployment: str
    model: str
    api_mode: str
    api_version: str
    timeout_seconds: int = 20

    @classmethod
    def from_env(cls) -> "AzureOpenAISettings | None":
        """Load settings from environment variables."""
        _load_dotenv_if_present()

        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        explicit_model = os.getenv("AZURE_OPENAI_MODEL", "").strip()
        model = explicit_model or deployment
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
        api_mode = _resolve_api_mode(
            explicit_model=explicit_model,
            deployment=deployment,
            api_version=api_version,
        )
        timeout = int(os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "20"))

        if not api_key or not endpoint:
            return None
        if api_mode == "responses" and not model:
            return None
        if api_mode == "chat_completions" and not deployment:
            return None
        return cls(
            api_key=api_key,
            endpoint=endpoint,
            deployment=deployment,
            model=model,
            api_mode=api_mode,
            api_version=api_version or _default_api_version(api_mode),
            timeout_seconds=timeout,
        )


def _load_dotenv_if_present() -> Path | None:
    """Load a nearby .env file without overriding already-exported variables."""
    env_path = _find_dotenv()
    if env_path is None:
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value and value[0] not in {"'", '"'} and " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)

    return env_path


def _find_dotenv() -> Path | None:
    """Search upward from common execution roots until a .env file is found."""
    visited: set[Path] = set()
    search_roots = [
        Path.cwd(),
        Path(__file__).resolve().parent,
    ]

    for root in search_roots:
        current = root if root.is_dir() else root.parent
        for directory in (current, *current.parents):
            if directory in visited:
                continue
            visited.add(directory)
            candidate = directory / ".env"
            if candidate.is_file():
                return candidate
    return None


def _resolve_api_mode(explicit_model: str, deployment: str, api_version: str) -> str:
    """Choose the Azure inference surface that matches the configured deployment."""
    explicit = os.getenv("AZURE_OPENAI_API_MODE", "").strip().lower()
    if explicit in {"responses", "chat_completions"}:
        return explicit

    normalized_version = api_version.strip().lower()
    normalized_name = (explicit_model or deployment).strip().lower()
    if normalized_version in {"v1", "preview"}:
        return "responses"
    if explicit_model:
        return "responses"
    if normalized_name.startswith(("gpt-5", "o1", "o3", "o4")):
        return "responses"
    return "chat_completions"


def _default_api_version(api_mode: str) -> str:
    """Return conservative defaults that keep current Azure integrations working."""
    if api_mode == "responses":
        return "2025-04-01-preview"
    return "2024-10-21"


def _build_request_components(
    settings: AzureOpenAISettings,
    payload: Dict[str, Any],
) -> tuple[str, bytes, Dict[str, str]]:
    """Build the final HTTP request tuple for the selected Azure API surface."""
    if settings.api_mode == "responses":
        url = _responses_url(settings)
    else:
        deployment = parse.quote(settings.deployment, safe="")
        url = (
            f"{settings.endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={settings.api_version}"
        )

    return (
        url,
        json.dumps(payload).encode("utf-8"),
        {
            "Content-Type": "application/json",
            "api-key": settings.api_key,
        },
    )


def _responses_url(settings: AzureOpenAISettings) -> str:
    """Resolve the correct Responses API URL for preview and v1 styles."""
    version = settings.api_version.strip().lower()
    if not version or version == "v1":
        return f"{settings.endpoint}/openai/v1/responses"
    if version == "preview":
        return f"{settings.endpoint}/openai/v1/responses?api-version=preview"
    return f"{settings.endpoint}/openai/responses?api-version={settings.api_version}"


def _default_requester(
    settings: AzureOpenAISettings,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Call Azure OpenAI chat completions endpoint with stdlib HTTP client."""
    url, body, headers = _build_request_components(settings, payload)
    req = request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=settings.timeout_seconds) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Azure OpenAI HTTP {exc.code}: {exc.reason} | url={url} | body={body_text or '<empty>'}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Azure OpenAI connection error: {exc.reason} | url={url}") from exc


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


def _build_payload(
    settings: AzureOpenAISettings,
    ir: CanonicalProblemIR,
    report: VerificationReport,
) -> Dict[str, Any]:
    """Construct the request body for the configured Azure API surface."""
    instructions = "You are an auditing assistant for symbolic verification reports in computational mechanics."
    prompt = _build_prompt(ir, report)

    if settings.api_mode == "responses":
        return {
            "model": settings.model,
            "instructions": instructions,
            "input": prompt,
            "max_output_tokens": 240,
            "store": False,
            "text": {"format": {"type": "text"}},
        }

    return {
        "messages": [
            {
                "role": "system",
                "content": instructions,
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 240,
    }


def _extract_response_text(settings: AzureOpenAISettings, response: Dict[str, Any]) -> str:
    """Extract normalized text from chat-completions or responses payloads."""
    if settings.api_mode == "chat_completions":
        return (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

    output_text = response.get("output_text", "")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    collected: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            text = content.get("text", "")
            if isinstance(text, str) and text.strip():
                collected.append(text.strip())

    return "\n".join(collected).strip()


def enrich_report_with_llm(
    report: VerificationReport,
    ir: CanonicalProblemIR,
    settings: Optional[AzureOpenAISettings] = None,
    requester: Optional[Callable[[AzureOpenAISettings, Dict[str, Any]], Dict[str, Any]]] = None,
) -> VerificationReport:
    """Append Azure OpenAI diagnostic notes into the report (mandatory path)."""
    resolved_settings = settings or AzureOpenAISettings.from_env()
    if resolved_settings is None:
        raise RuntimeError(
            "Missing Azure OpenAI configuration: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
            "and AZURE_OPENAI_DEPLOYMENT (or AZURE_OPENAI_MODEL for Responses API) are required."
        )

    call = requester or _default_requester
    payload = _build_payload(resolved_settings, ir, report)

    try:
        response = call(resolved_settings, payload)
        content = _extract_response_text(resolved_settings, response)
        if content:
            report.diagnostics.append(f"LLM(AzureOpenAI) audit note: {content}")
        else:
            raise RuntimeError("Azure OpenAI returned empty content for mandatory LLM diagnostics.")
    except Exception as exc:
        raise RuntimeError(f"Mandatory LLM diagnostics failed (Azure OpenAI): {exc}") from exc

    return report
