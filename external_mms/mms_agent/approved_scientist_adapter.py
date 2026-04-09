"""Deterministic adapter from approved scientist payload to canonical content."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .schemas import ApprovedScientistPayload


class ApprovedScientistAdapter:
    """Adapter for the currently supported approved scientist JSON format."""

    @staticmethod
    def load(path: str | Path) -> ApprovedScientistPayload:
        """Load and validate an approved scientist payload from disk."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return ApprovedScientistPayload.model_validate(data)

    @staticmethod
    def extract(payload: ApprovedScientistPayload) -> Dict[str, Any]:
        """Extract canonical raw sections from payload deterministically."""
        return {
            "problem_id": payload.problem_id,
            "mode": payload.mode,
            "problem_statement": payload.problem_statement,
            "coordinates": payload.coordinates,
            "unknowns": payload.unknowns,
            "definitions": [d.model_dump() for d in payload.definitions],
            "strong_form": payload.strong_form,
            "weak_form": payload.weak_form,
            "domain": payload.domain,
            "boundaries": payload.boundaries,
            "boundary_conditions": [bc.model_dump() for bc in payload.boundary_conditions],
            "initial_conditions": payload.initial_conditions,
            "parameters": payload.parameters,
            "assumptions": payload.assumptions,
            "ambiguities": payload.declared_ambiguities,
        }
