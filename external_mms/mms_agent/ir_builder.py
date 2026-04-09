"""IR builder for operator-oriented canonical representation."""

from __future__ import annotations

from typing import Any, Dict, List

from .schemas import BoundaryConditionIR, CanonicalProblemIR, DefinitionIR


class IRBuilder:
    """Build canonical IR from adapted approved payload content."""

    @staticmethod
    def build(extracted: Dict[str, Any]) -> CanonicalProblemIR:
        """Build a validated canonical IR."""
        equations: List[Dict[str, Any]] = extracted["strong_form"].get("equations", [])
        geometry = {
            "domain": extracted["domain"],
            "boundaries": extracted["boundaries"],
        }
        return CanonicalProblemIR(
            problem_id=extracted["problem_id"],
            mode=extracted["mode"],
            problem_statement=extracted["problem_statement"],
            coordinates=extracted["coordinates"],
            unknowns=extracted["unknowns"],
            definitions=[DefinitionIR.model_validate(d) for d in extracted["definitions"]],
            equations=equations,
            geometry=geometry,
            boundary_conditions=[BoundaryConditionIR.model_validate(bc) for bc in extracted["boundary_conditions"]],
            initial_conditions=extracted["initial_conditions"],
            parameters=extracted["parameters"],
            assumptions=extracted["assumptions"],
            ambiguities=extracted["ambiguities"],
            weak_form=extracted["weak_form"],
        )
