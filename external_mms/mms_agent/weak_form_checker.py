"""Structural weak-form consistency checker for v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .schemas import CanonicalProblemIR


@dataclass
class WeakFormCheckResult:
    """Weak form structural consistency outcome."""

    consistent: bool
    diagnostics: List[str]


class WeakFormChecker:
    """Perform declarative/structural consistency checks for weak form."""

    @staticmethod
    def check(ir: CanonicalProblemIR) -> WeakFormCheckResult:
        """Check structural consistency between strong form, boundaries and weak form text."""
        diagnostics: List[str] = []

        eqs = ir.equations
        has_div_p = any(str(eq.get("lhs", "")).strip().startswith("Div(P)") for eq in eqs)
        diagnostics.append(f"Strong form declares Div(P)=0: {has_div_p}.")

        wf_lhs = ir.weak_form.get("lhs", "")
        wf_rhs = ir.weak_form.get("rhs", "")
        lhs_ok = "inner(P" in wf_lhs and "grad(v)" in wf_lhs
        rhs_ok = "Gamma_N" in wf_rhs and "tbar" in wf_rhs
        diagnostics.append(f"Weak-form LHS structure valid: {lhs_ok}.")
        diagnostics.append(f"Weak-form RHS structure valid: {rhs_ok}.")

        bc_names = {bc.boundary for bc in ir.boundary_conditions}
        partition_ok = {"Gamma_D", "Gamma_N", "Gamma_rest"}.issubset(bc_names)
        diagnostics.append(f"Boundary partition references complete: {partition_ok}.")

        return WeakFormCheckResult(consistent=has_div_p and lhs_ok and rhs_ok and partition_ok, diagnostics=diagnostics)
