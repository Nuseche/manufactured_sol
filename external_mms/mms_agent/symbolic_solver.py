"""Symbolic solver for ansatz coefficients under strict constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from sympy import Eq, Matrix, Poly, nonlinsolve, simplify, solve

from .ansatz_generator import AnsatzCandidate
from .strict_preservation import StrictConstraints


@dataclass
class SolveResult:
    """Result of symbolic solve attempt."""

    solved: bool
    substitutions: Dict
    diagnostics: List[str]


class SymbolicSolver:
    """Solve coefficient constraints symbolically with deterministic fallback."""

    @staticmethod
    def _eq_to_expr(eq: Eq):
        return simplify(eq.lhs - eq.rhs)

    @staticmethod
    def _collect_coeff_equations(exprs: List, vars_: List) -> List:
        coeff_eqs: List = []
        for expr in exprs:
            num = expr.as_numer_denom()[0]
            try:
                poly = Poly(simplify(num), *vars_)
                coeff_eqs.extend(poly.coeffs())
            except Exception:
                coeff_eqs.append(simplify(num))
        return [simplify(c) for c in coeff_eqs]

    @staticmethod
    def solve_constraints(constraints: StrictConstraints, ansatz: AnsatzCandidate, vars_: List) -> SolveResult:
        """Solve strict symbolic equations with exact methods only."""
        diagnostics: List[str] = []
        exprs = [SymbolicSolver._eq_to_expr(eq) for eq in constraints.equations]
        coeff_eqs = SymbolicSolver._collect_coeff_equations(exprs, vars_)
        coeff_eqs = [e for e in coeff_eqs if simplify(e) != 0]

        diagnostics.append(f"Constructed {len(coeff_eqs)} coefficient equations.")
        if not coeff_eqs:
            return SolveResult(solved=False, substitutions={}, diagnostics=diagnostics + ["No nontrivial equations generated."])

        try:
            sol = solve(coeff_eqs, ansatz.coeffs, dict=True)
            if sol:
                diagnostics.append("Solved with sympy.solve.")
                return SolveResult(solved=True, substitutions=sol[0], diagnostics=diagnostics)
        except Exception as exc:
            diagnostics.append(f"sympy.solve failed: {exc}")

        try:
            nsol = nonlinsolve(coeff_eqs, ansatz.coeffs)
            for candidate in nsol:
                if hasattr(candidate, "free_symbols") and not candidate.free_symbols:
                    substitutions = {c: v for c, v in zip(ansatz.coeffs, list(candidate))}
                    diagnostics.append("Solved with sympy.nonlinsolve.")
                    return SolveResult(solved=True, substitutions=substitutions, diagnostics=diagnostics)
            diagnostics.append("nonlinsolve returned no fully determined symbolic tuple.")
        except Exception as exc:
            diagnostics.append(f"sympy.nonlinsolve failed: {exc}")

        return SolveResult(solved=False, substitutions={}, diagnostics=diagnostics + ["Search space exhausted without exact symbolic solve."])
