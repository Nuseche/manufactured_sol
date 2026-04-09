"""Symbolic solver for ansatz coefficients under strict constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from sympy import Eq, Rational, Symbol, count_ops, nonlinsolve, simplify, solve, sympify

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
        if not hasattr(eq, "lhs") or not hasattr(eq, "rhs"):
            return 0 if bool(eq) else 1
        return eq.lhs - eq.rhs

    @staticmethod
    def _collect_collocation_equations(exprs: List, vars_: List) -> List:
        sample_vals = [Rational(0), Rational(1, 2), Rational(1)]
        collocated: List = []
        for expr in exprs:
            num = expr.as_numer_denom()[0]
            if not vars_:
                collocated.append(simplify(num))
                continue
            pending = [({}, 0)]
            while pending:
                subs_map, idx = pending.pop()
                if idx == len(vars_):
                    collocated.append(simplify(num.subs(subs_map)))
                    continue
                var = vars_[idx]
                for val in sample_vals:
                    nxt = dict(subs_map)
                    nxt[var] = val
                    pending.append((nxt, idx + 1))
        cleaned = [simplify(e) for e in collocated if simplify(e) != 0]
        return list(dict.fromkeys(cleaned))

    @staticmethod
    def solve_constraints(
        constraints: StrictConstraints,
        ansatz: AnsatzCandidate,
        vars_: List,
        parameter_subs: Optional[Dict[Symbol, float]] = None,
    ) -> SolveResult:
        """Solve strict symbolic equations with exact methods only."""
        diagnostics: List[str] = []
        exprs = [sympify(SymbolicSolver._eq_to_expr(eq)) for eq in constraints.equations]
        if parameter_subs:
            exprs = [expr.subs(parameter_subs) for expr in exprs]
        coeff_eqs = SymbolicSolver._collect_collocation_equations(exprs, vars_)

        diagnostics.append(f"Constructed {len(coeff_eqs)} coefficient equations.")
        if not coeff_eqs:
            return SolveResult(solved=False, substitutions={}, diagnostics=diagnostics + ["No nontrivial equations generated."])

        if len(coeff_eqs) > 120:
            diagnostics.append("Equation system exceeds v1 deterministic complexity budget (>120 equations).")
            return SolveResult(
                solved=False,
                substitutions={},
                diagnostics=diagnostics + ["Search space exhausted without exact symbolic solve."],
            )

        max_ops = max(int(count_ops(eq)) for eq in coeff_eqs)
        if max_ops > 600:
            diagnostics.append("Equation complexity exceeds v1 deterministic complexity budget (count_ops>600).")
            return SolveResult(
                solved=False,
                substitutions={},
                diagnostics=diagnostics + ["Search space exhausted without exact symbolic solve."],
            )

        try:
            sol = solve(coeff_eqs, ansatz.coeffs, dict=True, simplify=False, check=False, manual=True)
            if sol:
                diagnostics.append("Solved with sympy.solve.")
                return SolveResult(solved=True, substitutions=sol[0], diagnostics=diagnostics)
        except Exception as exc:
            diagnostics.append(f"sympy.solve failed: {exc}")

        if len(coeff_eqs) <= 12 and len(ansatz.coeffs) <= 8:
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
        else:
            diagnostics.append("Skipped nonlinsolve due to deterministic complexity guard.")

        return SolveResult(solved=False, substitutions={}, diagnostics=diagnostics + ["Search space exhausted without exact symbolic solve."])
