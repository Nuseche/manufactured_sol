"""Report builder and strict-preservation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from sympy import Matrix, simplify

from .ansatz_generator import AnsatzGenerator
from .geometry import build_cook_geometry
from .operator_registry import OperatorRegistry, coverage_ok
from .residual_checker import ResidualChecker
from .schemas import CanonicalProblemIR, ManufacturedSolutionCandidate, VerificationReport
from .strict_preservation import StrictPreservationBuilder
from .symbolic_context import build_symbolic_context
from .symbolic_solver import SymbolicSolver
from .weak_form_checker import WeakFormChecker


@dataclass
class PipelineResult:
    """Internal pipeline output container."""

    report: VerificationReport


class ReportBuilder:
    """Main pipeline orchestrator for strict-preservation verification."""

    @staticmethod
    def _operator_coverage(ir: CanonicalProblemIR) -> bool:
        expressions: List[str] = [d.rhs for d in ir.definitions]
        expressions.extend([str(e.get("lhs", "")) for e in ir.equations])
        expressions.extend([ir.weak_form.get("lhs", ""), ir.weak_form.get("rhs", "")])
        supported = OperatorRegistry.default().supported_names()
        known = {"Integral_Omega", "Integral_Gamma_N"}
        return coverage_ok(expressions, supported=supported, known_symbols=known)

    @staticmethod
    def build(ir: CanonicalProblemIR, mode: str) -> VerificationReport:
        """Run strict-preservation symbolic verification and return report."""
        if mode != "strict_preservation":
            return VerificationReport(
                status="unsupported_ir",
                ir_valid=True,
                operator_coverage_ok=False,
                strict_preservation=False,
                exact_candidate_found=False,
                strong_form_residual_zero=False,
                boundary_conditions_satisfied=False,
                weak_form_structurally_consistent=False,
                declared_ambiguities=ir.ambiguities,
                diagnostics=[f"Unsupported mode: {mode}."],
            )

        diagnostics: List[str] = []
        op_ok = ReportBuilder._operator_coverage(ir)
        diagnostics.append(f"Operator coverage check: {op_ok}.")

        weak_res = WeakFormChecker.check(ir)
        diagnostics.extend(weak_res.diagnostics)

        if not op_ok:
            return VerificationReport(
                status="unsupported_ir",
                ir_valid=True,
                operator_coverage_ok=False,
                strict_preservation=True,
                exact_candidate_found=False,
                strong_form_residual_zero=False,
                boundary_conditions_satisfied=False,
                weak_form_structurally_consistent=weak_res.consistent,
                declared_ambiguities=ir.ambiguities,
                diagnostics=diagnostics,
            )

        ctx = build_symbolic_context(ir)
        geom = build_cook_geometry(ir.geometry)
        ansatzes = AnsatzGenerator.generate(ir, ctx)

        best_candidate: Optional[ManufacturedSolutionCandidate] = None
        strong_ok = False
        bc_ok = False

        for ans in ansatzes:
            diagnostics.append(f"Trying ansatz: {ans.name}.")
            strict_constraints = StrictPreservationBuilder.build(
                ir=ir,
                coords=ctx.coords,
                param_syms=ctx.parameters,
                ansatz=ans,
                geom=geom,
            )
            diagnostics.extend(strict_constraints.diagnostics)
            solve_result = SymbolicSolver.solve_constraints(strict_constraints, ans, list(ctx.coords))
            diagnostics.extend(solve_result.diagnostics)

            if not solve_result.solved:
                continue

            u_sub = Matrix([simplify(expr.subs(solve_result.substitutions)) for expr in ans.u])
            nontrivial = any(simplify(expr) != 0 for expr in u_sub)
            if not nontrivial:
                diagnostics.append("Discarded trivial solved candidate.")
                continue

            residual = ResidualChecker.check(ir, ctx.coords, ctx.parameters, u_sub, geom)
            diagnostics.extend(residual.diagnostics)
            if residual.strong_form_zero and residual.boundary_zero and residual.initial_conditions_ok and residual.definitions_consistent:
                strong_ok = True
                bc_ok = True
                best_candidate = ManufacturedSolutionCandidate(
                    unknown_expressions={"u_x": str(u_sub[0]), "u_y": str(u_sub[1])},
                    coefficients={str(k): str(v) for k, v in solve_result.substitutions.items()},
                    exact=True,
                    nontrivial=True,
                    diagnostics=[f"Accepted ansatz {ans.name} as exact symbolic solution."],
                )
                break

        if best_candidate is not None:
            status = "success"
            exact_found = True
        else:
            status = "infeasible_under_strict_preservation"
            exact_found = False
            diagnostics.append(
                "No exact nontrivial manufactured solution found in deterministic v1 ansatz spaces under strict preservation."
            )

        return VerificationReport(
            status=status,
            ir_valid=True,
            operator_coverage_ok=op_ok,
            strict_preservation=True,
            exact_candidate_found=exact_found,
            strong_form_residual_zero=strong_ok,
            boundary_conditions_satisfied=bc_ok,
            weak_form_structurally_consistent=weak_res.consistent,
            declared_ambiguities=ir.ambiguities,
            diagnostics=diagnostics,
            candidate_solution=best_candidate,
        )
