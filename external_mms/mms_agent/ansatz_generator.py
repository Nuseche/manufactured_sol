"""Deterministic ansatz generation for displacement unknowns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sympy import Matrix, Symbol, symbols

from .schemas import CanonicalProblemIR
from .symbolic_context import SymbolicContext


@dataclass
class AnsatzCandidate:
    """Candidate displacement ansatz with symbolic coefficients."""

    u: Matrix
    coeffs: List[Symbol]
    name: str


class AnsatzGenerator:
    """Generate deterministic low-order polynomial ansatz candidates."""

    @staticmethod
    def generate(ir: CanonicalProblemIR, ctx: SymbolicContext) -> List[AnsatzCandidate]:
        """Generate deterministic candidate list for u=(u_x,u_y)."""
        x = ctx.coord_symbols[ir.coordinates[0]]
        y = ctx.coord_symbols[ir.coordinates[1]]

        a0, a1, a2, b0, b1, b2 = symbols("a0 a1 a2 b0 b1 b2", real=True)
        # Enforce u=0 on x=0 exactly by construction.
        u_linear = Matrix(
            [
                x * (a0 + a1 * x + a2 * y),
                x * (b0 + b1 * x + b2 * y),
            ]
        )

        c0, c1, c2, c3, d0, d1, d2, d3 = symbols("c0 c1 c2 c3 d0 d1 d2 d3", real=True)
        u_quadratic = Matrix(
            [
                x * (c0 + c1 * x + c2 * y + c3 * y**2),
                x * (d0 + d1 * x + d2 * y + d3 * y**2),
            ]
        )

        return [
            AnsatzCandidate(u=u_linear, coeffs=[a0, a1, a2, b0, b1, b2], name="linear_x_factored"),
            AnsatzCandidate(
                u=u_quadratic,
                coeffs=[c0, c1, c2, c3, d0, d1, d2, d3],
                name="quadratic_x_factored",
            ),
        ]
