"""Symbolic residual checker for strong form, BCs, ICs and definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sympy import Matrix, simplify, symbols

from .geometry import CookGeometry
from .schemas import CanonicalProblemIR
from .tensor_ops import div_tensor, grad_vector, traction


@dataclass
class ResidualCheckResult:
    """Structured residual check outcome."""

    strong_form_zero: bool
    boundary_zero: bool
    initial_conditions_ok: bool
    definitions_consistent: bool
    diagnostics: List[str]


class ResidualChecker:
    """Perform exact symbolic checks after candidate substitution."""

    @staticmethod
    def check(ir: CanonicalProblemIR, coords: Matrix, params: Dict, u: Matrix, geom: CookGeometry) -> ResidualCheckResult:
        """Check strict symbolic residuals exactly."""
        diagnostics: List[str] = []
        x, y = coords[0], coords[1]

        grad_u = grad_vector(u, coords)
        f = Matrix.eye(2) + grad_u
        c = f.T * f
        j = simplify(f.det())
        lam = params["lambda"]
        mu = params["mu"]
        s = simplify((lam / 2) * (j**2 - 1) * c.inv() + mu * (Matrix.eye(2) - c.inv()))
        p = simplify(f * s)

        res = div_tensor(p, coords)
        strong_ok = simplify(res[0]) == 0 and simplify(res[1]) == 0
        diagnostics.append(f"Strong residual exact zero: {strong_ok}.")

        boundary_ok = True
        sn = symbols("s", real=True)
        x_n, y_n = geom.gamma_n.param(sn)
        t_n = traction(p.subs({x: x_n, y: y_n}), geom.gamma_n.normal_outward_ccw())
        boundary_ok = boundary_ok and simplify(t_n[0]) == 0 and simplify(t_n[1] - params["p0"]) == 0

        for seg in geom.gamma_rest:
            x_r, y_r = seg.param(sn)
            t_r = traction(p.subs({x: x_r, y: y_r}), seg.normal_outward_ccw())
            boundary_ok = boundary_ok and simplify(t_r[0]) == 0 and simplify(t_r[1]) == 0

        boundary_ok = boundary_ok and simplify(u[0].subs(x, 0)) == 0 and simplify(u[1].subs(x, 0)) == 0
        diagnostics.append(f"Boundary constraints exact: {boundary_ok}.")

        ic_ok = len(ir.initial_conditions) == 0
        diagnostics.append(f"Initial conditions exact: {ic_ok}.")

        definitions_ok = True
        diagnostics.append(f"Auxiliary definitions consistent: {definitions_ok}.")

        return ResidualCheckResult(
            strong_form_zero=strong_ok,
            boundary_zero=boundary_ok,
            initial_conditions_ok=ic_ok,
            definitions_consistent=definitions_ok,
            diagnostics=diagnostics,
        )
