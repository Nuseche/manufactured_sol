"""Strict-preservation symbolic constraints builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sympy import Eq, Matrix, Symbol, symbols

from .ansatz_generator import AnsatzCandidate
from .geometry import CookGeometry
from .schemas import BoundaryConditionIR, CanonicalProblemIR
from .tensor_ops import div_tensor, grad_vector, traction


@dataclass
class StrictConstraints:
    """Symbolic equalities required for strict-preservation mode."""

    equations: List
    diagnostics: List[str]


class StrictPreservationBuilder:
    """Construct strict symbolic constraints for a given ansatz."""

    @staticmethod
    def _constitutive(ir: CanonicalProblemIR, coords: Matrix, u: Matrix, param_syms: Dict[str, Symbol]) -> Dict[str, Matrix]:
        grad_u = grad_vector(u, coords)
        f = Matrix.eye(2) + grad_u
        c = f.T * f
        j = f.det()
        lam = param_syms["lambda"]
        mu = param_syms["mu"]
        s = (lam / 2) * (j**2 - 1) * c.inv() + mu * (Matrix.eye(2) - c.inv())
        p = f * s
        return {"F": f, "C": c, "J": j, "S": s, "P": p}

    @staticmethod
    def build(
        ir: CanonicalProblemIR,
        coords: Matrix,
        param_syms: Dict[str, Symbol],
        ansatz: AnsatzCandidate,
        geom: CookGeometry,
    ) -> StrictConstraints:
        """Build strong-form and BC exact constraints."""
        diagnostics: List[str] = []
        tensors = StrictPreservationBuilder._constitutive(ir, coords, ansatz.u, param_syms)
        p = tensors["P"]

        residual = div_tensor(p, coords)
        equations = [Eq(residual[0], 0), Eq(residual[1], 0)]
        diagnostics.append("Added exact strong-form residual constraints Div(P)=0.")

        x, y = coords[0], coords[1]
        equations.append(Eq(ansatz.u[0].subs(x, 0), 0))
        equations.append(Eq(ansatz.u[1].subs(x, 0), 0))
        diagnostics.append("Added exact Dirichlet constraints on Gamma_D.")

        s = symbols("s", real=True)
        x_n, y_n = geom.gamma_n.param(s)
        n_n = geom.gamma_n.normal_outward_ccw()
        t_n = traction(p.subs({x: x_n, y: y_n}), n_n)
        equations.append(Eq(t_n[0], 0))
        equations.append(Eq(t_n[1], param_syms["p0"]))
        diagnostics.append("Added exact Neumann constraints on Gamma_N.")

        for seg in geom.gamma_rest:
            x_r, y_r = seg.param(s)
            n_r = seg.normal_outward_ccw()
            t_r = traction(p.subs({x: x_r, y: y_r}), n_r)
            equations.append(Eq(t_r[0], 0))
            equations.append(Eq(t_r[1], 0))
        diagnostics.append("Added exact traction-free constraints on Gamma_rest.")

        if ir.initial_conditions:
            diagnostics.append("Initial conditions detected; strict exact IC constraints should be added in future extensions.")

        return StrictConstraints(equations=equations, diagnostics=diagnostics)
