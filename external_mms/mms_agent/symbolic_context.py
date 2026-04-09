"""Symbolic context builder for PDE definitions and equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sympy import Matrix, Symbol, symbols

from .schemas import CanonicalProblemIR


@dataclass
class SymbolicContext:
    """Container for symbolic entities used during verification."""

    coords: Matrix
    coord_symbols: Dict[str, Symbol]
    parameters: Dict[str, Symbol]
    parameter_values: Dict[str, float]
    identities: Dict[str, Matrix]


def build_symbolic_context(ir: CanonicalProblemIR) -> SymbolicContext:
    """Build baseline symbolic context from IR."""
    coord_syms = {name: symbols(name, real=True) for name in ir.coordinates}
    coords = Matrix([coord_syms[name] for name in ir.coordinates])

    params = {name: symbols(name, real=True) for name in ir.parameters.keys()}
    param_values = {name: float(meta["value"]) for name, meta in ir.parameters.items()}

    dim = len(ir.coordinates)
    identities = {f"I{dim}": Matrix.eye(dim), "I2": Matrix.eye(2)}

    return SymbolicContext(
        coords=coords,
        coord_symbols=coord_syms,
        parameters=params,
        parameter_values=param_values,
        identities=identities,
    )
