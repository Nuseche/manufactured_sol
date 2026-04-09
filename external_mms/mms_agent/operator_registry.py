"""Operator registry for symbolic expression evaluation and coverage checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Set

from sympy import Matrix, det, log, trace

from .tensor_ops import dot, matrix_inner, traction


@dataclass(frozen=True)
class OperatorRegistry:
    """Registry of supported symbolic operators for v1."""

    operators: Dict[str, Callable[..., Any]]

    @classmethod
    def default(cls) -> "OperatorRegistry":
        """Create registry with operators required by current benchmark."""
        return cls(
            operators={
                "grad": lambda u, coords: u.jacobian(coords),
                "Div": lambda p, coords: Matrix(
                    [sum(p[i, j].diff(coords[j]) for j in range(p.cols)) for i in range(p.rows)]
                ),
                "det": det,
                "transpose": lambda a: a.T,
                "inv": lambda a: a.inv(),
                "trace": trace,
                "dot": dot,
                "inner": matrix_inner,
                "traction": traction,
                "log": log,
                "matmul": lambda a, b: a * b,
            }
        )

    def eval_env(self) -> Dict[str, Callable[..., Any]]:
        """Evaluation environment for sympify/eval with operator callables."""
        return dict(self.operators)

    def supported_names(self) -> Set[str]:
        """Return supported operator names."""
        return set(self.operators.keys())


def extract_operator_tokens(expressions: Iterable[str]) -> Set[str]:
    """Extract function-like operator tokens from textual expressions."""
    pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    found: Set[str] = set()
    for expr in expressions:
        found.update(pattern.findall(expr))
    return found


def coverage_ok(expressions: Iterable[str], supported: Set[str], known_symbols: Set[str]) -> bool:
    """Check whether expression operators are covered by the registry."""
    tokens = extract_operator_tokens(expressions)
    unknown = {t for t in tokens if t not in supported and t not in known_symbols}
    return not unknown
