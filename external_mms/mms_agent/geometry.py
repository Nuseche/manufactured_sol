"""Geometry primitives for Cook membrane quadrilateral and boundary partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from sympy import Matrix, Rational, sqrt


Point = Tuple[float, float]


@dataclass(frozen=True)
class Segment:
    """Boundary segment represented by endpoints."""

    name: str
    start: Point
    end: Point

    def tangent(self) -> Matrix:
        """Return segment tangent vector (end-start)."""
        return Matrix([self.end[0] - self.start[0], self.end[1] - self.start[1]])

    def normal_outward_ccw(self) -> Matrix:
        """Return outward unit normal assuming CCW polygon orientation."""
        t = self.tangent()
        n = Matrix([t[1], -t[0]])
        norm = sqrt(n[0] ** 2 + n[1] ** 2)
        return n / norm

    def param(self, s):
        """Linear parametrization s in [0,1]."""
        return Matrix(
            [
                self.start[0] + (self.end[0] - self.start[0]) * s,
                self.start[1] + (self.end[1] - self.start[1]) * s,
            ]
        )


@dataclass
class CookGeometry:
    """Cook membrane geometry and boundary decomposition."""

    vertices: List[Point]
    gamma_d: Segment
    gamma_n: Segment
    gamma_rest: List[Segment]

    def bilinear_map(self, xi, eta) -> Matrix:
        """Bilinear map from [0,1]^2 -> physical quadrilateral."""
        v0, v1, v2, v3 = [Matrix(v) for v in self.vertices]
        n0 = (1 - xi) * (1 - eta)
        n1 = xi * (1 - eta)
        n2 = xi * eta
        n3 = (1 - xi) * eta
        return n0 * v0 + n1 * v1 + n2 * v2 + n3 * v3



def build_cook_geometry(geometry_dict: Dict) -> CookGeometry:
    """Build exact Cook geometry representation from IR geometry section."""
    vertices = [tuple(v) for v in geometry_dict["domain"]["vertices"]]
    boundaries = geometry_dict["boundaries"]

    gamma_d_eps = boundaries["Gamma_D"]["endpoints"]
    gamma_n_eps = boundaries["Gamma_N"]["endpoints"]
    gamma_d = Segment("Gamma_D", tuple(gamma_d_eps[0]), tuple(gamma_d_eps[1]))
    gamma_n = Segment("Gamma_N", tuple(gamma_n_eps[0]), tuple(gamma_n_eps[1]))

    # Remaining edges from polygon minus explicit Gamma_D/Gamma_N.
    edges = [
        Segment("e01", vertices[0], vertices[1]),
        Segment("e12", vertices[1], vertices[2]),
        Segment("e23", vertices[2], vertices[3]),
        Segment("e30", vertices[3], vertices[0]),
    ]
    known = {gamma_d.start, gamma_d.end, gamma_n.start, gamma_n.end}
    gamma_rest = [
        e
        for e in edges
        if not ({e.start, e.end} == {gamma_d.start, gamma_d.end} or {e.start, e.end} == {gamma_n.start, gamma_n.end})
    ]

    return CookGeometry(vertices=vertices, gamma_d=gamma_d, gamma_n=gamma_n, gamma_rest=gamma_rest)


def rationalize_point(point: Matrix) -> Matrix:
    """Convert point entries to Rational when possible."""
    return Matrix([Rational(point[0]), Rational(point[1])])
