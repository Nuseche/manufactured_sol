"""Geometry tests."""

from __future__ import annotations

from sympy import Rational

from mms_agent.geometry import build_cook_geometry


def test_cook_geometry_normals_and_mapping() -> None:
    geom = build_cook_geometry(
        {
            "domain": {"vertices": [[0.0, 0.0], [48.0, 44.0], [48.0, 60.0], [0.0, 44.0]]},
            "boundaries": {
                "Gamma_D": {"endpoints": [[0.0, 0.0], [0.0, 44.0]]},
                "Gamma_N": {"endpoints": [[48.0, 44.0], [48.0, 60.0]]},
                "Gamma_rest": {"type": "remaining_boundary"},
            },
        }
    )

    n_right = geom.gamma_n.normal_outward_ccw()
    assert float(n_right[0]) == 1.0
    assert float(n_right[1]) == 0.0
    assert len(geom.gamma_rest) == 2

    p = geom.bilinear_map(Rational(0), Rational(0))
    assert p[0] == 0
    assert p[1] == 0
