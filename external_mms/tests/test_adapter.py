"""Tests for approved scientist adapter and IR builder."""

from __future__ import annotations

import json
from pathlib import Path

from mms_agent.approved_scientist_adapter import ApprovedScientistAdapter
from mms_agent.ir_builder import IRBuilder


def test_adapter_extract_and_ir_build(tmp_path: Path) -> None:
    payload = {
        "approved": True,
        "source": "scientist",
        "problem_id": "cook_hyperelastic_plane_strain_psi1_v1",
        "mode": "strict_preservation",
        "problem_statement": "test",
        "dimension": 2,
        "coordinates": ["X", "Y"],
        "time_dependent": False,
        "unknowns": [{"name": "u", "kind": "vector", "shape": [2], "components": ["u_x", "u_y"]}],
        "domain": {"type": "quadrilateral", "vertices": [[0, 0], [48, 44], [48, 60], [0, 44]], "units": "mm", "thickness": 1.0},
        "definitions": [{"lhs": "F", "rhs": "I2 + grad(u)"}],
        "strong_form": {"equations": [{"lhs": "Div(P)", "rhs": [0, 0]}], "body_force": [0, 0]},
        "weak_form": {"lhs": "Integral_Omega(inner(P, grad(v)))", "rhs": "Integral_Gamma_N(dot(tbar, v))"},
        "boundaries": {
            "Gamma_D": {"type": "segment", "endpoints": [[0, 0], [0, 44]]},
            "Gamma_N": {"type": "segment", "endpoints": [[48, 44], [48, 60]]},
            "Gamma_rest": {"type": "remaining_boundary"},
        },
        "boundary_conditions": [
            {"type": "Dirichlet", "target": "u", "boundary": "Gamma_D", "value": [0, 0]},
            {"type": "Neumann", "target": "P*n", "boundary": "Gamma_N", "value": [0, "p0"]},
            {"type": "Neumann", "target": "P*n", "boundary": "Gamma_rest", "value": [0, 0]},
        ],
        "initial_conditions": [],
        "parameters": {
            "lambda": {"value": 432.099, "units": "MPa"},
            "mu": {"value": 185.185, "units": "MPa"},
            "p0": {"value": 20.0, "units": "MPa"},
            "t": {"value": 1.0, "units": "mm"},
        },
        "forcing_terms": {"body_force": [0, 0]},
        "assumptions": ["a"],
        "declared_ambiguities": ["b"],
        "objective_outputs": [],
        "consistency_warnings": [],
        "notes": "n",
    }
    path = tmp_path / "approved.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    model = ApprovedScientistAdapter.load(path)
    extracted = ApprovedScientistAdapter.extract(model)
    ir = IRBuilder.build(extracted)

    assert ir.problem_id == "cook_hyperelastic_plane_strain_psi1_v1"
    assert ir.coordinates == ["X", "Y"]
    assert len(ir.boundary_conditions) == 3
