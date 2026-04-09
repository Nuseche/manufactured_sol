"""Pydantic schemas for external_mms."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DefinitionIR(BaseModel):
    """A symbolic definition entry in the IR."""

    lhs: str
    rhs: str


class BoundaryConditionIR(BaseModel):
    """Boundary condition in canonical IR form."""

    type: Literal["Dirichlet", "Neumann"]
    target: str
    boundary: str
    value: List[Any]
    description: Optional[str] = None


class ApprovedScientistPayload(BaseModel):
    """Approved payload produced by scientist/judge."""

    approved: bool
    source: str
    problem_id: str
    mode: str
    problem_statement: str
    dimension: int
    coordinates: List[str]
    time_dependent: bool
    unknowns: List[Dict[str, Any]]
    domain: Dict[str, Any]
    definitions: List[DefinitionIR]
    strong_form: Dict[str, Any]
    weak_form: Dict[str, Any]
    boundaries: Dict[str, Any]
    boundary_conditions: List[BoundaryConditionIR]
    initial_conditions: List[Dict[str, Any]]
    parameters: Dict[str, Dict[str, Any]]
    forcing_terms: Dict[str, Any] = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)
    declared_ambiguities: List[str] = Field(default_factory=list)
    objective_outputs: List[str] = Field(default_factory=list)
    consistency_warnings: List[str] = Field(default_factory=list)
    notes: str = ""


class CanonicalProblemIR(BaseModel):
    """Operator-oriented canonical IR for symbolic verification."""

    problem_id: str
    mode: str
    problem_statement: str
    coordinates: List[str]
    unknowns: List[Dict[str, Any]]
    definitions: List[DefinitionIR]
    equations: List[Dict[str, Any]]
    geometry: Dict[str, Any]
    boundary_conditions: List[BoundaryConditionIR]
    initial_conditions: List[Dict[str, Any]]
    parameters: Dict[str, Dict[str, Any]]
    assumptions: List[str]
    ambiguities: List[str]
    weak_form: Dict[str, Any]


class ManufacturedSolutionCandidate(BaseModel):
    """Candidate manufactured solution."""

    unknown_expressions: Dict[str, str]
    coefficients: Dict[str, str]
    exact: bool
    nontrivial: bool
    diagnostics: List[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    """Structured final report for strict-preservation verification."""

    status: Literal[
        "success",
        "infeasible_under_strict_preservation",
        "unsupported_ir",
        "parse_error",
    ]
    ir_valid: bool
    operator_coverage_ok: bool
    strict_preservation: bool
    exact_candidate_found: bool
    strong_form_residual_zero: bool
    boundary_conditions_satisfied: bool
    weak_form_structurally_consistent: bool
    declared_ambiguities: List[str] = Field(default_factory=list)
    diagnostics: List[str] = Field(default_factory=list)
    candidate_solution: Optional[ManufacturedSolutionCandidate] = None
