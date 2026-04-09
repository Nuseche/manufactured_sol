"""SymPy tensor/vector operations for 2D symbolic mechanics."""

from __future__ import annotations

from sympy import Expr, Matrix, diff


def grad_vector(u: Matrix, coords: Matrix) -> Matrix:
    """Return Jacobian matrix grad(u) with shape (len(u), len(coords))."""
    return u.jacobian(coords)


def div_tensor(p: Matrix, coords: Matrix) -> Matrix:
    """Return Div(P) for a second-order tensor P in reference coordinates."""
    rows, cols = p.shape
    if cols != len(coords):
        raise ValueError("Tensor columns must match coordinate dimension.")
    return Matrix([sum(diff(p[i, j], coords[j]) for j in range(cols)) for i in range(rows)])


def matrix_inner(a: Matrix, b: Matrix) -> Expr:
    """Return Frobenius inner product a:b."""
    if a.shape != b.shape:
        raise ValueError("Matrix shapes must match for inner product.")
    return sum(a[i, j] * b[i, j] for i in range(a.rows) for j in range(a.cols))


def dot(a: Matrix, b: Matrix) -> Expr:
    """Return vector dot product."""
    if a.shape != b.shape:
        raise ValueError("Vector shapes must match for dot product.")
    return sum(a[i, 0] * b[i, 0] for i in range(a.rows))


def traction(p: Matrix, n: Matrix) -> Matrix:
    """Return traction vector P*n."""
    if n.cols != 1:
        raise ValueError("Normal must be a column vector.")
    return p * n
