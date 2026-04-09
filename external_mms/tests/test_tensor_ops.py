"""Tensor ops tests."""

from __future__ import annotations

from sympy import Matrix, symbols

from mms_agent.tensor_ops import div_tensor, grad_vector, traction


def test_grad_div_traction() -> None:
    x, y = symbols("x y")
    u = Matrix([x**2 + y, x * y])
    coords = Matrix([x, y])

    g = grad_vector(u, coords)
    assert g.shape == (2, 2)
    assert g[0, 0] == 2 * x

    p = Matrix([[x, y], [x * y, y]])
    d = div_tensor(p, coords)
    assert d.shape == (2, 1)
    assert d[0] == 2

    t = traction(p, Matrix([1, 0]))
    assert t[0] == x
    assert t[1] == x * y
