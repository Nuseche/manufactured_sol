# external_mms

Primera iteración real de un compilador/verificador simbólico para soluciones manufacturadas bajo preservación estricta del problema aprobado por `scientist/judge`.

## Alcance v1

- Entrada: `approved_scientist.json`.
- Adaptación determinista a IR canónica basada en operadores.
- Construcción de contexto simbólico con SymPy.
- Generación determinista de ansatz polinómicos de bajo orden.
- Ensamble de restricciones de preservación estricta:
  - residual fuerte exacto,
  - Dirichlet exacto,
  - Neumann exacto,
  - IC exacta (si aplica).
- Intento de resolución simbólica (`solve` / `nonlinsolve`) sin aceptar aproximaciones numéricas como resultado final.
- Verificación simbólica final y reporte estructurado.

## Honestidad técnica de v1

- El motor de búsqueda de ansatz es deliberadamente acotado (bajo orden polinómico).
- La verificación de forma débil en v1 es **estructural** (consistencia declarativa), no integración simbólica general en dominios arbitrarios.
- Si el espacio de búsqueda v1 no encuentra solución exacta no trivial bajo preservación estricta, se reporta:
  `infeasible_under_strict_preservation`.

## Estructura

```
external_mms/
  pyproject.toml
  README.md
  mms_agent/
    __init__.py
    schemas.py
    approved_scientist_adapter.py
    ir_builder.py
    symbolic_context.py
    operator_registry.py
    tensor_ops.py
    geometry.py
    ansatz_generator.py
    strict_preservation.py
    symbolic_solver.py
    residual_checker.py
    weak_form_checker.py
    report_builder.py
    cli.py
  tests/
    test_adapter.py
    test_geometry.py
    test_tensor_ops.py
    test_cook_strict_smoke.py
```

## Instalación

```bash
cd external_mms
python -m pip install -e .[dev]
```

## Ejecutar CLI

```bash
cd external_mms
python -m mms_agent.cli \
  --input examples/approved_scientist.json \
  --mode strict_preservation \
  --output examples/report.json
```

## Ejecutar tests

```bash
cd external_mms
pytest -q
```
