"""
The hyperopt "future tests" diagnostic, exercised through the contract.

``hyper_optimization/rewards.py::_set_central_value`` used to reach into a Keras model by name::

    for key, grid in model.x_in.items(): ...          # read the fit's own x grid
    model.get_layer("PDF_0").call = central_value     # freeze the reference PDF
    model.trainable = False; model.compile()

P2 replaced all of that with ``get_backend().view(model)`` plus ``bound_inputs``/``override``/
``freeze``.  The point of this test is that the *behaviour* is unchanged and the code no longer
needs a framework to exercise it: running the real function against the numpy double checks the
plumbing end to end (the grid it reads, the function it installs, the freezing) without
TensorFlow.

It also carries the provenance of the section it overrides: the generators nest the PDF model
under the name ``PDFs`` (``PDF_0`` until the 2023 rename, which missed this call site).  That name is
checked against the generators by
``test_model_roles.test_every_role_name_is_actually_created_by_n3fit``.  What is still *not*
repaired is the dispatch: nothing calls ``fit_future_tests`` (Q15 in the contract).
"""

import numpy as np
import pytest

from n3fit.tests.backend_conformance.testing_backends import NumpyDoubleBackend, NumpyModelView


class FakePDFSet:
    """Stands in for ``N3PDF``: a callable taking a grid and returning replica PDFs."""

    def __init__(self, value):
        self.value = np.asarray(value)
        self.seen_grid = None

    def __call__(self, grid):
        self.seen_grid = grid
        return self.value


@pytest.fixture
def backend(monkeypatch):
    """The numpy double, installed as the active backend for the duration of the test.

    Note *where* the patch goes: ``rewards`` does ``from n3fit.backends import get_backend``, so it
    holds its own reference to the function, bound when the module is imported.  Patching the
    facade only works if nothing imported ``rewards`` yet -- which is exactly what happened before:
    this test passed when it ran alone and silently exercised the *keras* adapter against the
    double's model when the suite ran it (``test_hyperopt`` imports ``rewards`` first), where its
    grid lookup found nothing.  Patch the module that uses the name.
    """
    double = NumpyDoubleBackend()
    from n3fit.hyper_optimization import rewards

    monkeypatch.setattr(rewards, "get_backend", lambda *args, **kwargs: double)
    return double


def test_set_central_value_reads_the_grid_and_overrides_the_reference(backend):
    from n3fit.hyper_optimization.rewards import _set_central_value

    grid = np.linspace(1e-3, 1.0, 4)
    graph = NumpyModelView(
        {"nn": {"value": 1.0}, "reference": {"value": 2.0}},
        inputs={
            "pdf_input": grid,
            "xgrid_integration": np.linspace(1e-3, 1.0, 6),
            "scaledx_x": grid,
        },
    )
    values = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)  # (replicas, x, flavours)
    n3pdf = FakePDFSet(values)

    _set_central_value(n3pdf, graph)

    # 1. the grid that was read is the fit's own x grid, not the integration one
    assert np.allclose(n3pdf.seen_grid, grid)

    # 2. the reference section now returns the replica average of the PDF set
    out = graph()["reference"]
    assert np.allclose(out, np.mean(values, axis=0, keepdims=True))

    # 3. the model is frozen for good
    assert graph.frozen is True


def test_set_central_value_refuses_a_model_with_no_grid(backend):
    """No grid is an error naming what the model does have, not an ``UnboundLocalError``."""
    from n3fit.hyper_optimization.rewards import _set_central_value

    graph = NumpyModelView({"nn": {"value": 1.0}}, inputs={"xgrid_integration": np.linspace(0, 1, 4)})
    with pytest.raises(ValueError, match="no x grid"):
        _set_central_value(FakePDFSet(np.ones((1, 4, 1))), graph)


def test_set_central_value_skips_the_integration_grid(backend):
    """The integration grid is in ``bound_inputs`` too and must not be picked (it is the
    largest of the two, and using it would silently change the diagnostic)."""
    from n3fit.hyper_optimization.rewards import _set_central_value

    graph = NumpyModelView(
        {"reference": {"value": 2.0}},
        inputs={"pdf_input": np.zeros(2), "xgrid_integration": np.ones(9)},
    )
    n3pdf = FakePDFSet(np.zeros((1, 9, 2)))

    _set_central_value(n3pdf, graph)

    assert n3pdf.seen_grid.shape == (2,), n3pdf.seen_grid
