"""
P2b: the two accessors that were still Keras vocabulary — ``predict()`` and ``split_replicas()``.

What this pins down:

* ``Model.__call__(inputs)`` is the contract's spelling of ``MetaModel.predict``: numpy in,
  numpy out, and **the same output shape** (nothing is squeezed: the call sites index the
  leading axis themselves).  ``None`` evaluates the graph as built, which is what the
  diagnostics do when every input is bound.
* ``Ensemble.__iter__``/``__getitem__``/``__len__`` replace ``split_replicas()``: iteration
  order is the replica order and indexing is 0-based, so ``ens[i]`` is n3fit's replica ``i+1``.
* the ensemble is an *iterable of models*, not a list of graphs: wrapping it in ``N3PDF`` (which
  checks ``isinstance(x, Iterable)``) must work, and so must viewing one of its members.

The behaviour is checked against the numpy reference implementation, so it runs everywhere; the
Keras adapter's own two entry points (``view`` idempotence, ``ensemble`` splitting a stacked
graph) are covered in ``test_keras_role_view.py``.
"""

import numpy as np
import pytest

from n3fit.tests.backend_conformance.testing_backends import (
    NumpyDoubleBackend,
    NumpyEnsemble,
    NumpyModelView,
)


@pytest.fixture
def backend():
    return NumpyDoubleBackend()


def _replica_graph(value):
    return {  # a "graph" of the double: one section whose evaluation is a constant
        "nn": {"value": np.full((1, 2), value)},
    }


def test_call_evaluates_the_graph(backend):
    """``view(...)(...)`` is the replacement for ``model.predict(...)``.

    The double's graphs are dictionaries of sections (see ``NumpyModelView.__call__``), so the
    assertion indexes the section; a real backend returns the graph's output array.
    """
    view = backend.view(NumpyModelView(_replica_graph(3.0)))
    assert np.allclose(view()["nn"], [[3.0, 3.0]])


def test_call_keeps_the_output_shape(backend):
    """No squeezing: the leading axis survives, because callers index it."""
    graph = {
        "nn": {"value": np.zeros((1, 3, 2))},
        "reference": {"override": lambda inputs: np.ones((1, 4, 2))},
    }
    view = backend.view(NumpyModelView(graph))
    out = view()
    assert out["nn"].shape == (1, 3, 2)
    assert out["reference"].shape == (1, 4, 2)


def test_ensemble_iterates_in_replica_order(backend):
    models = [NumpyModelView(_replica_graph(float(i))) for i in range(3)]
    ensemble = backend.ensemble(models)
    assert len(ensemble) == 3
    values = [float(model()["nn"][0, 0]) for model in ensemble]
    assert values == [0.0, 1.0, 2.0]
    assert float(ensemble[2]()["nn"][0, 0]) == 2.0


def test_ensemble_members_are_models(backend):
    """Each replica must be a contract model, not a raw graph: that is what lets n3fit stop
    calling ``.predict()``/``get_layer()`` on them."""
    ensemble = backend.ensemble([_replica_graph(1.0), _replica_graph(2.0)])
    for model in ensemble:
        assert isinstance(model, NumpyModelView)
        assert isinstance(model(), dict)  # callable == evaluated


def test_ensemble_is_an_iterable_of_models(backend):
    """``N3PDF`` accepts any iterable and stores it, then indexes and takes ``len()``."""
    ensemble = backend.ensemble([_replica_graph(1.0), _replica_graph(2.0)])
    assert isinstance(ensemble, NumpyEnsemble) or hasattr(ensemble, "__iter__")
    from collections.abc import Iterable

    assert isinstance(ensemble, Iterable)
    assert [m is ensemble[i] for i, m in enumerate(ensemble)] == [True, True]


def test_viewing_a_member_is_idempotent(backend):
    """n3fit may hold a member of an ensemble and hand it to ``view`` (the P2 idiom); that must
    return the model itself rather than wrapping it again."""
    model = backend.ensemble([_replica_graph(1.0)])[0]
    assert backend.view(model) is model


def test_slicing_an_ensemble_stays_an_ensemble(backend):
    ensemble = backend.ensemble([_replica_graph(float(i)) for i in range(4)])
    subset = ensemble[1:3]
    assert len(subset) == 2
    assert float(subset[0]()["nn"][0, 0]) == 1.0


def test_ensemble_weights_are_one_map_per_replica(backend):
    graph = {"nn": {"weights": {"dense/kernel": np.eye(2)}, "value": 1.0}}
    ensemble = backend.ensemble([NumpyModelView(graph), NumpyModelView(graph)])
    maps = ensemble.weights(role="nn")
    assert len(maps) == 2
    assert all("nn/dense/kernel" in m for m in maps)
