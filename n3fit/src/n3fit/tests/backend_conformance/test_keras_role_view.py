"""
The Keras adapter: role -> layer-name mapping, exercised without the framework.

``keras_backend/roles.py`` only uses the duck-typed part of a Keras model (``get_layer``,
``layers``, ``layer.weights``, ``compile``, ``summary``), so it can be tested here with a fake
graph.  That is deliberate: this module is on the path of every fit, and the bug P1 shipped
(a ``NameError`` at import time in ``capabilities.py``) is the kind of thing that only a
framework-free test can catch in an environment without keras.

The two behaviours worth pinning:

* ``override`` on a role whose layer does not exist raises instead of silently doing nothing.
  That is what ``hyper_optimization.rewards._set_central_value`` hit for two and a half years: the
  generators renamed the PDF layer (``PDF_0`` → ``PDFs``, commit b919773ef) and the override was
  the one call site nobody updated.  The failure must come from here, naming the layer it looked
  for (``n3fit-backend-contract.md`` §1.13a).
* ``bind_input("photon", ...)`` is a no-op in a graph without a photon -- the "fit without
  photons" case -- and compiles the graph when there is one.
"""

import numpy as np
import pytest

from n3fit.backends.base import (
    ROLE_NN,
    ROLE_OBJECTIVE,
    ROLE_PHOTON,
    ROLE_PREPROCESSING,
    ROLE_REFERENCE,
    ROLE_SUMRULE,
)
from n3fit.backends.keras_backend.roles import (
    INPUT_HELD_IN_GRAPH,
    ROLE_LAYER_NAMES,
    KerasEnsembleView,
    KerasModelView,
    as_view,
)


class FakeWeight:
    """One weight of a fake layer; ``name`` is what metaflow/metakeras produce (``up:0``)."""

    def __init__(self, name, value):
        self.name = name
        self._value = value

    def numpy(self):
        return self._value


class FakeLayer:
    def __init__(self, name, weights=(), built=True, has_photon_generator=True):
        self.name = name
        self.weights = list(weights)
        self.built = built
        self.compiled = False
        self.summaries = 0
        self.photon = None
        self.has_photon_generator = has_photon_generator

    def summary(self):
        self.summaries += 1

    def register_photon(self, grid):
        """``AddPhoton.register_photon``: stores the grid and marks the layer for a rebuild.

        The real layer only does this when it actually has a photon generator (``if
        self._photons_generator:``) -- a "photon layer" without one leaves ``built`` alone.
        """
        if self.has_photon_generator:
            self.photon = grid
            self.built = False


class FakeGraph:
    def __init__(self, layers, x_in=None):
        self.layers = list(layers)
        self._by_name = {layer.name: layer for layer in layers}
        self.x_in = x_in or {}
        self.trainable = True
        self.compile_calls = 0
        self.summaries = 0

    def get_layer(self, name):
        try:
            return self._by_name[name]
        except KeyError:
            raise ValueError(f"No such layer: {name}") from None

    def compile(self):
        self.compile_calls += 1

    def predict(self, inputs=None, **kwargs):
        """``MetaModel.predict``: numpy in, numpy out.  Records what it was called with."""
        self.predict_calls = getattr(self, "predict_calls", [])
        self.predict_calls.append((inputs, kwargs))
        return self.output

    def summary(self):
        self.summaries += 1


def preprocessing_layer():
    return FakeLayer(
        ROLE_LAYER_NAMES[ROLE_PREPROCESSING],
        weights=[
            FakeWeight("alpha_up:0", np.array([0.5])),
            FakeWeight("beta_up:0", np.array([3.0])),
            FakeWeight("alpha_down:0", np.array([-0.5])),
            # a weight that is not a preprocessing factor: must not appear in the paths
            FakeWeight("some_bias:0", np.array([0.0])),
        ],
    )


def test_preprocessing_weights_are_keyed_by_path():
    view = KerasModelView(FakeGraph([preprocessing_layer()]))
    weights = view.weights(role=ROLE_PREPROCESSING)
    assert set(weights) == {
        "preprocessing/alpha/up",
        "preprocessing/beta/up",
        "preprocessing/alpha/down",
    }, weights
    assert np.allclose(weights["preprocessing/alpha/up"], [0.5])
    assert np.allclose(weights["preprocessing/beta/up"], [3.0])


def test_weights_of_a_graph_without_that_section_are_empty():
    """An *implemented* role on a graph that has no such layer: empty, not an error."""
    view = KerasModelView(FakeGraph([]))  # no preprocessing layer at all
    assert view.weights(role=ROLE_PREPROCESSING) == {}


@pytest.mark.parametrize("role", [ROLE_NN, ROLE_OBJECTIVE, ROLE_SUMRULE])
def test_weights_of_an_unimplemented_role_raise(role):
    """``nn``/``objective``/``sumrule`` need the per-replica layout (P4); they must raise rather
    than return something plausible-looking."""
    view = KerasModelView(FakeGraph([]))
    with pytest.raises(NotImplementedError):
        view.weights(role=role)


def test_unknown_role_raises():
    view = KerasModelView(FakeGraph([]))
    with pytest.raises(ValueError):
        view.weights(role="nonsense")


def test_bound_inputs_are_converted_to_numpy():
    class Tensor:
        def __init__(self, value):
            self._value = value

        def numpy(self):
            return self._value

    graph = FakeGraph([], x_in={"pdf_input": Tensor(np.zeros(3)), "xgrid_integration": Tensor(np.ones(2))})
    view = KerasModelView(graph)
    bound = view.bound_inputs()
    assert set(bound) == {"pdf_input", "xgrid_integration"}
    assert isinstance(bound["pdf_input"], np.ndarray)
    # the model inputs are exactly the ones the generator builds (see model_gen.py)
    assert set(bound) <= set(INPUT_HELD_IN_GRAPH)


def test_bind_input_without_a_photon_is_a_no_op():
    graph = FakeGraph([])  # no add_photon layer: the fit has no photon
    view = KerasModelView(graph)
    view.bind_input("photon", np.zeros(3))
    assert graph.compile_calls == 0


def test_bind_input_registers_the_photon_and_rebuilds():
    """With a photon generator, registering a grid marks the layer unbuilt, so the graph is
    rebuilt -- every time, because the photon array's shape follows the grid."""
    layer = FakeLayer(ROLE_LAYER_NAMES[ROLE_PHOTON])
    graph = FakeGraph([layer])
    grid = np.linspace(0, 1, 4)
    KerasModelView(graph).bind_input("photon", grid)
    assert np.allclose(layer.photon, grid)
    assert layer.built is False
    assert graph.compile_calls == 1

    KerasModelView(graph).bind_input("photon", grid)
    assert graph.compile_calls == 2


def test_bind_input_without_a_generator_does_not_rebuild():
    """A photon layer with no generator (the ``photons: false`` case) leaves the graph built."""
    layer = FakeLayer(ROLE_LAYER_NAMES[ROLE_PHOTON], has_photon_generator=False)
    graph = FakeGraph([layer])
    KerasModelView(graph).bind_input("photon", np.zeros(3))
    assert layer.photon is None
    assert graph.compile_calls == 0


def test_override_without_the_reference_layer_says_what_is_missing():
    """A graph with no ``reference`` section: the override must raise from here, naming the layer
    it looked for.  (The real generators do create it -- ``test_model_roles`` checks every name in
    the table against them -- so this is the *other* half of the 2023 failure.)"""
    view = KerasModelView(FakeGraph([]))
    with pytest.raises(ValueError) as excinfo:
        view.override(ROLE_REFERENCE, lambda inputs: np.zeros(1))
    assert ROLE_LAYER_NAMES[ROLE_REFERENCE] in str(excinfo.value)


def test_override_sets_the_function_on_the_layer():
    layer = FakeLayer(ROLE_LAYER_NAMES[ROLE_REFERENCE])
    layer.input = {"pdf_input": np.zeros(3)}
    view = KerasModelView(FakeGraph([layer]))
    view.override(ROLE_REFERENCE, lambda inputs: np.array([1.0, 2.0]))
    # the adapter assigns a callable; without an ops object the values pass through unconverted
    assert np.allclose(layer.call(layer.input), [1.0, 2.0])


def test_freeze_makes_the_graph_non_trainable():
    # the Keras backend uses the ops object for conversions; here the plain path is enough
    graph = FakeGraph([])
    KerasModelView(graph).freeze()
    assert graph.trainable is False
    assert graph.compile_calls == 1


def test_summary_covers_every_present_role():
    nn = FakeLayer(ROLE_LAYER_NAMES[ROLE_NN])
    msr = FakeLayer(ROLE_LAYER_NAMES[ROLE_SUMRULE])
    graph = FakeGraph([nn, msr])
    KerasModelView(graph).summary()
    assert graph.summaries == 1
    assert nn.summaries == 1
    assert msr.summaries == 1


# --------------------------------------------------------------------------------------------
# P2b: evaluation and the ensemble
# --------------------------------------------------------------------------------------------
def test_call_delegates_to_predict_with_the_same_arguments():
    graph = FakeGraph([])
    graph.output = np.arange(6.0).reshape(1, 2, 3)
    view = KerasModelView(graph)
    grid = np.linspace(0, 1, 3)

    out = view({"pdf_input": grid})
    assert np.allclose(out, graph.output)
    assert graph.predict_calls == [({"pdf_input": grid}, {})]

    # no inputs: the graph as it was built (all inputs bound) -- the diagnostics' case
    view()
    assert graph.predict_calls[-1] == (None, {})


def test_ensemble_splits_a_stacked_graph():
    """``KerasEnsembleView.from_graph`` is the replacement for ``graph.split_replicas()``.

    The split itself must stay the backend's business -- the fake graph counts how often it was
    asked -- and the result must present each replica as a contract model.  (``Backend.ensemble``
    is a two-line delegate to this, so the logic is testable without a framework.)
    """
    replicas = [FakeGraph([]), FakeGraph([])]
    replicas[0].output = np.zeros((1, 2))
    replicas[1].output = np.ones((1, 2))

    class FakeStack:
        def __init__(self, replicas):
            self._replicas = replicas
            self.split_calls = 0

        def split_replicas(self):
            self.split_calls += 1
            return self._replicas

    stack = FakeStack(replicas)
    ensemble = KerasEnsembleView.from_graph(stack)

    assert stack.split_calls == 1
    assert len(ensemble) == 2
    assert [type(model).__name__ for model in ensemble] == ["KerasModelView", "KerasModelView"]
    assert np.allclose(ensemble[1](), np.ones((1, 2)))


def test_ensemble_from_models_and_from_an_ensemble():
    """A sequence of graphs, or an ensemble that already is one: both accepted, the second
    returned as it is rather than re-wrapped."""
    ensemble = KerasEnsembleView.from_graph([FakeGraph([]), FakeGraph([])])
    assert len(ensemble) == 2
    assert KerasEnsembleView.from_graph(ensemble) is ensemble


def test_as_view_is_idempotent():
    graph = FakeGraph([])
    first = as_view(graph)
    assert as_view(first) is first
