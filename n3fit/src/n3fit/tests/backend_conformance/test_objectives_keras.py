"""
P3 against the real Keras layers: the same numbers as before the refactor.

The offline tests (``test_objectives_offline.py``) pin the *contract* semantics with fake layers;
this file pins the thing a refactor of a live code path actually has to promise -- that a term
built from an :class:`ObjectiveSpec` computes exactly what the legacy layer computed, and that the
data/scalar updates do exactly what the legacy methods did.  It is the P3 acceptance test: if these
pass, no fit output changes.

Every comparison is against the legacy spelling *as it was before P3*:

==========================================  ====================================================
P3 (contract)                               legacy (the code this replaced)
==========================================  ====================================================
``objective(spec)`` where ``spec`` carries   ``LossInvcovmat(invcovmat, y_true, mask, covmat=…)``
``data={"invcovmat", "covmat", "target"}``
``spec.options["multiplier"] = c0``          ``LossPositivity(c=c0)``
``term.set_data(covmat=C1 + C2)``            ``layer.add_covmat(C2)`` (with ``covmat=C1``)
``term.set_data(mask=m)``                    ``layer.update_mask(m)``
``term.set_scalar("multiplier", c0 * m)``    ``LagrangeCallback``'s ``w.assign(w * m)``
``spec.options["alpha"]``                    ``LossPositivity(alpha=…)``
==========================================  ====================================================

Runs only where keras is installed; the rest of the conformance suite covers the contract itself.
"""

import numpy as np
import pytest

pytest.importorskip("keras")

from n3fit.backends import get_backend  # noqa: E402
from n3fit.backends import operations as legacy_ops  # noqa: E402
from n3fit.backends.base import ObjectiveSpec  # noqa: E402
from n3fit.layers import losses  # noqa: E402

NDATA = 4
NREPLICAS = 2


@pytest.fixture(scope="module")
def backend():
    return get_backend("keras")


@pytest.fixture
def inputs():
    """A positive-definite covariance, its inverse, target data and a prediction."""
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(NDATA, NDATA))
    covmat = raw @ raw.T / NDATA + np.eye(NDATA) * 0.1
    target = rng.normal(size=(1, 1, NDATA))
    prediction = rng.normal(size=(1, NREPLICAS, NDATA))
    return {
        "covmat": covmat,
        "invcovmat": np.linalg.inv(covmat),
        "target": target,
        "prediction": legacy_ops.numpy_to_tensor(prediction),
    }


def chi2_spec(inputs, name="LHC_exp", **kwargs):
    data = {k: inputs[k] for k in ("invcovmat", "covmat", "target")}
    return ObjectiveSpec(kind="chi2", name=name, data=data, **kwargs)


def test_chi2_term_from_a_spec_is_the_legacy_layer(backend, inputs):
    """The value a spec-built term gives is the value the legacy layer gave."""
    term = backend.objective(chi2_spec(inputs))
    legacy = losses.LossInvcovmat(
        legacy_ops.numpy_to_tensor(inputs["invcovmat"]), legacy_ops.numpy_to_tensor(inputs["target"])
    )
    assert np.allclose(np.asarray(term(inputs["prediction"])), np.asarray(legacy(inputs["prediction"])))


def test_chi2_term_with_a_mask_is_the_legacy_layer(backend, inputs):
    mask = np.array([1.0, 0.0, 1.0, 1.0])  # 1-D: what the legacy constructor takes
    term = backend.objective(chi2_spec(inputs, mask=mask))
    legacy = losses.LossInvcovmat(
        legacy_ops.numpy_to_tensor(inputs["invcovmat"]), legacy_ops.numpy_to_tensor(inputs["target"]), mask=mask
    )
    assert np.allclose(np.asarray(term(inputs["prediction"])), np.asarray(legacy(inputs["prediction"])))
    # the read-back is the legacy layer's mask weight (shape (1, 1, ndata)), which is what the
    # k-fold diagnostic copies into the last model
    assert np.allclose(term.mask(), np.asarray(legacy.mask))


def test_set_data_mask_is_update_mask(backend, inputs):
    term = backend.objective(chi2_spec(inputs, mask=np.array([1.0, 0.0, 1.0, 1.0])))
    legacy = losses.LossInvcovmat(
        legacy_ops.numpy_to_tensor(inputs["invcovmat"]),
        legacy_ops.numpy_to_tensor(inputs["target"]),
        mask=np.array([1.0, 0.0, 1.0, 1.0]),
    )
    new_mask = np.array([[[0.0, 1.0, 1.0, 0.0]]])  # the weight shape, as rewards.py passes it
    term.set_data(mask=new_mask)
    legacy(inputs["prediction"])  # the legacy layer needs a forward pass before update_mask
    legacy.update_mask(new_mask)
    assert np.allclose(np.asarray(term(inputs["prediction"])), np.asarray(legacy(inputs["prediction"])))


def test_set_data_covmat_sum_is_add_covmat(backend, inputs):
    """The k-fold diagnostic: ``inv(C + P)``.

    The call site passes the *sum* (the adapter inverts what it is given), and that must reproduce
    ``LossInvcovmat.add_covmat``, which inverted ``self._covmat + covmat`` for a layer built with
    ``covmat=self._covmat``.  Both the updated kernel and the resulting loss are compared.
    """
    pdf_covmat = np.eye(NDATA) * 0.5
    term = backend.objective(chi2_spec(inputs))
    term(inputs["prediction"])  # build the layer (Keras builds weights on first call)
    term.set_data(covmat=term.spec.data["covmat"] + pdf_covmat)

    legacy = losses.LossInvcovmat(
        legacy_ops.numpy_to_tensor(inputs["invcovmat"]),
        legacy_ops.numpy_to_tensor(inputs["target"]),
        covmat=inputs["covmat"],
    )
    legacy(inputs["prediction"])  # build
    legacy.add_covmat(pdf_covmat)

    assert np.allclose(np.asarray(term._layer.kernel), np.asarray(legacy.kernel))
    assert np.allclose(np.asarray(term(inputs["prediction"])), np.asarray(legacy(inputs["prediction"])))


@pytest.mark.parametrize(
    "kind,legacy_class",
    [("positivity", losses.LossPositivity), ("integrability", losses.LossIntegrability)],
)
def test_lagrange_terms_are_the_legacy_layers(backend, inputs, kind, legacy_class):
    """The initial multiplier is a build-time option in the spec and a scalar afterwards."""
    initial = 3.0
    term = backend.objective(ObjectiveSpec(kind=kind, name=f"POS_{kind}", options={"multiplier": initial}))
    legacy = legacy_class(c=initial, name=f"legacy_{kind}")
    assert np.allclose(np.asarray(term(inputs["prediction"])), np.asarray(legacy(inputs["prediction"])))

    # ``set_scalar`` *sets*: the schedule computes the new value, so the equivalent legacy
    # operation is the callback's in-place ``w.assign(w * factor)``
    factor = 1.5
    term.set_scalar("multiplier", initial * factor)
    legacy.kernel.assign(np.asarray(legacy.kernel) * factor)
    assert np.allclose(np.asarray(term(inputs["prediction"])), np.asarray(legacy(inputs["prediction"])))
    assert term.spec.options["multiplier"] == initial * factor


def test_positivity_alpha_option_reaches_the_layer(backend, inputs):
    """``alpha`` is the one option that is not a scalar: it changes the shape of the penalty."""
    alpha = 1e-3
    term = backend.objective(
        ObjectiveSpec(kind="positivity", name="POS", options={"multiplier": 1.0, "alpha": alpha})
    )
    legacy = losses.LossPositivity(c=1.0, alpha=alpha)
    assert np.allclose(np.asarray(term(inputs["prediction"])), np.asarray(legacy(inputs["prediction"])))
    # and the default alpha differs, i.e. the option really is what is being passed
    default = backend.objective(ObjectiveSpec(kind="positivity", name="POS_2", options={"multiplier": 1.0}))
    assert not np.allclose(np.asarray(default(inputs["prediction"])), np.asarray(term(inputs["prediction"])))


# --------------------------------------------------------------------------------------------
# Term discovery, and what a term consumes
# --------------------------------------------------------------------------------------------
# The graphs are duck-typed rather than built with ``MetaModel``, for the reason
# ``test_keras_role_view.py`` gives: the view only uses the duck-typed part of a graph
# (``layers``, ``input_tensors``) and this keeps the test on the adapter rather than on the
# legacy model wrapper.  The layers are real, with real weights.
class FakeGraph:
    """The part of a Keras graph the view uses."""

    def __init__(self, layers, inputs=None):
        self.layers = list(layers)
        self.input_tensors = inputs if inputs is not None else {"observables": "the input"}

    def get_layer(self, name):
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise ValueError(f"no such layer: {name}")


def _term_layers(backend, inputs):
    chi2 = backend.objective(chi2_spec(inputs, name="LHC_exp"))._layer
    positivity = backend.objective(
        ObjectiveSpec(kind="positivity", name="POS_XUQ", options={"multiplier": 1.0})
    )._layer
    return chi2, positivity


def test_the_view_finds_every_term_not_just_the_experimental_ones(backend, inputs):
    """Discovery is by layer class, not by name.

    The legacy lookup in n3fit was ``get_layer_re(".*_exp$")``, which found the experimental chi2
    terms and *silently missed* positivity and integrability (they are not named that way).  That
    was invisible while the only caller was the unreachable ``fit_future_tests``; the k-fold
    multiplier reset needs the penalty terms, so it has to be right.
    """
    chi2, positivity = _term_layers(backend, inputs)
    view = backend.view(FakeGraph([chi2, positivity]))
    terms = view.objectives()
    assert sorted(terms) == ["LHC_exp", "POS_XUQ"]
    assert terms["LHC_exp"].kind == "chi2"
    assert terms["POS_XUQ"].kind == "positivity"
    # addressable by name, kind reconstructed from the layer it wraps
    assert view.objective("POS_XUQ").kind == "positivity"
    with pytest.raises(ValueError, match="no objective named"):
        view.objective("not_a_term")


def test_an_adopted_lagrange_term_can_be_reset(backend, inputs):
    """The k-fold reset, in contract terms: ``set_scalar`` on a term found in a graph.

    This is what replaces ``MetaModel.reset_layer_weights_to`` (a name-based poke at the weights of
    a graph the caller does not own).  The layer here has *never been called*, which is the state a
    freshly built k-fold graph is in.
    """
    _, positivity = _term_layers(backend, inputs)
    term = backend.view(FakeGraph([positivity])).objective("POS_XUQ")
    # a negative prediction, so the penalty is non-trivial (elu of a positive number)
    negative = backend.ops.numpy_to_tensor(-np.ones((1, NREPLICAS, NDATA)))
    before = np.asarray(positivity(negative))
    assert np.all(before > 0)

    term.set_scalar("multiplier", 2.5)
    assert np.allclose(np.asarray(positivity.kernel), [2.5])
    assert np.allclose(np.asarray(positivity(negative)), 2.5 * before)


def test_an_adopted_plain_layer_is_not_a_term(backend, inputs):
    """A layer that is not a loss is not adopted, even if it is named like one."""
    decoy = backend.ops.as_layer(lambda x: x, name="LHC_exp")
    assert backend.view(FakeGraph([decoy])).objectives() == {}


def test_prediction_before_uses_what_the_adapter_fed_the_term(backend, inputs):
    """A term knows the prediction it was applied to, because the adapter applied it.

    n3fit's graphs are eager, so this edge exists nowhere in the framework: the legacy
    ``MetaModel(model.input, layer.input)`` asked Keras 2 for it and cannot work in Keras 3.
    """
    from n3fit.backends.keras_backend.roles import as_view

    prediction = backend.ops.numpy_to_tensor(np.zeros((1, NREPLICAS, NDATA)))
    term = backend.objective(chi2_spec(inputs))
    term.apply(prediction)  # as the generators do while assembling the graph

    built = []

    def builder(input_tensors, output_tensors):
        built.append((input_tensors, output_tensors))
        return FakeGraph([term._layer], input_tensors)

    graph = FakeGraph([term._layer])
    view = as_view(graph, ops=backend.ops, model_builder=builder)
    diagnostic = view.prediction_before("LHC_exp")
    assert built == [(graph.input_tensors, prediction)]
    assert isinstance(diagnostic, type(view))

    # a term the adapter never applied: named clearly, not silently mis-built
    _, positivity = _term_layers(backend, inputs)
    view = as_view(FakeGraph([positivity]), ops=backend.ops, model_builder=builder)
    with pytest.raises(ValueError, match="was not applied through this adapter"):
        view.prediction_before("POS_XUQ")


# --------------------------------------------------------------------------------------------
# What model_gen does now
# --------------------------------------------------------------------------------------------
class RecordingBackend:
    """The real backend, recording the specs n3fit asks it to build."""

    def __init__(self, backend):
        self._backend = backend
        self.specs = []

    def objective(self, spec):
        self.specs.append(spec)
        return self._backend.objective(spec)

    def __getattr__(self, name):
        return getattr(self._backend, name)


@pytest.fixture
def recorded_backend(backend, monkeypatch):
    """``model_gen`` asks the facade for a backend *inside* the call, so patching the facade
    attribute is enough (unlike ``rewards``, which binds the name at import time -- see
    ``test_central_value``)."""
    import n3fit.backends

    recorder = RecordingBackend(backend)
    monkeypatch.setattr(n3fit.backends, "get_backend", lambda *args, **kwargs: recorder)
    return recorder


def test_a_chi2_wrapper_asks_for_a_chi2_spec(recorded_backend, inputs):
    """The wrapper says *which kind of term* it is, not which layer class implements it."""
    from n3fit.model_gen import ObservableWrapper

    mask = np.array([1.0, 0.0, 1.0, 1.0])
    wrapper = ObservableWrapper(
        "LHC",
        [],
        None,
        [NDATA],
        invcovmat=inputs["invcovmat"],
        covmat=inputs["covmat"],
        data=inputs["target"],
    )
    # P4: the wrapper hands back the *term*, not a loss layer or a callable (``_generate_loss``
    # until the flip) -- the engine applies it.
    term = wrapper._build_objective(mask)

    assert len(recorded_backend.specs) == 1
    spec = recorded_backend.specs[0]
    assert spec.kind == "chi2"
    assert spec.name == "LHC"
    assert np.allclose(spec.data["invcovmat"], inputs["invcovmat"])
    assert np.allclose(spec.mask, mask)

    # and the term's graph-mode spelling computes what the legacy layer computed for the same
    # (zero) prediction
    zero_prediction = recorded_backend.ops.numpy_to_tensor(np.zeros((1, NREPLICAS, NDATA)))
    out = term.apply(zero_prediction)
    legacy = losses.LossInvcovmat(
        legacy_ops.numpy_to_tensor(inputs["invcovmat"]),
        legacy_ops.numpy_to_tensor(inputs["target"]),
        mask=mask,
    )
    assert np.allclose(np.asarray(out), np.asarray(legacy(zero_prediction)))


def test_a_penalty_wrapper_asks_for_a_penalty_spec_with_its_initial_multiplier(recorded_backend):
    """The Lagrange multiplier is the term's *initial value*: a build-time option that is also a
    scalar (``set_scalar`` moves it during the fit)."""
    from n3fit.model_gen import ObservableWrapper

    wrapper = ObservableWrapper(
        "POS_XUQ", [], None, [NDATA], multiplier=184.8, positivity=True
    )
    wrapper._build_objective(None)
    spec = recorded_backend.specs[0]
    assert spec.kind == "positivity"
    assert spec.options["multiplier"] == 184.8

    wrapper = ObservableWrapper("INT", [], None, [NDATA], multiplier=10.0, integrability=True)
    wrapper._build_objective(None)
    assert recorded_backend.specs[1].kind == "integrability"
    assert recorded_backend.specs[1].options["multiplier"] == 10.0
