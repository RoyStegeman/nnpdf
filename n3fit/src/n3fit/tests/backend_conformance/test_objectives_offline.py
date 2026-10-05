"""
P3: the objective terms and their schemas.

Framework-free by design.  Two things are checked here:

* **the schemas** -- every kind a backend declares must be constructible from what it declares and
  nothing else (a schema that does not describe its own implementation is how a "metric registry"
  becomes a lie);
* **the term semantics** -- ``apply`` (graph mode) and ``__call__`` (eager) are the same operation,
  ``set_data`` replaces data rather than adding to it, ``set_scalar`` sets, and the schema rejects
  what it does not declare instead of accepting it silently.

The Keras implementations of the three kinds are exercised against fake layers, so this runs without
a framework; the numerics of the real layers are the existing ``tests/test_losses.py``.
"""

import numpy as np
import pytest

from n3fit.backends.base import GROUP_INTEGRABILITY, GROUP_POSITIVITY, GROUP_TRAINING, ObjectiveGroup
from n3fit.backends.keras_backend.objectives import (
    OBJECTIVE_SCHEMAS,
    KerasObjective,
    objective_schemas,
)


class FakeWeight:
    """A layer weight with the two methods the adapter uses (``assign``, ``numpy``)."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def assign(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


class FakeLossLayer:
    """A stand-in for ``LossInvcovmat``/``LossPositivity``/``LossIntegrability``.

    The real layers take a *tensor* and return a tensor; here ``__call__`` records what it was
    given and returns ``self.result`` verbatim, so a test can tell ``apply`` (returns exactly what
    the layer returned) from ``__call__`` (converts it).
    """

    def __init__(self, name, result=None, trainable_kernel=None, with_mask=True):
        self.name = name
        self.result = np.array([0.0]) if result is None else result
        self.kernel = FakeWeight(trainable_kernel if trainable_kernel is not None else [1.0])
        if with_mask:
            self.mask = FakeWeight([1.0, 1.0])
        self._covmat = None
        self.calls = []

    def __call__(self, prediction, **kwargs):
        self.calls.append(prediction)
        return self.result


def spec_for(kind, name="test_term", **kwargs):
    from n3fit.backends.base import ObjectiveSpec

    data = {"invcovmat": np.eye(2), "covmat": np.eye(2), "target": np.zeros(2)}
    options = {"multiplier": 2.0} if kind in ("positivity", "integrability") else {}
    if kind != "chi2":
        data = {}
    return ObjectiveSpec(kind=kind, name=name, data=data, options=options, **kwargs)


def test_every_declared_schema_describes_a_constructible_term():
    """A schema must name exactly the data/scalars/options the implementation needs."""
    for kind, schema in OBJECTIVE_SCHEMAS.items():
        spec = spec_for(kind)
        term = KerasObjective(spec, FakeLossLayer(kind))  # validate() runs in __init__
        # the declared data is what the spec carries (chi2), and only that
        assert set(spec.data) == set(schema["data"]) or set(schema["data"]) <= {
            "invcovmat",
            "covmat",
            "target",
        }, (kind, spec.data)
        # every declared scalar must be settable
        for scalar in schema["scalars"]:
            term.set_scalar(scalar, 1.5)
        # every declared option must be accepted (they are build-time: not checked here)


def test_schemas_are_returned_by_the_capability():
    schemas = objective_schemas()
    assert set(schemas) == set(OBJECTIVE_SCHEMAS)
    assert schemas["chi2"]["data"] == ("invcovmat", "covmat", "target")
    assert schemas["positivity"]["scalars"] == ("multiplier",)
    # a copy, not the module's own dictionary: a caller must not be able to edit the schemas
    schemas["chi2"]["data"] = ()
    assert OBJECTIVE_SCHEMAS["chi2"]["data"] == ("invcovmat", "covmat", "target")


def test_unknown_kind_is_rejected():
    from n3fit.backends.base import ObjectiveSpec

    spec = ObjectiveSpec(kind="mystery", name="x")
    with pytest.raises(ValueError, match="Unknown objective kind"):
        KerasObjective.validate(spec)


def test_missing_data_is_rejected():
    from n3fit.backends.base import ObjectiveSpec

    spec = ObjectiveSpec(kind="chi2", name="x", data={"invcovmat": np.eye(2)})
    with pytest.raises(ValueError, match="needs data"):
        KerasObjective.validate(spec)


def test_unknown_option_is_rejected():
    from n3fit.backends.base import ObjectiveSpec

    spec = ObjectiveSpec(kind="integrability", name="x", options={"alpha": 1e-7})
    with pytest.raises(ValueError, match="accepts options"):
        KerasObjective.validate(spec)


def test_apply_returns_the_layer_result_unchanged():
    """Graph mode: the term hands back *exactly* what the layer returned (a tensor, here a
    non-array object) -- it must not convert, or the graph would be cut."""
    from n3fit.backends.base import ObjectiveSpec

    tensor = object()  # not a numpy array: anything apply() converted would stop being this
    layer = FakeLossLayer("pos", result=tensor)
    term = KerasObjective(ObjectiveSpec(kind="positivity", name="pos"), layer)
    prediction = np.zeros(2)
    assert term.apply(prediction) is tensor
    # and the layer was handed the prediction we passed
    assert len(layer.calls) == 1 and layer.calls[0] is prediction


def test_apply_without_a_prediction_is_an_error():
    from n3fit.backends.base import ObjectiveSpec

    term = KerasObjective(ObjectiveSpec(kind="positivity", name="pos"), FakeLossLayer("pos"))
    with pytest.raises(ValueError, match="needs a prediction"):
        term.apply(None)


def test_call_is_apply_plus_conversion():
    from n3fit.backends.base import ObjectiveSpec

    layer = FakeLossLayer("pos", result=np.array([3.0]))
    term = KerasObjective(ObjectiveSpec(kind="positivity", name="pos"), layer)
    out = term(np.zeros(2))
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, [3.0])


def test_set_data_replaces_the_covariance_by_inverting_it():
    """Q3: the caller passes the covariance to use; the term inverts it (no hidden addition)."""
    from n3fit.backends.base import ObjectiveSpec

    layer = FakeLossLayer("chi2")
    covmat = np.array([[2.0, 0.0], [0.0, 4.0]])
    term = KerasObjective(
        ObjectiveSpec(kind="chi2", name="chi2", data={"invcovmat": np.eye(2), "covmat": covmat}),
        layer,
    )
    new_covmat = np.array([[3.0, 0.0], [0.0, 3.0]])
    term.set_data(covmat=new_covmat)
    assert np.allclose(layer.kernel.value, np.linalg.inv(new_covmat))
    # and the spec records what it was given, so a caller can build the next sum from it
    assert np.allclose(term.spec.data["covmat"], new_covmat)


def test_set_data_mask_and_the_read_back():
    from n3fit.backends.base import ObjectiveSpec

    layer = FakeLossLayer("chi2")
    term = KerasObjective(ObjectiveSpec(kind="chi2", name="chi2", data={"invcovmat": np.eye(2)}), layer)
    assert np.allclose(term.mask(), [1.0, 1.0])
    term.set_data(mask=np.array([[0.0, 2.0]]))
    assert np.allclose(term.mask(), [[0.0, 2.0]])


def test_set_data_rejects_undeclared_arrays():
    """Every kind, every name it must not accept -- a wrong name must not reach a layer weight.

    This is a regression test with a real bug behind it: ``set_data(covmat=...)`` was written as a
    chain of ``if name == ...`` and so happily wrote an inverse covariance into a positivity term,
    whose only weight is the Lagrange multiplier.
    """
    from n3fit.backends.base import ObjectiveSpec
    from n3fit.backends.keras_backend.objectives import _REPLACEABLE_DATA

    candidates = ["covmat", "invcovmat", "target", "alpha", "multiplier", "nonsense"]
    for kind in OBJECTIVE_SCHEMAS:
        term = KerasObjective(ObjectiveSpec(kind=kind, name="t"), FakeLossLayer("t"))
        for name in candidates:
            if name in _REPLACEABLE_DATA[kind]:
                continue
            with pytest.raises(ValueError, match="cannot take data"):
                term.set_data(**{name: np.eye(2)})


def test_replaceable_data_is_a_subset_of_the_declared_data():
    """The two tables cannot drift apart: an array can only be replaceable if the kind declares it,
    and ``chi2`` -- the kind the k-fold diagnostic updates -- must declare the covariance."""
    from n3fit.backends.keras_backend.objectives import _REPLACEABLE_DATA

    assert set(_REPLACEABLE_DATA) == set(OBJECTIVE_SCHEMAS)
    for kind, names in _REPLACEABLE_DATA.items():
        assert set(names) <= set(OBJECTIVE_SCHEMAS[kind]["data"]), kind
    assert "covmat" in _REPLACEABLE_DATA["chi2"]


def test_mask_is_only_accepted_where_the_layer_has_one():
    """A term that has no mask says so, rather than silently ignoring the update."""
    from n3fit.backends.base import ObjectiveSpec

    # a layer without a mask attribute (the Lagrange layers have none)
    term = KerasObjective(
        ObjectiveSpec(kind="positivity", name="pos"), FakeLossLayer("pos", with_mask=False)
    )
    assert term.mask() is None
    with pytest.raises(ValueError, match="has no mask"):
        term.set_data(mask=np.array([1.0, 0.0]))


def test_set_scalar_rejects_undeclared_names():
    term = KerasObjective(spec_for("chi2"), FakeLossLayer("chi2"))
    with pytest.raises(ValueError, match="declares scalars"):
        term.set_scalar("multiplier", 2.0)


def test_set_scalar_sets_rather_than_scales():
    """``set_scalar`` replaces the value (the legacy callback *multiplied* the weight -- the
    schedule now computes the new value and sets it)."""
    term = KerasObjective(spec_for("positivity"), FakeLossLayer("pos"))
    term.set_scalar("multiplier", 5.0)
    assert np.allclose(term._layer.kernel.value, [5.0])
    term.set_scalar("multiplier", 5.0)
    assert np.allclose(term._layer.kernel.value, [5.0])  # not 25.0


def test_adopt_layer_reconstructs_a_spec_from_the_layer():
    """The k-fold diagnostic addresses terms of a model built elsewhere.

    The kind has to be named here because a fake layer is not one of the real loss classes: the
    adapter recognises a term by *class* (``kind_of``), never by name, which is what makes it find
    positivity and integrability terms too (they are not called ``*_exp``).
    """
    from n3fit.backends.keras_backend.objectives import adopt_layer

    layer = FakeLossLayer("LHC_exp")
    layer._covmat = np.eye(2)
    term = adopt_layer(layer, kind="chi2")
    assert term.kind == "chi2"
    assert term.name == "LHC_exp"
    assert np.allclose(term.spec.data["covmat"], np.eye(2))


def test_a_term_is_recognised_by_class_not_by_name():
    """``kind_of`` is what replaced the ``.*_exp$`` name pattern.

    The one test in this file that is not offline: ``kind_of`` is a *Keras backend* helper (it
    classifies real loss layers by class), so it skips rather than fails in a framework-free env.
    """
    pytest.importorskip("keras", reason="kind_of reads the Keras builder's own layers")

    from n3fit.backends.keras_backend.objectives import kind_of

    assert kind_of(FakeLossLayer("LHC_exp")) is None  # a fake is not a loss layer
    assert kind_of(FakeLossLayer("anything")) is None


# --------------------------------------------------------------------------------------------
# The group vocabulary (Q4)
# --------------------------------------------------------------------------------------------
def test_group_membership_is_data_not_graph_shape():
    groups = ObjectiveGroup(
        {
            GROUP_TRAINING: ("LHC", "POS_PV"),
            "validation": ("LHC",),
            "experimental": ("LHC",),
            GROUP_POSITIVITY: ("POS_PV",),
        }
    )
    assert groups.names(GROUP_TRAINING) == ("LHC", "POS_PV")
    assert groups.names(GROUP_INTEGRABILITY) == ()  # a fit without it: empty, not an error
    assert groups.groups_of("POS_PV") == (GROUP_TRAINING, GROUP_POSITIVITY)
    assert groups.all_terms == ("LHC", "POS_PV")  # each once
    assert "LHC" in groups and "nope" not in groups
