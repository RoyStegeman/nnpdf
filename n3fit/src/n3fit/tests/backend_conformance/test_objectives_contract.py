"""
The objective contract, exercised against *every* backend in the conformance suite.

``test_objectives_offline.py`` pins the semantics with fake layers; ``test_objectives_keras.py``
pins the numbers against the legacy layers.  This file is the third leg: whatever a backend
*declares* in ``Capabilities.objectives`` must be buildable from a spec that satisfies that
declaration -- and nothing else -- and must behave like a term of the contract.

That is the check that keeps the declaration from becoming fiction: a schema that does not describe
its own implementation is how a "metric registry" ends up as a list of names that only one backend
can honour.
"""

import numpy as np
import pytest

from n3fit.backends.base import ObjectiveSpec

SIZE = 3


def spec_from_schema(kind, schema, name="term", **overrides):
    """The minimal spec the schema promises is enough: every declared array, scalars initialised.

    Note that the initial value of a declared scalar is passed through ``options`` -- that is
    documented in the contract and is how the Lagrange multiplier gets its starting value
    (``model_trainer._LM_initial_and_multiplier``).
    """
    data = {}
    for key in schema["data"]:
        data[key] = np.eye(SIZE) if "covmat" in key else np.zeros(SIZE)
    options = {scalar: 1.0 for scalar in schema["scalars"]}
    data.update(overrides.pop("data", {}))
    options.update(overrides.pop("options", {}))
    return ObjectiveSpec(kind=kind, name=name, data=data, options=options, **overrides)


def test_every_declared_kind_is_buildable_from_its_own_schema(backend):
    declared = backend.capabilities.objectives
    assert declared, "a backend must declare at least one objective kind"
    for kind, schema in declared.items():
        assert set(schema) >= {"data", "scalars", "options"}, kind
        term = backend.objective(spec_from_schema(kind, schema, name=f"term_{kind}"))
        assert term.kind == kind
        assert term.name == f"term_{kind}"


def test_a_declared_kind_can_be_evaluated(backend):
    """A term is a function of a prediction: this is the whole of its evaluation contract."""
    for kind, schema in backend.capabilities.objectives.items():
        term = backend.objective(spec_from_schema(kind, schema, name=f"term_{kind}"))
        prediction = np.zeros((1, 1, SIZE))
        value = np.asarray(term(prediction))
        assert np.all(np.isfinite(value)), (kind, value)
        # a zero prediction against the zero target of the schema's minimal spec is a zero term
        assert np.allclose(value, 0.0), (kind, value)
        # and a non-trivial prediction is not zero: the term is wired to its input
        assert not np.allclose(np.asarray(term(-np.ones((1, 1, SIZE)))), 0.0), kind


def test_declared_scalars_can_be_set_and_undeclared_ones_are_rejected(backend):
    for kind, schema in backend.capabilities.objectives.items():
        term = backend.objective(spec_from_schema(kind, schema, name=f"term_{kind}"))
        for scalar in schema["scalars"]:
            term.set_scalar(scalar, 2.0)
        with pytest.raises(ValueError):
            term.set_scalar("not_a_scalar", 2.0)


def test_undeclared_data_is_rejected(backend):
    """``set_data`` must never write into whatever weight happens to be there."""
    for kind, schema in backend.capabilities.objectives.items():
        term = backend.objective(spec_from_schema(kind, schema, name=f"term_{kind}"))
        with pytest.raises(ValueError):
            term.set_data(not_a_field=np.eye(SIZE))


def test_a_spec_missing_declared_data_is_rejected(backend):
    for kind, schema in backend.capabilities.objectives.items():
        for missing in schema["data"]:
            incomplete = spec_from_schema(kind, schema, name=f"term_{kind}")
            del incomplete.data[missing]
            with pytest.raises(ValueError):
                backend.objective(incomplete)


def test_an_unknown_kind_is_rejected(backend):
    with pytest.raises(ValueError):
        backend.objective(ObjectiveSpec(kind="not_a_kind", name="x"))


def test_masking_follows_the_schema(backend):
    """Terms over data mask their points; the others have nothing to mask.

    The k-fold reset writes a mask between models, so "which kinds take a mask" has to be
    answerable from the declaration -- and the answer is derived from it (a term with ``data`` has
    a mask), not tabulated a second time.
    """
    for kind, schema in backend.capabilities.objectives.items():
        term = backend.objective(spec_from_schema(kind, schema, name=f"term_{kind}"))
        if schema["data"]:
            assert term.mask() is not None, kind
            term.set_data(mask=np.ones(term.mask().shape))
        else:
            assert term.mask() is None, kind
            with pytest.raises(ValueError):
                term.set_data(mask=np.ones(SIZE))
