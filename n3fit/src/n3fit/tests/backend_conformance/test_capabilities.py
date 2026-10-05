"""
Tests of the capability declarations.

Two kinds of check live here:

* **generic** (all backends): the declared capabilities have the structure the contract
  requires, so that ``n3fit.checks``/``hyper_scan`` can validate user input against them;
* **anti-drift** (Keras backend): the declared capabilities agree with the legacy
  dictionaries that are still the source of truth for the fitting code, so that the two
  cannot silently diverge while both exist (P1/P2 will remove the legacy copies).
"""

import pytest

from .conftest import TEST_DOUBLE


def test_capabilities_have_the_contracted_structure(backend):
    caps = backend.capabilities

    # the vocabulary registries
    assert len(caps.parametrizations) > 0
    assert len(caps.optimizers) > 0
    for kind, schema in caps.parametrizations.items():
        assert "options" in schema, f"parametrization {kind!r} declares no options schema"
    for name, schema in caps.optimizers.items():
        assert "options" in schema, f"optimizer {name!r} declares no options schema"
        assert isinstance(
            schema["is_iterative"], bool
        ), f"optimizer {name!r} must declare is_iterative"
        assert "requires" in schema, f"optimizer {name!r} must declare what it requires"
        # D13: whether the cross-validation stopper applies is an *optimizer* property
        assert isinstance(
            schema["uses_validation_stopping"], bool
        ), f"optimizer {name!r} must declare uses_validation_stopping"

    # feature flags
    assert isinstance(caps.train_n_replicas_together, bool)
    assert isinstance(caps.supports_weight_mutation, bool)
    assert isinstance(caps.supports_tensorboard, bool)
    assert isinstance(caps.requires_eager_workaround, bool)
    assert isinstance(caps.dtypes, frozenset)
    assert "float32" in caps.dtypes

    # derivative support is what gates Hessian error propagation
    assert caps.derivatives <= {"gradient", "jacobian", "hessian"}


def test_state_exposes_the_eager_toggle(backend):
    """``BackendState.set_eager`` is what the hyperopt rewards workaround needs."""
    assert callable(backend.state.set_eager)
    backend.state.set_eager(True)
    backend.state.set_eager(False)


def test_solver_options_are_defaults_not_required_arguments(backend):
    """Whatever a backend declares as optimizer options must be usable as-is."""
    for name, schema in backend.capabilities.optimizers.items():
        assert isinstance(schema["options"], dict), f"optimizer {name!r} options must be a mapping"


def test_test_double_declares_a_non_default_solver():
    """The double exists partly to keep the suite honest about non-SGD optimizers."""
    from n3fit.backends import get_backend

    backend = get_backend(TEST_DOUBLE)
    assert backend.capabilities.optimizers["levenberg_marquardt"]["uses_validation_stopping"] is False


def test_keras_capabilities_match_the_legacy_dictionaries():
    """The Keras capabilities must list exactly what the legacy dicts provide.

    Until P1/P2 move the fitting code onto ``capabilities``, the legacy dictionaries
    (``MetaModel.optimizers``, ``MetaLayer.initializers``, ``base_layers.layers``) are
    still what actually runs, so the two must agree.
    """
    pytest.importorskip("keras")
    from n3fit.backends.keras_backend.MetaLayer import initializers as legacy_initializers
    from n3fit.backends.keras_backend.MetaModel import optimizers as legacy_optimizers
    from n3fit.backends.keras_backend.base_layers import layers as legacy_layers
    from n3fit.backends.keras_backend.capabilities import build_capabilities

    caps = build_capabilities()

    assert set(caps.optimizers) == set(legacy_optimizers)
    assert set(caps.initializers) == set(legacy_initializers)
    # architectures are the layer vocabulary minus the structural layers
    assert set(caps.parametrizations) == set(legacy_layers) - {"dropout", "concatenate"}

    # and the defaults must be the legacy ones, not retyped
    for name, (_cls, args) in legacy_optimizers.items():
        assert caps.optimizers[name]["options"] == args
    for name, (_cls, args) in legacy_initializers.items():
        assert caps.initializers[name] == args

# --------------------------------------------------------------------------------------------
# The ops surface vs the contract
# --------------------------------------------------------------------------------------------
# ``test_ops.py`` pins the *numerics* of the operations.  What follows pins the *vocabulary*:
# the contract lists the primitives n3fit layers may use (``base.Ops``), and a backend that
# has them under different names is not yet conformant -- the layers cannot be moved onto
# ``ops.*`` until the names line up.
#
# The Keras backend still exposes the historical namespace (``op_log``, ``tensor_product``,
# ``numpy_to_tensor``, ...).  P1 added ``keras_backend/ops.py``, which maps one onto the
# other; the pairs that were *not* renames are handled there and documented in its docstring:
#
#   contract name      legacy Keras name            how P1 settled it
#   log                op_log                       alias
#   tensordot          tensor_product               alias
#   constant           numpy_to_tensor              alias
#   to_numpy           variable_to_numpy            alias
#   as_input           numpy_to_input               alias
#   cast, ones, zeros, matmul, ...                 re-export from ``keras.ops``
#   multiply/subtract  op_multiply/op_subtract      **not** synonyms (the legacy ones are
#                                                   list-taking *layers*): implemented with
#                                                   ``keras.ops``, legacy names keep the
#                                                   layer behaviour
#   gather             op_gather_keep_dims          **not** synonyms: the legacy one keeps the
#                                                   indexed axis (gather + expand_dims)
#   flatten                                                              reshape
#   scatter_update     scatter_to_one              **not** synonyms (one-hot write vs scatter):
#                                                   ``KerasOps.scatter_update`` implements the
#                                                   contract's semantics, the legacy name stays
#                                                   legacy-only
#   splitter           tensor_splitter             alias (served by the legacy namespace, which
#                                                   exports ``splitter`` as the alias itself)
#
# The table below is now **empty**: every name the contract requires is served by the Keras
# backend.  That is not how it was recorded for most of P1/P2 -- ``scatter_update`` and
# ``splitter`` were listed here as open until they were closed -- and the entries survived in
# the table because this test *skips* where keras is not importable, which is every environment
# the refactor was developed in until keras 3 was installed alongside a working backend.  The
# check is only as honest as its last real run, which is the point of the strictness in both
# directions below.
#
# The test is strict in both directions: a name the backend does not serve fails (a regression),
# and a recorded gap that has been closed fails too (remove it from the table, so that nobody has
# to guess how much of P1 is done).
KNOWN_PROTOCOL_GAPS = {}


def contract_ops_names():
    from n3fit.backends.base import Ops

    return {name for name in dir(Ops) if not name.startswith("_")}


def test_ops_surface_matches_the_contract(backend):
    missing = sorted(name for name in contract_ops_names() if not hasattr(backend.ops, name))
    recorded = KNOWN_PROTOCOL_GAPS.get(backend.name, {})

    new_gaps = sorted(set(missing) - set(recorded))
    assert not new_gaps, (
        f"backend {backend.name!r} does not provide {new_gaps}, which the contract requires\n"
        f"and which are not recorded in KNOWN_PROTOCOL_GAPS. Either the backend must provide "
        f"them or the contract must drop them."
    )

    closed = sorted(set(recorded) - set(missing))
    assert not closed, (
        f"backend {backend.name!r} now provides {closed}: remove them from "
        f"KNOWN_PROTOCOL_GAPS (and update the comment above it) so that the table keeps "
        f"describing the real state of the migration."
    )

    # and the *other* direction: names the backend has that the contract does not mention
    # are not an error, but they must not be the only place a contract name lives
    for contract_name, legacy_name in recorded.items():
        if legacy_name:
            assert hasattr(backend.ops, legacy_name), (
                f"the recorded legacy equivalent {legacy_name!r} of {contract_name!r} is gone "
                f"from backend {backend.name!r}: update KNOWN_PROTOCOL_GAPS"
            )


def test_ops_names_used_by_the_suite_exist_everywhere(backend):
    """Guard against silent coverage loss in ``test_ops.py``.

    Those checks call operations by name and skip what a backend does not have (they were
    inherited that way); if a name stops existing the coverage would vanish quietly.  This
    makes the missing name explicit instead.
    """
    import re
    from pathlib import Path

    suite = Path(__file__).with_name("test_ops.py").read_text()
    used = set(re.findall(r"\bops\.([a-zA-Z_][a-zA-Z0-9_]*)", suite))
    assert used, "the scan found no calls into ops.* in test_ops.py: has it been renamed?"
    missing = sorted(name for name in used if not hasattr(backend.ops, name))
    assert not missing, (
        f"backend {backend.name!r} does not implement {missing}, which test_ops.py calls: "
        f"those checks would be silently skipped for this backend"
    )
