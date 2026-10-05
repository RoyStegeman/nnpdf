"""
The capabilities declared by the Keras backend.

This is the single place where the *vocabulary* the Keras backend supports is declared:
which parametrizations/architectures, which optimizers (with the arguments they accept and
what they require), which initializers, activations and regularizers, which objective
kinds, and the optional feature flags.

P0 keeps this as a thin, declarative wrapper around the dictionaries that already exist in
``MetaModel``/``MetaLayer``/``base_layers``, so that the two cannot drift: where possible
the values are *derived* from the legacy dicts rather than retyped.  P1 switches
``n3fit.checks`` and ``n3fit.hyper_optimization.hyper_scan`` from the legacy dicts to
this object.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

import keras

from n3fit.backends.keras_backend.MetaLayer import initializers as _initializers
from n3fit.backends.keras_backend.MetaModel import optimizers as _optimizers
from n3fit.backends.keras_backend.base_layers import (
    custom_activations as _custom_activations,
    layers as _layers,
    regularizers as _regularizers,
)

# Layer names declared in the ``layers`` dictionary that are *not* parametrizations
# (dropout and concatenation are structural, they are not things one fits).
_STRUCTURAL_LAYERS = {"dropout", "concatenate"}


def _objective_schema():
    """The declared objective kinds and their schemas (P3).

    Imported lazily from ``objectives.py`` for the same reason as the other vocabularies: this
    module must be buildable with the framework stubbed, and the objectives module imports no
    framework at all.
    """
    from n3fit.backends.keras_backend.objectives import objective_schemas

    return objective_schemas()


def _optimizer_schema():
    """Build the optimizer declarations out of the legacy ``MetaModel.optimizers`` dictionary.

    Everything the Keras backend offers today is a first-order iterative optimizer: it
    requires a gradient and it is meant to be stopped by the validation criterion.
    Second-order and direct optimizers arrive with the P7 work described in the design
    documents; when they do, they will be added here with
    ``uses_validation_stopping=False``.
    """
    schema = {}
    for name, (_class, args) in _optimizers.items():
        schema[name] = {
            "options": dict(args),
            "is_iterative": True,
            "requires": {"gradient"},
            "uses_validation_stopping": True,
        }
    return schema


def _parametrization_schema():
    """The architectures offered today, with the options ``ReplicaSettings`` accepts."""
    common = {
        "initializer": str,
        "dropout_rate": float,
        "regularizer": str,
        "regularizer_args": dict,
        "seeds": tuple,
    }
    schema = {}
    for kind in sorted(set(_layers) - _STRUCTURAL_LAYERS):
        schema[kind] = {"options": {"nodes": tuple, "activations": tuple, **common}}
    return schema


def _tensorflow_version():
    """The installed tensorflow version as a tuple of ints, or None."""
    try:
        import tensorflow
    except ImportError:  # pragma: no cover - only when keras runs on another framework
        return None
    parts = []
    for piece in tensorflow.__version__.split(".")[:2]:
        digits = "".join(char for char in piece if char.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _requires_eager_workaround():
    """Whether this installation needs the historical ``tf`` < 2.4 eager workaround.

    The workaround belongs to code that builds a model outside a training loop -- historically
    ``hyperopt_optimization.rewards.fit_future_tests`` (deleted in P4); on those old versions
    that fails unless eager execution is on.  Keeping the version sniff *here* means n3fit can ask the question
    (``capabilities.requires_eager_workaround``) without importing tensorflow itself.
    """
    if keras_backend_name() != "tensorflow":
        return False
    version = _tensorflow_version()
    return version is not None and version < (2, 4)


@dataclass(frozen=True)
class KerasCapabilities:
    """The capabilities of the Keras backend (see :class:`n3fit.backends.base.Capabilities`)."""

    parametrizations: Mapping[str, Any] = field(default_factory=_parametrization_schema)
    optimizers: Mapping[str, Any] = field(default_factory=_optimizer_schema)
    objectives: Mapping[str, Any] = field(default_factory=_objective_schema)
    derivatives: frozenset = frozenset({"gradient"})
    initializers: Mapping[str, Any] = field(
        default_factory=lambda: {name: dict(args) for name, (_cls, args) in _initializers.items()}
    )
    activations: frozenset = frozenset(
        # the Keras built-ins that the runcards use, plus our own
        {"linear", "sigmoid", "tanh", "relu", "elu", "gelu", "softplus"}
        | set(_custom_activations)
    )
    regularizers: Mapping[str, Any] = field(
        default_factory=lambda: {name: dict(args) for name, (_cls, args) in _regularizers.items()}
    )
    dtypes: frozenset = frozenset({"float32", "float64"})
    train_n_replicas_together: bool = True
    supports_weight_mutation: bool = True
    # The tensorboard callback is documented (keras_backend/callbacks.py) as tensorflow-only
    supports_tensorboard: bool = field(default_factory=lambda: keras_backend_name() == "tensorflow")
    fast_single_replica_convolution: bool = True
    requires_eager_workaround: bool = field(default_factory=_requires_eager_workaround)


def build_capabilities():
    """Return the :class:`KerasCapabilities` of this installation."""
    return KerasCapabilities()


def keras_backend_name():
    """The Keras backend in use (``tensorflow``, ``torch``, ``jax``, ...)."""
    return keras.backend.backend()
