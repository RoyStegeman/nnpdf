"""The capabilities declared by the raw JAX + optax backend (P6).

This is the single place where the *vocabulary* the JAX backend supports is declared:
which parametrizations/architectures, which optimizers (with the arguments they accept and
what they require), which initializers, activations and regularizers, which objective
kinds, and the optional feature flags.

Two deliberate asymmetries with the Keras backend, both documented rather than hidden:

* the optimizer registry holds only ``Adam`` and ``SGD`` (the P6 decision, Q2): every name
  here is a promise the parity test keeps, and the other six Keras names all have optax
  equivalents, so widening this is mechanical later;
* ``parametrizations``/``initializers``/``regularizers`` declare the same *names* the Keras
  backend declares, while ``Backend.parametrization`` stays ``_staged`` (as it is for Keras
  too): validation of a runcard's vocabulary is backend-independent, and an unsupported
  *build* fails at build time with the phase that implements it, not at validation time
  with a vocabulary error.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = ["JaxCapabilities", "build_capabilities"]


def _objective_schema():
    """The declared objective kinds and their schemas (same shape as the Keras ones)."""
    from n3fit.backends.jax_backend.objectives import objective_schemas

    return objective_schemas()


def _optimizer_schema():
    """The optimizers this backend implements, with the options they accept.

    Defaults mirror the Keras registry's (``MetaModel.optimizers`` plus the injected
    ``clipnorm = 1.0``), so a runcard means the same thing under either backend.  ``beta1``/
    ``beta2``/``epsilon`` are accepted for Adam because optax and Keras disagree on the
    default epsilon (1e-8 vs 1e-7) and the parity test pins the Keras value.
    """
    return {
        "Adam": {
            "options": {
                "learning_rate": 0.01,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-7,
                "clipnorm": 1.0,
            },
            "is_iterative": True,
            "requires": {"gradient"},
            "uses_validation_stopping": True,
        },
        "SGD": {
            "options": {
                "learning_rate": 0.01,
                "momentum": 0.0,
                "nesterov": False,
                "clipnorm": 1.0,
            },
            "is_iterative": True,
            "requires": {"gradient"},
            "uses_validation_stopping": True,
        },
    }


def _parametrization_schema():
    """The architectures offered, with the options a spec may carry.

    The option *names* (and their meaning) are the Keras backend's; the JAX backend accepts
    the same vocabulary at validation time, while ``Backend.parametrization`` stays staged
    for both backends until P7.
    """
    common = {
        "initializer": str,
        "dropout_rate": float,
        "regularizer": str,
        "regularizer_args": dict,
        "seeds": tuple,
    }
    options = {"nodes": tuple, "activations": tuple, **common}
    return {
        "dense": {"options": dict(options)},
        "dense_per_flavour": {"options": dict(options)},
        "LSTM": {"options": dict(options)},
    }


@dataclass(frozen=True)
class JaxCapabilities:
    """The capabilities of the JAX backend (see :class:`n3fit.backends.base.Capabilities`)."""

    parametrizations: Mapping[str, Any] = field(default_factory=_parametrization_schema)
    optimizers: Mapping[str, Any] = field(default_factory=_optimizer_schema)
    objectives: Mapping[str, Any] = field(default_factory=_objective_schema)
    derivatives: frozenset = frozenset({"gradient"})
    # Vocabulary parity with the Keras backend (see the module docstring): sampling arrives
    # with ``Backend.parametrization`` in P7; nothing validates option values today.
    initializers: Mapping[str, Any] = field(
        default_factory=lambda: {"random_uniform": {}, "glorot_uniform": {}, "glorot_normal": {}}
    )
    activations: frozenset = frozenset(
        {"linear", "sigmoid", "tanh", "relu", "elu", "gelu", "softplus"}
    )
    regularizers: Mapping[str, Any] = field(
        default_factory=lambda: {"l1_l2": {"l1": 0.0, "l2": 0.0}}
    )
    dtypes: frozenset = frozenset({"float32", "float64"})
    # D2: replicas are trained in an explicit per-replica loop (a ``jit``ted step each), not
    # in one stacked graph -- which is also what lets replicas have different architectures.
    train_n_replicas_together: bool = False
    supports_weight_mutation: bool = True
    supports_tensorboard: bool = False
    fast_single_replica_convolution: bool = True
    requires_eager_workaround: bool = False


def build_capabilities():
    """Return the :class:`JaxCapabilities` of this installation."""
    return JaxCapabilities()
