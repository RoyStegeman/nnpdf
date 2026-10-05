"""
The ``n3fit`` backend package.

This package has two faces during the migration to the backend contract
(see ``proposed/n3fit-backend-contract.md``):

**The new interface** -- importable *without* any deep-learning framework installed:

    from n3fit.backends import get_backend, available_backends, register_backend
    from n3fit.backends.base import ParametrizationSpec, OptimizerSpec, Capabilities, ...

**The legacy interface** -- the Keras-flavoured names that the rest of n3fit has been
using until now (``MetaModel``, ``MetaLayer``, ``operations``, ``Input``, ``Lambda``,
``callbacks``, ...), kept working unchanged through a module-level ``__getattr__`` so
that the framework is imported *lazily*, on first use.  New code should use the new
interface; the legacy names will be re-pointed at ``get_backend()`` between P1 and P4.
"""

import importlib

from n3fit.backends.base import (  # noqa: F401  (re-exported for convenience)
    GROUP_EXPERIMENTAL,
    GROUP_INTEGRABILITY,
    GROUP_POSITIVITY,
    GROUP_TRAINING,
    GROUP_VALIDATION,
    OPTIMIZATION_GROUPS,
    REPORT_GROUPS,
    ROLE_NN,
    ROLE_OBJECTIVE,
    ROLE_PHOTON,
    ROLE_PREPROCESSING,
    ROLE_REFERENCE,
    ROLE_SUMRULE,
    ROLES,
    Backend,
    Capabilities,
    FitResult,
    Objective,
    ObjectiveGroup,
    ObjectiveSpec,
    ParametrizationSpec,
    ShapeSpec,
    OptimizerSpec,
)
from n3fit.backends.registry import (  # noqa: F401
    available_backends,
    get_backend,
    importable_backends,
    register_backend,
    selected_backend_name,
)

__all__ = [
    "available_backends",
    "importable_backends",
    "selected_backend_name",
    "get_backend",
    "register_backend",
    "Backend",
    "Capabilities",
    "FitResult",
    "Objective",
    "ObjectiveSpec",
    "ParametrizationSpec",
    "ShapeSpec",
    "OptimizerSpec",
    "ROLE_NN",
    "ROLE_OBJECTIVE",
    "ROLE_PHOTON",
    "ROLE_PREPROCESSING",
    "ROLE_REFERENCE",
    "ROLE_SUMRULE",
    "ROLES",
    "GROUP_TRAINING",
    "GROUP_VALIDATION",
    "GROUP_EXPERIMENTAL",
    "GROUP_POSITIVITY",
    "GROUP_INTEGRABILITY",
    "OPTIMIZATION_GROUPS",
    "REPORT_GROUPS",
    "ObjectiveGroup",
]

# Legacy names, resolved lazily from the keras backend on first use.
# ``name -> (module, attribute)``; ``None`` as attribute means "the module itself".
_LEGACY = {
    # these three are *submodules* of the keras backend package (whose __init__ is empty),
    # exactly as the historical `from ... import callbacks, constraints, operations` resolved
    # them -- hence attribute=None
    "callbacks": ("n3fit.backends.keras_backend.callbacks", None),
    "constraints": ("n3fit.backends.keras_backend.constraints", None),
    "operations": ("n3fit.backends.keras_backend.operations", None),
    "MetaLayer": ("n3fit.backends.keras_backend.MetaLayer", "MetaLayer"),
    "MetaModel": ("n3fit.backends.keras_backend.MetaModel", "MetaModel"),
    "NN_PREFIX": ("n3fit.backends.keras_backend.MetaModel", "NN_PREFIX"),
    "NN_LAYER_ALL_REPLICAS": ("n3fit.backends.keras_backend.MetaModel", "NN_LAYER_ALL_REPLICAS"),
    "PREPROCESSING_LAYER_ALL_REPLICAS": (
        "n3fit.backends.keras_backend.MetaModel",
        "PREPROCESSING_LAYER_ALL_REPLICAS",
    ),
    "Concatenate": ("n3fit.backends.keras_backend.base_layers", "Concatenate"),
    "Input": ("n3fit.backends.keras_backend.base_layers", "Input"),
    "Lambda": ("n3fit.backends.keras_backend.base_layers", "Lambda"),
    "base_layer_selector": ("n3fit.backends.keras_backend.base_layers", "base_layer_selector"),
    "regularizer_selector": ("n3fit.backends.keras_backend.base_layers", "regularizer_selector"),
    "clear_backend_state": ("n3fit.backends.keras_backend.internal_state", "clear_backend_state"),
    "get_physical_gpus": ("n3fit.backends.keras_backend.internal_state", "get_physical_gpus"),
    "set_eager": ("n3fit.backends.keras_backend.internal_state", "set_eager"),
    "set_initial_state": ("n3fit.backends.keras_backend.internal_state", "set_initial_state"),
    "MultiInitializer": ("n3fit.backends.keras_backend.multi_initializer", "MultiInitializer"),
}

__all__ += sorted(_LEGACY)

_legacy_announced = False


def __getattr__(name):
    """Resolve the legacy names on first use (PEP 562).

    Keeping these lazy is what makes ``import n3fit.backends`` framework-free, and
    therefore what allows ``n3fit.checks``, the documentation and the conformance suite
    to run in an environment where the framework is not even installed.
    """
    global _legacy_announced
    try:
        module_name, attribute = _LEGACY[name]
    except KeyError as err:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from err

    module = importlib.import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value  # cache: subsequent lookups are normal attribute access

    if not _legacy_announced:
        # Kept for backwards compatibility: this used to be printed when the package was
        # imported.  It now happens on the first use of a legacy (framework) name.
        _legacy_announced = True
        print("Using Keras backend")

    return value


def __dir__():
    return sorted(set(globals()) | set(_LEGACY))
