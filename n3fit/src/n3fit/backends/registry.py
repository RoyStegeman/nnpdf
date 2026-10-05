"""
Discovery and instantiation of n3fit backends.

This module is **framework-free**: importing it (and :mod:`n3fit.backends.base`) must not
import keras, tensorflow, torch or jax.  The framework is imported only when a backend is
actually requested, which is what allows ``n3fit.checks`` (parameter validation), the
documentation and the conformance suite to run without any of them installed.

Backends are looked up, in order:

1. an explicit ``name`` argument,
2. the ``N3FIT_BACKEND`` environment variable,
3. the ``KERAS_BACKEND`` environment variable, *but only if it names a registered backend*.
   ``KERAS_BACKEND`` historically selects the framework (``tensorflow``/``torch``/``jax``)
   for Keras itself, so ``KERAS_BACKEND=jax`` must keep meaning "the keras backend, running
   on jax" rather than "a backend called jax" -- Keras reads that variable by itself,
4. the default, ``"keras"``.

Third parties can add a backend with :func:`register_backend`; the mapping from name to
importable location is deliberately tiny so that it can later be replaced by entry points
without touching call sites.
"""

import functools
import importlib
import importlib.util
import logging
import os

log = logging.getLogger(__name__)

__all__ = [
    "available_backends",
    "importable_backends",
    "selected_backend_name",
    "get_backend",
    "register_backend",
    "DEFAULT_BACKEND",
]

DEFAULT_BACKEND = "keras"

# name -> "module.path:ClassName".  Deliberately data, not imports: see the module docstring.
_BUILTIN_BACKENDS = {
    "keras": "n3fit.backends.keras_backend.backend:KerasBackend",
}

_registry = dict(_BUILTIN_BACKENDS)


def register_backend(name, location):
    """Register a backend under ``name``.

    Parameters
    ----------
        name: str
            name to be used in ``N3FIT_BACKEND``
        location: str
            importable location in the form ``"module.path:ClassName"``
    """
    if ":" not in location:
        raise ValueError(
            f"Backend location must look like 'module.path:ClassName', got {location!r}"
        )
    _registry[name] = location


def available_backends():
    """Return the names of all *registered* backends (not necessarily importable)."""
    return tuple(sorted(_registry))


@functools.lru_cache(maxsize=None)
def _location_is_importable(location):
    """Whether a backend implementation can really be imported in this environment.

    The cheap ``find_spec`` check comes first, but it is not enough: a backend module exists
    on disk whether or not its framework is installed, and it is the *import* that fails.
    So this imports the module (and therefore its framework), which is why the result is
    cached per location.
    """
    module_name = location.split(":", 1)[0]
    try:
        if importlib.util.find_spec(module_name) is None:
            return False
    except (ImportError, ValueError):
        return False
    try:
        importlib.import_module(module_name)
    except Exception:  # pylint: disable=broad-except  (any failure means "not usable")
        return False
    return True


def importable_backends():
    """Registered backends that can actually be imported in this environment.

    Note that this imports the backend implementations (and hence their frameworks); the
    result is cached, so it is cheap to call repeatedly.
    """
    return tuple(name for name, loc in sorted(_registry.items()) if _location_is_importable(loc))


def _resolve_name(name):
    """Apply the selection order of the module docstring."""
    if name is not None:
        return name
    if value := os.environ.get("N3FIT_BACKEND"):
        return value.lower()
    if (value := os.environ.get("KERAS_BACKEND")) and (candidate := value.lower()) in _registry:
        # See point 3 of the module docstring: this variable belongs to Keras, and we only
        # interpret it when it happens to name one of our backends.
        return candidate
    return DEFAULT_BACKEND


def selected_backend_name(name=None):
    """Return the name of the backend that :func:`get_backend` would instantiate.

    Useful without paying for the import (for logging, reporting, and tests of the
    selection rules themselves).
    """
    return _resolve_name(name)


def get_backend(name=None):
    """Return an instance of the selected :class:`n3fit.backends.base.Backend`.

    Parameters
    ----------
        name: str, optional
            backend name; if not given, the environment is inspected (see module docstring)

    Returns
    -------
        backend: Backend
            a fresh backend object (not cached: the capabilities and the state object are
            cheap, and a fresh object avoids stale state between tests and hyperopt trials)
    """
    requested = _resolve_name(name)
    try:
        location = _registry[requested]
    except KeyError as err:
        importable = ", ".join(importable_backends()) or "none"
        raise ValueError(
            f"Backend {requested!r} is not registered. Registered: "
            f"{', '.join(available_backends())} (importable in this environment: {importable}). "
            f"Select one with the N3FIT_BACKEND environment variable or register it with "
            f"n3fit.backends.register_backend()."
        ) from err

    module_name, class_name = location.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as err:
        raise ImportError(
            f"Backend {requested!r} is registered as {location!r} but its implementation "
            f"could not be imported. This usually means the required framework is not "
            f"installed in this environment."
        ) from err

    backend = getattr(module, class_name)()
    log.info("Using n3fit backend: %s", backend.name)
    return backend
