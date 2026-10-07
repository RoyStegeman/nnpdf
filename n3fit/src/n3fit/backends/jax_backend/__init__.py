"""The raw JAX + optax implementation of the n3fit backend contract (P6).

This package must never import Keras: it is the proof that the contract is
framework-agnostic (D9), not a second spelling of the Keras backend.  The import ban
(``tests/test_backend_imports.py``) already lists ``jax``/``optax`` as frameworks that may
only appear under ``backends/<name>_backend/``, so that rule covers this package with no
test change.
"""
