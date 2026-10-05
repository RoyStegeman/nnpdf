"""
Parametrization of the conformance suite over the available backends.

Every test in this directory takes the ``backend`` fixture and must pass for *any* backend.
The list is:

* every *importable* registered backend (so, in a normal environment, ``keras``),
* plus the numpy test double, which is registered here to keep the suite honest: it makes
  it impossible for a test to depend on Keras-specific behaviour.

Adding a real second backend therefore requires no change to this file.
"""

import pytest

from n3fit.backends import get_backend, register_backend
from n3fit.backends.registry import importable_backends

TEST_DOUBLE = "numpy_test_double"

register_backend(TEST_DOUBLE, "n3fit.tests.backend_conformance.testing_backends:NumpyDoubleBackend")


def conformance_backends():
    """Names of the backends the conformance suite runs against."""
    names = list(importable_backends())
    if TEST_DOUBLE not in names:
        names.append(TEST_DOUBLE)
    return sorted(names)


@pytest.fixture(params=conformance_backends())
def backend(request):
    """An instance of each backend the suite runs against.

    A backend whose framework is not installed is skipped rather than failed: the suite is
    meant to run in every environment, and to *grow* coverage as backends become available.
    """
    try:
        return get_backend(request.param)
    except ImportError as err:  # pragma: no cover - depends on the environment
        pytest.skip(f"backend {request.param!r} is not importable here: {err}")
