"""
Tests of the backend registry: discovery, selection, registration, and the promise that
importing the backend package does not require a framework.
"""

import subprocess
import sys

import pytest

from n3fit.backends import available_backends, get_backend, register_backend
from n3fit.backends.registry import importable_backends, selected_backend_name

from .conftest import TEST_DOUBLE


def test_registry_lists_backends():
    names = available_backends()
    assert "keras" in names
    assert TEST_DOUBLE in names


def test_get_backend_by_name_returns_the_object():
    backend = get_backend(TEST_DOUBLE)
    assert backend.name == TEST_DOUBLE
    for attribute in ("ops", "capabilities", "state"):
        assert hasattr(backend, attribute)
    assert isinstance(backend.version_info(), dict)


def test_unknown_backend_error_is_informative():
    with pytest.raises(ValueError) as err:
        get_backend("definitely_not_a_backend")
    message = str(err.value)
    # the message must tell the user what *is* available and how to choose
    assert "definitely_not_a_backend" in message
    for name in importable_backends():
        assert name in message
    assert "N3FIT_BACKEND" in message


def test_register_backend_validates_the_location():
    with pytest.raises(ValueError):
        register_backend("broken", "not_a_location")


def test_register_then_use(monkeypatch):
    """A third party can add a backend through the public entry point."""
    register_backend(
        "alias_of_double", "n3fit.tests.backend_conformance.testing_backends:NumpyDoubleBackend"
    )
    assert get_backend("alias_of_double").name == "numpy_test_double"


def test_backend_selection_from_environment(monkeypatch):
    monkeypatch.setenv("N3FIT_BACKEND", TEST_DOUBLE)
    assert get_backend().name == TEST_DOUBLE


def test_keras_backend_env_var_alone_selects_the_keras_backend(monkeypatch):
    """``KERAS_BACKEND`` selects the *framework* for the Keras backend, not an n3fit backend.

    So ``KERAS_BACKEND=jax`` must keep selecting the ``keras`` backend (which will itself run
    on jax); it must not be interpreted as "a backend called jax", which does not exist.
    Asserted on the *selection* rather than on an instance, so that this rule is checked in
    environments where the framework is not installed.
    """
    monkeypatch.delenv("N3FIT_BACKEND", raising=False)
    monkeypatch.setenv("KERAS_BACKEND", "jax")
    assert selected_backend_name() == "keras"


def test_n3fit_backend_env_var_always_wins(monkeypatch):
    monkeypatch.setenv("KERAS_BACKEND", "jax")
    monkeypatch.setenv("N3FIT_BACKEND", TEST_DOUBLE)
    assert selected_backend_name() == TEST_DOUBLE
    assert get_backend().name == TEST_DOUBLE


def test_n3fit_backend_env_var_with_unknown_name_fails(monkeypatch):
    monkeypatch.setenv("N3FIT_BACKEND", "jax")
    with pytest.raises(ValueError):
        get_backend()


def test_importing_the_backend_package_needs_no_framework():
    """The package (and the contract) must import with the frameworks blocked.

    This is checked in a subprocess with an import blocker rather than by inspecting
    ``sys.modules``, so that it is meaningful also in an environment where the frameworks
    *are* installed (which is the case that matters: a regression here would otherwise go
    unnoticed in CI).
    """
    script = f"""
import sys

BANNED = {{"keras", "tensorflow", "torch", "jax", "optax", "flax"}}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BANNED:
            raise ImportError("blocked by the test: " + name)
        return None


sys.meta_path.insert(0, Blocker())
import n3fit.backends
import n3fit.backends.base
import n3fit.backends.registry
import n3fit.backends.registry as registry

# register a backend *without* any framework available, and use it: this also exercises
# register_backend() in a fresh process
registry.register_backend(
    "{TEST_DOUBLE}", "n3fit.tests.backend_conformance.testing_backends:NumpyDoubleBackend"
)
backend = n3fit.backends.get_backend("{TEST_DOUBLE}")
print(backend.name)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ":".join(sys.path)},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert TEST_DOUBLE in result.stdout
