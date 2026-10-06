"""
The Keras backend's *declarations* must be buildable without a framework installed.

This is a small but load-bearing check, added after P1 shipped a bug that no other test here
could see: ``capabilities.py`` referenced a helper defined further down the file, so importing
it raised ``NameError`` -- i.e. **every Keras fit would have died at import**.  The
Keras-dependent test in ``test_capabilities.py`` skipped in any environment without keras, and
the framework-free tests never touch the Keras modules, so nothing failed.

The check runs in a subprocess with ``keras`` and the three vocabulary modules stubbed (the
stubs are minimal on purpose: the *values* are checked against the real dictionaries by
``test_capabilities.test_keras_capabilities_match_the_legacy_dictionaries`` when keras is
present).  What is verified here is what does not depend on the framework: the module imports,
the declarations are built, and they have the structure the contract requires.
"""

import subprocess
import sys

SCRIPT = '''
import sys
import types

# ---- minimal stubs, installed *before* the module under test is imported
keras = types.ModuleType("keras")
keras.__version__ = "0.0.0-stub"
keras.backend = types.SimpleNamespace(backend=lambda: "tensorflow")
sys.modules["keras"] = keras

def stub(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module

ONE_SOLVER = {"Stub": (None, {"learning_rate": 0.01, "clipnorm": 1.0})}
stub("n3fit.backends.keras_backend.MetaModel", optimizers=ONE_SOLVER)
stub("n3fit.backends.keras_backend.MetaLayer", initializers={"stub_init": (None, {})})
stub(
    "n3fit.backends.keras_backend.base_layers",
    layers={"dense": (None, {}), "dropout": (None, {}), "concatenate": (None, {})},
    regularizers={"l1_l2": (None, {"l1": 0.0, "l2": 0.0})},
    custom_activations={"square_activation": None},
)

from n3fit.backends.keras_backend.capabilities import build_capabilities

capabilities = build_capabilities()

# the structure the contract requires of any backend (see test_capabilities.py)
assert set(capabilities.optimizers) == {"Stub"}, capabilities.optimizers
optimizer = capabilities.optimizers["Stub"]
assert set(optimizer) == {"options", "is_iterative", "requires", "uses_validation_stopping"}, optimizer
assert isinstance(optimizer["options"], dict) and isinstance(optimizer["is_iterative"], bool)
assert isinstance(optimizer["uses_validation_stopping"], bool)

# structural layers are not parametrizations
assert set(capabilities.parametrizations) == {"dense"}, capabilities.parametrizations
assert "stub_init" in capabilities.initializers
assert "square_activation" in capabilities.activations
assert isinstance(capabilities.supports_tensorboard, bool)
assert isinstance(capabilities.requires_eager_workaround, bool)
print("capabilities build with all frameworks stubbed")
'''



def test_keras_capabilities_are_buildable_without_a_framework():
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ":".join(sys.path)},
        check=False,
    )
    assert result.returncode == 0, (
        "building the Keras capabilities must not require an installed framework, and must not "
        "raise at import time (P1 shipped a NameError here that only a real environment would "
        f"have caught).\\n--- stdout ---\\n{result.stdout}\\n--- stderr ---\\n{result.stderr}"
    )
    assert "capabilities build" in result.stdout


def test_debug_state_configuration_does_not_assume_tensorflow_for_jax():
    script = r'''
import sys
import types

keras = types.ModuleType("keras")
keras.backend = types.SimpleNamespace(
    backend=lambda: "jax",
    clear_session=lambda: None,
    set_floatx=lambda dtype: None,
)
keras.utils = types.SimpleNamespace(set_random_seed=lambda seed: None)
sys.modules["keras"] = keras
sys.modules["jax"] = types.ModuleType("jax")

from n3fit.backends.keras_backend.internal_state import set_initial_state
set_initial_state(debug=True)
print("jax debug-state setup passed")
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ":".join(sys.path)},
        check=False,
    )
    assert result.returncode == 0, (
        "debug initialization must not reference TensorFlow under the JAX backend."
        f"\\n--- stdout ---\\n{result.stdout}\\n--- stderr ---\\n{result.stderr}"
    )
    assert "jax debug-state setup passed" in result.stdout
