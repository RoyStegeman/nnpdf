"""
The P2 surface: the role vocabulary and the ``Model`` access API.

These tests are framework-free by design: the behaviour is checked against the numpy test
double, so it verifies the *contract* rather than Keras.  What is pinned down:

* the role vocabulary is closed and adapters agree with it (no backend may invent a role),
* ``weights(role=...)`` is keyed by path and filtered by the first path component,
* a rebound input still evaluates (``bind_input`` must not break the graph -- in the Keras
  adapter this is the ``register_photon`` + ``compile`` behaviour, which is exactly the step
  that is easy to get wrong),
* ``override`` + ``freeze`` replace a section and make the graph non-trainable.  This is the
  ``_set_central_value`` path in ``hyper_optimization/rewards.py``: it is the behaviour the
  hyperopt diagnostic relies on, so it is worth a test that does not need TensorFlow.
"""

import numpy as np
import pytest

import n3fit

from n3fit.backends.base import (
    ROLE_NN,
    ROLE_OBJECTIVE,
    ROLE_PHOTON,
    ROLE_PREPROCESSING,
    ROLE_REFERENCE,
    ROLE_SUMRULE,
    ROLES,
    role_of,
)
from n3fit.tests.backend_conformance.testing_backends import NumpyDoubleBackend, NumpyModelView

EXPECTED_ROLES = {
    "nn",
    "preprocessing",
    "objective",
    "photon",
    "reference",
    "sumrule",
}


def test_role_vocabulary_is_the_agreed_one():
    """The six roles agreed for P2, and nothing else."""
    assert set(ROLES) == EXPECTED_ROLES
    assert {
        ROLE_NN,
        ROLE_PREPROCESSING,
        ROLE_OBJECTIVE,
        ROLE_PHOTON,
        ROLE_REFERENCE,
        ROLE_SUMRULE,
    } == EXPECTED_ROLES


def test_roles_are_unique():
    assert len(ROLES) == len(set(ROLES))


def test_role_of_reads_the_first_path_component():
    assert role_of("preprocessing/alpha/up") == ROLE_PREPROCESSING
    assert role_of("nn/dense/kernel") == ROLE_NN
    assert role_of("reference") == ROLE_REFERENCE


def test_keras_role_table_uses_only_declared_roles():
    """The Keras adapter's name table must not smuggle in an undeclared role.

    ``roles.py`` is deliberately framework-free, so this holds in every environment -- not only
    where keras is installed.
    """
    from n3fit.backends.keras_backend.roles import ROLE_LAYER_NAMES

    assert set(ROLE_LAYER_NAMES) <= set(ROLES)
    # The roles P2 routes through the adapter:
    assert ROLE_LAYER_NAMES[ROLE_PHOTON] == "add_photon"
    assert ROLE_LAYER_NAMES[ROLE_PREPROCESSING] == "preprocessing_factor"
    assert ROLE_LAYER_NAMES[ROLE_REFERENCE] == "PDFs"


def test_every_role_name_is_actually_created_by_n3fit():
    """Anti-drift: each name in the table must be a layer some n3fit code really creates.

    This is the test the 2023 rename would have failed.  ``PDF_0`` was renamed to ``PDFs`` in
    commit b919773ef and ``_set_central_value`` was missed, so the role pointed at a layer that
    had not existed for two and a half years -- and nothing noticed, because nothing ran the
    override.  Names are collected from the parsed sources: the ``name=`` keyword of every call,
    plus the two module-level layer-name constants.
    """
    import ast
    from pathlib import Path

    from n3fit.backends.keras_backend.roles import ROLE_LAYER_NAMES

    package = Path(n3fit.__file__).parent
    created = set()
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        if str(relative).startswith("tests/") or "__pycache__" in str(relative):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # name="..." given to a layer or a model
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if (
                        keyword.arg == "name"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        created.add(keyword.value.value)
            # module-level constants that hold a layer name (all_NNs, preprocessing_factor)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str) and any(
                    isinstance(target, ast.Name) and target.id.endswith("_ALL_REPLICAS")
                    for target in node.targets
                ):
                    created.add(node.value.value)

    missing = {
        role: name for role, name in ROLE_LAYER_NAMES.items() if name not in created
    }
    assert not missing, (
        f"these roles point at layers no n3fit code creates: {missing}\n"
        f"Either the generator was renamed (update ROLE_LAYER_NAMES) or the role is stale."
    )


def _double_graph():
    """A two-section double: an ``nn`` with one weight and a ``reference`` with a value."""
    return NumpyModelView(
        {
            ROLE_NN: {"weights": {"dense/kernel": np.arange(4.0).reshape(2, 2)}, "value": 1.0},
            ROLE_REFERENCE: {"value": 2.0},
        },
        inputs={"pdf_input": np.zeros(3), "xgrid_integration": np.linspace(0, 1, 3)},
    )


@pytest.fixture
def view():
    """The double's model view.

    These tests use the *reference implementation* of the contract rather than the parametrized
    ``backend`` fixture: ``view`` takes a graph produced by the backend implementing it, so
    there is no backend-independent graph to hand around.  What is backend-independent -- the
    vocabulary and the Keras name table -- is checked above without any graph at all, and the
    Keras adapter is checked against a fake graph in ``test_keras_role_view.py``.
    """
    return NumpyDoubleBackend().view(_double_graph())


def test_weights_are_keyed_by_path_and_filtered_by_role(view):
    everything = view.weights()
    assert "nn/dense/kernel" in everything, everything
    only_nn = view.weights(role=ROLE_NN)
    assert list(only_nn) == ["nn/dense/kernel"]
    assert np.allclose(only_nn["nn/dense/kernel"], np.arange(4.0).reshape(2, 2))
    # a role with no weights is empty, not an error
    assert view.weights(role=ROLE_SUMRULE) == {}


def test_bound_inputs_are_numpy(view):
    bound = view.bound_inputs()
    assert set(bound) == {"pdf_input", "xgrid_integration"}
    assert all(isinstance(value, np.ndarray) for value in bound.values())


def test_bind_input_still_evaluates(view):
    """Rebinding an input must leave a graph that evaluates -- the register_photon failure mode."""
    new_grid = np.linspace(0, 2, 3)
    view.bind_input("pdf_input", new_grid)
    assert np.allclose(view.bound_inputs()["pdf_input"], new_grid)
    assert view()  # the graph still produces something


def test_override_replaces_a_section(view):
    cv = np.array([7.0])

    def central_value(inputs):
        return cv

    view.override(ROLE_REFERENCE, central_value)
    assert np.allclose(view()[ROLE_REFERENCE], cv)


def test_override_unknown_role_is_an_error(view):
    """A role the graph does not have must fail loudly.

    Silently doing nothing would make a diagnostic quietly wrong -- exactly what happened when the
    PDF layer was renamed out from under ``_set_central_value`` (``n3fit-backend-contract.md``
    §1.13a), so the error is part of the contract.
    """
    with pytest.raises(ValueError):
        view.override(ROLE_PHOTON, lambda inputs: np.zeros(1))


def test_freeze_makes_the_graph_non_trainable(view):
    view.freeze()
    assert view.frozen is True
