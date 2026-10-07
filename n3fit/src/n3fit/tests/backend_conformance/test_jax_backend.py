"""
P6: the raw JAX + optax backend -- the members only this backend can pin down.

The contract suite (this directory's ``backend`` fixture) already runs every shared promise
against ``jax``; what is left is what the contract cannot say: the exact optimizer registry
(Adam + SGD, with the Keras-matching defaults the parity test depends on), the
functional-model mechanics, the engine's freeze/override/recompile rules, the weight-file
round trip, and D9 (no Keras import anywhere in this package).
"""

import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optax")

from n3fit.backends import get_backend  # noqa: E402
from n3fit.backends.base import (  # noqa: E402
    GROUP_TRAINING,
    ObjectiveSpec,
    OptimizerSpec,
    ParametrizationSpec,
)
from n3fit.backends.jax_backend.models import (  # noqa: E402
    JaxEnsemble,
    JaxEnsembleView,
    JaxModel,
)


@pytest.fixture(scope="module")
def backend():
    return get_backend("jax")


def _tiny_model(w0=(1.0, -2.0)):
    """One replica, one output, two weights: ``pred = w * x`` with ``x = (1, 1)``.

    The smallest model the engine can train: chi2 against a zero target with an identity
    covariance, so every SGD step shrinks the weights and any movement at all is visible.
    """

    def apply(params, inputs):
        return params["w"] * inputs["x"]

    return JaxModel(
        [{"w": np.array(w0, dtype="float32")}],
        {"LHC": apply},
        inputs={"x": np.ones(2, dtype="float32")},
        name="tiny",
    )


def _tiny_term(backend, **overrides):
    data = {
        "invcovmat": np.eye(2, dtype="float32"),
        "covmat": np.eye(2, dtype="float32"),
        "target": np.zeros(2, dtype="float32"),
    }
    spec = ObjectiveSpec(kind="chi2", name="LHC", data=data, **overrides)
    return backend.objective(spec)


def _tiny_run(backend, model, steps, optimizer="SGD", hooks=(), monitor_every=None, **options):
    ensemble = backend.ensemble(model)
    terms = {"LHC": _tiny_term(backend)}
    groups = {GROUP_TRAINING: ["LHC"]}
    run = backend.optimizer(OptimizerSpec(optimizer, options)).run
    return run(ensemble, terms, groups, steps=steps, monitor_every=monitor_every, hooks=hooks)


# -- identity -----------------------------------------------------------------------------


def test_backend_reports_its_name_and_versions(backend):
    assert backend.name == "jax"
    info = backend.version_info()
    assert info["backend"] == "jax"
    assert set(info) == {"backend", "jax", "jaxlib", "optax"}


def test_no_keras_import_in_this_package():
    """D9: the JAX backend is an implementation, not an adapter -- it must not import Keras.

    (Docstrings may *mention* Keras -- the parity work is defined against it -- so this scans
    import statements, not words.)
    """
    import ast

    package = pathlib.Path(__file__).parent.parent.parent / "backends" / "jax_backend"
    offenders = {}
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text())
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                hits.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                hits.append(node.module or "")
        hits = [hit for hit in hits if hit.split(".")[0] == "keras"]
        if hits:
            offenders[path.name] = hits
    assert offenders == {}


# -- capabilities -------------------------------------------------------------------------


def test_declares_adam_and_sgd_only(backend):
    """P6 promises exactly the two names the parity test exercises (proposal §3.4)."""
    assert sorted(backend.capabilities.optimizers) == ["Adam", "SGD"]


def test_optimizer_defaults_match_the_keras_registry(backend):
    """A runcard means the same under either backend: ``clipnorm = 1.0`` on both names, and
    Adam's epsilon pinned to the Keras value (1e-7), not the optax default (1e-8)."""
    adam = backend.capabilities.optimizers["Adam"]
    sgd = backend.capabilities.optimizers["SGD"]
    assert adam["options"]["clipnorm"] == 1.0
    assert sgd["options"]["clipnorm"] == 1.0
    assert adam["options"]["epsilon"] == 1e-7
    assert adam["options"]["beta1"] == 0.9
    assert adam["options"]["beta2"] == 0.999
    for declared in (adam, sgd):
        assert declared["is_iterative"] is True
        assert declared["requires"] == {"gradient"}


def test_derivatives_is_gradient_only(backend):
    """Jacobians and Hessians arrive in P7; until then the engine differentiates directly."""
    assert set(backend.capabilities.derivatives) == {"gradient"}


def test_check_feasible_accepts_the_declared_combination(backend):
    kind = next(iter(backend.capabilities.parametrizations))
    backend.check_feasible(
        ParametrizationSpec(kind=kind),
        OptimizerSpec("Adam", {}),
        ObjectiveSpec(kind="chi2", name="LHC"),
    )


@pytest.mark.parametrize("slot", ["parametrization", "optimizer", "objective"])
def test_check_feasible_rejects_what_is_not_declared(backend, slot):
    """Each undeclared member fails with a message naming the backend's own set."""
    kind = next(iter(backend.capabilities.parametrizations))
    specs = {
        "parametrization": ParametrizationSpec(kind=kind),
        "optimizer": OptimizerSpec("Adam", {}),
        "objective": ObjectiveSpec(kind="chi2", name="LHC"),
    }
    matches = {
        "parametrization": "no parametrization 'nope'",
        "optimizer": "no optimizer 'RMSprop'",
        "objective": "no objective kind 'nope'",
    }
    if slot == "parametrization":
        specs["parametrization"] = ParametrizationSpec(kind="nope")
    elif slot == "optimizer":
        specs["optimizer"] = OptimizerSpec("RMSprop", {})
    else:
        specs["objective"] = ObjectiveSpec(kind="nope", name="LHC")
    with pytest.raises(NotImplementedError, match=matches[slot]):
        backend.check_feasible(
            specs["parametrization"], specs["optimizer"], specs["objective"]
        )


# -- models -------------------------------------------------------------------------------


def test_view_is_the_identity_for_jax_models_and_refuses_the_rest(backend):
    model = _tiny_model()
    assert backend.view(model) is model
    with pytest.raises(ValueError, match="the 'keras' backend"):
        backend.view(object())


def test_ensemble_accepts_a_mapping_a_model_or_a_sequence(backend):
    training = _tiny_model()
    assert isinstance(backend.ensemble({GROUP_TRAINING: training}), JaxEnsemble)
    assert isinstance(backend.ensemble(training), JaxEnsemble)
    assert isinstance(backend.ensemble([training]), JaxEnsembleView)
    assert isinstance(backend.ensemble(backend.ensemble([training])), JaxEnsembleView)
    with pytest.raises(ValueError, match="a JAX ensemble is built from"):
        backend.ensemble(42)


def test_weights_round_trip_through_set_weights(backend):
    ensemble = backend.ensemble(_tiny_model())
    (before,) = ensemble.weights()
    edited = {path: value * 2 + 1 for path, value in before.items()}
    ensemble.set_weights([edited])
    (after,) = ensemble.weights()
    assert np.array_equal(after["w"], before["w"] * 2 + 1)


# -- objectives ---------------------------------------------------------------------------


def test_objective_validation_mirrors_the_keras_rules(backend):
    with pytest.raises(ValueError, match="Unknown objective kind 'nope'"):
        backend.objective(ObjectiveSpec(kind="nope", name="LHC"))
    with pytest.raises(ValueError, match="needs data"):
        backend.objective(ObjectiveSpec(kind="chi2", name="LHC", data={}))
    with pytest.raises(ValueError, match="accepts options"):
        backend.objective(
            ObjectiveSpec(kind="positivity", name="POS", options={"nope": 1.0})
        )
    term = backend.objective(
        ObjectiveSpec(kind="positivity", name="POS", options={"multiplier": 10.0})
    )
    assert term.scalar("multiplier") == 10.0
    term.set_scalar("multiplier", 11.0)
    assert term.scalar("multiplier") == 11.0
    with pytest.raises(ValueError, match="declares scalars"):
        term.set_scalar("nope", 1.0)


# -- the engine ---------------------------------------------------------------------------


def test_optimizer_rejects_names_it_does_not_implement(backend):
    with pytest.raises(NotImplementedError, match="optimizer not implemented: 'RMSprop'"):
        backend.optimizer(OptimizerSpec("RMSprop", {}))


def test_run_rejects_a_missing_step_budget(backend):
    ensemble = backend.ensemble(_tiny_model())
    optimizer = backend.optimizer(OptimizerSpec("SGD", {"learning_rate": 0.1}))
    with pytest.raises(ValueError, match="needs a number of steps"):
        optimizer.run(ensemble, {}, {GROUP_TRAINING: []}, steps=None, monitor_every=None)


def test_run_rejects_training_terms_it_was_not_given(backend):
    ensemble = backend.ensemble(_tiny_model())
    optimizer = backend.optimizer(OptimizerSpec("SGD", {"learning_rate": 0.1}))
    with pytest.raises(ValueError, match=r"\['LHC'\]"):
        optimizer.run(ensemble, {}, {GROUP_TRAINING: ["LHC"]}, steps=1, monitor_every=None)


def test_sgd_shrinks_the_tiny_problem(backend):
    """The engine trains: chi2 against zero with an identity covariance must shrink ``w``."""
    model = _tiny_model()
    result = _tiny_run(backend, model, 5, learning_rate=0.1)
    (after,) = result.parameters
    assert np.all(np.abs(after["w"]) < np.abs([1.0, 2.0]))
    assert result.diagnostics == {"monitored_steps": []}


def test_adam_trains_too(backend):
    model = _tiny_model()
    result = _tiny_run(backend, model, 5, optimizer="Adam", learning_rate=0.1)
    (after,) = result.parameters
    assert np.all(np.abs(after["w"]) < np.abs([1.0, 2.0]))


def test_frozen_model_fires_hooks_but_keeps_its_weights(backend):
    """Freeze is the engine's read of the flag: evaluations and hooks run, updates do not."""
    seen = []

    class Rec:
        def on_monitored_step(self, ctx):
            seen.append(ctx.step)

        def on_train_end(self):
            seen.append("end")

    model = _tiny_model()
    model.freeze()
    before = {path: value.copy() for path, value in model.weights().items()}
    result = _tiny_run(backend, model, 3, hooks=[Rec()], monitor_every=1, learning_rate=0.1)
    (after,) = result.parameters
    assert all(np.array_equal(after[path], before[path]) for path in before)
    assert seen == [1, 2, 3, "end"]
    assert result.diagnostics == {"monitored_steps": [1, 2, 3]}


def test_overridden_model_cannot_train_but_can_be_diagnosed(backend):
    """An override swaps the section for a fixed function: no gradient, so no training --
    unless the model is frozen, which is the diagnostics spelling."""
    model = _tiny_model()
    model.override("w", lambda inputs: np.zeros(2))
    optimizer = backend.optimizer(OptimizerSpec("SGD", {"learning_rate": 0.1}))
    ensemble = backend.ensemble(model)
    terms = {"LHC": _tiny_term(backend)}
    groups = {GROUP_TRAINING: ["LHC"]}
    with pytest.raises(NotImplementedError, match="has no gradient to train"):
        optimizer.run(ensemble, terms, groups, steps=1, monitor_every=None)
    model.freeze()
    result = optimizer.run(ensemble, terms, groups, steps=1, monitor_every=None)
    assert result.diagnostics == {"monitored_steps": []}


def test_only_trainable_paths_move(backend):
    """A model built with a trainable subset keeps the other paths bit-identical."""

    def apply(params, inputs):
        return params["a"] * inputs["x"] + params["b"] * inputs["x"]

    model = JaxModel(
        [{"a": np.array([1.0, 1.0], dtype="float32"), "b": np.array([1.0, 1.0], dtype="float32")}],
        {"LHC": apply},
        inputs={"x": np.ones(2, dtype="float32")},
        trainable=["a"],
        name="subset",
    )
    result = _tiny_run(backend, model, 5, optimizer="Adam", learning_rate=0.1)
    (after,) = result.parameters
    assert not np.array_equal(after["a"], [1.0, 1.0])
    assert np.array_equal(after["b"], [1.0, 1.0])


def test_mid_run_data_change_takes_effect_without_restart(backend):
    """The generation guard: a hook that swaps term data mid-run must move training, not
    lose to the compiled step's stale closure (the JAX half of the P6 staleness finding)."""

    class MaskSwap:
        def on_monitored_step(self, ctx):
            if ctx.step == 2:
                self.terms["LHC"].set_data(mask=[1.0, 0.0])

        def on_train_end(self):
            pass

    swap = MaskSwap()
    model = _tiny_model()
    ensemble = backend.ensemble(model)
    terms = {"LHC": _tiny_term(backend)}
    swap.terms = terms
    trace = {}

    class Trace:
        def on_monitored_step(self, ctx):
            trace[ctx.step] = {k: v.copy() for k, v in ensemble.weights()[0].items()}

        def on_train_end(self):
            pass

    backend.optimizer(OptimizerSpec("SGD", {"learning_rate": 0.1})).run(
        ensemble, terms, {GROUP_TRAINING: ["LHC"]}, steps=3, monitor_every=1,
        hooks=[swap, Trace()],
    )
    in_run_update = {
        path: trace[3][path] - trace[2][path] for path in trace[2]
    }

    # The same third step, taken fresh with the new mask from the start.
    fresh = _tiny_model()
    fresh_ensemble = backend.ensemble(fresh)
    fresh_ensemble.set_weights([trace[2]])
    fresh_terms = {"LHC": _tiny_term(backend, mask=[1.0, 0.0])}
    backend.optimizer(OptimizerSpec("SGD", {"learning_rate": 0.1})).run(
        fresh_ensemble, fresh_terms, {GROUP_TRAINING: ["LHC"]}, steps=1, monitor_every=None
    )
    (fresh_after,) = fresh_ensemble.weights()
    for path in in_run_update:
        assert np.array_equal(in_run_update[path], fresh_after[path] - trace[2][path])


# -- state --------------------------------------------------------------------------------


def test_state_configure_clear_and_devices(backend):
    state = backend.state
    state.configure(seed=123, deterministic=True)
    assert state.devices()
    state.clear()  # re-seeds from the configured seed; must not raise
    with pytest.raises(ValueError, match="float32.*float64"):
        state.configure(dtype="float16")


def test_state_float64_flag_is_reversible(backend):
    """``float64`` flips a process-global XLA flag: the test restores it whatever happens."""
    state = backend.state
    try:
        state.configure(dtype="float64")
        assert jax.config.read("jax_enable_x64") is True
    finally:
        state.configure(dtype="float32")
    assert jax.config.read("jax_enable_x64") is False


def test_tensorboard_hook_is_an_explained_no(backend):
    with pytest.raises(NotImplementedError, match="no tensorflow graph"):
        backend.tensorboard_hook("/tmp/logs")


def test_parametrization_and_model_are_staged_for_p7(backend):
    with pytest.raises(NotImplementedError, match="arrives in P7"):
        backend.parametrization(ParametrizationSpec(kind="dense"), {})
    with pytest.raises(NotImplementedError, match="arrives in P7"):
        backend.model({}, {}, {}, name="m")


# -- persistence --------------------------------------------------------------------------


def test_save_and_load_round_trip_one_replica(backend, tmp_path):
    ensemble = backend.ensemble(_tiny_model())
    path = tmp_path / "tiny.weights.npz"
    backend.save(ensemble, path)
    (saved,) = ensemble.weights()
    ensemble.set_weights([{path: value * 3 for path, value in saved.items()}])
    backend.load(ensemble, path)
    (restored,) = ensemble.weights()
    assert all(np.array_equal(restored[path], saved[path]) for path in saved)


def test_saving_a_multi_replica_ensemble_is_an_error(backend, tmp_path):
    ensemble = backend.ensemble([_tiny_model(), _tiny_model()])
    with pytest.raises(ValueError, match="one replica"):
        backend.save(ensemble, tmp_path / "nope.weights.npz")


def test_loading_into_a_missing_replica_is_an_error(backend, tmp_path):
    ensemble = backend.ensemble(_tiny_model())
    path = tmp_path / "tiny.weights.npz"
    backend.save(ensemble, path)
    with pytest.raises(ValueError, match="replicas"):
        backend.load(ensemble, path, replica=5)
