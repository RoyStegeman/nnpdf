"""
P6: the JAX engine trains the same fit as the Keras engine (the P6 golden).

Same skeleton as ``p4_golden.py`` but backend-built: a small dense core with two heads
(chi2 + positivity), three role models, a Lagrange schedule, and stopping with best-weight
restore -- built once per backend from one problem and one shared initialization, trained
through the *same* n3fit-side hooks, and compared per step.

Methodology (P4's: tolerances measured, not guessed):

* the initial weights are set identically on both sides through the P5 path maps, so the
  initializer streams -- statistically equivalent but bit-different by design -- are out
  of the comparison;
* SGD runs the full 30-step window with the schedule firing three times: the updates are
  un-normalized, so float32 association noise stays noise, and the measured agreement is
  ~1e-7 (weights) with bit-identical per-step losses;
* Adam runs a short window because its ``lr / eps`` gain turns that same ~1e-8 gradient
  noise into visible update differences on near-zero gradient entries (measured: median
  weight agreement 7e-7, a dozen amplified outliers to 4e-3, per-step losses to 4e-6).
  Both gradient computations are provably right -- JAX matches a float64 finite-difference
  reference and Keras matches JAX -- so the bar is bulk-plus-losses, not max-entry.

What each test guards (mutation-checked: reverting the fix fails the test):

* the SGD window guards the per-tensor ``clipnorm`` (a global-norm clip diverges 2x in one
  step) and the mid-run multiplier handoff (a stale capture diverges the trajectory after
  the first firing);
* the Adam window guards epsilon, the moments, and moment preservation across the Keras
  engine's mid-run recompile;
* the stopping test guards the stop decision, the best-weight snapshot, and the restore.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("keras")
pytest.importorskip("optax")

from n3fit.backends import MetaModel, get_backend  # noqa: E402
from n3fit.backends import operations as op  # noqa: E402
from n3fit.backends.base import (  # noqa: E402
    GROUP_EXPERIMENTAL,
    GROUP_TRAINING,
    GROUP_VALIDATION,
    ObjectiveSpec,
    OptimizerSpec,
)
from n3fit.backends.jax_backend.models import JaxModel  # noqa: E402
from n3fit.stopping import FitRecord, LagrangeHook, LogHook, StoppingHook  # noqa: E402

DIN, HID, NDATA, NPOS = 4, 6, 5, 3
LR = 1e-3
NREP = 2

# Measured agreement (see the module docstring); the assertions below hold ~10-100x of it.
RTOL_SGD_LOSSES = 1e-6
RTOL_SGD_WEIGHTS = 1e-6
RTOL_ADAM_LOSSES = 1e-4
RTOL_ADAM_WEIGHTS_MAX = 2e-2
RTOL_ADAM_WEIGHTS_MEDIAN = 1e-5
RTOL_FORWARD = 1e-6
ATOL = 1e-8


def problem(seed=7):
    """One synthetic problem: an input grid plus a chi2 dataset per role."""
    rng = np.random.default_rng(seed)
    out = {"xgrid": rng.normal(size=(1, DIN)).astype("float32")}
    for name in ["LHC", "LHC_val", "LHC_exp"]:
        raw = rng.normal(size=(NDATA, NDATA)).astype("float32")
        cov = (raw @ raw.T / NDATA + np.eye(NDATA, dtype="float32") * 0.5).astype("float32")
        out[name] = {
            "covmat": cov,
            "invcovmat": np.linalg.inv(cov).astype("float32"),
            "target": rng.normal(size=(NDATA,)).astype("float32"),
        }
    return out


CANON = [(DIN, HID), (HID,), (HID, NDATA), (NDATA,), (HID, NPOS), (NPOS,)]
NAMES = ["kernel", "bias"] * 3


def init_values(seed=123):
    """The shared initialization, in canonical path order (``nn/{i}/{kernel,bias}``)."""
    rng = np.random.default_rng(seed)
    return [rng.normal(size=shape).astype("float32") * 0.5 for shape in CANON]


def _chi2_specs(prob):
    return {
        name: ObjectiveSpec(
            kind="chi2",
            name=name,
            data={
                "invcovmat": prob[name]["invcovmat"],
                "covmat": prob[name]["covmat"],
                "target": prob[name]["target"],
            },
        )
        for name in ["LHC", "LHC_val", "LHC_exp"]
    }


GROUPS = {
    GROUP_TRAINING: ["LHC", "POS"],
    GROUP_VALIDATION: ["LHC_val"],
    GROUP_EXPERIMENTAL: ["LHC_exp"],
}


def build_keras(prob, nrep, inits):
    """The parity problem as nested Keras graphs (production nesting, toy widths).

    One ``NN_{i}`` branch per replica under an ``all_NNs`` model, re-applied per role: the
    store then yields exactly the ``nn/{i}/{kernel,bias}`` paths the JAX side declares, so
    the shared init and the weight interchange below compare identical layouts.
    """
    import keras
    from keras.layers import Identity

    backend = get_backend("keras")

    def make_branch(i):
        inner = keras.layers.Input(batch_size=1, shape=(DIN,))
        hidden = keras.layers.Dense(HID, activation="tanh")(inner)
        head_chi2 = keras.layers.Dense(NDATA, activation="linear")(hidden)
        head_pos = keras.layers.Dense(NPOS, activation="linear")(hidden)
        return keras.Model(inner, [head_chi2, head_pos], name=f"NN_{i}")

    branches = [make_branch(i) for i in range(nrep)]
    for branch in branches:
        weights = []
        for layer in branch.layers:
            if layer.weights:
                weights.extend(layer.weights)
        assert len(weights) == len(inits)
        for var, val in zip(weights, inits):
            var.assign(val)

    x = keras.layers.Input(batch_size=1, shape=(DIN,), name="x")
    x.tensor_content = op.numpy_to_tensor(prob["xgrid"])
    applied = [branch(x) for branch in branches]
    flat = [tensor for pair in applied for tensor in pair]
    all_nns = keras.Model(x, flat, name="all_NNs")

    def role_graph(out_names):
        final = all_nns(x)
        # flat order per replica is [chi2 head, positivity head]
        per_output = {}
        for name in out_names:
            take = 0 if name.startswith("LHC") else 1
            per_output[name] = [final[2 * i + take] for i in range(nrep)]
        stacked = {
            name: keras.layers.Lambda(
                lambda zs: keras.ops.stack(zs, axis=1), name=f"stack_{name}"
            )(heads)
            for name, heads in per_output.items()
        }
        named = [Identity(name=name)(tensor) for name, tensor in stacked.items()]
        return MetaModel({"x": x}, named)

    graphs = {
        GROUP_TRAINING: role_graph(["LHC", "POS"]),
        GROUP_VALIDATION: role_graph(["LHC_val"]),
        GROUP_EXPERIMENTAL: role_graph(["LHC_exp"]),
    }
    specs = _chi2_specs(prob)
    specs["POS"] = ObjectiveSpec(kind="positivity", name="POS", options={"multiplier": 10.0})
    terms = {name: backend.objective(spec) for name, spec in specs.items()}
    return backend.ensemble(graphs), terms, dict(GROUPS)


def build_jax(prob, nrep, inits):
    """The parity problem as functional JAX models over the same paths and inits."""
    import jax.numpy as jnp

    backend = get_backend("jax")
    paths = [f"nn/{i}/{name}" for i, name in enumerate(NAMES)]

    def head(kernel, bias):
        def apply(params, inputs):
            hidden = jnp.tanh(inputs["x"] @ params[paths[0]] + params[paths[1]])
            return (hidden @ params[paths[kernel]] + params[paths[bias]])[0]

        return apply

    head_chi2, head_pos = head(2, 3), head(4, 5)
    # Every role model shares the one parameter list: training moves all roles at once.
    params = [{path: val for path, val in zip(paths, inits)} for _ in range(nrep)]
    models = {
        GROUP_TRAINING: JaxModel(
            params, {"LHC": head_chi2, "POS": head_pos}, inputs={"x": prob["xgrid"]}, name="tr"
        ),
        GROUP_VALIDATION: JaxModel(
            params, {"LHC_val": head_chi2}, inputs={"x": prob["xgrid"]}, name="vl"
        ),
        GROUP_EXPERIMENTAL: JaxModel(
            params, {"LHC_exp": head_chi2}, inputs={"x": prob["xgrid"]}, name="ex"
        ),
    }
    specs = _chi2_specs(prob)
    specs["POS"] = ObjectiveSpec(kind="positivity", name="POS", options={"multiplier": 10.0})
    terms = {name: backend.objective(spec) for name, spec in specs.items()}
    return backend.ensemble(models), terms, dict(GROUPS)


class _Recorder:
    """Per-step training losses plus the validation/experimental evaluations."""

    def __init__(self):
        self.logs = {}

    def on_monitored_step(self, ctx):
        self.logs[ctx.step] = {k: np.asarray(v).copy() for k, v in ctx.logs.items()}
        self.logs[ctx.step]["_vl"] = {
            k: np.asarray(v).copy() for k, v in ctx.evaluate(GROUP_VALIDATION).items()
        }
        self.logs[ctx.step]["_ex"] = {
            k: np.asarray(v).copy() for k, v in ctx.evaluate(GROUP_EXPERIMENTAL).items()
        }

    def on_train_end(self):
        pass


def _assert_logs_close(ka, kb, rtol, atol=ATOL):
    assert sorted(ka) == sorted(kb)
    for step in sorted(ka):
        for key, va in ka[step].items():
            vb = kb[step][key]
            items = va.items() if isinstance(va, dict) else [(key, va)]
            for name, x in items:
                y = vb[name] if isinstance(vb, dict) else vb
                assert np.allclose(x, y, rtol=rtol, atol=atol), (key, name, step, x, y)


def _assert_weights_close(wa, wb, rtol, atol=ATOL):
    assert [sorted(m) for m in wa] == [sorted(m) for m in wb]
    for ma, mb in zip(wa, wb):
        for path in ma:
            assert np.allclose(ma[path], mb[path], rtol=rtol, atol=atol), path


def test_shared_init_gives_identical_forward_values():
    """Same paths, same values, same predictions: the comparison starts from equality."""
    prob, inits = problem(), init_values()
    keras_ens, keras_terms, _ = build_keras(prob, NREP, [v.copy() for v in inits])
    jax_ens, jax_terms, _ = build_jax(prob, NREP, [v.copy() for v in inits])
    keras_weights, jax_weights = keras_ens.weights(), jax_ens.weights()
    assert [sorted(m) for m in keras_weights] == [sorted(m) for m in jax_weights]
    for ma, mb in zip(keras_weights, jax_weights):
        for path in ma:
            assert np.array_equal(ma[path], mb[path]), path
    for backend_name, ens, terms in [("keras", keras_ens, keras_terms), ("jax", jax_ens, jax_terms)]:
        evaluators = get_backend(backend_name).optimizer(OptimizerSpec("SGD", {})).evaluate
        if backend_name == "keras":
            keras_eval = {
                group: evaluators(ens, terms, group)
                for group in [GROUP_TRAINING, GROUP_VALIDATION, GROUP_EXPERIMENTAL]
            }
        else:
            for group in [GROUP_TRAINING, GROUP_VALIDATION, GROUP_EXPERIMENTAL]:
                for name, value in evaluators(ens, terms, group).items():
                    assert np.allclose(
                        value, keras_eval[group][name], rtol=RTOL_FORWARD, atol=ATOL
                    ), (group, name)


def test_sgd_parity_with_lagrange_schedule():
    """30 SGD steps, the schedule firing three times: losses bit-identical, weights ~1e-7."""
    runs = {}
    for backend_name, build in [("keras", build_keras), ("jax", build_jax)]:
        ensemble, terms, groups = build(problem(), NREP, init_values())
        lag = LagrangeHook(terms, {"POS": 1.1}, period=10)
        rec = _Recorder()
        result = (
            get_backend(backend_name)
            .optimizer(OptimizerSpec("SGD", {"learning_rate": LR}))
            .run(ensemble, terms, groups, steps=30, monitor_every=1, hooks=[lag, rec])
        )
        runs[backend_name] = (rec.logs, terms["POS"].scalar("multiplier"), lag.applied["POS"],
                              result.parameters, result.diagnostics)
    (k_logs, k_mult, k_applied, k_weights, k_diag), (j_logs, j_mult, j_applied, j_weights, j_diag) = (
        runs["keras"], runs["jax"])
    _assert_logs_close(k_logs, j_logs, RTOL_SGD_LOSSES)
    assert k_mult == pytest.approx(j_mult, rel=1e-6)
    assert k_applied == j_applied == 3
    _assert_weights_close(k_weights, j_weights, RTOL_SGD_WEIGHTS)
    assert k_diag == j_diag


def test_adam_parity_with_lagrange_schedule():
    """5 Adam steps with a firing inside: losses tight, weights bulk-tight with amplified
    outliers (see the module docstring for the measured justification)."""
    runs = {}
    for backend_name, build in [("keras", build_keras), ("jax", build_jax)]:
        ensemble, terms, groups = build(problem(), NREP, init_values())
        lag = LagrangeHook(terms, {"POS": 1.1}, period=3)
        rec = _Recorder()
        result = (
            get_backend(backend_name)
            .optimizer(OptimizerSpec("Adam", {"learning_rate": LR}))
            .run(ensemble, terms, groups, steps=5, monitor_every=1, hooks=[lag, rec])
        )
        runs[backend_name] = (rec.logs, terms["POS"].scalar("multiplier"), lag.applied["POS"],
                              result.parameters, result.diagnostics)
    (k_logs, k_mult, k_applied, k_weights, k_diag), (j_logs, j_mult, j_applied, j_weights, j_diag) = (
        runs["keras"], runs["jax"])
    _assert_logs_close(k_logs, j_logs, RTOL_ADAM_LOSSES)
    assert k_mult == pytest.approx(j_mult, rel=1e-6)
    assert k_applied == j_applied == 1
    assert [sorted(m) for m in k_weights] == [sorted(m) for m in j_weights]
    rels = []
    for ma, mb in zip(k_weights, j_weights):
        for path in ma:
            a, b = (np.asarray(m[path], dtype=np.float64) for m in (ma, mb))
            assert np.allclose(a, b, rtol=RTOL_ADAM_WEIGHTS_MAX, atol=ATOL), path
            rels.extend(float(v) for v in (np.abs(a - b) / np.maximum(np.abs(b), 1e-12)).ravel())
    assert float(np.median(rels)) < RTOL_ADAM_WEIGHTS_MEDIAN
    assert k_diag == j_diag


def test_stopping_hook_stops_both_engines_together():
    """The stop decision, the best snapshot, and the restore agree across backends.

    ``stopping_delta`` is large enough that no step after the first can improve on it, yet
    small enough that the first step arms the counter -- so both fits must stop at the
    fourth monitored step with the first step's weights restored (cf. the P4 golden's
    stopping case).  The timing hook rides along: it is backend-agnostic by construction.
    """
    runs = {}
    for backend_name, build in [("keras", build_keras), ("jax", build_jax)]:
        ensemble, terms, groups = build(problem(), NREP, init_values())
        record = FitRecord()
        stopping = StoppingHook(
            record, ensemble,
            ndata={"LHC": [NDATA] * NREP}, vl_ndata={"LHC_val": [NDATA] * NREP},
            positivity_terms=[], total_steps=30,
            stopping_patience=2, stopping_delta=100.0, threshold_chi2=1e18, monitor_every=1,
        )
        timer = LogHook()
        result = (
            get_backend(backend_name)
            .optimizer(OptimizerSpec("SGD", {"learning_rate": LR}))
            .run(ensemble, terms, groups, steps=30, monitor_every=1, hooks=[stopping, timer])
        )
        runs[backend_name] = (result, record, timer)
    (k_result, k_record, k_timer), (j_result, j_record, j_timer) = runs["keras"], runs["jax"]
    assert k_result.diagnostics == j_result.diagnostics == {"monitored_steps": [1, 2, 3, 4]}
    assert k_record.e_best_chi2 == j_record.e_best_chi2 == [0, 0]
    assert k_record.stop_epochs == j_record.stop_epochs == [3, 3]
    _assert_weights_close(k_result.parameters, j_result.parameters, RTOL_SGD_WEIGHTS)
    # The first monitored step arms the timer; the rest append -- 4 steps, 3 entries, both sides.
    assert len(k_timer.all_times) == len(j_timer.all_times) == 3


def test_weight_files_interchange_between_backends(tmp_path):
    """A file written by either backend loads into the other, exactly (DoD §1)."""
    keras_backend, jax_backend = get_backend("keras"), get_backend("jax")
    keras_ens, _, _ = build_keras(problem(), NREP, init_values())
    jax_ens, _, _ = build_jax(problem(), NREP, init_values())

    # Keras -> JAX: replica 0's file fills JAX replica 1 bit-exactly.
    keras_file = tmp_path / "keras.weights.npz"
    keras_backend.save(keras_ens.graph(GROUP_TRAINING), keras_file)
    jax_backend.load(jax_ens, keras_file, replica=1)
    saved, loaded = keras_ens.weights()[0], jax_ens.weights()[1]
    assert set(saved) == set(loaded)
    assert all(np.array_equal(saved[path], loaded[path]) for path in saved)

    # JAX -> Keras: edited JAX weights round-trip back bit-exactly.
    edited = {path: value * 1.5 + 0.1 for path, value in jax_ens.weights()[0].items()}
    jax_ens.set_weights([edited, jax_ens.weights()[1]])
    jax_file = tmp_path / "jax.weights.npz"
    jax_backend.save(jax_ens.model(GROUP_TRAINING), jax_file)
    keras_backend.load(keras_ens, jax_file, replica=0)
    back = keras_ens.weights()[0]
    assert all(np.array_equal(edited[path], back[path]) for path in edited)

    # And the cross-loaded replica still evaluates identically on both sides.
    _, jax_terms, _ = build_jax(problem(), NREP, init_values())
    jax_backend.load(jax_ens, jax_file, replica=0)
    jax_eval = jax_backend.optimizer(OptimizerSpec("SGD", {})).evaluate(jax_ens, jax_terms, GROUP_VALIDATION)
    _, keras_terms, _ = build_keras(problem(), NREP, init_values())
    keras_eval = (
        keras_backend.optimizer(OptimizerSpec("SGD", {})).evaluate(keras_ens, keras_terms, GROUP_VALIDATION)
    )
    for name in jax_eval:
        assert np.allclose(jax_eval[name][0], keras_eval[name][0], rtol=RTOL_FORWARD, atol=ATOL)
