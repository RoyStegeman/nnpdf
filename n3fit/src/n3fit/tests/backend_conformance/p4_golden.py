#!/usr/bin/env python
"""
P4 golden-step fixture: generator (legacy) and checker (contract).

P4 replaces *the loop* (``MetaModel.perform_fit`` + the Keras callbacks) with ``Optimizer.run`` and
stops the model from being the loss.  Numbers must not move, and "must not move" has to be checked
against the code that is being replaced, not against the replacement.  This script defines one
small, fully synthetic problem that exercises everything the loop does, runs it through whichever
implementation is on ``PYTHONPATH``, and either writes the answer down (``legacy``) or compares it
with the recorded one (``contract``).

Usage
-----
Write the reference from a **pristine** checkout (no P4), e.g. ``af339ef``::

    KERAS_BACKEND=jax PYTHONPATH=/var/tmp/baseline/n3fit/src:<vp>:<data> \\
        python p4_golden.py legacy --out p4_golden.json

Then, in the patched tree::

    KERAS_BACKEND=jax PYTHONPATH=<patched>/n3fit/src:<vp>:<data> \\
        python p4_golden.py contract --fixture p4_golden.json

The problem
-----------
A real PDF model (``generate_pdf_model``: real ``all_NNs``/``preprocessing_factor`` layers, so the
weight get/set path that ``Stopping`` restores through is the production one), reduced by one
``Dense`` per observable, then the production mask and loss layers, in the three-role arrangement
``model_trainer`` uses:

* training: chi2 + positivity + integrability,
* validation: chi2 + positivity,
* experimental: chi2 only,

and each role's terms are its *own* terms, named per role (``POS``/``INT`` for training,
``POS_val`` for validation, ``LHC``/``LHC_val``/``LHC_exp`` for the chi2 -- n3fit already names the
chi2 that way).  That is forced by the contract's flat ``terms`` mapping and it is what the legacy
did too: one loss layer per graph, and the Lagrange callback scaled only the training one.

with per-replica masks that select the same *number* of points but different *points* -- a rewrite
that collapses the replica axis, or applies one replica's mask to another, changes every number.

Two cases are recorded:

``main``
    30 steps, no early stopping, Lagrange period 10, so the schedule fires twice.  Checks the
    per-step losses (the numbers ``Stopping`` sees), the multiplier series, and the final weights.
``stopping``
    30 steps allowed, a ``stopping_delta`` no step can beat: the fit stops early (step 4) *and*
    restores the weights snapshotted at its best epoch (step 0), which is the path a rewrite is
    most likely to break.

Conditioning, and why the tolerances are what they are
-----------------------------------------------------
Two implementations that sum the same terms in different orders cannot be bit-identical in float32,
and how much that matters depends on the problem: the per-step gap is one ulp in one weight, and
this toy's dynamics *stretch* it.  Measured on ``main`` (30 steps, lr 1e-3):

* per step the two loops differ in a single float32 ulp in one weight (``p4_probe_where.py``, kept
  with the P4 tooling);
* after 30 steps: median weight difference 1.6e-6 relative, worst entry 1.6e-4 relative, i.e. 1.6
  units of ``rtol=1e-4, atol=1e-6`` (``p4_probe_wdist.py``);
* everything that decides anything is far tighter: the per-step scalars and the validation
  quantities agree to ~3e-5 relative and the multipliers to 6e-8, and ``stop_epoch``,
  ``e_best_chi2`` and the positivity statuses are equal;
* a *larger* learning rate makes this toy chaotic rather than the comparison stricter: at lr=1e-2 a
  one-ulp perturbation of one initial weight separates the legacy loop from itself by 0.56 over 30
  steps, and the argmin behind ``e_best_chi2`` flips (``p4_probe_chaos.py``).  The fixture therefore
  uses lr=1e-3, where a 1e-7 perturbation stays 1e-7 -- the difference measured here is the
  implementation's, not the problem's.

So the weights carry ``rtol=1e-3`` (10x the measured worst entry, still 10x tighter than anything a
logic change produces: a wrong multiplier moves them by tens of percent) and everything else keeps
``1e-4``.
"""

import argparse
import json
import platform
import sys

import numpy as np

NREP = 2
NX = 3
NOBS = 6
NFL = 14
# Per-replica splits: same number of points, different points.
TRAIN_IDX = [np.array([0, 1, 2, 3]), np.array([0, 1, 4, 5])]
VALID_IDX = [np.array([4, 5]), np.array([2, 3])]
NDATA_TR = 4
NDATA_VL = 2

# The shipped "examples" values, so the multiplier series is a realistic one.
POS_INITIAL, INT_INITIAL, INT_ALPHA = 184.8, 10.0, 2.0
LAGRANGE_PERIOD = 10
# Learning rate.  Measured, not guessed: the engine and the legacy loop differ by ~1 float32 ulp
# per step (a different summation order for the same terms), and this toy fit's dynamics amplify a
# perturbation of that size.  ``p4_probe_chaos.py`` measures the amplification by perturbing one
# initial weight: at 0.01 the toy separates *from itself* by 0.56 over 30 steps (and the argmin that
# sets ``e_best_chi2`` flips), at 1e-3 it is neutral (a 1e-7 perturbation stays 1.1e-7) so the
# implementation difference, not the conditioning, is what the fixture measures.
DEFAULT_LR = 1e-3
PUSH_MULTIPLIER = 1.1

# ``threshold_chi2``/``threshold_positivity`` are deliberately huge: the stopping algorithm only
# starts counting once the validation chi2 is below its threshold, and a synthetic problem does not
# land at chi2 ~ 1, so a realistic threshold would leave the counting (and the best-weight
# bookkeeping and the restore) untested.  A real fit uses 10.0 / 1e-6; the arithmetic being compared
# is the same code either way.
CASES = {
    "main": {"epochs": 30, "patience": 1000, "threshold_chi2": 1e12, "threshold_positivity": 1e12},
    # ``stopping``: a chi2 threshold the fit can never clear, so the "how many monitored steps in a
    # row without a new best" counter is the only thing that can end the run -- the early-stop path,
    # the per-replica counters, the snapshot/restore and the truncation of the recorded history.
    # ``stopping``: a ``stopping_delta`` big enough that no step can improve on the best by that
    # much, so the "steps in a row without a new best" counter is what ends the run.  This is the
    # early-stop path: the per-replica counters, the per-replica best-epoch snapshot, the restore
    # of the best weights at the end, and a truncated history.
    "stopping": {
        "epochs": 30,
        "patience": 2,
        # Wide enough that the trajectory (which improves by ~1e6/step) can never improve on the
        # best by that much: the first step passes (it beats ``INITIAL_CHI2 = 1e9``), the rest do
        # not, so the counter runs and the fit stops early.
        "stopping_delta": 1e7,
        "threshold_chi2": 1e12,
        "threshold_positivity": 1e12,
    },
}


def problem():
    """The synthetic problem: fixed, and identical for both implementations."""
    rng = np.random.default_rng(20251004)
    xgrid = np.linspace(1e-3, 0.9, NX).reshape(1, NX, 1).astype("float64")
    kernel = rng.normal(size=(NX * NFL, NOBS)).astype("float32")
    bias = (rng.normal(size=(NOBS,)) * 0.1).astype("float32")
    raw = rng.normal(size=(NOBS, NOBS))
    covmat = (raw @ raw.T / NOBS + np.eye(NOBS) * 0.5).astype("float32")
    target = rng.normal(size=(NOBS,)).astype("float32")

    def block(idx):
        return covmat[np.ix_(idx, idx)], target[idx]

    tr_cov = np.stack([block(i)[0] for i in TRAIN_IDX])
    tr_tgt = np.stack([block(i)[1] for i in TRAIN_IDX])
    vl_cov = np.stack([block(i)[0] for i in VALID_IDX])
    vl_tgt = np.stack([block(i)[1] for i in VALID_IDX])
    full = np.arange(NOBS)
    ex_cov = np.stack([block(full)[0]] * NREP)
    ex_tgt = np.stack([block(full)[1]] * NREP)

    def full_mask(idx):
        mask = np.zeros((NREP, NOBS), dtype=bool)
        for i, points in enumerate(idx):
            mask[i, points] = True
        return mask

    return {
        "xgrid": xgrid,
        "kernel": kernel,
        "bias": bias,
        "roles": {
            "training": {
                "mask": full_mask(TRAIN_IDX),
                "invcovmat": np.linalg.inv(tr_cov).astype("float32"),
                "covmat": tr_cov.astype("float32"),
                "target": tr_tgt[np.newaxis].astype("float32"),
                "name": "LHC",
                "suffix": "",
            },
            "validation": {
                "mask": full_mask(VALID_IDX),
                "invcovmat": np.linalg.inv(vl_cov).astype("float32"),
                "covmat": vl_cov.astype("float32"),
                "target": vl_tgt[np.newaxis].astype("float32"),
                "name": "LHC_val",
                "suffix": "_val",
            },
            "experimental": {
                "mask": full_mask([full] * NREP),
                "invcovmat": np.linalg.inv(ex_cov).astype("float32"),
                "covmat": ex_cov.astype("float32"),
                "target": ex_tgt[np.newaxis].astype("float32"),
                "name": "LHC_exp",
                "suffix": "_exp",
            },
        },
        # The `reporting` list `Stopping` parses: chi2 datasets carry ndata per replica.
        "reporting": [
            {
                "name": "LHC",
                "count_chi2": True,
                "ndata": [NDATA_TR] * NREP,
                "ndata_vl": [NDATA_VL] * NREP,
                "positivity": False,
                "integrability": False,
            },
            {"name": "POS", "positivity": True, "integrability": False},
            {"name": "INT", "positivity": True, "integrability": True},
        ],
    }


def environment():
    """The environment that fixes every number this file records or compares.

    The problem is a float32 fit whose dynamics are chaotic in the initial weights (see the module
    docstring), and the initial weights are whatever the Keras initializers draw on whichever
    framework Keras currently runs on: a different keras/jax version, or a different framework
    backend, changes the random stream and with it the entire trajectory -- including the discrete
    decisions (stop epoch, best epochs) recorded here.  The fixture is therefore only meaningful
    together with the environment that produced it; ``test_optimizer_run`` gates on this dict.
    """
    import keras

    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "keras": keras.__version__,
        "keras_backend": keras.config.backend(),
    }
    try:
        import jax

        env["jax"] = jax.__version__
    except ImportError:  # a different framework runs Keras here
        pass
    return env


def _natural_key(text):
    """Sort ``section/10/x`` after ``section/2/x``: numbers compare as numbers.

    The weight maps moved from ``{section: [arrays]}`` to ``{section/index/name}`` paths (P5),
    and a lexicographic sort would interleave index 10..15 between 1 and 2 -- the order must be
    the map's own (build order) for the two serializations to flatten alike.
    """
    import re

    return tuple(
        int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", str(text))
    )


def _flatten(obj):
    """Numbers out of nested dicts/lists, in a stable order (for comparing two fixtures)."""
    if isinstance(obj, dict):
        out = []
        for key in sorted(obj, key=_natural_key):
            out.extend(_flatten(obj[key]))
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            out.extend(_flatten(item))
        return out
    return [float(obj)]


def as_json(obj):
    """Nested dicts/lists of arrays -> plain JSON (``get_replica_weights`` returns a dict)."""
    if isinstance(obj, dict):
        return {k: as_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_json(v) for v in obj]
    return np.asarray(obj, dtype="float64").tolist()


def states_of(record):
    return [
        {"step": int(step), "losses": {k: round(float(v), 10) for k, v in sorted(losses.items())}}
        for step, losses in record
    ]


def build_pdf_layers(prob):
    """The real PDF model plus the observable reduction, wired to one concrete input tensor.

    Returns ``(input_tensors, prediction, layers)`` where ``layers`` holds the pieces whose weights
    the fixture checks (the core, the observable and the three penalty multipliers).
    """
    import keras

    from n3fit.model_gen import ReplicaSettings, generate_pdf_model
    from n3fit.backends import operations as ops

    fake_fl = [
        {"fl": fl, "largex": [0, 1], "smallx": [1, 2]}
        for fl in ["u", "ubar", "d", "dbar", "c", "g", "s", "sbar"]
    ]
    replicas = [
        ReplicaSettings(nodes=[8], activations=["linear"], seed=100 + i) for i in range(NREP)
    ]
    pdf_model = generate_pdf_model(replicas, flav_info=fake_fl, fitbasis="FLAVOUR")

    # A concrete x grid (the production input has a ``None`` x axis, which would make the reduction
    # below unbuildable; the values are what matter for the fixture).
    x = keras.layers.Input(batch_size=1, shape=(NX, 1), name="pdf_input")
    x.tensor_content = ops.numpy_to_tensor(prob["xgrid"])
    input_dict, full_pdf = pdf_model.apply_as_layer({"pdf_input": x})

    flat = keras.layers.Reshape((NREP, NX * NFL), name="flatten_x_flavour")(full_pdf)
    prediction = keras.layers.Dense(
        NOBS,
        name="observable",
        kernel_initializer=keras.initializers.Constant(prob["kernel"]),
        bias_initializer=keras.initializers.Constant(prob["bias"]),
    )(flat)
    return input_dict, prediction, pdf_model


# ---------------------------------------------------------------------------------------------
# legacy: the pre-P4 loop -- loss-output models + Keras callbacks
# ---------------------------------------------------------------------------------------------
def run_legacy(case):
    """The reference: the pre-P4 loop, which exists only in a pre-P4 checkout.

    Since P4 the loop it records (``MetaModel.perform_fit`` + the three Keras callbacks + the legacy
    ``Stopping``) is *deleted from this tree*, so this function cannot run here by construction --
    the fixture is what is checked in, and regenerating it needs the older revision.  The guard
    below turns the import error into an instruction.
    """
    try:
        import n3fit.stopping as _legacy_stopping  # the marker of a pre-P4 tree

        _legacy_stopping.Stopping
    except ImportError as e:
        raise RuntimeError(
            "the legacy loop this records was deleted in P4: run this on a pre-P4 checkout "
            "(e.g. 'git stash' the P4 patch) to regenerate p4_golden.json"
        ) from e
    from n3fit.backends import MetaModel, operations as ops
    from n3fit.backends.keras_backend import callbacks as n3callbacks
    from n3fit.layers.losses import LossInvcovmat, LossIntegrability, LossPositivity
    from n3fit.layers.mask import Mask
    from n3fit.stopping import Stopping

    prob = problem()
    data = prob["roles"]
    input_dict, prediction, pdf_model = build_pdf_layers(prob)

    def role_model(role, terms):
        masked = Mask(
            bool_mask=data[role]["mask"], name=f"mask_{role}"
        )(prediction)
        layers = []
        for term in terms:
            if term == "chi2":
                layers.append(
                    LossInvcovmat(
                        ops.numpy_to_tensor(data[role]["invcovmat"]),
                        ops.numpy_to_tensor(data[role]["target"]),
                        covmat=data[role]["covmat"],
                        name=data[role]["name"],
                    )
                )
            elif term == "positivity":
                layers.append(LossPositivity(c=POS_INITIAL, name="POS"))
            elif term == "integrability":
                layers.append(LossIntegrability(c=INT_INITIAL, name="INT"))
        model = MetaModel(input_dict, [layer(masked) for layer in layers])
        model.compile(
            optimizer_name="RMSprop",
            learning_rate=case.get("learning_rate", DEFAULT_LR),
            clipnorm=1.0,
        )
        return model, layers

    models = {}
    terms = {}
    training_kinds = ["chi2", "positivity", "integrability"]
    if case.get("order") == "reversed":
        # Floor control: the *reference* implementation with the same terms summed in the opposite
        # order.  ``_default_loss`` is ``nansum(y_pred)`` over the graph's outputs, so the output
        # order is the association of a float32 sum -- this is how much the legacy loop separates
        # from itself when only that order changes.
        training_kinds = list(reversed(training_kinds))
    models["training"], terms["training"] = role_model("training", training_kinds)
    models["validation"], terms["validation"] = role_model("validation", ["chi2", "positivity"])
    models["experimental"], terms["experimental"] = role_model("experimental", ["chi2"])

    stopping = Stopping(
        models["validation"],
        prob["reporting"],
        pdf_model,
        total_epochs=case["epochs"],
        stopping_patience=case["patience"],
        stopping_delta=case.get("stopping_delta", 0.0),
        threshold_positivity=case["threshold_positivity"],
        threshold_chi2=case["threshold_chi2"],
    )

    record = []
    original_monitor = stopping.monitor_chi2  # bound before patching, else this recurses

    def monitor(training_info, epoch, print_stats=False):
        record.append((epoch, dict(training_info)))
        back = original_monitor(training_info, epoch, print_stats=print_stats)
        # The validation chi2 the stopping decides on, and the training chi2 it parses out of the
        # pre-update logs: recorded, because their *order* (pre-update training, post-update
        # validation) is what a rewrite is most likely to get wrong.
        state = stopping._history.get_state(epoch)
        # NOTE the asymmetry, which is the thing a rewrite gets wrong: the *training* numbers here
        # are the per-term scalars `correct_logs` reconstructs (summed over replicas, and over the
        # batch of 1), while the *validation* numbers are the raw per-replica arrays of
        # ``compute_losses``.
        parsed[int(epoch)] = {
            "vl_chi2": as_json(state.vl_chi2),
            "vl_loss": as_json(state.vl_loss),
        }
        return back

    stopping.monitor_chi2 = monitor
    parsed = {}

    # The raw per-term, per-replica values before any update: what the engine's ``evaluate`` must
    # return, and the reference for the scalar-summed training logs of step 0.
    initial_per_replica = as_json(models["training"].compute_losses())

    callbacks = [
        n3callbacks.StoppingCallback(stopping),
        n3callbacks.LagrangeCallback(
            ["POS", "INT"], [PUSH_MULTIPLIER, PUSH_MULTIPLIER], update_freq=LAGRANGE_PERIOD
        ),
    ]
    models["training"].perform_fit(epochs=case["epochs"], callbacks=callbacks)

    return {
        "states": states_of(record),
        "initial_per_replica": initial_per_replica,
        "parsed": parsed,
        "stop_epoch": int(stopping.stop_epoch),
        "e_best_chi2": [None if e is None else int(e) for e in stopping.e_best_chi2],
        "positivity_statuses": [str(s) for s in stopping.positivity_statuses],
        "weights": [as_json(pdf_model.get_replica_weights(i)) for i in range(NREP)],
        "multipliers": {
            "POS": as_json(terms["training"][1].get_weights()[0]),
            "INT": as_json(terms["training"][2].get_weights()[0]),
        },
    }


# ---------------------------------------------------------------------------------------------
# contract: the P4 engine -- prediction-output models, terms applied by the engine, hooks
# ---------------------------------------------------------------------------------------------
def run_contract(case):
    from n3fit.backends import get_backend, operations as ops
    from n3fit.backends.base import (
        GROUP_TRAINING,
        GROUP_VALIDATION,
        ObjectiveGroup,
        ObjectiveSpec,
        OptimizerSpec,
    )
    from n3fit.layers.mask import Mask
    from n3fit.stopping import FitRecord, LagrangeHook, StoppingHook

    prob = problem()
    data = prob["roles"]
    input_dict, prediction, pdf_model = build_pdf_layers(prob)
    backend = get_backend("keras")

    terms = {}

    def chi2_spec(name, role):
        return ObjectiveSpec(
            kind="chi2",
            name=name,
            data={
                "invcovmat": ops.numpy_to_tensor(data[role]["invcovmat"]),
                "covmat": data[role]["covmat"],
                "target": ops.numpy_to_tensor(data[role]["target"]),
            },
        )

    def role_graph(role, kinds):
        """A prediction-output graph: one named output per *prediction* (terms applied here).

        The legacy graphs had one output per *loss*; here the masked prediction each term consumes
        is an output, so the engine can apply the term itself (and P7 can get the residuals).  The
        penalties both measure the same tensor, which is why the spec says which output it wants.

        Every term gets its **own name**: the training and validation penalties are different terms
        in the legacy too (one ``LossPositivity`` layer per graph, and the Lagrange callback scaled
        only the training one), and the contract identifies a term by its name in a flat mapping.
        """
        suffix = "" if role == "training" else "_val" if role == "validation" else "_exp"
        masked = Mask(bool_mask=data[role]["mask"], name=f"mask_{role}")(prediction)
        outputs = {}
        for kind in kinds:
            if kind == "chi2":
                name = data[role]["name"]
                spec = chi2_spec(name, role)
            elif kind == "positivity":
                name = f"POS{suffix}"
                spec = ObjectiveSpec(
                    kind="positivity",
                    name=name,
                    options={"multiplier": POS_INITIAL},
                    prediction=name,
                )
            else:
                name = f"INT{suffix}"
                spec = ObjectiveSpec(
                    kind="integrability",
                    name=name,
                    options={"multiplier": INT_INITIAL},
                    prediction=name,
                )
            terms[name] = backend.objective(spec)
            outputs[name] = masked
        # ``Identity`` only to give every output its own tensor *and* its own name: numerics and
        # gradients are untouched (the engine needs names to bind term -> prediction).
        named = {name: Identity(name=name)(tensor) for name, tensor in outputs.items()}
        graph = MetaModel(input_dict, list(named.values()))
        return graph, list(named)

    from keras.layers import Identity

    from n3fit.backends import MetaModel

    graphs = {}
    graphs["training"], _ = role_graph("training", ["chi2", "positivity", "integrability"])
    graphs["validation"], _ = role_graph("validation", ["chi2", "positivity"])
    graphs["experimental"], _ = role_graph("experimental", ["chi2"])

    group_map = {
        GROUP_TRAINING: (data["training"]["name"], "POS", "INT"),
        GROUP_VALIDATION: (data["validation"]["name"], "POS_val"),
        "experimental": (data["experimental"]["name"],),
    }
    if case.get("order") == "reversed":
        # Control: the *same* engine, the same terms, summed in the opposite order.  The compiled
        # loss adds the per-output losses up in the order the terms are listed, and float32 addition
        # is not associative, so this is the fixture's floor for "any two orderings of this fit".
        group_map = {key: tuple(reversed(value)) for key, value in group_map.items()}
    groups = ObjectiveGroup(group_map)
    if case.get("perturb"):
        # One-ulp relative perturbation of one initial weight: the control for the trajectory
        # comparison (how much does *any* implementation change over 30 steps of this problem?).
        weight = pdf_model.trainable_weights[0]
        weight.assign(np.asarray(weight) * (1.0 + float(case["perturb"])))

    ensemble = backend.ensemble(graphs, weights_graph=pdf_model)

    record = FitRecord()
    stopping = StoppingHook(
        record,
        ensemble,
        ndata={"LHC": [NDATA_TR] * NREP},
        vl_ndata={"LHC_val": [NDATA_VL] * NREP},
        # The positivity check reads the *validation* graph's penalty (legacy
        # ``Positivity.__call__`` -> ``fitstate.validation``), so it is named by that graph: the
        # contract identifies a term by its name, and the legacy had one POS layer per graph.
        positivity_terms=["POS_val"],
        total_steps=case["epochs"],
        stopping_patience=case["patience"],
        stopping_delta=case.get("stopping_delta", 0.0),
        threshold_chi2=case["threshold_chi2"],
        threshold_positivity=case["threshold_positivity"],
        monitor_every=1,
    )
    lagrange = LagrangeHook(
        {"POS": terms["POS"], "INT": terms["INT"]},
        {"POS": PUSH_MULTIPLIER, "INT": PUSH_MULTIPLIER},
        period=LAGRANGE_PERIOD,
    )

    optimizer = backend.optimizer(
        OptimizerSpec(
            "RMSprop", {"learning_rate": case.get("learning_rate", DEFAULT_LR), "clipnorm": 1.0}
        )
    )
    result = optimizer.run(
        ensemble,
        terms,
        groups.terms,
        steps=case["epochs"],
        monitor_every=1,
        hooks=[stopping, lagrange],
    )

    states = []
    parsed = {}
    for index, fitstep in enumerate(record.steps):
        # The engine counts updates from 1 and keeps per-replica values per term; the legacy logs
        # carried one scalar per term (the sum over replicas), which is what `correct_logs` gave.
        losses = {
            name: float(np.sum(np.asarray(value))) for name, value in fitstep.tr_terms.items()
        }
        losses["loss"] = float(sum(losses.values()))
        states.append(
            {"step": index, "losses": {k: round(v, 10) for k, v in sorted(losses.items())}}
        )
        parsed[str(index)] = {
            "vl_chi2": as_json(fitstep.vl_chi2),
            "vl_loss": as_json(fitstep.vl_loss),
        }

    return {
        "states": states,
        "parsed": parsed,
        "stop_epoch": stopping.stop_epoch,
        "e_best_chi2": [None if e is None else int(e) for e in record.e_best_chi2],
        "positivity_statuses": [str(s) for s in record.positivity_statuses],
        "weights": [as_json(p) for p in result.parameters],
        "multipliers": {
            "POS": as_json(terms["POS"].scalar("multiplier")),
            "INT": as_json(terms["INT"].scalar("multiplier")),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["legacy", "contract"])
    parser.add_argument("--out", help="write the fixture here (legacy)")
    parser.add_argument("--fixture", help="the recorded fixture to compare against (contract)")
    args = parser.parse_args()

    if args.mode == "legacy":
        fixture = {
            "env": environment(),
            "cases": {name: run_legacy(case) for name, case in CASES.items()},
        }
        text = json.dumps(fixture, indent=1, sort_keys=True)
        if args.out:
            with open(args.out, "w") as handle:
                handle.write(text + "\n")
            print(f"wrote {args.out}")
        else:
            print(text)
        return 0

    if args.mode == "contract":
        fixture = json.load(open(args.fixture))
        if (recorded := fixture.get("env")) != (current := environment()):
            print("WARNING: environment differs from the one that recorded the fixture")
            for key in sorted(set(recorded) | set(current)):
                want, got = recorded.get(key), current.get(key)
                if want != got:
                    print(f"  {key}: fixture={want} current={got}")
            print("  numbers are only comparable in the recorded environment")
        contract = {name: run_contract(case) for name, case in CASES.items()}
        np.set_printoptions(precision=8)
        failures = 0
        for name in CASES:
            want, got = fixture["cases"][name], contract[name]
            print(f"=== {name} ===")
            for key in ("stop_epoch",):
                ok = want[key] == got[key]
                failures += not ok
                verdict = "OK" if ok else "MISMATCH"
                print(f"  {key}: fixture={want[key]} contract={got[key]} {verdict}")
            for key in ("e_best_chi2", "positivity_statuses"):
                ok = want[key] == got[key]
                failures += not ok
                verdict = "OK" if ok else "MISMATCH"
                print(f"  {key}: fixture={want[key]} contract={got[key]} {verdict}")
            for key in ("multipliers", "weights"):
                flat_want = np.array(_flatten(want[key]))
                flat_got = np.array(_flatten(got[key]))
                # ``weights`` carries the accumulated float32 trajectory, so it is compared with a
                # looser tolerance than the scalars.  Measured (lr = 1e-3, 30 steps): median
                # 1.6e-6 relative, 13/80 entries over 0.1 tolerance units, worst entry 1.6e-4
                # relative.  The gap per *step* is a single float32 ulp in one weight; this toy's
                # dynamics stretch it (``p4_probe_floor.py``).  Nothing that decides anything
                # (stop epoch, best epochs, statuses, multipliers, per-term losses) is in this key.
                tol = 1e-3 if key == "weights" else 1e-4
                ok = flat_want.shape == flat_got.shape and np.allclose(
                    flat_want, flat_got, rtol=tol, atol=1e-6
                )
                failures += not ok
                worst = np.max(
                    np.abs(flat_want - flat_got) / np.maximum(np.abs(flat_want), 1e-30)
                )
                verdict = "OK" if ok else "MISMATCH"
                print(f"  {key}: max rel diff={worst:.3e} {verdict}")
            for key in ("vl_chi2", "vl_loss"):
                steps = sorted(want["parsed"], key=int)
                want_values = np.array([want["parsed"][step][key] for step in steps])
                got_values = np.array([got["parsed"][step][key] for step in steps])
                ok = want_values.shape == got_values.shape and np.allclose(
                    want_values, got_values, rtol=1e-4, atol=1e-6
                )
                failures += not ok
                worst = np.max(
                    np.abs(want_values - got_values) / np.maximum(np.abs(want_values), 1e-30)
                )
                verdict = "OK" if ok else "MISMATCH"
                print(f"  {key}: shape {want_values.shape} rel diff={worst:.3e} {verdict}")
            want_logs = np.array([_flatten(step["losses"]) for step in want["states"]])
            got_logs = np.array([_flatten(step["losses"]) for step in got["states"]])
            ok = want_logs.shape == got_logs.shape and np.allclose(
                want_logs, got_logs, rtol=1e-4, atol=1e-6
            )
            failures += not ok
            if ok:
                print("  training logs (per-step scalars): OK")
            else:
                diff = np.abs(want_logs - got_logs) / np.maximum(np.abs(want_logs), 1e-30)
                first = int(np.argmax(diff > 1e-4)) // want_logs.shape[1]
                print(
                    f"  training logs: max rel diff={diff.max():.3e}, first bad step={first}"
                    f" (fixture={want_logs[first]}, contract={got_logs[first]})"
                )
            n_steps_want, n_steps_got = len(want["states"]), len(got["states"])
            ok = n_steps_want == n_steps_got
            failures += not ok
            verdict = "OK" if ok else "MISMATCH"
            print(f"  monitored steps: fixture={n_steps_want} contract={n_steps_got} {verdict}")
        print("RESULT:", "OK" if not failures else f"{failures} MISMATCHES")
        return 0 if not failures else 1

    raise SystemExit("unknown mode")


if __name__ == "__main__":
    sys.exit(main())
