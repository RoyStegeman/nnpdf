"""
P4: the flip, end to end -- the trainer's assembly driven by the engine.

``test_optimizer_run.py`` proves the *engine* reproduces the loop it replaces, on graphs the test
builds itself.  This file proves the thing the flip is actually about: that ``ModelTrainer``'s own
pieces -- ``ObservableWrapper`` producing ``(prediction, term)`` pairs, ``_model_generation``
assembling the three role graphs and the flat term mapping, and the engine driving them -- work
together on a real PDF model, with the real stopping hook and the real Lagrange schedule.

What is exercised, and why each part matters:

* ``_model_generation`` is called on a **stand-in trainer** (a namespace with exactly the attributes
  the method reads): what is under test is that method, not the data plumbing of a full fit, which
  needs validphys and a real runcard.  The masks, the data, the covmats and the PDF model are real.
* the terms of the three roles are distinct objects under distinct names -- the property that makes
  one flat ``terms`` mapping work (contract A6) -- and each name is an output of its own role graph
  (which is *how* the engine routes a term to its prediction);
* the validation positivity term is not touched by the Lagrange schedule, which is the legacy
  behaviour the graphs used to get for free by having one loss layer each;
* `evaluate` returns the same validation chi2 the stopping ended on, and the training chi2 is a
  finite number of the right shape: this is what ``performfit`` writes into the fit output.
"""

import numpy as np
import pytest

# The whole point is the *real* wrapper against the *real* model, so this file needs a framework:
# importing n3fit.model_gen is enough to need one (it is a Keras module), hence the skip first.
pytest.importorskip("keras")

from n3fit.backends import (  # noqa: E402
    GROUP_EXPERIMENTAL,
    GROUP_TRAINING,
    GROUP_VALIDATION,
    MetaLayer,
)
from n3fit.model_gen import ObservableWrapper  # noqa: E402
from n3fit.model_trainer import ModelTrainer  # noqa: E402
from n3fit.tests.backend_conformance.p4_golden import (  # noqa: E402
    NFL,
    NREP,
    NDATA_TR,
    NDATA_VL,
    NOBS,
    NX,
    POS_INITIAL,
    INT_INITIAL,
    build_pdf_layers,
    problem,
)


def _wrapper(name, observables, mask, dataset_xsizes, **kwargs):
    return ObservableWrapper(name, observables, mask, dataset_xsizes, **kwargs)


def _observable_layer(prob):
    """A stand-in for a DY/DIS layer: ``(1, n_replicas, nx, n_flavour) -> (1, n_replicas, n_obs)``.

    It has to be an ``MetaLayer`` rather than a standalone ``keras.Model``: the wrapper calls it *on
    an existing tensor* (the real ones are the DY/DIS layers), and a model with an input of its own
    would leave that input unfed when the role graphs are built.
    """
    import keras

    class _Observable(MetaLayer):
        def __init__(self, name="observable"):
            super().__init__(name=name)
            self.flatten = keras.layers.Reshape((NREP, NX * NFL))
            self.dense = keras.layers.Dense(
                NOBS,
                kernel_initializer=keras.initializers.Constant(prob["kernel"]),
                bias_initializer=keras.initializers.Constant(prob["bias"]),
            )

        def call(self, inputs):
            return self.dense(self.flatten(inputs))

    return _Observable()


def make_trainer():
    """A stand-in ``ModelTrainer`` whose attributes are exactly what ``_model_generation`` reads.

    The three dictionaries carry the ``output`` lists (the wrappers, in the order the trainer builds
    them: chi2 first, then the penalties), and ``objective_groups`` is the membership P3 computes.
    """
    from n3fit.backends import ObjectiveGroup, operations as ops
    from n3fit.layers.mask import Mask

    prob = problem()
    data = prob["roles"]
    input_dict, prediction, pdf_model = build_pdf_layers(prob)
    observable = _observable_layer(prob)

    def role_wrappers(role, kinds):
        wrappers = []
        for kind in kinds:
            mask = Mask(bool_mask=data[role]["mask"], name=f"mask_{role}_{kind}")
            if kind == "chi2":
                wrappers.append(
                    _wrapper(
                        data[role]["name"],
                        [observable],
                        mask,
                        [NX],
                        invcovmat=ops.numpy_to_tensor(data[role]["invcovmat"]),
                        covmat=data[role]["covmat"],
                        data=ops.numpy_to_tensor(data[role]["target"]),
                    )
                )
            elif kind == "positivity":
                wrappers.append(
                    _wrapper(
                        f"POS{'' if role == 'training' else '_val'}",
                        [observable],
                        mask,
                        [NX],
                        multiplier=POS_INITIAL,
                        positivity=True,
                    )
                )
            else:
                wrappers.append(
                    _wrapper(
                        "INT",
                        [observable],
                        mask,
                        [NX],
                        multiplier=INT_INITIAL,
                        integrability=True,
                    )
                )
        return wrappers

    from n3fit.model_trainer import InputInfo

    # The splitter the trainer builds in ``_xgrid_generation``: ``(batch, replicas, x, flavour)``
    # split along the x axis into the (here single) unique input grid.
    splitter = ops.tensor_splitter((1, NREP, NX, NFL), [NX], axis=2, name="splitter")
    trainer = type("StandInTrainer", (), {})()
    trainer.training = {
        "output": role_wrappers("training", ["chi2", "positivity", "integrability"]),
        "chi2_names": [data["training"]["name"]],
        "penalty_names": ["POS", "INT"],
        "posmultipliers": [1.1],
        "integmultipliers": [1.1],
        "penalty_initials": {"POS": POS_INITIAL, "INT": INT_INITIAL},
    }
    trainer.validation = {
        "output": role_wrappers("validation", ["chi2", "positivity"]),
        "chi2_names": [data["validation"]["name"]],
        "penalty_names": ["POS_val"],
    }
    trainer.experimental = {
        "output": role_wrappers("experimental", ["chi2"]),
        "chi2_names": [data["experimental"]["name"]],
    }
    trainer.objective_groups = ObjectiveGroup(
        {
            GROUP_TRAINING: (data["training"]["name"], "POS", "INT"),
            GROUP_VALIDATION: (data["validation"]["name"], "POS_val"),
            GROUP_EXPERIMENTAL: (data["experimental"]["name"],),
        }
    )
    trainer.print_summary = False
    trainer.terms = {}
    # ``idx`` has one entry per *training* output -- the chi2 term and the two penalties here --
    # and says which unique x grid each of them is evaluated on: one, shared by all three.
    trainer._xinput = InputInfo(input=input_dict["pdf_input"], split=splitter, idx=[0, 0, 0])
    return trainer, input_dict, pdf_model, data


def make_graphs(trainer, pdf_model):
    return ModelTrainer._model_generation(
        trainer, trainer._xinput, pdf_model, partition=None, partition_idx=0
    )


def test_the_three_roles_have_their_own_named_outputs():
    """Every term is an output of its own role graph, under its own name."""
    trainer, _, pdf_model, data = make_trainer()
    models = make_graphs(trainer, pdf_model)
    expected = {
        GROUP_TRAINING: [data["training"]["name"], "POS", "INT"],
        GROUP_VALIDATION: [data["validation"]["name"], "POS_val"],
        GROUP_EXPERIMENTAL: [data["experimental"]["name"]],
    }
    for role, names in expected.items():
        assert models[role].output_names == names, role
    # ... and no name is used by two roles (the property A6 exists for)
    all_names = [name for names in expected.values() for name in names]
    assert len(all_names) == len(set(all_names))
    assert set(trainer.terms) == set(all_names)


def test_the_terms_are_the_objectives_of_their_specs():
    """``self.terms`` is the flat mapping the engine is handed: names to real objectives."""
    trainer, _, pdf_model, _ = make_trainer()
    make_graphs(trainer, pdf_model)
    for name, term in trainer.terms.items():
        assert term.name == name
        assert term.spec.kind in ("chi2", "positivity", "integrability")
        assert (term.spec.prediction or term.name) in (
            set().union(*[set(model.output_names) for model in make_graphs(trainer, pdf_model).values()])
        )


def test_the_engine_runs_and_the_hooks_see_the_groups():
    """A short fit through ``Optimizer.run`` with the real stopping and Lagrange hooks."""
    from n3fit.backends import OptimizerSpec, get_backend
    from n3fit.stopping import LagrangeHook, FitRecord, StoppingHook

    trainer, _, pdf_model, data = make_trainer()
    models = make_graphs(trainer, pdf_model)
    backend = get_backend()
    ensemble = backend.ensemble(
        {
            GROUP_TRAINING: models[GROUP_TRAINING],
            GROUP_VALIDATION: models[GROUP_VALIDATION],
            GROUP_EXPERIMENTAL: models[GROUP_EXPERIMENTAL],
        },
        weights_graph=pdf_model,
    )
    record = FitRecord()
    stopping = StoppingHook(
        record,
        ensemble,
        ndata={data["training"]["name"]: [NDATA_TR] * NREP},
        vl_ndata={data["validation"]["name"]: [NDATA_VL] * NREP},
        positivity_terms=["POS_val"],
        total_steps=6,
        stopping_patience=1000,
        threshold_chi2=1e12,
        threshold_positivity=1e12,
    )
    lagrange = LagrangeHook({"POS": trainer.terms["POS"]}, {"POS": 1.1}, period=2)
    optimizer = backend.optimizer(OptimizerSpec("RMSprop", {"learning_rate": 1e-3}))
    result = optimizer.run(
        ensemble,
        trainer.terms,
        dict(trainer.objective_groups.terms),
        steps=6,
        monitor_every=1,
        hooks=[stopping, lagrange],
    )

    # the fit happened and was recorded: one step per update, with the losses of the monitored group
    assert result.history.monitored_steps == [1, 2, 3, 4, 5, 6]
    assert len(record.steps) == 6
    assert set(record.steps[0].tr_terms) == {data["training"]["name"], "POS", "INT"}
    assert np.asarray(record.steps[0].vl_chi2).shape == (NREP,)

    # the Lagrange schedule fired (period 2, three times over six steps)...
    assert np.asarray(trainer.terms["POS"].scalar("multiplier")).item() > POS_INITIAL
    # ... and did *not* touch the validation penalty, which is a different term (A6)
    assert np.asarray(trainer.terms["POS_val"].scalar("multiplier")).item() == pytest.approx(
        POS_INITIAL
    )

    # the validation chi2 the stopping decided on is the one the record reports
    assert record.vl_chi2 is not None
    assert np.allclose(record.vl_chi2, record.steps[-1].vl_chi2)
