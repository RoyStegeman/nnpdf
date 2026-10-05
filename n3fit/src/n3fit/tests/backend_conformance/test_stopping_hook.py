"""
P4: the stopping hook's decisions, and the record the fit reports afterwards.

No framework is involved.  The hook is arithmetic over the engine's ``StepContext`` -- it decides on
numbers the engine hands it and it never sees a model -- so it is tested here against the numpy
double, with a scripted history.  Two of these tests are regressions for bugs the P4 golden fixture
found in the first version of the hook; both are invisible unless a fit stops early *and* never
improves, which is why the fixture has a case that does exactly that:

* ``e_best_chi2`` must fall back to the epoch a replica stopped at when it never improved -- the
  legacy ``Stopping.e_best_chi2`` is "best, or last", and ``io/writer.py`` writes the list into the
  replica summary;
* the fallback is an *epoch*, so what the hook records is zero-based, like the legacy ``epoch``
  (one-based numbers appear only in ``stop_epoch`` and in ``FitStep.step``).
"""

import numpy as np
import pytest

from n3fit.backends.base import GROUP_VALIDATION
from n3fit.stopping import POS_BAD, POS_OK, FitRecord, StoppingHook
from n3fit.tests.backend_conformance.testing_backends import NumpyEnsemble, NumpyModelView

TOTAL_STEPS = 10


def weight_list(mappings):
    """Per-replica ``{path: list}`` -- plain Python, so it can be compared with ``==``."""
    return [
        {path: np.asarray(value).tolist() for path, value in mapping.items()}
        for mapping in mappings
    ]


def make_ensemble(n_replicas=2):
    """A two-replica ensemble whose weights identify the replica (so a restore is visible)."""
    models = [
        NumpyModelView({"core": {"weights": {"w": np.full(3, float(1 + replica))}}})
        for replica in range(n_replicas)
    ]
    return NumpyEnsemble(models)


class FakeContext:
    """The parts of ``StepContext`` that a stopping decision reads."""

    def __init__(self, step, logs, validation):
        self.step = step
        self.logs = logs
        self._validation = validation
        self.stopped = False

    def evaluate(self, group):
        assert group == GROUP_VALIDATION, "the hook decides on the validation group"
        return self._validation

    def stop(self):
        self.stopped = True


def make_hook(ensemble, **kwargs):
    settings = {
        "ndata": {"LHC": [4, 4]},
        "vl_ndata": {"LHC_val": [2, 2]},
        "positivity_terms": ["POS_val"],
        "total_steps": TOTAL_STEPS,
        "stopping_patience": 2,
        "threshold_chi2": 1e12,
        "threshold_positivity": 1e12,
    }
    settings.update(kwargs)
    return StoppingHook(FitRecord(), ensemble, **settings)


def validation(chi2=100.0, positivity=1.0):
    """The engine's ``{term: per-replica array}`` for the validation group."""
    return {
        "LHC_val": np.array([chi2, chi2]),
        "POS_val": np.array([positivity, positivity]),
    }


def test_a_fit_that_never_improves_reports_the_last_epoch():
    """No step ever beats the initial chi2: the best epoch is the fallback, not ``None``."""
    hook = make_hook(make_ensemble(), threshold_chi2=1.0)  # the counter can never start
    for step in range(1, TOTAL_STEPS + 1):
        ctx = FakeContext(step, {"LHC": np.array([1.0, 1.0])}, validation())
        hook.on_monitored_step(ctx)
        assert not ctx.stopped, "a fit that never passes keeps going until the budget runs out"
    hook.on_train_end()
    assert hook.record.e_best_chi2 == [TOTAL_STEPS - 1] * 2
    assert hook.record.positivity_statuses == [POS_BAD] * 2


def test_an_early_stop_keeps_the_epochs_zero_based_and_restores_the_best_weights():
    """One step beats the initial chi2, the rest do not: stop, and give the weights back."""
    ensemble = make_ensemble()
    hook = make_hook(ensemble)
    snapshot = ensemble.weights().copy()
    for step in range(1, TOTAL_STEPS + 1):
        ctx = FakeContext(step, {"LHC": np.array([1.0, 1.0])}, validation())
        hook.on_monitored_step(ctx)
        if ctx.stopped:
            break
    else:  # pragma: no cover - the fixture would be wrong, not the hook
        raise AssertionError("the fit should have stopped early")

    assert ctx.step == 4, "one pass, then patience + 1 steps without a new best"
    assert hook.record.stop_epoch == 4  # one-based: the number of updates taken
    assert hook.record.stop_epochs == [3, 3]  # zero-based: the epoch that completed
    assert hook.record.e_best_chi2 == [0, 0]  # the best epoch, which is what a consumer wants
    assert hook.record.positivity_statuses == [POS_OK, POS_OK]
    assert len(hook.record.steps) == 4

    # ``on_train_end`` is what puts the best weights back, and the ones to put back are the ones
    # from step 0 -- before the first update, since that is the step that was never beaten.
    for replica in range(len(ensemble)):
        ensemble[replica].sections["core"]["weights"]["w"] = np.zeros(3)
    hook.on_train_end()
    assert weight_list(ensemble.weights()) == weight_list(snapshot)
    assert np.asarray(hook.record.steps[0].tr_terms["LHC"]).shape == (2,)


def test_the_positivity_terms_are_names_of_the_evaluated_group():
    """A name that the validation group does not carry is a wiring error, and says so."""
    hook = make_hook(make_ensemble(), positivity_terms=["POS"])
    ctx = FakeContext(1, {"LHC": np.array([1.0, 1.0])}, validation())
    with pytest.raises(KeyError, match="not in the evaluated group"):
        hook.on_monitored_step(ctx)
