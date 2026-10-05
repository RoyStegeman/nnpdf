"""
Stopping the fit: the criterion, the penalties and the record (P4).

The criterion itself is unchanged from the loop this replaces and never left n3fit: ``StoppingHook``
takes the values the engine computed for one monitored step, decides whether the fit should stop,
remembers the best weights per replica and restores them at the end.  What changed is *where the
numbers come from*: the hook is handed ``{term name: per-replica array}`` for a group
(``StepContext.logs`` for the training objective, ``ctx.evaluate(...)`` for the validation one)
instead of parsing a Keras history dictionary by suffix (D11).

The legacy ``Stopping``/``FitState``/``FitHistory``/``Positivity`` classes and their
``{name}_loss``/``val_loss`` parsers are gone: ``MetaModel.perform_fit`` and the three Keras
callbacks -- the loop they belonged to -- were replaced by ``Optimizer.run`` plus the hooks below,
and the hooks are what this module now owns end to end.
"""

import logging
import math
from dataclasses import dataclass
from time import time

import numpy as np

from n3fit.backends.base import GROUP_VALIDATION

log = logging.getLogger(__name__)

# Put a very big number here so that we for sure discard this run
# AND we have a clear marker that something went wrong, not just a bad fit
TERRIBLE_CHI2 = 1e10
INITIAL_CHI2 = 1e9

# Pass/veto keys
POS_OK = "POS_PASS"
POS_BAD = "POS_VETO"
THRESHOLD_POS = 1e-6


# --------------------------------------------------------------------------------------------
# P4: the stopping rule, as a hook
# --------------------------------------------------------------------------------------------
# The arithmetic below is the legacy ``Stopping.monitor_chi2``, step for step; what changed is what
# it is computed *on*:
#
# * the losses arrive as a ``{term name: per-replica array}`` mapping computed by the engine, so
#   there is no ``f"{name}_loss"`` key to construct and no ``val_loss``/``loss`` suffix to guess
#   (D11).  The names the hook watches are handed to it by n3fit -- the same lists P3 built for the
#   objective groups -- instead of being re-derived from the dataset names;
# * the validation evaluation is an explicit ``ctx.evaluate(...)`` request, so the hook decides when
#   a forward pass happens instead of the engine paying for one on every step (the reason the
#   legacy path forced ``compute_losses()`` every step under jax);
# * the best weights are snapshotted through ``ctx.weights`` / ``Ensemble.set_weights``, so the
#   snapshot is a numpy mapping of weight paths and not a list of Keras tensors.


@dataclass
class FitStep:
    """One monitored step of a fit: what the stopping decided on, and what it looked like."""

    step: int
    """The engine's step: the number of optimizer updates taken so far (counts from 1).

    The legacy ``epoch`` is ``step - 1`` -- it counted completed updates from zero -- and the
    reporting names that read like epochs (``stop_epoch``, ``e_best_chi2``, the keys of
    ``chi2exps_json``) keep the legacy convention so the output files do not change.
    """

    tr_loss: float
    tr_terms: dict
    tr_ndata: tuple
    vl_loss: "np.ndarray"
    vl_chi2: "np.ndarray"
    vl_partial: dict

    @property
    def chi2(self):
        """The training chi2: the same arithmetic the reported one uses, on this step's terms."""
        return _parse_chi2(self.tr_terms, self.tr_ndata)[0]


class FitRecord:
    """The record of a fit, filled by :class:`StoppingHook`.

    The two read-side methods the reporting needs and cannot recompute from the record alone are
    the training chi2 (:meth:`evaluate_training`, which is one ``evaluate`` of the engine) and the
    validation chi2 of the last step (:attr:`vl_chi2`); both are attached when the fit is
    assembled by ``model_trainer``.

    This is what the fit reports afterwards, and it deliberately keeps the names the consumers
    already use (``stop_epoch``, ``e_best_chi2``, ``positivity_statuses``, ``chi2exps_json()``), so
    ``performfit.py`` and ``io/writer.py`` -- and the on-disk formats they write -- do not change.
    Unlike the legacy ``FitState``/``FitHistory`` pair it stores *parsed* numbers: there is no loss
    key convention left to parse (D11).
    """

    def __init__(self):
        self.steps: list[FitStep] = []
        self.final_step = None
        self._training_evaluation = None
        #: The two things a consumer cannot recompute from the recorded steps: the training chi2
        #: of the fitted model (one engine evaluation, attached by ``model_trainer``) and the
        #: validation chi2 the fit ended on.  Both are properties of the *fit*, not of a step.
        self.vl_chi2 = None
        self.stopping_patience = 0
        self.total_steps = 0
        self.positivity_statuses = []
        self.multipliers = {}
        # Filled (by reference) by the stopping hook, which owns the decisions: the per-replica
        # best epoch so far, and the epoch each replica stopped at.
        self.best_epochs = []
        self.stop_epochs = []

    @property
    def e_best_chi2(self):
        """Best epoch per replica -- or the epoch the replica stopped at, when it never improved.

        The fallback is not cosmetic: the legacy ``Stopping.e_best_chi2`` is "best, or last", and
        ``io/writer.py`` writes this list into the replica summary.
        """
        return [
            best if best is not None else last
            for best, last in zip(self.best_epochs, self.stop_epochs)
        ]

    def register(self, fitstep):
        self.final_step = fitstep.step
        self.steps.append(fitstep)

    @property
    def stop_epoch(self):
        """The step the fit stopped at, one-based -- the legacy ``Stopping.stop_epoch`` (-1 if the
        fit recorded nothing)."""
        return -1 if self.final_step is None else self.final_step

    def evaluate_training(self):
        """The training chi2 of the fitted model (legacy ``Stopping.evaluate_training``).

        It cannot be read off the recorded steps: the numbers in them are the ones that *informed*
        each update, not the ones the final weights produce.  ``model_trainer`` sets this to a
        callable of no arguments -- one ``Optimizer.evaluate`` of the training group -- so that a
        consumer keeps calling ``record.evaluate_training()`` as it did before.
        """
        if self._training_evaluation is None:
            raise RuntimeError("this record has no training evaluation attached")
        return self._training_evaluation()

    def get(self, index):
        """The ``index``-th registered step (0-based, in monitored-step order)."""
        try:
            return self.steps[index]
        except IndexError as e:
            raise ValueError(
                f"only {len(self.steps)} monitored steps were recorded, asked for {index}"
            ) from e

    def chi2exps_json(self, i_replica=0, log_each=100):
        """The ``chi2exps.log`` payload: every ``log_each`` monitored steps, per replica.

        Same keys and same values as the legacy ``Stopping.chi2exps_json``: the key of a step is
        the legacy *epoch* (zero-based, so the 100th monitored step is ``99``), the training loss
        is the scalar the step was decided on, and the validation loss and per-experiment chi2 are
        per replica.  ``io/writer.py`` writes this straight to disk.
        """
        json_dict = {}
        for index in range(log_each - 1, len(self.steps), log_each):
            fitstep = self.steps[index]
            json_dict[fitstep.step - 1] = {
                "training_loss": fitstep.tr_loss,
                "validation_loss": np.asarray(fitstep.vl_loss).tolist(),
                "validation_chi2s": {
                    k: np.take(v, i_replica) for k, v in fitstep.vl_partial.items()
                },
            }
        return json_dict


class PositivityCheck:
    """Whether the penalty values are below the threshold -- the legacy :class:`Positivity`."""

    def __init__(self, threshold, positivity_terms):
        self.threshold = threshold
        self.positivity_terms = list(positivity_terms)

    @property
    def positivity_sets(self):
        """The terms checked (legacy name: the *datasets* whose loss was checked)."""
        return self.positivity_terms

    def __call__(self, values):
        """``values`` is the engine's ``{term: per-replica array}`` for the validation group."""
        passed = True
        for name in self.positivity_terms:
            if name not in values:
                raise KeyError(
                    f"the positivity term {name!r} is not in the evaluated group "
                    f"{sorted(values)}: the names handed to the hook must be the ones the "
                    f"validation group uses (a penalty that exists in several role graphs is named "
                    f"per role, e.g. 'POS' and 'POS_val')"
                )
            passed &= values[name] < self.threshold
        return np.array(passed)


class StoppingHook:
    """Cross-validation stopping, driven by monitored steps (replaces ``Stopping``).

    The arithmetic is the one in ``Stopping.monitor_chi2``, step for step -- the thresholds, the
    two-part improvement test, the per-replica counters, the best-weight snapshot and the restore
    on stop.  The one deliberate change is the unit of ``patience``: it is given in *optimizer
    steps* and converted to monitored steps with ``ceil(patience / monitor_every)``, so choosing a
    larger ``monitor_every`` samples the chi2 history more coarsely without changing when a fit
    stops (contract §1.3).

    Parameters
    ----------
        record: FitRecord
            filled as the fit goes, and read afterwards.
        ensemble: Ensemble
            the source/target of the best-weight snapshots (per replica).
        ndata: dict
            ``{term name: n_points}`` for the training chi2 terms.
        vl_ndata: dict
            the same for the validation chi2 terms; ``None`` when there is no validation set, in
            which case the training terms are watched instead (the legacy behaviour).
        positivity_terms: list
            names of the terms whose value must stay below ``threshold_positivity``.
        total_steps: int
            the fit's budget, in optimizer steps.
        stopping_patience: int
            in optimizer steps (7000 by default, as in the legacy code).
        monitor_every: int
            steps between monitored steps; used only to convert the patience.
    """

    def __init__(
        self,
        record,
        ensemble,
        ndata,
        vl_ndata=None,
        positivity_terms=(),
        total_steps=0,
        stopping_patience=7000,
        stopping_delta=0.0,
        threshold_chi2=10.0,
        threshold_positivity=THRESHOLD_POS,
        dont_stop=False,
        monitor_every=1,
    ):
        self.record = record
        self._ensemble = ensemble
        self.tr_ndata = dict(ndata)
        self.vl_ndata = None if vl_ndata is None else dict(vl_ndata)
        if self.vl_ndata is None:
            # no validation data: watch the training set (legacy: vl_ndata points at tr_ndata)
            self.vl_ndata = dict(self.tr_ndata)
        self._positivity = PositivityCheck(threshold_positivity, positivity_terms)
        self._threshold_chi2 = threshold_chi2
        self._stopping_delta = stopping_delta
        self._dont_stop = dont_stop
        self.total_steps = total_steps

        n_replicas = len(ensemble)
        self._n_replicas = n_replicas
        self._stopping_degrees = np.zeros(n_replicas, dtype=int)
        self._counts = np.zeros(n_replicas, dtype=int)
        self._dont_stop_me_now = np.ones(n_replicas, dtype=bool)
        self._stop_epochs = [max(total_steps - 1, 0)] * n_replicas
        self._best_epochs = [None] * n_replicas
        self._best_weights = [None] * n_replicas
        self._best_val_chi2s = [INITIAL_CHI2] * n_replicas
        self._stop_now = False
        self.stopping_patience = int(math.ceil(stopping_patience / max(monitor_every, 1)))
        record.best_epochs = self._best_epochs
        record.stop_epochs = self._stop_epochs
        record.positivity_statuses = [POS_BAD] * n_replicas
        # what the penalties (``patience``) and the reporting read off the fit as a whole
        record.stopping_patience = self.stopping_patience
        record.total_steps = total_steps

    # -- the record's view of the hook's state ------------------------------------------------
    @property
    def stop_epoch(self):
        """Step in which the fit is stopped (the record owns the number; legacy: ``stop_epoch``)."""
        return self.record.stop_epoch

    @property
    def positivity_status(self):
        return self.record.positivity_statuses

    def stop_here(self):
        return False if self._dont_stop else self._stop_now

    def make_stop(self):
        """Stop, and put the best weights back (legacy ``make_stop``)."""
        self._stop_now = True
        self._restore_best_weights()

    def _restore_best_weights(self):
        weights = self._ensemble.weights()
        for i_replica, best in enumerate(self._best_weights):
            if best is not None:
                weights[i_replica] = best
        self._ensemble.set_weights(weights)

    # -- the loop's entry points --------------------------------------------------------------
    def on_monitored_step(self, ctx):
        """Decide whether to stop, exactly as ``Stopping.monitor_chi2`` did."""
        training_loss = _total(ctx.logs)
        if np.isnan(training_loss) or not np.isfinite(training_loss):
            log.warning(" > NaN found, stopping activated")
            ctx.stop()
            return

        validation = ctx.evaluate(GROUP_VALIDATION)
        vl_loss = _sum_terms(validation, list(validation))
        vl_chi2, vl_partial = _parse_chi2(validation, self.vl_ndata)

        fitstep = FitStep(
            step=ctx.step,
            tr_loss=float(training_loss),
            tr_terms=dict(ctx.logs),
            tr_ndata=tuple(self.tr_ndata),
            vl_loss=vl_loss,
            vl_chi2=vl_chi2,
            vl_partial=vl_partial,
        )
        self.record.register(fitstep)
        self.record.vl_chi2 = vl_chi2

        passes = self._counts | (vl_chi2 < self._threshold_chi2)
        passes &= vl_loss < self._best_val_chi2s
        passes &= vl_loss < (np.asarray(self._best_val_chi2s) - self._stopping_delta)
        passes &= self._positivity(validation)
        passes &= self._dont_stop_me_now

        self._stopping_degrees += self._counts

        for i_replica in np.where(passes)[0]:
            self._best_epochs[i_replica] = ctx.step - 1  # zero-based, as the legacy epochs were
            self.record.positivity_statuses[i_replica] = POS_OK
            self._best_val_chi2s[i_replica] = vl_loss[i_replica]
            self._best_weights[i_replica] = self._ensemble.weights()[i_replica]
            self._stopping_degrees[i_replica] = 0
            self._counts[i_replica] = 1

        stop_replicas = self._counts & (self._stopping_degrees > self.stopping_patience)
        for i_replica in np.where(stop_replicas)[0]:
            self._stop_epochs[i_replica] = ctx.step - 1  # zero-based, as the legacy ``epoch``
            self._counts[i_replica] = 0
            self._dont_stop_me_now[i_replica] = False

        if min(self._stopping_degrees) > self.stopping_patience:
            self.make_stop()
            ctx.stop()

    def on_train_end(self):
        """A fit that ran out of budget is still a fit: restore the best weights (legacy
        ``StoppingCallback.on_train_end`` -> ``make_stop``)."""
        self.make_stop()


def _total(values):
    """The total over terms and replicas -- the scalar the legacy logs carried in ``"loss"``."""
    return float(np.sum([np.sum(np.asarray(v)) for v in values.values()]))


def _sum_terms(values, names):
    """Per-replica sum over the named terms (legacy ``compute_losses()["loss"]``)."""
    arrays = [np.asarray(values[name]) for name in names]
    return np.sum(arrays, axis=0)


class LagrangeHook:
    """The Lagrange multiplier schedule, as a hook (replaces ``LagrangeCallback``).

    Legacy behaviour it must keep: the multiplier of each penalty term is *scaled* by a constant
    every ``period`` steps, so the term's value after ``n`` firings is ``initial * multiplier**n``.
    The difference is where the schedule is anchored: the callback used ``(epoch + 1) % period``,
    i.e. it fired at the last step of each period, while here a monitored step at index ``step``
    has ``step // period`` periods behind it and catches up by exactly the firings it owes.  For
    ``monitor_every = 1`` (the default) the two are the same thing -- which is what the golden
    fixture checks -- and for ``monitor_every > 1`` the catch-up is what keeps the schedule exact
    when the interval does not divide the period (contract §1.3).

    Parameters
    ----------
        terms: mapping
            ``{term name: Objective}``; the terms must declare a ``multiplier`` scalar.
        multipliers: mapping
            ``{term name: push factor}``, as ``_LM_initial_and_multiplier`` computes it.
        period: int
            steps between firings (100 for positivity/integrability in a production fit).
    """

    def __init__(self, terms, multipliers, period=100):
        missing = set(multipliers) - set(terms)
        if missing:
            raise ValueError(f"multipliers for unknown terms: {sorted(missing)}")
        self._terms = terms
        self._multipliers = dict(multipliers)
        self._period = int(period)
        self._applied = {name: 0 for name in multipliers}
        self._current = {name: None for name in multipliers}

    @property
    def applied(self):
        """``{term: number of firings so far}`` -- for tests and for the report."""
        return dict(self._applied)

    def on_monitored_step(self, ctx):
        for name, multiplier in self._multipliers.items():
            owed = ctx.step // self._period - self._applied[name]
            if owed <= 0:
                continue
            current = self._current[name]
            if current is None:
                current = self._terms[name].scalar("multiplier")
            for _ in range(owed):  # one multiplication per firing, as the callback did
                current = current * multiplier
            self._current[name] = current
            self._applied[name] += owed
            self._terms[name].set_scalar("multiplier", current)

    def on_train_end(self):
        """Nothing to close: the scalar is left where the schedule put it."""


class LogHook:
    """Timing summary of the fit (replaces ``TimerCallback``; D11 keeps nothing else).

    Per the round-24 decision this logs *only* the timing: the per-experiment chi2 line that the
    legacy ``Stopping.print_current_stats`` emitted is expendable, and ``chi2exps.log`` (written by
    ``io/writer.py`` from the ``FitRecord``) is where those numbers live anyway.
    """

    def __init__(self, count_range=100):
        self.all_times = []
        self.starting_time = None
        self.last_time = 0.0

    def on_monitored_step(self, ctx):  # pylint: disable=unused-argument
        # The step context is ignored on purpose: this hook measures the *engine*, so it must not
        # evaluate anything the fit would not have evaluated otherwise (which is what the legacy
        # ``compute_losses()`` every step was doing under jax).
        new_time = time()
        if self.starting_time is None:
            self.starting_time = new_time
        else:
            self.all_times.append(new_time - self.last_time)
        self.last_time = new_time

    def on_train_end(self):
        if not self.all_times:
            return
        total_time = self.last_time - self.starting_time
        # Same window as the legacy timer: skip the first 110 steps, where compilation dominates.
        window = self.all_times[min(110, len(self.all_times) - 1) :]
        mean = np.mean(window)
        std = np.std(window)
        log.info(f"> > Average time per step: {mean:.5} +- {std:.5} s")
        log.info(f"> > > Total time: {total_time/60:.5} min")


def _parse_chi2(values, ndata):
    """``{term: per-replica chi2}`` and the total, exactly as ``parse_losses`` computed them.

    Legacy detail worth keeping: the *total* is ``sum(losses) / sum(npoints)``, not
    ``sum(loss / npoints)`` -- the difference matters for datasets of different sizes, and it is
    what the stopping threshold was tuned against.
    """
    partial = {}
    total = 0.0
    total_points = 0
    for name, npoints in ndata.items():
        npoints = np.asarray(npoints)  # one entry per replica (legacy ``parse_ndata`` did this)
        values_array = np.asarray(values[name])
        partial[name] = values_array / np.maximum(npoints, 1)
        total = total + values_array
        total_points = total_points + npoints
    return total / np.maximum(total_points, 1), partial
