"""The contract's :class:`Optimizer`, implemented for raw JAX + optax (P6).

This module is where the JAX backend's ``Optimizer.run`` lives: the same loop the Keras
engine runs (P4), with the same step/hook/history semantics, because the hooks *are* the
same objects (``n3fit.stopping``) and they observe step indices, pre-update logs and
evaluation calls.  The loop below mirrors ``keras_backend/optimizer.py`` deliberately --
same chunking, same ``StepContext``, same ``FitResult`` -- so the two engines can only
differ in arithmetic, which is what the parity test measures.

What is genuinely different (the D9 payoff):

* the gradient is ``jax.grad`` of ``sum(term(prediction))`` over the training group --
  there is no ``train_on_batch``, no compiled-loss trick and no zero-targets hack,
  because JAX exposes autodiff directly instead of hiding it behind ``fit``;
* replicas are trained in an explicit per-replica loop with a ``jit``ted step each (D2),
  not in one stacked graph -- which is also what lets replicas have different shapes;
* term *data* (covariances, masks, targets) is captured in the compiled step, while the
  Lagrange scalars are threaded as arguments, so the schedule never triggers a recompile;
  a ``set_data`` mid-run re-traces instead of silently using stale data (see below).
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax

from n3fit.backends.base import GROUP_TRAINING, FitResult, StepContext
from n3fit.backends.jax_backend.models import JaxWeightsView

log = logging.getLogger(__name__)

__all__ = ["JaxHistory", "JaxOptimizer"]


def _clip_by_norm(grad, clipnorm):
    """Per-tensor clip-by-norm with exact Keras ``clipnorm`` semantics.

    Keras's ``clipnorm`` clips the gradient of *each weight individually* to
    ``g * clipnorm / max(||g||, clipnorm)`` (with a NaN-safe norm), *before*
    any momentum/Adam accumulator update.  It is emphatically NOT a
    global-norm clip (that is Keras's ``global_clipnorm``, which n3fit does
    not use).  Mirror the formula exactly -- the parity test trains through
    it, and ``optax.clip_by_global_norm`` observably diverges from it.
    """
    l2sum = jnp.sum(jnp.square(grad))
    pred = l2sum > 0
    # Two-tap where trick to bypass NaN gradients (same as Keras).
    l2sum_safe = jnp.where(pred, l2sum, jnp.ones_like(l2sum))
    l2norm = jnp.where(pred, jnp.sqrt(l2sum_safe), l2sum)
    return (grad * clipnorm) / jnp.maximum(l2norm, clipnorm)


class JaxOptimizer:
    """The contract's :class:`Optimizer` for JAX (first-order families; P7 adds the rest).

    ``is_iterative()`` is ``True`` for every name in the JAX registry, and the requirements
    are declared in :attr:`Capabilities.optimizers` (``requires={"gradient"}``); n3fit
    validates a request against that map, so this class never has to reject anything that
    is not a name it does not implement.
    """

    #: Options each optimizer accepts (``spec.options`` beyond these are warned, not run).
    _ACCEPTED_OPTIONS = {
        "Adam": {"learning_rate", "beta1", "beta2", "epsilon", "clipnorm"},
        "SGD": {"learning_rate", "momentum", "nesterov", "clipnorm"},
    }

    def __init__(self, spec):
        self.spec = spec
        self._groups = None
        if spec.name not in self._ACCEPTED_OPTIONS:
            raise NotImplementedError(
                f"[JaxOptimizer] optimizer not implemented: {spec.name!r} "
                f"(the backend declares {sorted(self._ACCEPTED_OPTIONS)})"
            )

    def __repr__(self):
        return f"<JaxOptimizer: {self.spec.name} {dict(self.spec.options)}>"

    # -- capabilities -------------------------------------------------------------------------
    def is_iterative(self):
        return True

    def min_monitor_interval(self):
        """JAX can honour any interval: a "step" here is one optax update per replica."""
        return 1

    # -- construction ---------------------------------------------------------------------------
    def _make_transformation(self):
        """The optax update rule for the spec (the ``MetaModel.compile`` half of training).

        Defaults come from the backend's own declared registry (so the declaration and the
        implementation cannot drift); the spec's options override them where the optimizer
        accepts them.  An option it does not accept is *reported*, not dropped: quietly
        ignoring a learning rate is indistinguishable from a fit that ignores its runcard.
        """
        from n3fit.backends.jax_backend.capabilities import build_capabilities

        try:
            declared = build_capabilities().optimizers[self.spec.name]
        except KeyError as err:  # pragma: no cover - __init__ already rejected the name
            raise NotImplementedError(
                f"[JaxOptimizer] optimizer not implemented: {self.spec.name!r}"
            ) from err
        accepted = self._ACCEPTED_OPTIONS[self.spec.name]
        options = dict(declared["options"])
        for key, value in self.spec.options.items():
            if value is None:
                continue
            if key in accepted:
                options[key] = value
            else:
                log.warning("the optimizer %s does not take %r: ignored", self.spec.name, key)
        # ``clipnorm`` is deliberately NOT in the optax chain: Keras applies it to
        # each gradient tensor *before* the accumulator update, which a chain
        # entry cannot reproduce for SGD-with-momentum either (chain order) --
        # so _take_step clips the raw grads explicitly (see _clip_by_norm).
        clipnorm = options.get("clipnorm")
        if not (clipnorm and clipnorm > 0):
            clipnorm = None
        parts = []
        if self.spec.name == "Adam":
            parts.append(
                optax.adam(
                    options["learning_rate"],
                    b1=options.get("beta1", 0.9),
                    b2=options.get("beta2", 0.999),
                    eps=options.get("epsilon", 1e-7),
                )
            )
        else:
            parts.append(
                optax.sgd(
                    options["learning_rate"],
                    momentum=options.get("momentum", 0.0),
                    nesterov=options.get("nesterov", False),
                )
            )
        return parts[0], clipnorm

    # -- the loop -------------------------------------------------------------------------------
    def run(self, ensemble, terms, groups, *, steps, monitor_every, hooks=()):
        """Minimize ``groups["training"]`` over ``ensemble``, firing ``hooks`` every ``monitor_every``."""
        if steps is None:
            raise ValueError("an iterative optimizer needs a number of steps")
        model = ensemble.model(GROUP_TRAINING)
        if model.overridden and not model.frozen:
            raise NotImplementedError(
                "this model was overridden by a fixed numpy function, so it has no "
                "gradient to train; freeze it for diagnostics or train the unmodified model"
            )
        self._groups = dict(groups)
        training_names = tuple(groups[GROUP_TRAINING])
        missing = [name for name in training_names if name not in terms]
        if missing:
            raise ValueError(
                f"the training group names terms this run was not given: {missing}"
            )
        transformation, clipnorm = self._make_transformation()
        states = [transformation.init(model.params[i]) for i in range(model.n_replicas)]
        step_fn, generations = self._compile_step(model, terms, training_names)

        weights = JaxWeightsView(model)
        history = JaxHistory()
        stop = [False]

        # Hooks that need the ensemble rather than the weights get it once, here.
        for hook in hooks:
            if hasattr(hook, "on_train_start"):
                hook.on_train_start(ensemble)

        def evaluate(group):
            return self.evaluate(ensemble, terms, group)

        def request_stop():
            stop[0] = True

        step = 0
        monitor_every = int(monitor_every) if monitor_every else None
        while step < steps:
            # Term data captured in the compiled step is fixed for a run *unless* a hook
            # replaced it mid-run (no in-tree hook does -- the Lagrange schedule moves
            # scalars, which are arguments, not captures -- but a stale closure must never
            # silently win over an explicit ``set_data``).
            current = _generations(terms, training_names)
            if current != generations:
                step_fn, generations = self._compile_step(model, terms, training_names)
            chunk = min(monitor_every or steps - step, steps - step)
            logs = None
            for offset in range(chunk):
                if monitor_every is not None and offset == chunk - 1:
                    # The losses that inform *this* step's update: one forward pass, pre-update.
                    logs = self.evaluate(ensemble, terms, GROUP_TRAINING)
                if not model.frozen:
                    states = self._take_step(
                        model, terms, training_names, transformation, states, step_fn, clipnorm
                    )
                step += 1
            if monitor_every is None:
                continue
            ctx = StepContext(
                step=step,
                logs=logs,
                weights=weights,
                evaluate=evaluate,
                stop=request_stop,
            )
            history.register(ctx)
            for hook in hooks:
                hook.on_monitored_step(ctx)
            if stop[0]:
                break
        for hook in hooks:
            hook.on_train_end()
        return FitResult(
            parameters=ensemble.weights(),
            history=history,
            diagnostics=history.diagnostics(),
        )

    def _compile_step(self, model, terms, training_names):
        """The ``jit``ted per-replica ``(value, grad)`` plus the data generations it saw."""
        outputs = model._outputs  # pylint: disable=protected-access  (same backend)

        def loss_fn(params, inputs, scalars):
            predictions = {
                name: apply(params, inputs) for name, apply in outputs.items()
            }
            total = 0.0
            for name in training_names:
                term = terms[name]
                batched = jnp.asarray(predictions[_output_of(term)])[None, None, :]
                if term.kind in ("positivity", "integrability"):
                    value = term.forward(batched, {"multiplier": scalars[name]})
                else:
                    value = term.forward(batched)
                total = total + jnp.sum(value)
            return total

        return jax.jit(jax.value_and_grad(loss_fn)), _generations(terms, training_names)

    def _take_step(
        self, model, terms, training_names, transformation, states, step_fn, clipnorm
    ):
        """One optimizer update per replica (the unit the contract counts in)."""
        scalars = {
            name: jnp.asarray(terms[name]._multiplier)  # pylint: disable=protected-access
            for name in training_names
            if terms[name].kind in ("positivity", "integrability")
        }
        trainable = model.trainable_paths
        new_states = []
        for replica in range(model.n_replicas):
            params = model.params[replica]
            _loss, grads = step_fn(params, model._inputs, scalars)  # pylint: disable=protected-access
            if clipnorm is not None:
                grads = {
                    path: _clip_by_norm(grad, clipnorm)
                    for path, grad in grads.items()
                }
            updates, state = transformation.update(grads, states[replica], params)
            # Non-trainable paths keep their value: their *update* is zeroed (zeroing the
            # gradient would not do it -- Adam's moments would still move the parameter).
            frozen = set(params) - trainable
            if frozen:
                updates = {
                    path: jnp.zeros_like(update) if path in frozen else update
                    for path, update in updates.items()
                }
            updated = optax.apply_updates(params, updates)
            for path, value in updated.items():
                params[path] = value
            new_states.append(state)
        return new_states

    # -- evaluation -----------------------------------------------------------------------------
    def evaluate(self, ensemble, terms, group):
        """Per-term, per-member values of ``group`` -- terms applied by name.

        ``group`` names either a role (``"training"``/``"validation"``/``"experimental"``,
        evaluated on that role's model) or a reporting group (``"positivity"``/
        ``"integrability"``, evaluated on the training model, which is where those terms
        live).  Membership comes from the groups the current run was given (:meth:`run`),
        which is the only place n3fit states it.
        """
        try:
            model = ensemble.model(group)
        except KeyError:
            model = ensemble.model(GROUP_TRAINING)
        names = self._members(group, terms, model)
        stacked = {}
        for replica in range(model.n_replicas):
            predictions = model.predict_replica(replica)
            for output, value in predictions.items():
                stacked.setdefault(output, []).append(np.asarray(value))
        batched = {
            output: np.asarray(values)[None, :, :]
            for output, values in stacked.items()
        }
        return {
            name: np.asarray(terms[name].forward(batched[_output_of(terms[name])]))
            for name in names
        }

    def _members(self, group, terms, model):
        """The terms of ``group``: the run's groups when they know it, else the model's own names."""
        if self._groups is not None and group in self._groups:
            return [name for name in self._groups[group] if name in terms]
        if group in terms:  # a bare term name is a group of one -- handy in tests and diagnostics
            return [group]
        return [name for name in terms if _output_of(terms[name]) in set(model.output_names)]

    def jacobian(self, model, terms, group=GROUP_TRAINING):
        raise NotImplementedError("'JaxOptimizer.jacobian' arrives in P7 (parameter-space solvers)")

    def hessian(self, model, terms, group=GROUP_TRAINING):
        raise NotImplementedError("'JaxOptimizer.hessian' arrives in P7 (error propagation)")


class JaxHistory:
    """The contract's :class:`History`: what the hooks saw, in order."""

    def __init__(self):
        self.monitored_steps = []
        self.losses = {}
        self.evaluations = {}

    def register(self, ctx):
        self.monitored_steps.append(ctx.step)
        for name, value in ctx.logs.items():
            self.losses.setdefault(name, []).append(value)

    def record(self, group, values):
        """Keep an evaluation a hook asked for (used for reporting, not for decisions)."""
        self.evaluations.setdefault(group, []).append(values)

    def diagnostics(self):
        return {"monitored_steps": list(self.monitored_steps)}


def _output_of(term):
    """The model output a term consumes: its own name unless it declares another prediction."""
    prediction = getattr(term.spec, "prediction", None)
    return prediction or term.spec.name


def _generations(terms, names):
    """The data generations of the named terms (see :meth:`JaxOptimizer.run`)."""
    return {
        name: terms[name]._generation  # pylint: disable=protected-access  (same backend)
        for name in names
    }
