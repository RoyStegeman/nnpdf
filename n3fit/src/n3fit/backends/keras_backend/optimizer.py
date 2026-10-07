"""The contract's :class:`Optimizer`, implemented for Keras.

This module is where P4's ``Optimizer.run`` lives: the loop that n3fit used to get from
``MetaModel.perform_fit`` (Keras' ``fit``) plus the three callbacks in ``keras_backend.callbacks``.
It is deliberately *the* place in the backend that is allowed to drive the framework's training
machinery, and the only place that knows how a step is taken.

Two facts about Keras 3 shape the implementation, and both are findings of P4 rather than
preferences -- they are written down here because they bound what the contract can ask for:

1. **There is no public, backend-agnostic autodiff API.**  ``keras.ops`` has no ``gradient`` and
   ``keras.src.backend`` does not export one either, so a hand-written loop cannot compute its own
   gradients without reaching for ``jax.grad``/``tf.GradientTape``/``torch.autograd`` -- i.e.
   without the backend *becoming* framework-specific, which is what n3fit must not do.  The
   supported entry point for "one optimizer update" is therefore ``Model.train_on_batch``, which
   runs the framework's own ``train_step``.  The consequence for the design: the objective is
   handed to the framework as the compiled loss (built here, from the *terms*), while the same
   terms are applied by the engine itself when it evaluates a group for monitoring.  n3fit never
   sees either spelling.
2. **Keras averages what it reports.**  Per-step losses in Keras' ``history`` are running averages
   over batches, which is why the legacy loop needed ``CallbackStep.correct_logs`` to reconstruct
   "the loss of this step".  The engine does not reconstruct anything: it evaluates the monitored
   group itself, one forward pass, at the point in the step where the value is defined (before the
   update).  That is also what makes ``monitor_every > 1`` meaningful -- the legacy
   ``per-step``-log convention has no answer for it.

Step semantics (contract §1.3): ``steps`` counts optimizer updates, ``monitor_every`` how often the
hooks fire.  For ``monitor_every = k`` the engine takes ``k`` updates and, immediately before the
last one, evaluates the monitored group -- so a hook sees the losses that informed the update of
"its" step, which for ``k = 1`` (the default) is exactly the legacy convention.
"""

import inspect
import logging
import math

from .MetaModel import optimizers
from .ops import KerasOps
from .roles import tensor_to_numpy
from n3fit.backends.base import (
    GROUP_TRAINING,
    FitResult,
    StepContext,
)


class KerasWeightsView:
    """A mutable handle on the weights of one graph (contract :class:`WeightsView`).

    ``get()`` returns the weight store's ``{path: array}`` mapping (P5), so a hook can snapshot
    (``get``) and restore (``update``) without knowing anything about the graph.  ``assign``
    addresses one weight by its store path (``role/index/name``), or -- as a fallback for
    graphs where it is unambiguous -- by a layer name that owns a single weight.
    """

    def __init__(self, graph, ops=None):
        self._graph = graph
        self._ops = ops if ops is not None else KerasOps()

    def get(self, role=None):
        """A copy of the weights (``role`` selects a section reader, see ``roles.py``).

        Without a role this is the store's mapping for replica 0; a full per-replica snapshot is
        the ensemble's job (``Ensemble.weights()``).
        """
        if role is not None:
            from n3fit.backends.keras_backend.roles import as_view

            return as_view(self._graph).weights(role)
        from n3fit.backends.keras_backend.weights import weight_map

        return weight_map(self._graph, replica=0)

    def assign(self, path, value):
        """Set one weight in place, by store path or by single-weight layer name."""
        from n3fit.backends.keras_backend.weights import weight_slots

        slots = {slot.path: slot for slot in weight_slots(self._graph, replica=0)}
        if path in slots:
            slots[path].write(value, replica=0)
            return
        layer = self._graph.get_layer(path)
        if len(layer.weights) != 1:
            raise ValueError(
                f"{path!r} holds {len(layer.weights)} weights; address them individually"
            )
        layer.weights[0].assign(self._ops.numpy_to_tensor(value))

    def update(self, values):
        """Set several weights in place; the inverse of :meth:`get`."""
        for path, value in values.items():
            self.assign(path, value)


log = logging.getLogger(__name__)


class KerasOptimizer:
    """The contract's :class:`Optimizer` for Keras (first-order families; P7 adds the rest).

    ``is_iterative()`` is ``True`` for every name in the Keras registry, and the requirements are
    declared in :attr:`Capabilities.optimizers` (``requires={"gradient"}``); n3fit validates a
    request against that map, so this class never has to reject anything by name.
    """

    def __init__(self, spec, ops=None):
        self.spec = spec
        self._groups = None
        # The contract's ``Ops`` surface (``to_numpy``/``sum``/``zeros``/``numpy_to_tensor``), not
        # the legacy module: the engine is written against the same names everything else uses.
        self._ops = ops if ops is not None else KerasOps()
        if spec.name not in optimizers:
            raise NotImplementedError(
                f"[KerasOptimizer] optimizer not implemented: {spec.name!r} "
                f"(the backend declares {sorted(optimizers)})"
            )

    def __repr__(self):
        return f"<KerasOptimizer: {self.spec.name} {dict(self.spec.options)}>"

    # -- capabilities -------------------------------------------------------------------------
    def is_iterative(self):
        return True

    def min_monitor_interval(self):
        """Keras can honour any interval: a "step" here is one ``train_on_batch`` call."""
        return 1

    # -- the loop -----------------------------------------------------------------------------
    def _make_optimizer(self, name):
        """Instantiate the framework's optimizer (the construction ``MetaModel.compile`` did).

        The registry in :mod:`MetaModel` holds ``name -> (class, default kwargs)``; the defaults are
        copied (the legacy mutated them, so one fit's learning rate leaked into the next) and the
        spec's options override them where the optimizer accepts them.  An option it does not accept
        is *reported*, not dropped: quietly ignoring a learning rate is what the legacy did for
        ``Adamax``, and it is indistinguishable from a fit that ignores its runcard.
        """
        try:
            opt_class, defaults = optimizers[name]
        except KeyError as e:
            raise NotImplementedError(
                f"[KerasOptimizer] optimizer not implemented: {name!r} "
                f"(the backend declares {sorted(optimizers)})"
            ) from e
        accepted = set(inspect.signature(opt_class.__init__).parameters)
        options = dict(defaults)
        for key, value in self.spec.options.items():
            if value is None:
                continue
            if key in accepted:
                options[key] = value
            else:
                log.warning("the optimizer %s does not take %r: ignored", name, key)
        return opt_class(**options)

    def _loss_for(self, terms, names):
        """The compiled loss: a mapping from output name to the term applied to that output.

        Post-P4 a model outputs *predictions*; the terms are applied here, so the framework's
        gradient lands on ``sum(term(prediction))`` over the group -- the same function the legacy
        graphs computed inside the model and Keras summed through ``_default_loss = nansum``.
        """

        by_output = {}
        for name in names:
            by_output.setdefault(_output_of(terms[name]), []).append(terms[name])

        def as_scalar(group_terms):
            def loss(y_true, y_pred):  # pylint: disable=unused-argument
                # Several terms on one prediction add up, which is what Keras did by summing the
                # outputs of a graph whose every output was already a loss.
                return self._ops.sum(sum(term.apply(y_pred) for term in group_terms))

            return loss

        return {output: as_scalar(group_terms) for output, group_terms in by_output.items()}

    def _graph(self, ensemble, group):
        """The graph a group lives in: the role graph when the group *is* a role, else training."""
        try:
            return ensemble.graph(group)
        except KeyError:
            return ensemble.graph(GROUP_TRAINING)

    def _inputs(self, graph):
        return graph._parse_input(None)  # pylint: disable=protected-access

    def _targets(self, graph):
        """Zero targets: the terms ignore ``y_true``, but the framework insists on a target."""
        import numpy as np

        return [self._ops.numpy_to_tensor(np.zeros((1, 1))) for _ in graph.output_shape]

    def run(self, ensemble, terms, groups, *, steps, monitor_every, hooks=()):
        """Minimize ``groups["training"]`` over ``ensemble``, firing ``hooks`` every ``monitor_every``."""
        if steps is None:
            raise ValueError("an iterative optimizer needs a number of steps")
        graph = self._graph(ensemble, GROUP_TRAINING)
        self._groups = dict(groups)
        training_names = tuple(groups[GROUP_TRAINING])
        # The optimizer instance is kept: a mid-run recompile (below) must reuse it, or the
        # accumulator state (Adam's moments, the iteration count) would silently reset.
        optimizer = self._make_optimizer(self.spec.name)
        graph.compile(
            optimizer=optimizer,
            loss=self._loss_for(terms, training_names),
        )
        generations = _generations(terms, training_names)
        inputs = self._inputs(graph)
        targets = self._targets(graph)

        weights = KerasWeightsView(graph, ops=self._ops)
        history = KerasHistory()
        stop = [False]

        # Hooks that need the graph rather than the weights get it once, here (``Hook`` allows an
        # ``on_train_start``; tensorboard needs a handle on the model it logs).
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
            # Term state captured in the compiled train step is fixed for a run *unless* a
            # hook moved it (the Lagrange schedule moves the multiplier every period): the
            # jitted step closes over the term layers' weights *by value*, so without a
            # recompile training would silently keep the old value while ``scalar()`` and
            # the eager evaluations report the new one.  Recompiling with the same
            # optimizer instance keeps the accumulator state; the loss closures are rebuilt
            # over the same terms, so only the captured values change.
            current = _generations(terms, training_names)
            if current != generations:
                graph.compile(
                    optimizer=optimizer,
                    loss=self._loss_for(terms, training_names),
                )
                generations = current
            chunk = min(monitor_every or steps - step, steps - step)
            logs = None
            for offset in range(chunk):
                if monitor_every is not None and offset == chunk - 1:
                    # The losses that inform *this* step's update: one forward pass, pre-update.
                    logs = self.evaluate(ensemble, terms, GROUP_TRAINING)
                graph.train_on_batch(inputs, targets)
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

    # -- evaluation ---------------------------------------------------------------------------
    def evaluate(self, ensemble, terms, group):
        """Per-term, per-member values of ``group`` -- one forward pass, terms applied by name.

        ``group`` names either a role (``"training"``/``"validation"``/``"experimental"``, evaluated
        on that role's graph) or a reporting group (``"positivity"``/``"integrability"``, evaluated
        on the training graph, which is where those terms live).  Membership comes from the groups
        the current run was given (:meth:`run`), which is the only place n3fit states it: the
        contract's ``evaluate`` signature cannot carry it, and inventing a second membership rule
        here is exactly what P3 removed.
        """
        graph = self._graph(ensemble, group)
        names = self._members(group, terms, graph)
        outputs = graph(self._inputs(graph))
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        by_output = dict(zip(graph.output_names, outputs))
        return {
            name: tensor_to_numpy(
                terms[name].apply(by_output[_output_of(terms[name])]), ops=self._ops
            )
            for name in names
        }

    def _members(self, group, terms, graph):
        """The terms of ``group``: the run's groups when they know it, else the graph's own names."""
        if self._groups is not None and group in self._groups:
            return [name for name in self._groups[group] if name in terms]
        if group in terms:  # a bare term name is a group of one -- handy in tests and diagnostics
            return [group]
        return [name for name in terms if _output_of(terms[name]) in set(graph.output_names)]

    def jacobian(self, model, terms, group=GROUP_TRAINING):
        raise NotImplementedError("'KerasOptimizer.jacobian' arrives in P7 (parameter-space solvers)")

    def hessian(self, model, terms, group=GROUP_TRAINING):
        raise NotImplementedError("'KerasOptimizer.hessian' arrives in P7 (error propagation)")


class KerasHistory:
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
    """The graph output a term consumes: its own name unless it declares another prediction."""
    prediction = getattr(term.spec, "prediction", None)
    return prediction or term.spec.name


def _generations(terms, names):
    """The state generations of the named terms (see :meth:`KerasOptimizer.run`)."""
    return {
        name: terms[name]._generation  # pylint: disable=protected-access  (same backend)
        for name in names
    }


def _step_budget(steps, monitor_every):
    """How many chunks of updates the loop takes (kept separate: it is the loop's arithmetic)."""
    if monitor_every is None:
        return 1
    return int(math.ceil(steps / monitor_every))
