"""
The Keras implementation of the n3fit backend contract.

**P0/P1 scope.**  This is an *adapter*: it exposes what already exists (the ``operations``
namespace as ``ops``, through :mod:`n3fit.backends.keras_backend.ops`; the legacy state
functions as ``state``; the legacy dictionaries as ``capabilities``) through the interface of
:mod:`n3fit.backends.base`.  The members that require the P2-P5 refactor are declared with an
explicit ``NotImplementedError`` naming the phase, so that call sites fail loudly and
informatively rather than subtly.

Nothing in n3fit is required to use the not-yet-implemented members: the legacy code path
still works untouched, and this object is used by the registry, by ``n3fit.checks`` and the
other P1 call sites, and by the conformance suite.
"""

from collections.abc import Mapping

import keras

from n3fit.backends.keras_backend.capabilities import build_capabilities, keras_backend_name
from n3fit.backends.keras_backend.ops import KerasOps
from n3fit.backends.keras_backend.roles import KerasEnsembleView, KerasRoleEnsemble, as_view
from n3fit.backends.keras_backend.internal_state import (
    clear_backend_state,
    get_physical_gpus,
    set_eager,
    set_initial_state,
)


def _staged(member, phase, note=""):
    """Build the ``NotImplementedError`` for a member whose implementation is staged."""
    message = f"'{member}' is not implemented by the Keras backend yet (arrives in {phase})."
    if note:
        message += f" {note}"
    return NotImplementedError(message)


def _graph_replica_count(graph):
    """Count the explicit replica axis of an n3fit PDF graph (or return one for a single model)."""
    outputs = getattr(graph, "output", None)
    first = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    shape = getattr(first, "shape", None)
    if shape is not None and len(shape) >= 4 and shape[1] is not None:
        return int(shape[1])
    return 1


class KerasState:
    """Global state of the Keras backend (see :class:`n3fit.backends.base.BackendState`).

    P0: a direct delegation to the functions in ``internal_state``, which is what n3fit
    still calls.  P1 switches ``performfit`` over to this object.
    """

    def configure(
        self,
        *,
        dtype="float32",
        threads=None,  # noqa: ARG002  (accepted by the contract, not used by this backend)
        seed=None,
        deterministic=False,
        eager=None,
        max_cores=None,
    ):
        """Configure the process-wide state.

        Maps onto the legacy entry point: ``deterministic``/``eager``/``max_cores`` and the
        seed are handled by ``set_initial_state``, and the floating point type by
        ``keras.backend.set_floatx``.
        """
        set_initial_state(
            debug=deterministic,
            external_seed=seed,
            max_cores=max_cores,
            double_precision=(dtype == "float64"),
        )
        if eager is not None:
            self.set_eager(eager)

    def set_eager(self, enabled):
        """Eager execution on/off (a no-op for the eager-by-default frameworks)."""
        set_eager(enabled)

    def clear(self):
        """Release the state between fits (hyperopt trials, k-folds)."""
        clear_backend_state()

    def devices(self):
        """Available compute devices."""
        return [device.name for device in get_physical_gpus()]


class KerasBackend:
    """The Keras implementation of :class:`n3fit.backends.base.Backend`."""

    name = "keras"

    def __init__(self):
        self.capabilities = build_capabilities()
        self.state = KerasState()
        # the contract's operations, adapted from the historical ``operations`` namespace;
        # the legacy names remain reachable through it (see ``keras_backend/ops.py``)
        self.ops = KerasOps()

    @property
    def version(self):
        return keras.__version__

    def version_info(self):
        """Library versions, for the fit output (see ``io.writer.version``).

        The ``keras``/``tensorflow``/``torch`` keys reproduce what that function used to
        collect by importing the frameworks itself -- including the ``backend='...'`` spelling
        -- so that the fit output does not change; ``backend`` and ``keras_backend`` are the
        backend-agnostic names, and are what a second backend would populate.
        """
        framework = keras_backend_name()
        info = {
            "backend": self.name,
            "keras_backend": framework,
            "keras": f"{keras.__version__} backend='{framework}'",
        }
        if framework == "tensorflow":
            import tensorflow  # pylint: disable=import-outside-toplevel

            info["tensorflow"] = tensorflow.__version__
        elif framework == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            # NOTE: this also fixes a typo in the code this replaces, which compared instead
            # of assigning (``versions["torch"] == torch.__version__``) and so never recorded
            # the version.  Only visible on torch backends.
            info["torch"] = torch.__version__
        return info

    # ------------------------------------------------------------------ active members
    def view(self, graph):
        """Expose an existing Keras graph through the contract's :class:`Model` API (P2).

        This is how n3fit gets role-based access (``weights(role=...)``, ``bind_input``,
        ``override``, ``freeze``, ``summary``) without knowing any layer name; the names live in
        ``keras_backend/roles.py``, on this side of the boundary.

        Idempotent: viewing something that is already a view returns it unchanged, so callers
        may mix raw graphs (from code that has not been migrated, or from a fit result) with
        contract models without checking which they hold.
        """
        # ``_model_builder`` lets the adapter construct diagnostic graphs (its
        # ``prediction_before``) without importing the framework itself.
        from n3fit.backends.keras_backend.MetaModel import MetaModel

        return as_view(graph, ops=self.ops, model_builder=lambda inputs, outputs: MetaModel(inputs, outputs))

    # ------------------------------------------------------------------ staged members
    # Declared so that the interface is complete and call sites fail informatively.
    def parametrization(self, spec, shapes):
        raise _staged(
            "KerasBackend.parametrization",
            "P2",
            "Until then use n3fit.model_gen.generate_pdf_model with ReplicaSettings.",
        )

    def model(self, shapes, inputs, outputs, *, name, roles=None):
        raise _staged(
            "KerasBackend.model",
            "P2",
            "Until then use n3fit.backends.MetaModel.",
        )

    def objective(self, spec):
        """Build the term for ``spec`` (P3).

        The layer classes come from ``n3fit.layers.losses`` and the conversion ops from this
        backend; the wrapper that makes them a contract :class:`Objective` lives in
        ``objectives.py`` and imports no framework itself.
        """
        from n3fit.layers import losses

        from n3fit.backends.keras_backend.objectives import build_objective

        layer_classes = {
            "chi2": losses.LossInvcovmat,
            "positivity": losses.LossPositivity,
            "integrability": losses.LossIntegrability,
        }
        try:
            layer_class = layer_classes[spec.kind]
        except KeyError:
            raise ValueError(
                f"unknown objective kind {spec.kind!r}; this backend implements "
                f"{sorted(layer_classes)}"
            ) from None
        return build_objective(spec, layer_class=layer_class, ops=self.ops)

    def ensemble(self, models, strategy=None, **kwargs):
        """The replicas of a model, as individual contract models (P2b).

        Two ways in, because n3fit is mid-migration: a sequence of models, or **a single graph
        that carries its replicas stacked** (what every call site has today) -- the legacy
        decomposition is ``MetaModel.split_replicas``, which makes one single-replica graph per
        replica and copies that replica's weights into it.  Doing the split here rather than at
        the call sites is what lets ``pdf_model.split_replicas()`` disappear from n3fit.

        ``strategy`` is unused so far: the legacy graph fixes how replicas share weights (P4,
        D2).
        """
        if isinstance(models, Mapping):
            # ``weights_graph``: the roles are built by re-applying the PDF model, so its per-replica
            # layers are where the fit's weights actually live (see KerasRoleEnsemble).
            return KerasRoleEnsemble(models, ops=self.ops, weights_graph=kwargs.get("weights_graph"))
        return KerasEnsembleView.from_graph(models, ops=self.ops)

    def optimizer(self, spec):
        """Build the engine for ``spec`` (P4).

        The implementation lives in ``optimizer.py``; this method only decides *which* class, which
        is what lets a second optimizer family (parameter-space solvers, P7) be added without
        touching n3fit or this facade.
        """
        from n3fit.backends.keras_backend.optimizer import KerasOptimizer

        return KerasOptimizer(spec, ops=self.ops)

    def check_feasible(self, parametrization, optimizer, objective):
        """Raise if this (parametrization, optimizer, objective) is not a buildable combination.

        What it is *for*: a combination that cannot work should fail here, with a message about the
        combination, instead of surfacing as a framework error several thousand lines deeper (the
        legacy behaviour was a ``NotImplementedError`` from inside ``MetaModel.compile``, which
        also lost the fact that the *optimizer* was the problem).

        What it checks here is the two things this backend can actually distinguish: the
        parametrization kind and the optimizer name must be ones it declared in
        :class:`KerasCapabilities`, and an optimizer that requires a Jacobian/Hessian needs a
        backend that can produce one (``Capabilities.derivatives``).  The data-side of a term is
        already validated when the term is built (``Backend.objective``).
        """
        capabilities = self.capabilities
        if parametrization.kind not in capabilities.parametrizations:
            raise NotImplementedError(
                f"the Keras backend has no parametrization {parametrization.kind!r}; it implements "
                f"{sorted(capabilities.parametrizations)}"
            )
        try:
            declared = capabilities.optimizers[optimizer.name]
        except KeyError:
            raise NotImplementedError(
                f"the Keras backend has no optimizer {optimizer.name!r}; it implements "
                f"{sorted(capabilities.optimizers)}"
            ) from None
        missing = set(declared.get("requires", ())) - {"gradient"} - set(capabilities.derivatives)
        if missing:
            raise NotImplementedError(
                f"the optimizer {optimizer.name!r} requires {sorted(missing)} and this backend "
                f"can only produce {sorted(capabilities.derivatives)}"
            )
        if objective.kind not in capabilities.objectives:
            raise NotImplementedError(
                f"the Keras backend has no objective kind {objective.kind!r}; it implements "
                f"{sorted(capabilities.objectives)}"
            )

    def tensorboard_hook(self, logdir, *, histogram_freq=0, profiling=False):
        """A hook that writes tensorboard logs, or raise if this backend cannot (P4/A3).

        Tensorboard is a Keras *callback* and only works on the tensorflow backend (upstream
        limitation), so n3fit cannot own it and cannot assume it: it asks, and here the answer is
        either a hook or a clear ``NotImplementedError``.
        """
        if not self.capabilities.supports_tensorboard:
            raise NotImplementedError(
                f"the Keras backend cannot drive tensorboard on '{keras_backend_name()}' "
                f"(it needs the tensorflow backend)"
            )
        from n3fit.backends.keras_backend.callbacks import gen_tensorboard_callback

        callback = gen_tensorboard_callback(
            logdir, profiling=profiling, histogram_freq=histogram_freq
        )

        import numpy as np

        from n3fit.backends.base import GROUP_TRAINING

        class _TensorboardHook:
            """Drive the tensorboard callback from the engine's per-step hook protocol.

            The callback is a *Keras* callback: it wants a model (``set_model``) and
            ``on_epoch_end`` with a ``{metric: value}`` mapping, which is why this adapter exists
            here rather than n3fit calling the callback itself.  The model arrives through
            ``on_train_start`` -- n3fit builds its hooks before the graphs exist -- and the step
            that is logged is the training group's, per term, as the legacy logs carried it.
            """

            def __init__(self, callback, histogram_freq):
                self._callback = callback
                self._histogram_freq = histogram_freq
                self._epoch = 0

            def on_train_start(self, ensemble):
                self._callback.set_model(ensemble.graph(GROUP_TRAINING))

            def on_monitored_step(self, ctx):
                self._epoch += 1
                logs = {name: float(np.sum(np.asarray(value))) for name, value in ctx.logs.items()}
                logs["loss"] = float(sum(logs.values()))
                self._callback.on_epoch_end(self._epoch - 1, logs=logs)

            def on_train_end(self):
                self._callback.on_train_end()

        return _TensorboardHook(callback, histogram_freq)

    def save(self, model, path):
        """Write the weights of one replica plus a manifest (the ``n3fit-weights/2`` schema, P5).

        The schema is one replica per file -- that is what the per-replica fit folders hold, what
        ``load:`` broadcasts and what ``load_weights_from_fit`` exchanges. ``model`` may be a
        one-replica contract ensemble, a contract :class:`Model`, or a raw graph. A raw stacked
        graph saves replica 0 and records its actual replica count in the manifest; a multi-replica
        ensemble is rejected so callers cannot mistake its maps for one file.
        """
        from n3fit.backends.keras_backend import weights as store
        from n3fit.backends.keras_backend.roles import KerasModelView

        if hasattr(model, "__len__") and hasattr(model, "weights"):
            if len(model) != 1:
                raise ValueError(
                    f"a weight file holds one replica; this ensemble has {len(model)} "
                    "(save the replicas one file each, as the fit folders them)"
                )
            values = model.weights()[0]
            n_replicas_graph = len(model)
        elif isinstance(model, KerasModelView):
            values = store.weight_map(model._graph, replica=0)
            n_replicas_graph = _graph_replica_count(model._graph)
        else:  # a raw graph
            values = store.weight_map(model, replica=0)
            n_replicas_graph = _graph_replica_count(model)
        manifest = store.make_manifest(values, n_replicas_graph=n_replicas_graph, replica=0)
        store.save_weight_file(path, values, manifest=manifest)

    def load(self, ensemble, path, replica=None):
        """Read a weight file back into ``ensemble`` (the ``n3fit-weights/2`` schema, P5).

        With ``replica=None`` the file's single replica is broadcast into *every* replica of the
        ensemble -- the legacy ``load:`` semantics (one warm-start file, identical replicas).
        With ``replica=i`` only that replica is filled -- the ``load_weights_from_fit`` semantics.
        The file must describe the same replica layout the graph was built with: a mismatch is an
        error naming the file, never a half-updated model.
        """
        from n3fit.backends.keras_backend import weights as store

        values, _manifest = store.load_weight_file(path)
        if replica is None:
            targets = list(range(len(ensemble)))
        else:
            if replica < 0 or replica >= len(ensemble):
                raise ValueError(
                    f"the ensemble has {len(ensemble)} replicas, cannot load replica {replica}"
                )
            targets = [replica]

        # Validate every target before changing the first one: replicas may have different
        # hyperopt architectures even though the weight paths are otherwise identical.
        for index in targets:
            ensemble._validate_replica(values, index)

        # Keep snapshots as a rollback path for unexpected framework assignment failures after
        # validation (e.g. a backend/device error).  Expected path/shape errors happen above.
        import numpy as np

        current = ensemble.weights()
        original = {
            index: {key: np.array(value, copy=True) for key, value in current[index].items()}
            for index in targets
        }
        try:
            for index in targets:
                ensemble._assign_replica(values, index)
        except Exception as err:
            rollback_errors = []
            for index in reversed(targets):
                try:
                    ensemble._assign_replica(original[index], index)
                except Exception as rollback_error:  # pragma: no cover - framework failure path
                    rollback_errors.append((index, rollback_error))
            if rollback_errors:  # pragma: no cover - framework failure path
                details = "; ".join(
                    f"replica {index}: {failure}" for index, failure in rollback_errors
                )
                raise RuntimeError(f"{err}; weight rollback also failed ({details})") from err
            raise
