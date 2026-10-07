"""The raw JAX + optax implementation of the n3fit backend contract (P6).

This is an implementation, not an adapter: models are explicit parameters plus pure
functions (:mod:`n3fit.backends.jax_backend.models`), terms are pure functions
(:mod:`n3fit.backends.jax_backend.objectives`), and the engine differentiates them with
``jax.grad`` through optax (:mod:`n3fit.backends.jax_backend.optimizer`).  Nothing here
imports Keras -- that is the point (D9).

The members that require model *construction* from specs (``parametrization``, ``model``)
are staged, exactly as they are for the Keras backend: n3fit still builds its graphs
through the legacy generators, and P7 is what makes the constructor real for both.
"""

from collections.abc import Mapping

import jax
import numpy as np

from n3fit.backends.jax_backend.capabilities import build_capabilities
from n3fit.backends.jax_backend.models import JaxEnsemble, JaxEnsembleView, JaxModel
from n3fit.backends.jax_backend.objectives import build_objective
from n3fit.backends.jax_backend.ops import JaxOps

__all__ = ["JaxBackend", "JaxState"]


def _staged(member, phase, note=""):
    """Build the ``NotImplementedError`` for a member whose implementation is staged."""
    message = f"'{member}' is not implemented by the JAX backend yet (arrives in {phase})."
    if note:
        message += f" {note}"
    return NotImplementedError(message)


class JaxState:
    """Global state of the JAX backend (see :class:`n3fit.backends.base.BackendState`)."""

    def __init__(self):
        self._dtype = "float32"
        self._seed = None
        self._base_key = jax.random.PRNGKey(0)
        self._deterministic = False
        self._eager = None

    @property
    def dtype(self):
        """The floating-point type new models and terms are built in."""
        return self._dtype

    def configure(
        self,
        *,
        dtype="float32",
        threads=None,  # noqa: ARG002  (accepted by the contract, not used by this backend)
        seed=None,
        deterministic=False,
        eager=None,
        max_cores=None,  # noqa: ARG002  (XLA schedules threads itself)
    ):
        """Configure the process-wide state.

        ``dtype`` selects the precision of everything built afterwards (``float64``
        enables ``jax_enable_x64``); call this before building any model or term, as the
        flag cannot move once arrays exist.  ``seed`` derives the base RNG key (replicas
        split from it); ``eager`` is recorded but needs no action, since JAX programs are
        always eagerly executable and compilation here is an explicit per-step ``jit``.
        """
        if dtype not in ("float32", "float64"):
            raise ValueError(
                f"the JAX backend builds in 'float32' or 'float64', got {dtype!r}"
            )
        self._dtype = dtype
        jax.config.update("jax_enable_x64", dtype == "float64")
        self._seed = seed
        self._base_key = jax.random.PRNGKey(0 if seed is None else seed)
        self._deterministic = bool(deterministic)
        self._eager = eager

    def set_eager(self, enabled):
        """Eager execution on/off (a no-op: JAX needs no eager workaround)."""
        self._eager = enabled

    def clear(self):
        """Release the state between fits (hyperopt trials, k-folds).

        The RNG is re-seeded from the configured seed and the compiled-step caches are
        dropped; the precision flag is left alone (moving it under live arrays is what the
        ``configure`` docstring warns about).
        """
        self._base_key = jax.random.PRNGKey(0 if self._seed is None else self._seed)
        if hasattr(jax, "clear_caches"):
            jax.clear_caches()

    def devices(self):
        """Available compute devices."""
        return [str(device) for device in jax.devices()]


class JaxBackend:
    """The raw JAX + optax implementation of :class:`n3fit.backends.base.Backend`."""

    name = "jax"

    def __init__(self):
        self.capabilities = build_capabilities()
        self.state = JaxState()
        self.ops = JaxOps()

    @property
    def version(self):
        return jax.__version__

    def version_info(self):
        """Library versions, for the fit output (see ``io.writer.version``)."""
        import jaxlib  # pylint: disable=import-outside-toplevel
        import optax  # pylint: disable=import-outside-toplevel

        return {
            "backend": self.name,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "optax": optax.__version__,
        }

    # ------------------------------------------------------------------ active members
    def view(self, graph):
        """The contract's :class:`Model` over ``graph`` -- the identity for JAX models.

        Idempotent, as the Keras one is.  A model this backend did not build (in
        particular a Keras graph) cannot be viewed here -- views mutate the object they
        wrap, and a JAX view cannot mutate a graph object -- so that is an error naming
        the backend that can, not a silent mis-wrap.
        """
        if isinstance(graph, JaxModel):
            return graph
        raise ValueError(
            f"the JAX backend views JAX models, not {type(graph).__name__}; "
            "a Keras graph needs the 'keras' backend"
        )

    def objective(self, spec):
        """Build the term for ``spec`` (the pure functions of ``objectives.py``)."""
        return build_objective(spec, dtype=self.state.dtype)

    def ensemble(self, models, strategy=None):
        """The replicas of ``models``, as individual contract models.

        A mapping of role names to models is a fit's role ensemble; a single model is the
        training role on its own; a sequence is an explicit replica list.  ``strategy`` is
        unused: replicas always train independently (D2).
        """
        if isinstance(models, (JaxEnsemble, JaxEnsembleView)):
            return models
        if isinstance(models, Mapping):
            return JaxEnsemble(models)
        if isinstance(models, JaxModel):
            from n3fit.backends.base import GROUP_TRAINING

            return JaxEnsemble({GROUP_TRAINING: models})
        try:
            return JaxEnsembleView(list(models))
        except TypeError as err:
            raise ValueError(
                "a JAX ensemble is built from a {role: JaxModel} mapping, a single "
                f"JaxModel or a sequence of them, not {type(models).__name__}"
            ) from err

    def optimizer(self, spec):
        """Build the engine for ``spec`` (``optimizer.py``; one class per optimizer family)."""
        from n3fit.backends.jax_backend.optimizer import JaxOptimizer

        return JaxOptimizer(spec)

    def check_feasible(self, parametrization, optimizer, objective):
        """Raise if this (parametrization, optimizer, objective) is not buildable here.

        The same two checks the Keras backend performs: the parametrization kind and the
        optimizer name must be ones this backend declared, and an optimizer that requires
        a Jacobian/Hessian needs a backend that can produce one.
        """
        capabilities = self.capabilities
        if parametrization.kind not in capabilities.parametrizations:
            raise NotImplementedError(
                f"the JAX backend has no parametrization {parametrization.kind!r}; it implements "
                f"{sorted(capabilities.parametrizations)}"
            )
        try:
            declared = capabilities.optimizers[optimizer.name]
        except KeyError:
            raise NotImplementedError(
                f"the JAX backend has no optimizer {optimizer.name!r}; it implements "
                f"{sorted(capabilities.optimizers)}"
            ) from None
        missing = set(declared.get("requires", ())) - {"gradient"} - set(
            capabilities.derivatives
        )
        if missing:
            raise NotImplementedError(
                f"the optimizer {optimizer.name!r} requires {sorted(missing)} and this backend "
                f"can only produce {sorted(capabilities.derivatives)}"
            )
        if objective.kind not in capabilities.objectives:
            raise NotImplementedError(
                f"the JAX backend has no objective kind {objective.kind!r}; it implements "
                f"{sorted(capabilities.objectives)}"
            )

    def tensorboard_hook(self, logdir, *, histogram_freq=0, profiling=False):
        """Tensorboard cannot be driven here (no TensorFlow graph to attach it to)."""
        raise NotImplementedError(
            "the JAX backend cannot drive tensorboard (it has no tensorflow graph)"
        )

    # ------------------------------------------------------------------ staged members
    def parametrization(self, spec, shapes):
        raise _staged(
            "JaxBackend.parametrization",
            "P7",
            "Until then models are built explicitly (see jax_backend.models).",
        )

    def model(self, shapes, inputs, outputs, *, name, roles=None):
        raise _staged(
            "JaxBackend.model",
            "P7",
            "Until then models are built explicitly (see jax_backend.models).",
        )

    # ------------------------------------------------------------------ persistence
    def save(self, model, path):
        """Write the weights of one replica plus a manifest (the ``n3fit-weights/2`` schema).

        The schema is one replica per file.  ``model`` may be a one-replica ensemble or a
        model (whose replica 0 is saved); a multi-replica ensemble is rejected so callers
        cannot mistake its maps for one file.  The bytes are written by the shared
        ``n3fit.backends._weight_files`` -- a file does not know which backend wrote it.
        """
        from n3fit.backends._weight_files import make_manifest, save_weight_file

        if isinstance(model, (JaxEnsemble, JaxEnsembleView)):
            if len(model) != 1:
                raise ValueError(
                    f"a weight file holds one replica; this ensemble has {len(model)} "
                    "(save the replicas one file each, as the fit folders them)"
                )
            values = model.weights()[0]
            n_replicas_graph = len(model)
        elif isinstance(model, JaxModel):
            values = model.weights()
            n_replicas_graph = model.n_replicas
        else:
            raise ValueError(
                "the JAX backend saves JAX models and ensembles, "
                f"not {type(model).__name__}"
            )
        manifest = make_manifest(values, n_replicas_graph=n_replicas_graph, replica=0)
        save_weight_file(path, values, manifest=manifest)

    def load(self, ensemble, path, replica=None):
        """Read a weight file back into ``ensemble`` (the ``n3fit-weights/2`` schema).

        With ``replica=None`` the file's single replica is broadcast into *every* replica
        of the ensemble (the legacy ``load:`` semantics); with ``replica=i`` only that
        replica is filled (the ``load_weights_from_fit`` semantics).  Every target is
        validated before the first one is touched, so a mismatch never half-loads a model.
        """
        from n3fit.backends._weight_files import load_weight_file

        values, _manifest = load_weight_file(path)
        if replica is None:
            targets = list(range(len(ensemble)))
        else:
            if replica < 0 or replica >= len(ensemble):
                raise ValueError(
                    f"the ensemble has {len(ensemble)} replicas, cannot load replica {replica}"
                )
            targets = [replica]

        for index in targets:
            ensemble._validate_replica(values, index)

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
                except Exception as rollback_error:  # pragma: no cover - assignment failure path
                    rollback_errors.append((index, rollback_error))
            if rollback_errors:  # pragma: no cover - assignment failure path
                details = "; ".join(
                    f"replica {index}: {failure}" for index, failure in rollback_errors
                )
                raise RuntimeError(f"{err}; weight rollback also failed ({details})") from err
            raise
