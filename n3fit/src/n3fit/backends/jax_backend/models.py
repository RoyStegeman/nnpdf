"""Functional models for the JAX backend: params + apply (P6).

A :class:`JaxModel` is not a graph object: it is an explicit parameter list (one
``{path: array}`` dict per replica, in the store's ``{role}/{index}/{name}`` grammar) plus
a mapping of output names to pure functions ``(params, inputs) -> prediction``.  There is
nothing to build, split or compile -- which is why ``view`` is the identity here and why
per-replica models are cheap snapshots rather than rebuilt graphs.

Sharing rules (these are the whole design, so they are stated once):

* role models of one fit **share** the per-replica parameter dicts: the engine updates
  the training model's dicts in place (entry by entry, never rebound) and the
  validation/experimental models see the update, exactly as the Keras role graphs share
  their trainable layers;
* per-replica members yielded by iterating an ensemble hold **their own dicts** referencing
  the arrays as they were when the ensemble was first iterated -- snapshots, like the
  graphs ``MetaModel.split_replicas`` builds, since JAX arrays are immutable and every
  update replaces a dict entry rather than mutating an array.
"""

import jax.numpy as jnp
import numpy as np

from n3fit.backends.base import GROUP_TRAINING, ROLES, role_of

__all__ = [
    "JaxEnsemble",
    "JaxEnsembleView",
    "JaxModel",
    "JaxWeightsView",
]


def _check_role(role):
    """Reject a role nobody declared: a typo must not look like an empty section."""
    if role is not None and role not in ROLES:
        raise ValueError(f"Unknown role {role!r}; known roles: {sorted(ROLES)}")


def _to_backend_array(value, dtype):
    return jnp.asarray(np.asarray(value), dtype=dtype)


class JaxModel:
    """The :class:`n3fit.backends.base.Model` contract over explicit JAX parameters.

    Parameters
    ----------
        params:
            one ``{path: array}`` dict per replica, taken **by reference**: values are
            converted to backend arrays in place and the list object is kept, so role
            models built over the same list share their weights (see the module
            docstring).  Pass copies (``[dict(replica) for replica in params]``) to isolate.
        outputs:
            ``{output name: callable}``; each callable takes ``(params, inputs)`` (one
            replica's dict, the bound inputs) and returns that replica's prediction for the
            output, a ``(ndata,)`` JAX array.
        inputs:
            bound constants (``{name: array}``), the functional spelling of the graph's
            held inputs.
        trainable:
            paths that may be updated (``None`` means all of them).
    """

    def __init__(
        self, params, outputs, *, inputs=None, name="model", shapes=None, trainable=None,
        dtype="float32",
    ):
        if not params:
            raise ValueError("a model needs at least one replica's parameters")
        if not outputs:
            raise ValueError("a model needs at least one output")
        self._dtype = np.dtype(dtype)
        self._params = list(params)
        for replica in self._params:
            for path, value in replica.items():
                replica[path] = _to_backend_array(value, self._dtype)
        self._outputs = dict(outputs)
        self._inputs = {
            key: _to_backend_array(value, self._dtype)
            for key, value in dict(inputs or {}).items()
        }
        self._name = name
        self.shapes = shapes
        known = set(self._params[0])
        self._trainable = set(known) if trainable is None else set(trainable)
        if not self._trainable <= known:
            raise ValueError(
                f"trainable paths {sorted(self._trainable - known)} are not parameters "
                f"of this model (it has {sorted(known)})"
            )
        self._frozen = False
        self._override = None

    # ------------------------------------------------------------------ introspection
    @property
    def name(self):
        return self._name

    @property
    def n_replicas(self):
        return len(self._params)

    @property
    def output_names(self):
        return tuple(self._outputs)

    @property
    def params(self):
        """The live parameter list (backend-internal; the engine updates it in place)."""
        return self._params

    @property
    def frozen(self):
        return self._frozen

    @property
    def overridden(self):
        """Whether a section was replaced by a fixed function (then there is no gradient)."""
        return self._override is not None

    @property
    def trainable_paths(self):
        """The paths the engine may update (all of them, unless restricted)."""
        return set(self._trainable)

    def summary(self):
        """Print the model: outputs, roles and parameter shapes (replica 0)."""
        print(f"{self._name}: {self.n_replicas} replica(s), outputs {self.output_names}")
        for path, value in self._params[0].items():
            print(f"  {path}: {tuple(value.shape)} {value.dtype}")

    # ------------------------------------------------------------------ evaluation
    def predict_replica(self, replica):
        """``{output: (ndata,) JAX array}`` for one replica (backend-internal)."""
        params = self._params[replica]
        if self._override is not None:
            _role, fn = self._override
            result = fn({key: np.asarray(value) for key, value in self._inputs.items()})
            if isinstance(result, dict):
                return {
                    name: _to_backend_array(result[name], self._dtype)
                    for name in self._outputs
                }
            array = _to_backend_array(result, self._dtype)
            return {name: array for name in self._outputs}
        return {
            name: jnp.asarray(apply(params, self._inputs), dtype=self._dtype)
            for name, apply in self._outputs.items()
        }

    def __call__(self, inputs=None):
        """Evaluate on numpy arrays and return numpy arrays (inference only).

        ``None`` evaluates the model as built; otherwise the given slots overlay the bound
        ones for this call.  Nothing is squeezed: a single-output model returns
        ``(1, n_replicas, ndata)`` and a multi-output one returns that per output name,
        which is what the diagnostics index into.
        """
        if inputs is not None:
            saved = self._inputs
            self._inputs = {
                **saved,
                **{k: _to_backend_array(v, self._dtype) for k, v in inputs.items()},
            }
            try:
                return self.__call__(None)
            finally:
                self._inputs = saved
        stacked = {}
        for name in self._outputs:
            per_replica = [self.predict_replica(i)[name] for i in range(self.n_replicas)]
            stacked[name] = np.asarray(
                jnp.expand_dims(jnp.stack(per_replica, axis=0), axis=0)
            )
        if len(stacked) == 1:
            return next(iter(stacked.values()))
        return stacked

    # ------------------------------------------------------------------ weights
    def _replica_map(self, replica):
        """A frozen ``{path: numpy}`` copy of one replica (snapshots must not move)."""
        return {
            path: np.array(value, copy=True) for path, value in self._params[replica].items()
        }

    def weights(self, role=None):
        """Weights as a numpy dict keyed by path (replica 0; the ensemble reads them all).

        A role filters by path prefix; a model simply has no paths for sections it was not
        built with, so that is ``{}`` rather than an error -- the sections of a functional
        model are explicit at construction, unlike the discovered sections of a graph.
        """
        _check_role(role)
        values = self._replica_map(0)
        if role is None:
            return values
        return {path: array for path, array in values.items() if role_of(path) == role}

    def parameters(self, role=None):
        """The *trainable* weights, in layout order (the space ``theta`` lives in)."""
        values = self.weights(role)
        return {path: values[path] for path in self._params[0] if path in values}

    def set_weights(self, values):
        """Restore replica 0 from a storage map (multi-replica restore is the ensemble's)."""
        self._assign(0, values, strict=True)

    def _validate(self, replica, values, *, strict=True):
        known = set(self._params[replica])
        provided = set(values)
        if provided - known:
            raise ValueError(
                f"this model has no weight {sorted(provided - known)[:3]}... "
                f"(it has {len(known)}: {sorted(known)[:3]}...)"
            )
        if strict and known - provided:
            raise ValueError(
                f"the weight map is missing {len(known - provided)} of this model's "
                f"{len(known)} weights, e.g. {sorted(known - provided)[:3]}"
            )
        for path, value in values.items():
            expected = tuple(self._params[replica][path].shape)
            if tuple(np.shape(value)) != expected:
                raise ValueError(
                    f"the weight {path!r} has shape {expected}, got {tuple(np.shape(value))}"
                )

    def _assign(self, replica, values, *, strict=True):
        self._validate(replica, values, strict=strict)
        for path, value in values.items():
            self._params[replica][path] = _to_backend_array(value, self._dtype)

    def weights_view(self):
        """A live handle on replica 0's weights (what the hooks mutate through)."""
        return JaxWeightsView(self)

    # ------------------------------------------------------------------ inputs
    def bound_inputs(self):
        """The constants currently bound to the model, as numpy arrays."""
        return {key: np.array(value, copy=True) for key, value in self._inputs.items()}

    def bind_input(self, name, value):
        """Bind (or replace) a constant input.

        Any slot may be bound: a functional model has no build step, so there is nothing
        to rebuild -- this is a deliberate superset of the Keras adapter, which can only
        rebind the photon.
        """
        self._inputs[name] = _to_backend_array(value, self._dtype)

    # ------------------------------------------------------------------ graph surgery
    def override(self, role, fn):
        """Replace the section playing ``role`` by the fixed numpy function ``fn``.

        ``fn`` maps the bound-inputs mapping to an array (or, for a multi-output model, to
        a mapping of output names to arrays); the conversion back is this model's business.
        A missing section is an error, as in the Keras adapter.  An overridden model still
        evaluates but can no longer be differentiated (the engine raises if it is asked to).
        """
        sections = {role_of(path) for replica in self._params for path in replica}
        if role not in sections:
            raise ValueError(
                f"no section in this model plays the role {role!r} "
                f"(it has {sorted(sections)}); cannot override it"
            )
        self._override = (role, fn)

    def freeze(self):
        """Make the model permanently non-trainable (the engine then only evaluates it)."""
        self._frozen = True

    def __repr__(self):
        return f"<JaxModel {self._name!r}: {self.n_replicas} replica(s)>"


class JaxWeightsView:
    """A mutable handle on replica 0's weights (contract :class:`WeightsView`)."""

    def __init__(self, model):
        self._model = model

    def get(self, role=None):
        """A copy of the weights (``role`` selects a section, see :meth:`JaxModel.weights`)."""
        return self._model.weights(role)

    def assign(self, path, value):
        """Set one weight in place, by store path."""
        if path not in self._model.params[0]:
            raise ValueError(
                f"this model has no weight {path!r} (it has "
                f"{sorted(self._model.params[0])[:3]}...)"
            )
        self._model._assign(0, {path: value}, strict=False)

    def update(self, values):
        """Set several weights in place; the inverse of :meth:`get`."""
        self._model._assign(0, dict(values), strict=False)


class JaxEnsembleView:
    """The contract's :class:`Ensemble` over an explicit sequence of models.

    Iteration order *is* the replica order.  Slices stay an ensemble rather than degrading
    to a bare list, so a caller cannot accidentally end up holding raw models.
    """

    def __init__(self, models):
        self._models = list(models)
        for model in self._models:
            if not isinstance(model, JaxModel):
                raise ValueError(
                    f"a JAX ensemble holds JaxModel objects, got {type(model).__name__}"
                )

    def __iter__(self):
        return iter(self._models)

    def __getitem__(self, replica):
        if isinstance(replica, slice):
            return JaxEnsembleView(self._models[replica])
        return self._models[replica]

    def __len__(self):
        return len(self._models)

    def weights(self, role=None):
        """One weight map per replica, in iteration order (``role`` filters by section)."""
        return [model.weights(role) for model in self._models]

    def set_weights(self, values):
        """Restore per-replica weights from storage maps (the inverse of :meth:`weights`)."""
        if len(values) != len(self._models):
            raise ValueError(
                f"expected {len(self._models)} replica weight maps, got {len(values)}"
            )
        for model, replica_weights in zip(self._models, values):
            model._validate(0, replica_weights)
        for model, replica_weights in zip(self._models, values):
            model._assign(0, replica_weights)

    def _validate_replica(self, values, replica):
        """Backend-internal preflight; this view stores one model per replica."""
        self._models[replica]._validate(0, values)

    def _assign_replica(self, values, replica):
        """Backend-internal: load one replica's storage map (``Backend.load``)."""
        self._models[replica]._assign(0, values)

    def __repr__(self):
        return f"<JaxEnsembleView: {len(self._models)} replicas>"


class JaxEnsemble:
    """The contract's :class:`Ensemble` for a *fit*: the role models over one set of weights.

    A fit is three models, not one.  The role models share their parameter list object (see
    the module docstring), so the optimizer updates the training model and the validation
    and experimental models -- which differ in their outputs, not their weights -- see the
    update.  ``weights()``/``set_weights()`` address the replicas of the weights model
    through the same path grammar a fit writes to disk.  Iterating the ensemble yields
    per-replica snapshots (cheap here: no graph is rebuilt); the engine never does it.
    """

    def __init__(self, models, *, weights_role=GROUP_TRAINING):
        if not models:
            raise ValueError("an ensemble needs at least one role model")
        self._models = dict(models)
        for role, model in self._models.items():
            if not isinstance(model, JaxModel):
                raise ValueError(
                    f"role {role!r} holds {type(model).__name__}, not a JaxModel"
                )
        if weights_role not in self._models:
            raise ValueError(
                f"no model for weights role {weights_role!r}; this ensemble has "
                f"{sorted(self._models)}"
            )
        self._weights_role = weights_role
        self._split = None

    @property
    def shapes(self):
        """Replica/tensor shapes, taken from the weights model (see the contract)."""
        return self._models[self._weights_role].shapes

    @property
    def n_replicas(self):
        return self._models[self._weights_role].n_replicas

    def graph(self, role=GROUP_TRAINING):
        """The object playing ``role`` (backend-internal; n3fit uses :meth:`model`).

        For this backend the model *is* the object -- there is no raw graph behind the
        view -- so this and :meth:`model` coincide.  It exists so backend-internal code
        reads the same way for both backends.
        """
        try:
            return self._models[role]
        except KeyError as e:
            raise KeyError(
                f"no model for role {role!r}; this ensemble has {sorted(self._models)}"
            ) from e

    def model(self, role=GROUP_TRAINING):
        """The role's model as a contract :class:`Model` (same object on every call)."""
        return self.graph(role)

    def _snapshots(self):
        """The replicas of the weights model, as single-replica snapshot models."""
        if self._split is None:
            source = self._models[self._weights_role]
            members = []
            for replica in range(source.n_replicas):
                # Fresh dicts referencing the current arrays: later updates replace the
                # live dicts' entries, so these keep pointing at the snapshot.
                member = JaxModel(
                    [dict(source.params[replica])],
                    # pylint: disable=protected-access  (same backend: snapshots share
                    # the callables and the bound inputs of the model they come from)
                    dict(source._outputs),
                    inputs=dict(source._inputs),
                    name=f"{source.name}[{replica}]",
                    shapes=source.shapes,
                    trainable=source.trainable_paths,
                )
                if source.frozen:
                    member.freeze()
                members.append(member)
            self._split = JaxEnsembleView(members)
        return self._split

    def __iter__(self):
        """The replicas of the weights model, in order (snapshots, see above)."""
        return iter(self._snapshots())

    def __getitem__(self, replica):
        return self._snapshots()[replica]

    def __len__(self):
        return self.n_replicas

    def weights(self, role=None):
        """One weight map per replica, in replica order.

        With ``role=None`` the maps are the storage maps of the weights model.  A ``role``
        reads that *role's* model instead -- mirroring ``KerasRoleEnsemble``, whose ``role``
        likewise names a role graph rather than a contract section (the two Keras ensemble
        classes disagree with each other here; this backend follows each one's own reading
        and the inconsistency is flagged for P7).
        """
        model = self._models[self._weights_role] if role is None else self.graph(role)
        return [model._replica_map(i) for i in range(model.n_replicas)]

    def set_weights(self, values):
        """Restore per-replica storage maps snapshotted by the stopping hook."""
        model = self._models[self._weights_role]
        if len(values) != model.n_replicas:
            raise ValueError(
                f"expected {model.n_replicas} replica weight maps, got {len(values)}"
            )
        for i, replica_weights in enumerate(values):
            model._validate(i, replica_weights)
        for i, replica_weights in enumerate(values):
            model._assign(i, replica_weights)

    def _validate_replica(self, values, replica):
        """Backend-internal preflight for ``Backend.load`` on the weights model."""
        self._models[self._weights_role]._validate(replica, values)

    def _assign_replica(self, values, replica):
        """Backend-internal: load one replica's storage map (``Backend.load``)."""
        self._models[self._weights_role]._assign(replica, values)

    def __repr__(self):
        return f"<JaxEnsemble: roles={sorted(self._models)}>"
