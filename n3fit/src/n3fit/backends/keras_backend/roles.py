"""
Role-based access to a Keras graph: the ``Model`` contract over the objects n3fit builds today.

Why this file exists (P2)
-------------------------
Everything outside the backend used to reach into the graph *by name*::

    model.get_layer("add_photon").register_photon(grid)          # model_trainer, vpinterface
    model.get_layer(PREPROCESSING_LAYER_ALL_REPLICAS) \\
         .get_weight_by_name(f"alpha_{flav}").numpy()            # vpinterface
    model.get_layer("PDF_0").call = central_value                # rewards
    for key, grid in model.x_in.items(): ...                     # rewards

Those names are Keras' business.  A second backend should not have to invent a layer called
``add_photon``, so n3fit asks for *roles* instead (:mod:`n3fit.backends.base` declares the
vocabulary) and this module is the Keras implementation of that translation.  The map lives
here, on the backend side of the boundary, and nowhere else.

Why a view object rather than methods on ``MetaModel``
------------------------------------------------------
``MetaModel`` *is* a Keras ``Model``, and Keras already defines ``weights`` (a property
returning a list of variables) and ``set_weights`` (taking a list).  The contract's
``weights(role=...)`` returns a dict keyed by path, so implementing it on the graph object
itself would collide with the very API Keras uses internally.  Hence: a small wrapper,
obtained from ``Backend.view(graph)``, which holds the graph and mutates it in place.

This module must not import Keras
---------------------------------
It is written against a *duck-typed* graph: it only uses ``graph.layers``,
``graph.get_layer(name)``, ``layer.name``, ``layer.weights``, ``layer.built``,
``graph.compile()``, ``graph.summary()`` and ``layer.summary()``.  Keeping it Keras-free is
what makes it testable in an environment without the framework (see
``tests/backend_conformance/test_keras_role_view_offline.py``), which matters because this is
code that runs inside a fit.
"""

import numpy as np

from n3fit.backends.base import (
    GROUP_TRAINING,
    ROLE_NN,
    ROLE_PHOTON,
    ROLE_PREPROCESSING,
    ROLE_REFERENCE,
    ROLE_SUMRULE,
    ROLES,
)

__all__ = [
    "KerasEnsembleView",
    "KerasModelView",
    "KerasRoleEnsemble",
    "ROLE_LAYER_NAMES",
    "INPUT_HELD_IN_GRAPH",
    "as_view",
]


#: role -> the name of the layer that plays it, as the current model generators name them.
#: This table is the *only* place where n3fit's roles meet Keras' layer names; when P3 makes
#: the model constructor declare roles explicitly, this map becomes the default for graphs
#: that were built the old way.
ROLE_LAYER_NAMES = {
    ROLE_NN: "all_NNs",
    ROLE_PREPROCESSING: "preprocessing_factor",
    ROLE_PHOTON: "add_photon",
    ROLE_SUMRULE: "impose_msr",
    # The PDF model itself, wrapped as a nested layer by ``MetaModel.apply_as_layer`` -- the
    # section ``_set_central_value`` replaces.
    #
    # Provenance, because the name matters and has been wrong here before: this layer was called
    # ``PDF_0`` until commit b919773ef (2023-10-23) renamed it to ``PDFs``; that commit updated
    # model_trainer.py and missed ``_set_central_value``, so the override has been raising
    # ``No such layer: PDF_0`` ever since.  The evidence that ``PDFs`` is the right target is
    # that the summary chain (removed in P2) navigated ``training.get_layer("PDFs")`` and then
    # ``.get_layer("all_NNs")`` on exactly the kind of model that ``_set_central_value`` is given.
    ROLE_REFERENCE: "PDFs",
}

#: Inputs whose value lives in the graph rather than in the data (the x grid, the integration
#: grid).  ``bound_inputs`` reads them; ``bind_input`` may replace the photon one.
INPUT_HELD_IN_GRAPH = ("pdf_input", "scaledx_x", "xgrid_integration")

def tensor_to_numpy(value, ops=None):
    """Backend tensor -> numpy, without assuming which backend produced it.

    Keras *variables* have ``.numpy()`` (``ops.to_numpy`` is the contract spelling of that), but an
    evaluated tensor does not: under jax it is a ``jax.Array`` and there is no such method, so
    anything without ``.numpy`` goes through ``np.asarray``.  The engine reports every value it
    evaluates through this function, and keeping one implementation is what stops the adapter and
    the engine from disagreeing about conversion.
    """
    if hasattr(value, "numpy"):
        if ops is not None:
            return ops.to_numpy(value)
        return value.numpy()
    return np.asarray(value)


def as_view(graph, ops=None, model_builder=None):
    """The contract's :class:`Model` over ``graph``, or ``graph`` itself if it already is one.

    Idempotent on purpose: n3fit may hold either a raw graph (a fit result, code that has not
    been migrated) or a contract model (a member of an ensemble), and should not have to care
    which.  ``Backend.view`` is this function.
    """
    if isinstance(graph, KerasModelView):
        return graph
    return KerasModelView(graph, ops=ops, model_builder=model_builder)


class KerasModelView:
    """The :class:`n3fit.backends.base.Model` contract over an existing Keras graph."""

    def __init__(self, graph, ops=None, model_builder=None):
        self._graph = graph
        self._ops = ops
        self._model_builder = model_builder

    # ------------------------------------------------------------------ introspection
    @property
    def graph(self):
        """The wrapped graph (escape hatch; n3fit must not need it)."""
        return self._graph

    @staticmethod
    def _check_role(role):
        """Reject a role nobody declared: a typo must not look like an empty section."""
        if role is not None and role not in ROLES:
            raise ValueError(f"Unknown role {role!r}; known roles: {sorted(ROLES)}")

    def _role_layer(self, role):
        """The layer playing ``role``, or None if this graph has no such section.

        A graph legitimately has no photon (the fit has no photon) and no sum rule (the runcard
        does not impose one), so absence is normal here; the callers decide whether it matters.
        """
        self._check_role(role)
        name = ROLE_LAYER_NAMES.get(role)
        if name is None:
            # declared in the contract but this adapter has no name for it: never a silent no-op
            raise ValueError(f"role {role!r} is declared but has no layer name in this backend")
        try:
            return self._graph.get_layer(name)
        except (ValueError, KeyError, AttributeError):
            return None

    def summary(self):
        """Print the model and each role section it contains.

        Replaces the chain ``training.summary(); get_layer("PDFs").summary();
        get_layer("all_NNs").summary(); get_layer("impose_msr").summary()`` that used to live
        in ``model_trainer``: n3fit asks for one summary of the whole thing and the backend
        decides what that means.  (The four prints are expendable per D11; this keeps them
        available behind one call.)
        """
        self._graph.summary()
        for role in (ROLE_NN, ROLE_SUMRULE, ROLE_PREPROCESSING):
            layer = self._role_layer(role)
            if layer is None:
                continue
            print(f"--- role {role!r} ({layer.name}) ---")
            layer.summary()

    # ------------------------------------------------------------------ evaluation
    def __call__(self, inputs=None, **kwargs):
        """Evaluate the graph on numpy arrays and return numpy arrays.

        This is ``MetaModel.predict`` behind the contract, with the same semantics *including
        the shapes*: nothing is squeezed or averaged away, because the call sites index the
        result themselves (``y[0]``, ``np.concatenate(..., axis=0)``).  ``inputs`` needs only
        the slots that are not bound constants; ``None`` evaluates the graph as it was built,
        which is what the diagnostics do.
        """
        return self._graph.predict(inputs, **kwargs)

    # ------------------------------------------------------------------ weights
    def weights(self, role=None):
        """Weights as a numpy dict keyed by path (see the contract's path grammar).

        P2 implements the role n3fit actually reads today, ``preprocessing`` (the
        ``alpha``/``beta`` exponents per flavour).  The ``nn``, ``objective`` and ``sumrule``
        roles need the per-replica weight layout, which is the P4 work (the replica axis is
        currently folded into every kernel by ``MultiInitializer``); asking for them raises
        rather than returning something plausible-looking.

        Note the three distinct outcomes: an *unknown* role is a ``ValueError`` (a typo), a
        declared-but-unimplemented role is a ``NotImplementedError`` (work not done yet), and an
        implemented role with no matching weights is ``{}`` (a fit without that section).
        """
        self._check_role(role)
        if role is None:
            return dict(self.weights(ROLE_PREPROCESSING))
        if role == ROLE_PREPROCESSING:
            return self._preprocessing_weights()
        raise NotImplementedError(
            f"role {role!r} weights are not available yet (P4 defines the per-replica layout); "
            f"P2 implements {ROLE_PREPROCESSING!r}"
        )

    def _preprocessing_weights(self):
        """``{"preprocessing/alpha/<flavour>": array, "preprocessing/beta/<flavour>": array}``.

        The layer stores one weight per flavour, named ``alpha_<flavour>`` / ``beta_<flavour>``
        (see ``n3fit/layers/preprocessing.py``); the flavour string is kept verbatim so that the
        paths match the flavour names the runcard uses.
        """
        layer = self._role_layer(ROLE_PREPROCESSING)
        out = {}
        if layer is None:
            return out
        for weight in layer.weights:
            name = weight.name.split(":")[0].split("/")[-1]  # "alpha_up:0" -> "alpha_up"
            if name.startswith(("alpha_", "beta_")):
                out[f"{ROLE_PREPROCESSING}/{name.split('_', 1)[0]}/{name.split('_', 1)[1]}"] = (
                    self._to_numpy(weight)
                )
        return out

    def _to_numpy(self, value):
        """Backend tensor -> numpy (see :func:`tensor_to_numpy`)."""
        return tensor_to_numpy(value, ops=self._ops)

    # ------------------------------------------------------------------ inputs
    def bound_inputs(self):
        """The constants the graph carries as inputs, as numpy arrays.

        ``MetaModel`` collects these in ``x_in`` (inputs given a value at construction time, or
        carrying a ``tensor_content``).  Exposed by *role-free name* because they are data
        slots, not sections; the values are converted through the contract's ``ops.to_numpy``.
        """
        x_in = getattr(self._graph, "x_in", None)
        if x_in is None:
            return {}
        return {name: self._to_numpy(value) for name, value in x_in.items()}

    def bind_input(self, name, value):
        """Bind (or replace) a constant input.

        NOTE: a rebind *always* rebuilds the graph, because ``AddPhoton.register_photon``
        deliberately marks the layer unbuilt (the photon array's shape depends on the grid it
        was computed from -- see ``n3fit/layers/rotations.py``).  This mirrors what the two
        call sites did before P2.

        For the photon role this is what ``AddPhoton.register_photon`` + ``compile`` did: the
        layer recomputes its photon array from the new grid and the graph is rebuilt.  A graph
        with no photon section is a no-op -- that is the "fit without photons" case, and the
        caller does not have to know which it is.
        """
        if name != "photon":
            raise NotImplementedError(
                f"bind_input({name!r}): only the 'photon' input is rebindable in P2; "
                f"static inputs get their value at construction time"
            )
        layer = self._role_layer(ROLE_PHOTON)
        if layer is None:
            return
        layer.register_photon(value)
        if not getattr(layer, "built", True):
            self._graph.compile()

    # ------------------------------------------------------------------ objectives
    def objectives(self):
        """The model's loss terms, keyed by name, as contract objectives (P3).

        This is how n3fit talks about the terms of a model it did not build itself (the k-fold
        diagnostic takes models out of a fit; the k-fold reset puts the multipliers back): the
        *names* are the ones n3fit gave them when it built the graph, the wrapping is this
        adapter's business, and the layer is never handed out.

        A layer is a term if it *is* one of this backend's loss layers (``kind_of``), not if its
        name matches a pattern.  Which of them belong to which objective group is n3fit's business
        and comes from ``ObjectiveGroup``; this method just says what is in the graph.
        """
        from n3fit.backends.keras_backend.objectives import adopt_layer, kind_of

        terms = {}
        for layer in self._graph.layers:
            kind = kind_of(layer)
            if kind is not None:
                terms[layer.name] = adopt_layer(layer, kind=kind)
        return terms

    def _term_layer(self, name):
        """The loss layer of the term called ``name`` (the graph lookup, in one place)."""
        from n3fit.backends.keras_backend.objectives import kind_of

        for layer in self._graph.layers:
            if layer.name == name and kind_of(layer) is not None:
                return layer
        return None

    def objective(self, name):
        """One term by name (raises if this model has no such term -- never a silent no-op)."""
        layer = self._term_layer(name)
        if layer is None:
            raise ValueError(
                f"this model has no objective named {name!r}; it has "
                f"{sorted(self.objectives())}"
            )
        from n3fit.backends.keras_backend.objectives import adopt_layer

        return adopt_layer(layer)

    def prediction_before(self, name):
        """The prediction a term consumes, as a model: evaluate the graph up to that term.

        The k-fold diagnostic needs the model's *predictions* (to build a covariance), not its
        loss, which is expressed by building a second model whose output is the tensor feeding the
        loss layer.  Constructing it is graph plumbing, so it lives here; the caller gets a
        contract :class:`Model` and never sees a layer.

        The tensor itself comes from this adapter's own record of what it fed each term
        (``objectives.fed_prediction``), not from the framework: n3fit's graphs are eager, so there
        is no edge to read.  The legacy spelling of this -- ``MetaModel(model.input, layer.input)``
        in the diary of the future-test diagnostic, deleted in P4 -- cannot work in Keras 3
        (``layer.input`` no longer
        exists), which is one more reason that function needs a decision (contract Q15) rather than
        a port.
        """
        from n3fit.backends.keras_backend.objectives import fed_prediction

        if self._model_builder is None:
            raise NotImplementedError(
                "this view was built without a model builder, so it cannot construct graphs"
            )
        layer = self._term_layer(name)
        if layer is None:
            raise ValueError(f"this model has no objective named {name!r}")
        prediction = fed_prediction(layer)
        if prediction is None:
            raise ValueError(
                f"the objective {name!r} was not applied through this adapter, so the tensor it "
                f"consumes is unknown; terms built with Backend.objective(spec).apply() do know it"
            )
        inputs = getattr(self._graph, "input_tensors", None)
        if inputs is None:
            raise ValueError(
                "the wrapped graph does not expose its inputs as a dict (MetaModel.input_tensors), "
                "so a diagnostic model cannot be built from it"
            )
        return as_view(
            self._model_builder(inputs, prediction),
            ops=self._ops,
            model_builder=self._model_builder,
        )

    # ------------------------------------------------------------------ graph surgery
    def override(self, role, fn):
        """Replace the section playing ``role`` by the fixed function ``fn``.

        ``fn`` maps a mapping of numpy arrays to a numpy array; the conversion to whatever the
        backend wants is *this* adapter's business (that is what let ``operations`` leave
        ``hyper_optimization/rewards.py``).  Unlike :meth:`bind_input`, a missing section is an
        error: overriding something that is not there would silently not happen.
        """
        layer = self._role_layer(role)
        if layer is None:
            raise ValueError(
                f"no layer in this graph plays the role {role!r} "
                f"(looked for {ROLE_LAYER_NAMES.get(role)!r}); cannot override it"
            )
        ops = self._ops

        def call(inputs, training=None, **kwargs):  # pylint: disable=unused-argument
            result = fn(inputs)
            return ops.numpy_to_tensor(result) if ops is not None else result

        layer.call = call
        return layer

    def freeze(self):
        """Make the graph permanently non-trainable (``trainable = False`` + rebuild)."""
        self._graph.trainable = False
        self._graph.compile()

    # ------------------------------------------------------------------ data
    def set_data(self, **arrays):
        raise NotImplementedError("objective data is set through the Objective (P3)")

    def parameters(self, role=None):
        raise NotImplementedError(
            "trainable parameters in layout order are P7 (parameter-space optimizers/Hessian)"
        )

    def set_weights(self, values):
        """Restore the weights of replica 0 from a storage map (the inverse of the store's read).

        A single-replica graph has exactly that; a stacked multi-replica graph is ambiguous by
        design, so the multi-replica restore lives on the ensemble (``Ensemble.set_weights``),
        and this only ever addresses the first replica.
        """
        from n3fit.backends.keras_backend.weights import assign_weight_map

        assign_weight_map(self._graph, values, replica=0)

    def weights_view(self):
        from n3fit.backends.keras_backend.optimizer import KerasWeightsView

        return KerasWeightsView(self._graph, ops=self._ops)


class KerasEnsembleView:
    """The contract's :class:`Ensemble` over the replicas of a Keras graph.

    Iteration order *is* the replica order: n3fit counts replicas from 1 and the underlying
    list is the one ``MetaModel.split_replicas()`` returns, so this is a rename, not a
    renumbering.
    """

    def __init__(self, models, ops=None):
        self._ops = ops
        self._models = [as_view(model, ops=ops) for model in models]

    @classmethod
    def from_graph(cls, models, ops=None):
        """The ensemble of ``models``, which may be a single graph carrying its replicas stacked.

        That stacked shape is what n3fit has today, and splitting it is the backend's business
        (``MetaModel.split_replicas`` makes one single-replica graph per replica and copies that
        replica's weights into it) — it is why ``pdf_model.split_replicas()`` can disappear from
        n3fit while the behaviour stays identical.  ``strategy`` has no meaning until P4: the
        legacy graph already fixes how replicas share weights.
        """
        if isinstance(models, cls):
            return models
        if hasattr(models, "split_replicas"):
            return cls(models.split_replicas(), ops=ops)
        return cls(models, ops=ops)

    def __iter__(self):
        """Yield the replicas in order, each one a single-replica model view."""
        return iter(self._models)

    def __getitem__(self, replica):
        """The ``replica``-th model (0-based, as everywhere in the contract).

        Slices are allowed and stay an ensemble rather than degrading to a bare list, so a
        caller cannot accidentally end up holding raw graphs.
        """
        if isinstance(replica, slice):
            return KerasEnsembleView(self._models[replica], ops=self._ops)
        return self._models[replica]

    def __len__(self):
        return len(self._models)

    def weights(self, role=None):
        """One weight map per replica, in iteration order.

        With ``role=None`` the maps are the storage maps of the weight store
        (:mod:`n3fit.backends.keras_backend.weights`): ``{path: numpy array}``, keyed so that the
        same file loads into any graph the same generator built (P5).  A ``role`` keeps the P2
        reader semantics (per-section, e.g. the preprocessing exponents).
        """
        if role is not None:
            return [model.weights(role) for model in self._models]
        from n3fit.backends.keras_backend.weights import weight_map

        return [weight_map(model._graph, replica=0) for model in self._models]

    def set_weights(self, values):
        """Restore per-replica weights from storage maps (the inverse of :meth:`weights`)."""
        from n3fit.backends.keras_backend.weights import assign_weight_map

        if len(values) != len(self._models):
            raise ValueError(f"expected {len(self._models)} replica weight maps, got {len(values)}")
        for model, replica_weights in zip(self._models, values):
            assign_weight_map(model._graph, replica_weights, replica=0)

    def _assign_replica(self, values, replica):
        """Backend-internal: load one replica's storage map (``Backend.load``)."""
        from n3fit.backends.keras_backend.weights import assign_weight_map

        assign_weight_map(self._models[replica]._graph, values, replica=0)

    def __repr__(self):
        return f"<KerasEnsembleView: {len(self._models)} replicas>"

class KerasRoleEnsemble:
    """The contract's :class:`Ensemble` for a *fit*: the role graphs over one set of weights.

    A fit is three graphs, not one.  ``training`` is the graph the optimizer updates; ``validation``
    and ``experimental`` share its weights (n3fit builds them by re-applying the same PDF model, so
    the trainable layers are the *same objects*) and differ in which data is masked in and which
    terms are attached.  n3fit passes all three in, keyed by role, and the engine evaluates a group
    on the graph that owns it -- that is the whole reason the contract's ``Ensemble`` grew
    ``model(role)`` in P4.

    ``weights()``/``set_weights()`` address the replicas of the graph the fit's weights live in,
    through the weight store (P5): one ``{path: numpy array}`` map per replica, the same mapping
    a fit writes to disk and ``Stopping``'s best-weight snapshot snapshots/restores.  Iterating
    the ensemble is the expensive operation (it *splits* the replicas into single-replica graphs)
    and the engine never does it.
    """

    def __init__(self, graphs, ops=None, weights_graph=None):
        if not graphs:
            raise ValueError("an ensemble needs at least one role graph")
        self._ops = ops
        self._graphs = dict(graphs)
        self._views = {}
        self._split = None
        # The graph the fit's weights are stored in.  It is *not* the training graph: n3fit builds
        # the role graphs by re-applying the PDF model as a layer (``apply_as_layer``), so the
        # trainable layers live in the PDF model and are shared by every role graph.  The legacy
        # snapshot/restore (``Stopping``) addressed exactly that model, and so does this.
        self._weights_graph = weights_graph if weights_graph is not None else self._graphs[GROUP_TRAINING]

    @property
    def shapes(self):
        """Replica/tensor shapes, taken from the training graph (see the contract)."""
        return as_view(self._training_graph).shapes

    @property
    def _training_graph(self):
        return self._graphs[GROUP_TRAINING]

    def graph(self, role=GROUP_TRAINING):
        """The raw graph playing ``role`` (backend-internal; n3fit uses :meth:`model`)."""
        try:
            return self._graphs[role]
        except KeyError as e:
            raise KeyError(
                f"no graph for role {role!r}; this ensemble has {sorted(self._graphs)}"
            ) from e

    def model(self, role=GROUP_TRAINING):
        """The role's graph as a contract :class:`Model` (same object on every call)."""
        if role not in self._views:
            self._views[role] = as_view(self.graph(role), ops=self._ops)
        return self._views[role]

    def __iter__(self):
        """The replicas of the training graph, in order (builds single-replica graphs)."""
        if self._split is None:
            self._split = KerasEnsembleView.from_graph(self._training_graph, ops=self._ops)
        return iter(self._split)

    def __getitem__(self, replica):
        if self._split is None:
            self._split = KerasEnsembleView.from_graph(self._training_graph, ops=self._ops)
        return self._split[replica]

    @property
    def n_replicas(self):
        """How many replicas the graphs carry.

        Taken from the first output's replica axis rather than from ``MetaModel.num_replicas``,
        which reads ``output.shape[1]`` and therefore only works for single-output graphs -- a
        *training* graph has one output per term since P4 (and always did, in the sense that its
        outputs were the terms).
        """
        outputs = self._weights_graph.output
        first = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        return int(first.shape[1])

    def __len__(self):
        return self.n_replicas

    def weights(self, role=None):
        """One weight map per replica, in replica order.

        With ``role=None`` the maps are the storage maps of the weight store
        (:mod:`n3fit.backends.keras_backend.weights`): ``{path: numpy array}``, the same mapping
        a fit writes to disk and a hook snapshots (P5).  A ``role`` reads the corresponding role
        graph instead (the trainable layers are shared, so the values are the same objects).
        """
        graph = self._weights_graph if role is None else self.graph(role)
        from n3fit.backends.keras_backend.weights import weight_map

        return [weight_map(graph, replica=i) for i in range(self.n_replicas)]

    def set_weights(self, values):
        """Restore per-replica weights from storage maps (the snapshot/restore the stopping hook needs)."""
        from n3fit.backends.keras_backend.weights import assign_weight_map

        if len(values) != self.n_replicas:
            raise ValueError(f"expected {self.n_replicas} replica weight maps, got {len(values)}")
        for i, replica_weights in enumerate(values):
            assign_weight_map(self._weights_graph, replica_weights, replica=i)

    def _assign_replica(self, values, replica):
        """Backend-internal: load one replica's storage map (``Backend.load``)."""
        from n3fit.backends.keras_backend.weights import assign_weight_map

        assign_weight_map(self._weights_graph, values, replica=replica)

    def __repr__(self):
        return f"<KerasRoleEnsemble: roles={sorted(self._graphs)}>"
