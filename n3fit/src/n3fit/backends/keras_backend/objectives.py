"""
The Keras implementation of the objective terms (P3).

Today each term *is* a layer inserted into the training graph.  That is an implementation detail of
this backend and stays one: what the contract requires is :class:`n3fit.backends.base.Objective` --
a term is a function of the model's prediction, with data it can be handed and scalars it can be
told to change.  This module is the translation between the two, and it is also where the legacy
names that n3fit used to know about (``add_covmat``, ``update_mask``, the ``lagMult`` weight
mutation) are interpreted in contract terms.

Why the layers are reused rather than rewritten
-----------------------------------------------
Numerically, P3 must not change anything: the same Keras layer with the same weights and the same
``call`` gives bit-identical losses, which is what lets the refactor be verified by comparing fit
outputs.  What changes is *who knows what*: n3fit no longer builds ``LossInvcovmat`` with four
positional arguments and no longer pokes its weights -- it builds an :class:`ObjectiveSpec` and
calls ``set_data``/``set_scalar``.

The covariance rule (Q3, contract §"Objective.set_data")
--------------------------------------------------------
``set_data(covmat=C)`` means "this is the covariance the term should use" -- the adapter inverts it.
The k-fold diagnostic therefore passes the sum ``covmat + pdf_covmat``, which is exactly what
``LossInvcovmat.add_covmat`` computed internally (``np.linalg.inv(self._covmat + covmat)``), with the
same operands in the same order.

This module imports no framework: the layers are injected.  That keeps it testable in an environment
without keras (see ``tests/backend_conformance/test_objectives_offline.py``), which matters because it
is on the path of every fit.
"""

import weakref

import numpy as np

__all__ = [
    "KerasObjective",
    "adopt_layer",
    "build_objective",
    "ensure_built",
    "fed_prediction",
    "has_mask",
    "kind_of",
    "objective_schemas",
    "OBJECTIVE_SCHEMAS",
]


#: The schemas this backend declares: which ``data`` arrays a spec must carry, which ``scalars``
#: :meth:`KerasObjective.set_scalar` accepts, and which build-time ``options`` are meaningful.
#: This is what n3fit validates a requested metric against (replacing its hard-coded list), and what
#: ``Capabilities.objectives`` exposes.
OBJECTIVE_SCHEMAS = {
    "chi2": {
        "data": ("invcovmat", "covmat", "target"),
        "scalars": (),
        "options": (),
    },
    "positivity": {
        "data": (),
        "scalars": ("multiplier",),
        "options": ("alpha",),
    },
    "integrability": {
        "data": (),
        "scalars": ("multiplier",),
        "options": (),
    },
}


#: Which of a kind's declared ``data`` arrays :meth:`KerasObjective.set_data` can replace, per
#: kind (``mask`` is accepted generically, and only where the layer has one).  ``invcovmat`` and
#: ``target`` are *build-time*: the layer bakes them into a tensor (``_y_true``) and into the kernel
#: at ``build``, and nothing needs to replace them afterwards.  Kept as its own table, and checked
#: against ``OBJECTIVE_SCHEMAS`` by the conformance test, because a name that reaches ``set_data``
#: without being declared here would land on a layer weight -- which is how ``set_data(covmat=...)``
#: used to be able to overwrite a positivity term's multiplier.
_REPLACEABLE_DATA = {
    "chi2": ("covmat",),
    "positivity": (),
    "integrability": (),
}


#: The prediction each term has been applied to, keyed by the term's layer.
#:
#: The adapter builds the training graphs, so it is the only thing that ever knows which tensor
#: feeds which term -- and it has to remember, because the framework does not.  n3fit's graphs are
#: **eager** (the inputs carry their values in ``tensor_content`` and the trainable parameters
#: live inside the layers), so there is no functional graph to walk and no ``layer.input`` edge to
#: read: that attribute is the Keras 2 spelling and does not exist in Keras 3.  The legacy k-fold
#: diagnostic asked for ``layer.input`` and had been broken since the migration -- it is dead code
#: (contract Q15), which is why nobody noticed.  Recording it here is what the replacement uses.
_FED_PREDICTIONS = weakref.WeakKeyDictionary()


def fed_prediction(layer):
    """The prediction this adapter applied ``layer`` to, or ``None`` if it never did."""
    return _FED_PREDICTIONS.get(layer)


def has_mask(kind):
    """Whether a term of ``kind`` masks its data points.

    Derived from the schema rather than tabulated a second time: a term over data (``chi2``) has a
    per-point mask weight, the Lagrange penalties have no data to mask.  The k-fold reset writes a
    mask into terms of a model built elsewhere, so which kinds take one has to be answerable
    without the layer in hand.
    """
    return bool(OBJECTIVE_SCHEMAS[kind]["data"])


def ensure_built(layer):
    """Build ``layer`` if Keras has not done it yet.

    These layers build entirely from what their constructor was given (the inverse covariance, the
    target, the initial multiplier, the mask) and ignore the input shape, which is why
    ``layer.build(None)`` is both valid and enough -- verified for all four classes in
    ``tests/backend_conformance/test_objectives_keras.py``.  It matters because the k-fold reset
    addresses the terms of a model it did not train in that call: without this, ``set_scalar`` on
    such a term would fail with ``'LossLagrange' object has no attribute 'kernel'`` (which is
    exactly how the legacy ``add_covmat`` failed if it was called before the first forward pass).
    """
    if not getattr(layer, "built", True):
        layer.build(None)


def objective_schemas():
    """The declared schemas, as ``Capabilities.objectives`` wants them."""
    return {kind: dict(schema) for kind, schema in OBJECTIVE_SCHEMAS.items()}


class KerasObjective:
    """A contract term over one of the legacy loss layers.

    The wrapper does three things and nothing else: validate a spec against the schema, translate
    contract calls into layer operations, and expose the layer's value as a numpy array.
    """

    def __init__(self, spec, layer):
        self.spec = spec
        self._layer = layer
        self._schema = OBJECTIVE_SCHEMAS[spec.kind]
        # Bumped by every ``set_data``/``set_scalar``: the engine's compiled train step
        # captures the term layer's weights by value under ``jit`` (``KERAS_BACKEND=jax``),
        # so a mid-run change must recompile or training silently keeps the old value.
        self._generation = 0

    # ------------------------------------------------------------------ validation
    @staticmethod
    def validate(spec):
        """Check ``spec`` against the schema for its kind.

        Called at construction time by the backend, so a mismatched spec fails where it is written
        rather than in the middle of a fit.
        """
        if spec.kind not in OBJECTIVE_SCHEMAS:
            raise ValueError(
                f"Unknown objective kind {spec.kind!r}; this backend declares "
                f"{sorted(OBJECTIVE_SCHEMAS)}"
            )
        schema = OBJECTIVE_SCHEMAS[spec.kind]
        missing = sorted(set(schema["data"]) - set(spec.data))
        if missing:
            raise ValueError(
                f"objective {spec.kind!r} needs data {missing}; the spec has {sorted(spec.data)}"
            )
        # ``options`` may also carry the *initial value* of a declared scalar -- the Lagrange
        # multiplier is the case that matters: it has a starting value (a build-time decision,
        # ``model_trainer._LM_initial_and_multiplier``) and changes during the fit (a scalar).
        # Spelling it once, as a scalar, keeps the two from drifting.
        allowed_options = set(schema["options"]) | set(schema["scalars"])
        unknown_options = sorted(set(spec.options) - allowed_options)
        if unknown_options:
            raise ValueError(
                f"objective {spec.kind!r} accepts options {sorted(allowed_options)} "
                f"(plus the scalars it declares); got {unknown_options}"
            )
        return spec

    # ------------------------------------------------------------------ the term
    def apply(self, prediction):
        """Apply the term to a prediction inside a graph, returning the backend's tensor.

        For this backend the term *is* a layer, so this is the layer's ``__call__`` -- the exact
        operation ``model_gen`` used to perform by importing the loss class.  Nothing is converted:
        the result flows on into the rest of the graph (it is the model's output today).
        """
        if prediction is None:
            raise ValueError("Objective.apply needs a prediction to score")
        # remember the edge: this is how the adapter can later say what a term consumes
        _FED_PREDICTIONS[self._layer] = prediction
        return self._layer(prediction)

    def __call__(self, prediction):
        """The term's value for a model prediction (numpy in, numpy out).

        ``apply`` plus the conversion; the eager path used by diagnostics and tests.
        """
        return self._as_numpy(self.apply(prediction))

    def _as_numpy(self, value):
        """Backend tensor -> numpy, without importing the backend (duck-typed)."""
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    # ------------------------------------------------------------------ data
    def set_data(self, **arrays):
        """Replace part of the term's data.

        ``covmat`` follows the contract's replace-semantics: the term recomputes its inverse from
        what it is given.  ``mask`` replaces the term's mask weight.  What each kind accepts is
        ``_REPLACEABLE_DATA[kind]`` plus ``mask``; anything else raises, because the alternative is
        writing a covariance into whatever weight happens to be there.
        """
        allowed = _REPLACEABLE_DATA[self.spec.kind]
        for name, value in arrays.items():
            if name == "mask":
                if not has_mask(self.spec.kind):
                    raise ValueError(
                        f"objective {self.spec.kind!r} has no mask; only terms over data have one"
                    )
                self._set_mask(np.asarray(value))
            elif name in allowed:
                self._set_covmat(np.asarray(value))
            else:
                accepted = ", ".join(list(allowed) + ["mask"])
                raise ValueError(
                    f"objective {self.spec.kind!r} cannot take data {name!r}; "
                    f"it accepts {accepted}"
                )

    def _set_covmat(self, covmat):
        """``inv(covmat)`` into the layer's kernel (the legacy ``add_covmat``, given the sum)."""
        inverse = np.linalg.inv(covmat)
        ensure_built(self._layer)
        self._layer.kernel.assign(inverse)
        self._generation += 1
        # keep the spec honest: n3fit reads spec.data["covmat"] to build the sum it passes
        self.spec.data["covmat"] = covmat

    def _set_mask(self, mask):
        """The legacy ``update_mask``."""
        ensure_built(self._layer)
        self._layer.mask.assign(mask)
        self._generation += 1

    # ------------------------------------------------------------------ scalars
    def set_scalar(self, name, value):
        """Set a scalar declared in the schema.

        The Lagrange multiplier lives in a non-trainable one-element weight named ``lagMult``; this
        is the contract spelling of what ``LagrangeCallback`` used to do by multiplying that weight
        in place.  Note the difference in meaning: ``set_scalar`` *sets*, the legacy callback
        *scaled*, so the callback (a Hook, in P4) computes the new value and sets it.
        """
        if name not in self._schema["scalars"]:
            raise ValueError(
                f"objective {self.spec.kind!r} declares scalars {self._schema['scalars']}; "
                f"cannot set {name!r}"
            )
        if name == "multiplier":
            ensure_built(self._layer)
            self._layer.kernel.assign(np.asarray([value], dtype=np.float32))
            self._generation += 1
            self.spec.options["multiplier"] = value
        else:  # pragma: no cover - the schema above has a single scalar; guards future kinds
            raise NotImplementedError(f"set_scalar({name!r}) is not implemented for this kind")

    def scalar(self, name):
        """The current value of a declared scalar (see the contract's :meth:`Objective.scalar`)."""
        if name not in self._schema["scalars"]:
            raise ValueError(
                f"objective {self.spec.kind!r} declares scalars {self._schema['scalars']}; "
                f"cannot read {name!r}"
            )
        ensure_built(self._layer)
        return float(np.asarray(self._layer.kernel).reshape(-1)[0])

    # ------------------------------------------------------------------ reading back
    def mask(self):
        """The mask currently in the term, as numpy (the read side of ``set_data(mask=…)``).

        ``None`` for the kinds that have no mask at all (the Lagrange penalties).  For a term over
        data this is the layer's mask *weight* -- shape ``(1, 1, ndata)`` -- because that is exactly
        what the legacy ``update_mask`` was handed and what the k-fold diagnostic copies between
        models.
        """
        if not has_mask(self.spec.kind):
            return None
        ensure_built(self._layer)
        return np.asarray(self._as_numpy(self._layer.mask))

    # ------------------------------------------------------------------ introspection
    @property
    def name(self):
        return self.spec.name

    @property
    def kind(self):
        return self.spec.kind

    def __repr__(self):
        return f"<KerasObjective {self.spec.kind} {self.spec.name!r}>"


def kind_of(layer):
    """The contract kind a legacy loss layer implements, or ``None`` if it is not a term.

    This is the *only* place that maps this backend's layer classes to the contract's vocabulary,
    and it is what lets the adapter recognise the terms of a graph without a naming convention:
    a layer is a term if it is one of these classes, whatever it is called.  (Before this, the
    adapter found terms by matching names against ``.*_exp$``, which found the experimental chi2
    and silently missed positivity and integrability -- they are not named that way.)
    """
    from n3fit.layers import losses

    if isinstance(layer, losses.LossPositivity):
        return "positivity"
    if isinstance(layer, losses.LossIntegrability):
        return "integrability"
    if isinstance(layer, losses.LossInvcovmat):
        return "chi2"
    return None


def adopt_layer(layer, kind=None):
    """Wrap a loss layer that already exists in a graph as a contract term (P3).

    Used where n3fit is handed a *model built elsewhere* -- the k-fold diagnostic, the k-fold
    multiplier reset -- and must address that model's terms.  The spec is reconstructed from what
    the layer holds rather than from a caller: the original covariance is kept for the sum (Q3),
    and the data is validated against the schema like any other spec.  ``kind`` defaults to what
    the layer class implements (``kind_of``).
    """
    from n3fit.backends.base import ObjectiveSpec

    if kind is None:
        kind = kind_of(layer)
    if kind is None:
        raise ValueError(f"{type(layer).__name__} {layer.name!r} is not an objective layer")
    if kind == "chi2":
        data = {"covmat": getattr(layer, "_covmat", None)}
        options = {}
    else:
        # the Lagrange kinds carry their multiplier in a one-element weight
        ensure_built(layer)
        data = {}
        options = {"multiplier": float(np.asarray(layer.kernel)[0])}
    spec = ObjectiveSpec(kind=kind, name=layer.name, data=data, options=options)
    return KerasObjective(spec, layer)


def build_objective(spec, *, layer_class, ops, multiplier=None):
    """Build the layer for ``spec`` and wrap it.

    ``layer_class`` is looked up by n3fit's spec kind, so this function does not import the loss
    layers (which import the backend facade): the caller passes the class.  ``ops`` is the backend's
    tensor conversion, used to build the data tensors the layer expects.
    """
    KerasObjective.validate(spec)
    kind = spec.kind
    if kind == "chi2":
        layer = layer_class(
            ops.numpy_to_tensor(spec.data["invcovmat"]),
            ops.numpy_to_tensor(spec.data["target"]),
            spec.mask,
            covmat=spec.data.get("covmat"),
            name=spec.name,
        )
    elif kind == "positivity":
        layer = layer_class(name=spec.name, c=spec.options.get("multiplier", multiplier or 1.0),
                            alpha=spec.options.get("alpha", 1e-7))
    elif kind == "integrability":
        layer = layer_class(name=spec.name, c=spec.options.get("multiplier", multiplier or 1.0))
    else:  # pragma: no cover - validate() already rejected unknown kinds
        raise NotImplementedError(kind)
    return KerasObjective(spec, layer)
