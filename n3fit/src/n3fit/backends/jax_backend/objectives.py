"""The contract's objective terms, as pure JAX functions (P6).

Each term is a function of a prediction shaped ``(batch, replicas, ndata)`` (the legacy
layers' convention, kept so that values agree with the Keras backend term by term):

* ``chi2``: ``sum_ij (target - pred)_i invcovmat_ij (target - pred)_j`` with a per-point
  mask, one value per replica -- the same bilinear form ``LossInvcovmat`` computes;
* ``positivity``: ``sum(elu(-multiplier * pred))`` over batch and points;
* ``integrability``: ``sum((multiplier * pred)^2)`` over batch and points.

Data lives in the term as JAX arrays (``float32`` unless the backend was configured for
``float64``); ``set_data``/``set_scalar`` replace them, exactly as the Keras adapter
replaces the layer weights.  The engine differentiates through :meth:`JaxObjective.forward`
-- the single source of truth both ``apply`` and ``__call__` wrap -- so the evaluated term
and the differentiated term cannot disagree.
"""

import jax.numpy as jnp
import numpy as np

__all__ = [
    "OBJECTIVE_SCHEMAS",
    "JaxObjective",
    "build_objective",
    "has_mask",
    "objective_schemas",
]

#: The schemas this backend declares, in the same shape the Keras backend declares them.
#: A backend that wants a new metric declares its kind here and implements the term.
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

#: Which of a kind's declared ``data`` arrays :meth:`JaxObjective.set_data` can replace
#: (``mask`` is accepted generically, and only where the kind has one).  ``invcovmat`` and
#: ``target`` are *build-time*; ``covmat`` is replaceable because the k-fold diagnostic adds
#: the PDF covariance to it.  Anything else raises -- the alternative is writing a covariance
#: into whatever storage happens to be there.
_REPLACEABLE_DATA = {
    "chi2": ("covmat",),
    "positivity": (),
    "integrability": (),
}


def objective_schemas():
    """The declared schemas, as ``Capabilities.objectives`` wants them."""
    return {kind: dict(schema) for kind, schema in OBJECTIVE_SCHEMAS.items()}


def has_mask(kind):
    """Whether a term of ``kind`` masks its data points.

    Derived from the schema rather than tabulated a second time: a term over data (``chi2``)
    has a per-point mask, the Lagrange penalties have no data to mask.
    """
    return bool(OBJECTIVE_SCHEMAS[kind]["data"])


def _as_backend_array(value, dtype):
    """Numpy in, JAX array in the backend's dtype out."""
    return jnp.asarray(np.asarray(value), dtype=dtype)


class JaxObjective:
    """A contract term over the pure functions of this module."""

    def __init__(self, spec, dtype="float32"):
        JaxObjective.validate(spec)
        self.spec = spec
        self._schema = OBJECTIVE_SCHEMAS[spec.kind]
        self._dtype = np.dtype(dtype)
        # Bumped by every ``set_data``: the engine captures term data in its compiled step
        # and re-traces when this moves, so a mid-run ``set_data`` can never silently lose
        # to a stale closure (``set_scalar`` does not bump it -- scalars are arguments).
        self._generation = 0
        kind = spec.kind
        if kind == "chi2":
            invcovmat = np.asarray(spec.data["invcovmat"])
            if invcovmat.ndim != 2:
                raise ValueError(
                    f"objective 'chi2' needs a 2D invcovmat, got shape {invcovmat.shape}"
                )
            target = np.asarray(spec.data["target"])
            self._ndata = int(target.shape[-1])
            if invcovmat.shape != (self._ndata, self._ndata):
                raise ValueError(
                    f"objective 'chi2': invcovmat has shape {invcovmat.shape} but the "
                    f"target has {self._ndata} points"
                )
            self._invcovmat = _as_backend_array(invcovmat, self._dtype)
            self._target = _as_backend_array(target, self._dtype)
            self._mask = _as_backend_array(self._initial_mask(spec.mask), self._dtype)
        else:
            self._ndata = 0
            self._multiplier = self._dtype.type(spec.options.get("multiplier", 1.0))
            self._alpha = float(spec.options.get("alpha", 1e-7))

    def _initial_mask(self, mask):
        """The starting mask: ones, unless the spec carries a real one.

        Mirrors the legacy layer (``LossInvcovmat.__init__``): ``None`` or an all-true mask
        means "no masking".  Always stored as ``(1, 1, ndata)`` -- the shape the k-fold
        reset writes and :meth:`mask` reads back.
        """
        if mask is None or bool(np.all(np.asarray(mask))):
            return np.ones((1, 1, self._ndata))
        reshaped = np.asarray(mask, dtype=np.float32).reshape((1, 1, -1))
        if reshaped.shape[-1] != self._ndata:
            raise ValueError(
                f"objective 'chi2': mask has {reshaped.shape[-1]} points but the target "
                f"has {self._ndata}"
            )
        return reshaped

    # ------------------------------------------------------------------ validation
    @staticmethod
    def validate(spec):
        """Check ``spec`` against the schema for its kind (same rules as the Keras one)."""
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
        # ``options`` may also carry the *initial value* of a declared scalar (the Lagrange
        # multiplier's starting value is a build-time decision; the schedule then moves it).
        allowed_options = set(schema["options"]) | set(schema["scalars"])
        unknown_options = sorted(set(spec.options) - allowed_options)
        if unknown_options:
            raise ValueError(
                f"objective {spec.kind!r} accepts options {sorted(allowed_options)} "
                f"(plus the scalars it declares); got {unknown_options}"
            )
        return spec

    # ------------------------------------------------------------------ the term
    def forward(self, prediction, scalars=None):
        """The term's value for a prediction, as a JAX array (the differentiated path).

        ``prediction`` is ``(batch, replicas, ndata)``; the result is one value per replica.
        ``scalars`` overrides the stored scalar values for this call (``{"multiplier": v}``)
        -- it is how the engine evaluates without recompiling when the Lagrange schedule
        moves the multiplier.
        """
        prediction = jnp.asarray(prediction, dtype=self._dtype)
        kind = self.spec.kind
        if kind == "chi2":
            residual = (self._target - prediction) * self._mask
            return jnp.einsum("bri,ij,brj->r", residual, self._invcovmat, residual)
        multiplier = self._multiplier if scalars is None else scalars.get("multiplier", self._multiplier)
        scaled = jnp.asarray(multiplier, dtype=self._dtype) * prediction
        if kind == "positivity":
            # ``elu(-pred)`` (``LossPositivity``): the sign is the penalty -- a positive
            # prediction must cost ~0 and a negative one must cost the multiplier times it.
            negated = -scaled
            penalty = jnp.where(
                negated > 0, negated, self._alpha * (jnp.exp(negated) - 1)
            )
            return jnp.sum(penalty, axis=(0, -1))
        return jnp.sum(scaled * scaled, axis=(0, -1))

    def apply(self, prediction):
        """Apply the term to a prediction, returning the backend's tensor.

        For this backend the term *is* a pure function, so this is :meth:`forward` -- the
        exact operation the engine differentiates through.  Nothing is converted.
        """
        if prediction is None:
            raise ValueError("Objective.apply needs a prediction to score")
        return self.forward(prediction)

    def __call__(self, prediction):
        """The term's value for a model prediction (numpy in, numpy out)."""
        return np.asarray(self.apply(prediction))

    # ------------------------------------------------------------------ data
    def set_data(self, **arrays):
        """Replace part of the term's data (same replace-semantics as the Keras term)."""
        allowed = _REPLACEABLE_DATA[self.spec.kind]
        if arrays:
            self._generation += 1
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
        """``inv(covmat)`` into the term (the legacy ``add_covmat``, given the sum)."""
        # Inverted on the host in float64, stored in the backend dtype -- the same two steps
        # the legacy layer performs (``np.linalg.inv`` upcasts; the weight casts back).
        inverse = np.linalg.inv(np.asarray(covmat, dtype=np.float64))
        self._invcovmat = _as_backend_array(inverse, self._dtype)
        # keep the spec honest: n3fit reads spec.data["covmat"] to build the sum it passes
        self.spec.data["covmat"] = covmat

    def _set_mask(self, mask):
        """The legacy ``update_mask`` (stored as ``(1, 1, ndata)``)."""
        reshaped = np.asarray(mask).reshape((1, 1, -1))
        if reshaped.shape[-1] != self._ndata:
            raise ValueError(
                f"objective 'chi2': mask has {reshaped.shape[-1]} points but the term "
                f"has {self._ndata}"
            )
        self._mask = _as_backend_array(reshaped, self._dtype)

    # ------------------------------------------------------------------ scalars
    def set_scalar(self, name, value):
        """Set a scalar declared in the schema (``set`` semantics, like the Keras term)."""
        if name not in self._schema["scalars"]:
            raise ValueError(
                f"objective {self.spec.kind!r} declares scalars {self._schema['scalars']}; "
                f"cannot set {name!r}"
            )
        if name == "multiplier":
            # Stored in the backend dtype: the Keras term keeps the multiplier in a float32
            # weight, so storing float64 here would make the two backends' Lagrange series
            # differ at 1e-7 from the very first firing.
            self._multiplier = self._dtype.type(value)
            self.spec.options["multiplier"] = value
        else:  # pragma: no cover - the schema above has a single scalar; guards future kinds
            raise NotImplementedError(f"set_scalar({name!r}) is not implemented for this kind")

    def scalar(self, name):
        """The current value of a declared scalar (the read side of :meth:`set_scalar`)."""
        if name not in self._schema["scalars"]:
            raise ValueError(
                f"objective {self.spec.kind!r} declares scalars {self._schema['scalars']}; "
                f"cannot read {name!r}"
            )
        return float(np.asarray(self._multiplier).reshape(-1)[0])

    # ------------------------------------------------------------------ reading back
    def mask(self):
        """The mask currently in the term, as numpy (``None`` for the kinds without one)."""
        if not has_mask(self.spec.kind):
            return None
        return np.asarray(self._mask)

    # ------------------------------------------------------------------ introspection
    @property
    def name(self):
        return self.spec.name

    @property
    def kind(self):
        return self.spec.kind

    def __repr__(self):
        return f"<JaxObjective {self.spec.kind} {self.spec.name!r}>"


def build_objective(spec, dtype="float32"):
    """Validate ``spec`` against the schema and wrap it (the backend calls this)."""
    return JaxObjective(spec, dtype=dtype)
