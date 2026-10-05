"""
A numpy-only *test double* implementing the backend contract.

This is **not** a supported n3fit backend: it has no neural networks and no training.  It
exists for two reasons:

1. so that the conformance suite can be run against more than one implementation, which is
   what keeps the suite backend-agnostic -- a test that only passes for the Keras backend
   belongs to the Keras backend, not to the contract; and
2. to be the *reference implementation of the surface*: it implements every name of
   :class:`n3fit.backends.base.Ops` in numpy, which is the cheapest way to find out whether
   the declared primitives are actually enough to write the n3fit layers.

It also serves as the smallest worked example of what a new backend has to provide: an
object with ``name``, ``ops``, ``capabilities`` and ``state``, registered through the
public ``n3fit.backends.register_backend`` (see ``conftest.py``).

Names not in the contract (``numpy_to_tensor``, ``op_multiply``, ``tensor_product``, ...)
are the *legacy* Keras-backend vocabulary; they are kept here because the conformance suite
exercises them today and the Keras backend exposes them.  Reconciling the two vocabularies
is P1 work (see README.md).
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from n3fit.backends.base import role_of


class NumpyLayer:
    """A stand-in for the backend's layer type.

    In a real backend a layer is an object that has been *built* against a shape and can be
    composed into a graph.  In numpy there is no graph, so this carries the function and
    its arguments and applies them on call; enough for the conformance suite to check what
    the contract promises (that the helper returns something callable on tensors).
    """

    def __init__(self, fn, args=(), kwargs=None):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs or {}

    def __call__(self, *tensors):
        return self.fn(*tensors, *self.args, **self.kwargs)


class NumpyOps:
    """The primitive operation set, implemented directly with numpy.

    The grouping follows :class:`n3fit.backends.base.Ops`, so the two can be compared
    line by line (``test_capabilities.py`` does exactly that).
    """

    # ------------------------------------------------------------------ tensor algebra
    einsum = staticmethod(np.einsum)
    tensordot = staticmethod(np.tensordot)
    matmul = staticmethod(np.matmul)
    sum = staticmethod(np.sum)

    @staticmethod
    def nansum(x, axis=None):
        """Like ``numpy.nansum``: NaN treated as 0."""
        return np.nansum(x, axis=axis)

    nan_to_num = staticmethod(np.nan_to_num)

    # ------------------------------------------------------------------ elementwise
    pow = staticmethod(np.power)
    log = staticmethod(np.log)
    clip = staticmethod(np.clip)
    absolute = staticmethod(np.absolute)
    tanh = staticmethod(np.tanh)

    @staticmethod
    def elu(x, alpha=1.0):
        x = np.asarray(x)
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def multiply(x, y):
        """Elementwise product.

        NOTE: not (yet) a member of the ``Ops`` protocol -- see the P0 report.  Careful: this
        is *not* the same operation as the legacy ``op_multiply`` (below), which follows the
        Keras layer convention of taking a single list of tensors.
        """
        return x * y

    @staticmethod
    def subtract(x, y):
        """Elementwise difference (see the note on ``multiply``/``op_subtract``)."""
        return x - y

    # ------------------------------------------------------------------ shape
    reshape = staticmethod(np.reshape)
    transpose = staticmethod(np.transpose)
    expand_dims = staticmethod(np.expand_dims)
    concatenate = staticmethod(np.concatenate)
    stack = staticmethod(np.stack)
    repeat = staticmethod(np.repeat)

    @staticmethod
    def split(x, indices, axis=0):
        return np.split(x, indices, axis=axis)

    @staticmethod
    def gather(x, indices, axis=0):
        return np.take(x, indices, axis=axis)

    @staticmethod
    def scatter_update(x, indices, values):
        """Out-of-place equivalent of ``variable[indices] = values`` along axis 0."""
        out = np.array(x, copy=True)
        out[indices] = values
        return out

    # ------------------------------------------------------------------ creation / conversion
    @staticmethod
    def constant(value, dtype=None):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype="float32"):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(shape, dtype="float32"):
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def cast(x, dtype):
        return np.asarray(x, dtype=dtype)

    @staticmethod
    def to_numpy(x):
        return np.asarray(x)

    # ------------------------------------------------------------------ layer helpers
    @staticmethod
    def as_layer(fn, **kwargs):
        return NumpyLayer(fn, kwargs=kwargs)

    @staticmethod
    def as_input(value, name=None):
        """A value-level stand-in for an input slot: numpy arrays are values, not slots."""
        return np.asarray(value)

    @staticmethod
    def splitter(shape, sizes, axis=0, name="splitter"):
        def split_it(x):
            return np.split(x, np.cumsum(sizes)[:-1], axis=axis)

        return NumpyLayer(split_it)

    # ------------------------------------------------------------------ legacy vocabulary
    # The names the Keras backend exposes and the code currently calls.  They are aliases
    # of the contract names wherever the semantics coincide, so that the two implementations
    # agree by construction rather than by convention.
    numpy_to_tensor = staticmethod(constant)
    tensor_to_numpy_or_python = staticmethod(to_numpy)
    op_log = log
    tensor_product = tensordot
    flatten = staticmethod(np.ravel)

    # The two "ops" below are Keras *layers* (keras.layers.multiply/subtract) applied through
    # a list of tensors, which is why they cannot simply alias the contract names above --
    # an example of a pair that needs a real adapter rather than a rename.
    op_multiply = staticmethod(lambda xs: xs[0] * xs[1])
    op_subtract = staticmethod(lambda xs: xs[0] - xs[1])

    @staticmethod
    def dict_to_numpy_or_python(mapping):
        return {key: np.asarray(value) for key, value in mapping.items()}

    @staticmethod
    def c_to_py_fun(op_name, name="dataset"):
        """The NNPDF operation vocabulary; delegate to the same source the Keras backend uses.

        (In the target design this mapping belongs to n3fit -- it is the physics vocabulary,
        not a backend primitive -- and the backend only provides the machinery around it.)
        """
        from validphys.convolution import OP

        try:
            operation = OP[op_name]
        except KeyError as err:
            raise ValueError(f"Operation {op_name} not recognised") from err

        def operate_on_tensors(tensor_list):
            return operation(*tensor_list)

        return operate_on_tensors


@dataclass(frozen=True)
class NumpyCapabilities:
    """The (deliberately tiny) capabilities of the test double."""

    parametrizations: Mapping[str, Any] = None
    optimizers: Mapping[str, Any] = None
    objectives: Mapping[str, Mapping[str, Any]] = None
    derivatives: frozenset = frozenset({"jacobian"})
    initializers: Mapping[str, Any] = None
    activations: frozenset = frozenset({"linear"})
    regularizers: Mapping[str, Any] = None
    dtypes: frozenset = frozenset({"float32", "float64"})
    train_n_replicas_together: bool = False
    supports_weight_mutation: bool = False
    supports_tensorboard: bool = False
    fast_single_replica_convolution: bool = False
    requires_eager_workaround: bool = False

    def __post_init__(self):
        # frozen dataclass: set the mutable defaults through object.__setattr__
        if self.parametrizations is None:
            object.__setattr__(
                self,
                "parametrizations",
                {"polynomial": {"options": {"degree": int}, "linear_in_parameters": False}},
            )
        if self.optimizers is None:
            object.__setattr__(
                self,
                "optimizers",
                {
                    "levenberg_marquardt": {
                        "options": {"max_iter": int},
                        "is_iterative": True,
                        "requires": {"jacobian"},
                        "uses_validation_stopping": False,
                    }
                },
            )
        if self.objectives is None:
            # The double declares two kinds, with the same *shape* of schema the Keras backend
            # declares (see P3: `objectives` is a schema mapping, not a set of names).
            object.__setattr__(self, "objectives", dict(NUMPY_OBJECTIVE_SCHEMAS))
        if self.initializers is None:
            object.__setattr__(self, "initializers", {"zeros": {"options": {}}})
        if self.regularizers is None:
            object.__setattr__(self, "regularizers", {})


class NumpyState:
    """No global state worth configuring; kept so the contract is honoured."""

    def __init__(self):
        self.configuration = {}

    def configure(self, **kwargs):
        self.configuration = dict(kwargs)

    def clear(self):
        self.configuration = {}

    def set_eager(self, enabled):
        """Recorded rather than acted upon: numpy is always eager."""
        self.configuration["eager"] = enabled

    def devices(self):
        return ["cpu"]


def schema_has_mask(kind):
    """Whether a kind's terms mask their data points (the contract rule: terms over data do)."""
    return bool(NUMPY_OBJECTIVE_SCHEMAS[kind]["data"])


#: The kinds the double implements.  Deliberately the same two shapes of term the Keras backend
#: declares (one over data, one over a scalar), so the conformance tests exercise both branches.
NUMPY_OBJECTIVE_SCHEMAS = {
    "chi2": {"data": ("invcovmat", "target"), "scalars": (), "options": ()},
    "positivity": {"data": (), "scalars": ("multiplier",), "options": ()},
}


class NumpyObjective:
    """The contract's :class:`Objective` in numpy (P3).

    Small on purpose, but *complete*: the double is what catches the day the contract's term
    semantics change, and it does that only if it implements every method a term has.  The two
    kinds it knows are real terms (a chi2 over a residual, a positivity penalty), not stubs, so
    the conformance tests can compare a backend's numbers against them.
    """

    def __init__(self, spec, backend):
        self.spec = spec
        self._backend = backend
        schema = backend.capabilities.objectives.get(spec.kind)
        if schema is None:
            raise ValueError(
                f"Unknown objective kind {spec.kind!r}; this backend declares "
                f"{sorted(backend.capabilities.objectives)}"
            )
        missing = [name for name in schema["data"] if name not in spec.data]
        if missing:
            raise ValueError(f"objective {spec.kind!r} needs data {missing}")
        # the same validation rule as the backend implementations: options may set the initial
        # value of a declared scalar, anything else is an error rather than a silent no-op
        allowed = set(schema["options"]) | set(schema["scalars"])
        unknown = sorted(set(spec.options) - allowed)
        if unknown:
            raise ValueError(
                f"objective {spec.kind!r} accepts options {sorted(allowed)}; got {unknown}"
            )
        self._scalars = {name: 0.0 for name in schema["scalars"]}
        for name, value in spec.options.items():
            if name in self._scalars:
                self._scalars[name] = float(value)
        self._data = dict(spec.data)
        # A term over data always has a mask: with no mask given it masks nothing, which is the
        # all-ones mask the Keras layers build themselves.  ``None`` means "this kind has no mask
        # at all" (the Lagrange penalties), and the two must not be confused -- the k-fold reset
        # copies the mask of one model's term into another's.
        if schema_has_mask(spec.kind):
            self._mask = np.ones_like(np.asarray(spec.data["target"]), dtype=float) \
                if spec.mask is None else np.asarray(spec.mask, dtype=float)
        else:
            self._mask = None

    # ------------------------------------------------------------------ the term
    def apply(self, prediction):
        """The double has one mode -- a tensor is an array here -- so this is ``__call__``."""
        return self.__call__(prediction)

    def __call__(self, prediction):
        x = np.asarray(prediction, dtype=float)
        if self.spec.kind == "chi2":
            residual = np.asarray(self._data["target"], dtype=float) - x
            if self._mask is not None:
                residual = residual * self._mask
            return np.einsum("...i,ij,...j->...", residual, self._data["invcovmat"], residual)
        if self.spec.kind == "positivity":
            return self._scalars["multiplier"] * float(np.sum(np.maximum(-x, 0.0)))
        raise NotImplementedError(self.spec.kind)  # pragma: no cover - guarded in __init__

    # ------------------------------------------------------------------ data and scalars
    def set_data(self, **arrays):
        if not arrays:
            return
        # a term over data has a mask (same derived rule as the Keras adapter and the schemas)
        replaceable = {"covmat", "mask"} if schema_has_mask(self.spec.kind) else set()
        unknown = sorted(set(arrays) - replaceable)
        if unknown:
            raise ValueError(
                f"objective {self.spec.kind!r} cannot take data {unknown}; "
                f"it accepts {sorted(replaceable)}"
            )
        for name, value in arrays.items():
            if name == "mask":
                if value is None:
                    raise ValueError("the mask of a term over data is an array, not None")
                self._mask = np.asarray(value, dtype=float)
            else:  # covmat: the term recomputes what it needs from the covariance it is given
                self._data["invcovmat"] = np.linalg.inv(np.asarray(value, dtype=float))
                self._data["covmat"] = np.asarray(value, dtype=float)

    def set_scalar(self, name, value):
        if name not in self._scalars:
            raise ValueError(
                f"objective {self.spec.kind!r} declares scalars {sorted(self._scalars)}; "
                f"cannot set {name!r}"
            )
        self._scalars[name] = float(value)

    def mask(self):
        return None if self._mask is None else np.array(self._mask)

    @property
    def kind(self):
        return self.spec.kind

    @property
    def name(self):
        return self.spec.name

    def __repr__(self):
        return f"<NumpyObjective {self.spec.kind} {self.spec.name!r}>"


class NumpyModelView:
    """A minimal, honest implementation of the contract's :class:`Model` (P2).

    It exists so the contract can be *exercised* without a framework: the conformance suite
    builds one of these around a dict of arrays and checks the promised behaviour (roles are
    validated, ``weights`` is keyed by path, ``bind_input`` substitutes and the graph still
    evaluates, ``override``+``freeze`` freeze a section).  It is deliberately the smallest thing
    that can be wrong: if the contract drifts, this double stops matching it.
    """

    def __init__(self, sections, inputs=None):
        self.sections = dict(sections)
        self.inputs = dict(inputs or {})
        self.frozen = False

    def summary(self):
        print(f"numpy double: sections {sorted(self.sections)}, inputs {sorted(self.inputs)}")

    def weights(self, role=None):
        weights = {}
        for name, section in self.sections.items():
            for key, value in section.get("weights", {}).items():
                weights[f"{name}/{key}"] = np.asarray(value)
        if role is None:
            return weights
        return {key: value for key, value in weights.items() if role_of(key) == role}

    def bound_inputs(self):
        return {name: np.asarray(value) for name, value in self.inputs.items()}

    def bind_input(self, name, value):
        if name in self.inputs:
            self.inputs[name] = np.asarray(value)
            return
        raise KeyError(f"no bound input {name!r}; the double has {sorted(self.inputs)}")

    def override(self, role, fn):
        for name, section in self.sections.items():
            if role_of(name) == role:
                section["override"] = fn
                return section
        raise ValueError(f"no section plays the role {role!r}")

    def freeze(self):
        self.frozen = True

    def __call__(self, inputs=None):
        """Evaluate the graph, numpy in and numpy out (``Model.__call__``).

        NOTE the one place this double is not literal: the contract types the result as a single
        array (a model has one output), while the double's graphs are *dictionaries of sections*
        (that is what makes ``override``/``weights`` testable here), so it returns the mapping
        keyed by section.  The property the tests need -- numpy in, numpy out, shapes preserved
        -- holds either way.  The point of ``bind_input`` is that this still works after it.
        """
        out = {}
        for name, section in self.sections.items():
            if "override" in section:
                out[name] = section["override"]({"inputs": {**self.inputs, **(inputs or {})}})
            else:
                out[name] = section.get("value")
        return out


class NumpyEnsemble:
    """The reference :class:`Ensemble`: a list of models, presented as the contract wants.

    The double implements it over an explicit list of models, which is what a second backend
    would do; the Keras adapter additionally *produces* that list from a stacked graph
    (``KerasEnsembleView``), which is the migration case.
    """

    def __init__(self, models):
        self._models = [
            model if isinstance(model, NumpyModelView) else NumpyModelView(model) for model in models
        ]

    def __iter__(self):
        return iter(self._models)

    def __getitem__(self, replica):
        if isinstance(replica, slice):
            return NumpyEnsemble(self._models[replica])
        return self._models[replica]

    def __len__(self):
        return len(self._models)

    def weights(self, role=None):
        return [model.weights(role) for model in self._models]

    def set_weights(self, values):
        """Write per-replica ``{path: array}`` maps back into the models (D7's mutable half).

        The counterpart of :meth:`weights`, and the reference behaviour for the adapters: the
        mapping a caller gets out of ``weights()`` can be edited in place and handed back.
        """
        if len(values) != len(self._models):
            raise ValueError(f"got {len(values)} weight maps for {len(self._models)} replicas")
        for model, mapping in zip(self._models, values):
            for path, array in mapping.items():
                section, _, key = path.rpartition("/")
                store = model.sections[section]["weights"]
                if key not in store:
                    raise KeyError(f"no weight {key!r} in section {section!r}")
                array = np.asarray(array)
                if array.shape != np.shape(store[key]):
                    raise ValueError(
                        f"weight {path!r} has shape {np.shape(store[key])}, got {array.shape}"
                    )
                store[key] = np.array(array, copy=True)


class NumpyDoubleBackend:
    """The test double itself."""

    name = "numpy_test_double"
    version = "0"

    def __init__(self):
        self.ops = NumpyOps()
        self.capabilities = NumpyCapabilities()
        self.state = NumpyState()

    def view(self, graph):
        """Wrap an already-built graph (a :class:`NumpyModelView`) in the contract's API.

        Idempotent, as the Keras one is: viewing a view returns it unchanged.
        """
        return graph if isinstance(graph, NumpyModelView) else NumpyModelView(graph)

    def objective(self, spec):
        """Build a term from a spec (P3): the double does not compile anything, so this is
        just validation + construction."""
        if getattr(spec, "mask", None) is not None and spec.kind != "chi2":
            raise ValueError(f"objective {spec.kind!r} does not take a mask")
        return NumpyObjective(spec, self)

    def ensemble(self, models):
        """The replicas of ``models``.

        The double takes an explicit sequence (a single object that carries its replicas
        stacked is a *legacy* shape, and only the Keras adapter has to know how to split it).
        """
        if isinstance(models, NumpyEnsemble):
            return models
        return NumpyEnsemble(models)

    def version_info(self):
        return {"backend": self.name, "numpy": np.__version__}
