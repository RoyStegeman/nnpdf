"""The contract's ``Ops``, implemented with raw JAX (P6).

Every n3fit custom layer is written against :class:`n3fit.backends.base.Ops` and nothing
else, so this module *is* the physics/backend boundary for this backend.  The numerics are
pinned by ``tests/backend_conformance/test_ops.py`` (run against every backend) and the
vocabulary by ``test_capabilities.py::test_ops_surface_matches_the_contract``.

The legacy names (``numpy_to_tensor``, ``op_log``, ``op_multiply``, ``tensor_product``,
``flatten``, ``c_to_py_fun``) are served here because ``test_ops.py`` still calls them and
``test_ops_names_used_by_the_suite_exist_everywhere`` requires every backend to have every
name the suite calls.  Their semantics are the legacy ones: ``op_multiply`` takes a *list*
of tensors (unlike the elementwise contract ``multiply``), and ``flatten`` takes whatever
``numpy.ravel`` takes.
"""

import jax.numpy as jnp
import numpy as np

__all__ = ["JaxOps", "JaxLayer"]


class JaxLayer:
    """A callable taking tensors to tensors (the contract's ``as_layer``/``splitter``)."""

    def __init__(self, fn, args=(), kwargs=None):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs or {}

    def __call__(self, *tensors):
        return self.fn(*tensors, *self.args, **self.kwargs)


def _arraylike(x):
    """What ``jnp`` will not take but numpy will: a sequence of tensors.

    The contract promises numpy semantics, and ``np.sum([t])``/``np.log([t])`` stack
    their input first -- including the legacy list-taking call sites ``test_ops.py`` kept.
    ``jnp`` rejects a bare list, so sequences are stacked here and nowhere else.
    """
    if isinstance(x, (list, tuple)):
        return jnp.asarray(x)
    return x


class JaxOps:
    """The contract's operations, implemented with ``jax.numpy``.

    Structural conformance is what the contract requires (``Ops`` is a ``Protocol``), so
    this is a plain object: no inheritance, no registration.
    """

    # ------------------------------------------------------------------ tensor algebra
    @staticmethod
    def einsum(subscripts, *tensors):
        return jnp.einsum(subscripts, *tensors)

    @staticmethod
    def tensordot(a, b, axes):
        return jnp.tensordot(a, b, axes=axes)

    @staticmethod
    def matmul(a, b):
        return jnp.matmul(a, b)

    @staticmethod
    def sum(x, axis=None, keepdims=False):
        return jnp.sum(_arraylike(x), axis=axis, keepdims=keepdims)

    @staticmethod
    def nansum(x, axis=None):
        return jnp.nansum(_arraylike(x), axis=axis)

    @staticmethod
    def nan_to_num(x):
        return jnp.nan_to_num(x)

    # ------------------------------------------------------------------ elementwise
    @staticmethod
    def pow(x, y):
        return jnp.power(x, y)

    @staticmethod
    def log(x):
        return jnp.log(_arraylike(x))

    @staticmethod
    def multiply(x, y):
        """Elementwise product (NOT the legacy list-taking ``op_multiply``)."""
        return jnp.multiply(x, y)

    @staticmethod
    def subtract(x, y):
        """Elementwise difference (see the note on ``multiply``)."""
        return jnp.subtract(x, y)

    @staticmethod
    def clip(x, lo, hi):
        return jnp.clip(x, lo, hi)

    @staticmethod
    def absolute(x):
        return jnp.abs(x)

    @staticmethod
    def tanh(x):
        return jnp.tanh(x)

    @staticmethod
    def elu(x, alpha=1.0):
        return jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))

    # ------------------------------------------------------------------ shape
    @staticmethod
    def reshape(x, shape):
        return jnp.reshape(x, shape)

    @staticmethod
    def transpose(x, axes):
        return jnp.transpose(x, axes)

    @staticmethod
    def expand_dims(x, axis):
        return jnp.expand_dims(x, axis)

    @staticmethod
    def concatenate(xs, axis):
        return jnp.concatenate(list(xs), axis=axis)

    @staticmethod
    def stack(xs, axis=0):
        return jnp.stack(list(xs), axis=axis)

    @staticmethod
    def split(x, indices, axis):
        return jnp.split(x, indices, axis=axis)

    @staticmethod
    def gather(x, indices, axis=0):
        """``numpy.take`` with the axis spelled out (no kept dims, unlike the Keras
        legacy ``op_gather_keep_dims``)."""
        return jnp.take(x, indices, axis=axis)

    @staticmethod
    def scatter_update(x, indices, values):
        """A copy of ``x`` with ``x[indices] = values`` along axis 0."""
        return jnp.asarray(x).at[indices].set(values)

    @staticmethod
    def repeat(x, n, axis=0):
        return jnp.repeat(x, n, axis=axis)

    # ------------------------------------------------------------------ creation / conversion
    @staticmethod
    def constant(value, dtype=None):
        return jnp.asarray(value, dtype=dtype)

    @staticmethod
    def zeros(shape):
        return jnp.zeros(shape)

    @staticmethod
    def ones(shape):
        return jnp.ones(shape)

    @staticmethod
    def cast(x, dtype):
        return jnp.asarray(x, dtype=dtype)

    @staticmethod
    def to_numpy(x):
        return np.asarray(x)

    # ------------------------------------------------------------------ layer helpers
    @staticmethod
    def as_layer(fn, **kwargs):
        return JaxLayer(fn, kwargs=kwargs)

    @staticmethod
    def as_input(value, name=None):
        """An input slot is just its (replaceable) value; the model holds the mapping."""
        return jnp.asarray(value)

    @staticmethod
    def splitter(shape, sizes, axis, name):
        """A layer splitting into chunks of ``sizes`` along ``axis`` (``shape``/``name``
        are descriptive: a functional backend needs neither to build nor to find it)."""

        def split_it(x):
            return jnp.split(x, np.cumsum(list(sizes))[:-1], axis=axis)

        return JaxLayer(split_it)

    # ------------------------------------------------------------------ legacy vocabulary
    # The names ``test_ops.py`` still calls (see the module docstring).  Aliases where the
    # semantics coincide, adapters where they do not.
    @staticmethod
    def numpy_to_tensor(value, dtype=None):
        return jnp.asarray(value, dtype=dtype)

    @staticmethod
    def tensor_to_numpy_or_python(value):
        return np.asarray(value)

    @staticmethod
    def dict_to_numpy_or_python(mapping):
        return {key: np.asarray(value) for key, value in mapping.items()}

    op_log = log
    tensor_product = tensordot

    @staticmethod
    def flatten(x):
        return jnp.ravel(_arraylike(x))

    @staticmethod
    def op_multiply(xs):
        """The legacy list-taking spelling (``keras.layers.multiply`` behaviour)."""
        return xs[0] * xs[1]

    @staticmethod
    def op_subtract(xs):
        return xs[0] - xs[1]

    @staticmethod
    def c_to_py_fun(op_name, name="dataset"):
        """The NNPDF operation vocabulary; the same source the other backends use.

        (In the target design this mapping belongs to n3fit -- it is the physics
        vocabulary, not a backend primitive -- and the backend only provides the machinery
        around it.)
        """
        from validphys.convolution import OP

        try:
            operation = OP[op_name]
        except KeyError as err:
            raise ValueError(f"Operation {op_name} not recognised") from err

        def operate_on_tensors(tensor_list):
            return operation(*tensor_list)

        return operate_on_tensors
