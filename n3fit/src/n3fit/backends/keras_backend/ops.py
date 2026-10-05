"""
The Keras implementation of the contract's ``Ops``: the adapter from the historical
``operations`` namespace onto the names the rest of n3fit will use.

Why this exists (P1, decision D15): the two namespaces overlap but are not the same, and
only *some* of the pairs are renames.  Writing the translation down once, here, means every
call site in ``layers/`` can move onto ``backend.ops.<contract name>`` without n3fit having
to know anything about Keras.

The not-a-rename cases, which are the reason this file is not a one-line re-export:

``scatter_update``
    The legacy ``scatter_to_one`` writes a *one-hot* along a new axis (a different signature
    and a different operation).  There is no existing function to alias, so the contract
    semantics -- "return a copy of ``x`` with ``x[indices] = values``" -- are built here out
    of ``keras.ops.scatter_update`` (or ``scatter`` on older Keras), which is exactly the
    operation the contract asks for.

``create_constant`` / ``multiply`` / ``subtract``
    The Keras spellings are a *variable/input creator* and two *layers taking a list*
    (``keras.layers.multiply``), which is not the elementwise function signature.  Elementwise
    multiplication and subtraction are expressed through ``keras.ops``, and the layer-shaped
    behaviour stays available under the legacy names.

``as_input``
    ``numpy_to_input`` is the historical entry point and is kept as the implementation, so
    that the two cannot drift.

Names that are aliases in the contract's *direction* only (``tensor_product`` →
``tensordot``, ``op_log`` → ``log``, ``variable_to_numpy`` → ``to_numpy``) remain available
under their legacy spellings because the un-migrated call sites still use them; they are
listed in ``LEGACY_ALIASES`` for exactly that reason, and they disappear in P2.
"""

from keras import ops as keras_ops

from n3fit.backends.keras_backend import operations as legacy_ops

__all__ = ["KerasOps", "LEGACY_ALIASES"]


# The contract name -> the legacy name it is an alias of.  Only *true* aliases belong here;
# anything whose semantics differ gets a method below.
LEGACY_ALIASES = {
    "log": "op_log",
    "tensordot": "tensor_product",
    "constant": "numpy_to_tensor",
    "to_numpy": "variable_to_numpy",
    "as_input": "numpy_to_input",
    "splitter": "tensor_splitter",
    "flatten": "flatten",
}

# Contract names that are re-exports of ``keras.ops`` (the backend-native spelling).
REEXPORTS = {
    "matmul": "matmul",
    "cast": "cast",
    "ones": "ones",
    "zeros": "zeros",
    "einsum": "einsum",
    "sum": "sum",
    "nansum": "nansum",
    "pow": "power",
    "clip": "clip",
    "absolute": "absolute",
    "tanh": "tanh",
    "elu": "elu",
    "reshape": "reshape",
    "transpose": "transpose",
    "expand_dims": "expand_dims",
    "concatenate": "concatenate",
    "stack": "stack",
    "split": "split",
    "take": "take",
    "repeat": "repeat",
    "nan_to_num": "nan_to_num",
}


class KerasOps:
    """The contract's operations, implemented with Keras.

    Structural conformance is what the contract requires (``Ops`` is a ``Protocol``), so this
    is a plain object: no inheritance, no registration.
    """

    def __init__(self, legacy=None):
        self._legacy = legacy if legacy is not None else legacy_ops
        for contract_name, keras_name in REEXPORTS.items():
            setattr(self, contract_name, getattr(keras_ops, keras_name))
        for contract_name, legacy_name in LEGACY_ALIASES.items():
            setattr(self, contract_name, getattr(self._legacy, legacy_name))

    # ------------------------------------------------------------------ not-a-rename cases
    def multiply(self, x, y):
        """Elementwise product.

        NOT ``legacy.op_multiply``: that one is ``keras.layers.multiply``, which takes a
        single list of tensors and returns a layer-style output; this is the elementwise
        function the layers need.
        """
        return keras_ops.multiply(x, y)

    def subtract(self, x, y):
        """Elementwise difference (see the note on ``multiply``)."""
        return keras_ops.subtract(x, y)

    def gather(self, x, indices, axis=0):
        """``numpy.take`` with the axis argument spelled out.

        Not ``legacy.op_gather_keep_dims``: that one keeps the indexed axis (it is
        ``gather`` + ``expand_dims``), which callers that want the numpy behaviour must not
        get by accident.
        """
        return keras_ops.take(x, indices, axis=axis)

    def scatter_update(self, x, indices, values):
        """Return a copy of ``x`` with ``x[indices] = values`` along axis 0.

        NOT ``legacy.scatter_to_one``, which builds a one-hot along a new axis.
        """
        if hasattr(keras_ops, "scatter_update"):
            return keras_ops.scatter_update(x, indices, values)
        return keras_ops.scatter(indices, values, shape=keras_ops.shape(x))

    def as_layer(self, fn, **kwargs):
        """Wrap a python function on tensors into a layer (the legacy ``as_layer``)."""
        return self._legacy.as_layer(fn, **kwargs)

    def numpy_to_tensor(self, value, **kwargs):
        """Legacy spelling, kept for the un-migrated call sites (P2 removes it)."""
        return self._legacy.numpy_to_tensor(value, **kwargs)

    def tensor_to_numpy_or_python(self, value):
        """Legacy spelling of ``to_numpy`` (P2 removes it)."""
        return self._legacy.tensor_to_numpy_or_python(value)

    def dict_to_numpy_or_python(self, values):
        """Legacy spelling (P2 removes it)."""
        return self._legacy.dict_to_numpy_or_python(values)

    def c_to_py_fun(self, op_name, name="dataset"):
        """The NNPDF convolution vocabulary.

        Kept in the backend for now because it is what the layers call today; in the target
        design this mapping belongs to n3fit (it is the physics vocabulary, not a backend
        primitive), which is a P2 change.
        """
        return self._legacy.c_to_py_fun(op_name, name=name)

    # ------------------------------------------------------------------ legacy-only helpers
    def __getattr__(self, name):
        """Fall back to the historical namespace for names the contract does not have.

        This is what lets the call sites move one by one: ``backend.ops.<contract name>``
        resolves here, and anything still spelled the old way keeps working until P2.  It
        also means a typo raises ``AttributeError`` from the legacy module, as before.
        """
        return getattr(self._legacy, name)
