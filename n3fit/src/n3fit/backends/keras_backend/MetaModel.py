"""
MetaModel class

Extension of the backend Model class containing some wrappers in order to absorb other
backend-dependent calls.
"""


import re

from keras import backend as K
from keras import optimizers as Kopt
from keras.models import Model
import numpy as np

from . import operations as ops

# Define in this dictionary new optimizers as well as the arguments they accept
# (with default values if needed be)
optimizers = {
    "RMSprop": (Kopt.RMSprop, {"learning_rate": 0.01}),
    "Adam": (Kopt.Adam, {"learning_rate": 0.01}),
    "Adagrad": (Kopt.Adagrad, {}),
    "Adadelta": (Kopt.Adadelta, {"learning_rate": 1.0}),
    "Adamax": (Kopt.Adamax, {}),
    "Nadam": (Kopt.Nadam, {"learning_rate": 0.001}),
    "Amsgrad": (Kopt.Adam, {"learning_rate": 0.01, "amsgrad": True}),
    "SGD": (Kopt.SGD, {"learning_rate": 0.01, "momentum": 0.0, "nesterov": False}),
}

NN_PREFIX = "NN"
NN_LAYER_ALL_REPLICAS = "all_NNs"
PREPROCESSING_LAYER_ALL_REPLICAS = "preprocessing_factor"

# Some keys need to work for everyone
for k, v in optimizers.items():
    v[1]["clipnorm"] = 1.0


class MetaModel(Model):
    """
    The model wraps keras.Model and adds some custom behaviour. Most notably it
    allows supplying constant values for input arguments, which are used when
    training and making predictions with the model (note that constants need to
    be explicitly registered as inputs, see
    https://github.com/keras-team/keras/issues/11912). These inputs can be
    passed in the ``input_values`` parameter, or gathered from the
    ``tensor_content`` attribute of the ``input_tensors``, which is set
    automatically when using the ``numpy_to_input`` function from
    :py:mod:`n3fit.backends.keras_backend.operations`.

    Parameters
    ----------
        input_tensors: dict[Any, tensorflow.keras.layers.Input]
            Input layer
        output_tensors: tensorflow.keras.layers.Layer
            Output layer
        input_values: dict[Any, array_like]
            Constant values for the input layer, to be supplied when making
            predictions with the model.

        **kwargs:
            keyword arguments to pass directly to Model
    """

    accepted_optimizers = optimizers

    def __init__(self, input_tensors, output_tensors, scaler=None, input_values=None, **kwargs):
        self.has_dataset = False
        self.required_slots = set()

        if input_values is None:
            input_values = {}

        if not isinstance(input_tensors, dict):
            raise TypeError("Expecting input_tensors to be a dict")

        if not isinstance(input_values, dict):
            raise TypeError("Expecting input_values to be a dict or None")

        x_in = {}
        # Go over the inputs. If we can deduce a constant value, either because
        # it is set in input_values or because it has a tensor_content, we
        # store it. Otherwise we mark the input as required when making
        # predictions.
        for k, v in input_tensors.items():
            if k in input_values:
                x_in[k] = input_values[k]
            elif hasattr(v, "tensor_content"):
                x_in[k] = ops.numpy_to_tensor(v.tensor_content)
            else:
                self.required_slots.add(k)
        super().__init__(input_tensors, output_tensors, **kwargs)

        self.input_tensors = input_tensors
        self.single_replica_generator = None

        self._scaler = scaler

        # Keras' __setattr__ would try to track the input dictionary as a TrackedDict
        # which is incompatible with jax, to avoid this problem, set the attribute directly
        object.__setattr__(self, "x_in", x_in)

    def _parse_input(self, extra_input=None):
        """Returns the input data the model was built with.
        Introduces the extra_input in the places asigned to the placeholders.

        If the model was generated with a scaler, the input will be scaled accordingly
        """
        if extra_input is None:
            if self.required_slots:
                raise ValueError(f"The following inputs must be provided: {self.required_slots}")
            return self.x_in

        if not isinstance(extra_input, dict):
            raise TypeError("extra_input should be a dict or None")

        if diff := (self.required_slots - extra_input.keys()):
            raise ValueError(f"The following inputs must be provided {diff}")

        if diff := (extra_input.keys() - (self.x_in.keys() | self.required_slots)):
            raise ValueError(f"The following inputs are unknown {diff}")

        if self._scaler is not None:
            extra_input = {k: self._scaler(i) for k, i in extra_input.items()}

        return {**self.x_in, **extra_input}

    def predict(self, x=None, **kwargs):
        """Call super().predict with the right input arguments"""
        x = self._parse_input(x)
        result = super().predict(x=x, **kwargs)
        return result

    def set_masks_to(self, names, val=0.0):
        """Set all mask value to the selected value
        Masks in MetaModel should be named {name}_mask

        Mask are layers with one single weight (shape=(1,)) that multiplies the input

        Parameters
        ----------
            names: list
                list of masks to look for
            val: float
                selected value of the mask
        """
        mask_val = [val]
        for name in names:
            mask_name = f"{name}_mask"
            mask_w = self.get_layer(mask_name).weights[0]
            mask_w.assign(mask_val)

    def reset_layer_weights_to(self, layer_names, reference_vals):
        """Set weights for the given layer to the given reference values

        The ``reference_vals`` values list must be a list of the same size
        of ``layer_names`` and it must consist of numpy arrays that perfectly
        align to the reference layer weights.
        In the special case of 1-weight layers it admits a scalar as input.

        Parameters
        ----------
            layer_names: list
                list of names of the layers to update weights
            reference_vals: list(float) or list(arrays)
                list of scalar or arrays to assign to each layer
        """
        for layer_name, values in zip(layer_names, reference_vals):
            if np.isscalar(values):
                values = np.array([[values]])
            layer = self.get_layer(layer_name)
            all_w = layer.weights
            for w, v in zip(all_w, values):
                w.assign(v)

    def apply_as_layer(self, x):
        """Apply the model as a layer"""
        all_input = {**self.input_tensors, **x}
        return all_input, super().__call__(all_input)

    def get_layer_re(self, regex):
        """Get all layers matching the given regular expression"""
        check = lambda x: re.match(regex, x.name)
        return list(filter(check, self.layers))

    def get_replica_weights(self, i_replica):
        """
        Get the weights of replica i_replica.

        This assumes that the only weights are in the
        layer types defined as the constants
            NN_LAYER_ALL_REPLICAS & PREPROCESSING_LAYER_ALL_REPLICAS

        Parameters
        ----------
            i_replica: int

        Returns
        -------
            dict
                dictionary with the weights of the replica
        """
        weights = {}
        for layer_type in [NN_LAYER_ALL_REPLICAS, PREPROCESSING_LAYER_ALL_REPLICAS]:
            layer = self.get_layer(layer_type)
            weights[layer_type] = get_layer_replica_weights(layer, i_replica)

        return weights

    def set_replica_weights(self, weights, i_replica=0):
        """
        Set the weights of replica i_replica.

        This assumes that the only weights are in layers called
        ``NN_{i_replica}`` and ``preprocessing_factor_{i_replica}``

        Parameters
        ----------
            weights: dict
                dictionary with the weights of the replica
            i_replica: int
                the replica number to set, defaulting to 0
        """
        for layer_type in [NN_LAYER_ALL_REPLICAS, PREPROCESSING_LAYER_ALL_REPLICAS]:
            layer = self.get_layer(layer_type)
            set_layer_replica_weights(layer=layer, weights=weights[layer_type], i_replica=i_replica)

    def split_replicas(self):
        """
        Split the single multi-replica model into a list of separate single replica models,
        maintaining the current state of the weights.

        Returns
        -------
            list
                list of single replica models
        """
        if self.single_replica_generator is None:
            raise ValueError("Trying to generate single replica models with no generator set.")
        replicas = []
        for i_replica in range(self.num_replicas):
            replica = self.single_replica_generator(i_replica)
            replica.set_replica_weights(self.get_replica_weights(i_replica))
            replicas.append(replica)

        return replicas

    @property
    def num_replicas(self):
        return self.output.shape[1]


def is_stacked_single_replicas(layer):
    """
    Check if the layer consists of stacked single replicas (Only happens for NN layers),
    to determine how to extract single replica weights.

    Parameters
    ----------
        layer: MetaLayer
            the layer to check

    Returns
    -------
        bool
            True if the layer consists of stacked single replicas
    """
    if not isinstance(layer, MetaModel):
        return False
    return f"{NN_PREFIX}_0" in [sublayer.name for sublayer in layer.layers]


def get_layer_replica_weights(layer, i_replica: int):
    """
    Get the weights for the given single replica ``i_replica``,
    from a ``layer`` that contains the weights of all the replicas.

    Note that the layer could be a complete NN with many separated sub_layers
    each of which containing weights for all replicas together.
    This functions separates the per-replica weights and returns the list of weight as if the
    input ``layer`` were made of _only_ replica ``i_replica``.

    Parameters
    ----------
        layer: MetaLayer
            the layer to get the weights from
        i_replica: int
            the replica number

    Returns
    -------
        weights: list
            list of weights for the replica
    """
    if is_stacked_single_replicas(layer):
        weights_ref = layer.get_layer(f"{NN_PREFIX}_{i_replica}").weights
        weights = [ops.variable_to_numpy(w) for w in weights_ref]
    else:
        weights = [ops.variable_to_numpy(w)[i_replica : i_replica + 1] for w in layer.weights]

    return weights


def set_layer_replica_weights(layer, weights, i_replica: int):
    """
    Set the weights for the given single replica ``i_replica``.
    When the input ``layer`` contains weights for many replicas, ensures that
    only those corresponding to replica ``i_replica`` are updated.

    Parameters
    ----------
        layer: MetaLayer
            the layer to set the weights for
        weights: list
            list of weights for the replica
        i_replica: int
            the replica number
    """
    if is_stacked_single_replicas(layer):
        layer.get_layer(f"{NN_PREFIX}_{i_replica}").set_weights(weights)
        return

    full_weights = [w.numpy() for w in layer.weights]
    for w_old, w_new in zip(full_weights, weights):
        w_old[i_replica : i_replica + 1] = w_new

    layer.set_weights(full_weights)
