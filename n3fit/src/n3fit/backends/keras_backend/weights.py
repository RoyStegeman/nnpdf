"""Weights, by path: the one place that knows how a graph stores them (P5).

The contract (D7) says a model's weights are ``{path: numpy array}``, and that the same mapping is
what a *fit writes to disk*, what a *hook snapshots*, and what a *parameter-space optimizer* sees
as ``theta``.  This module is that mapping for the Keras backend, and it exists because the three
consumers used to disagree: ``MetaModel.get_replica_weights`` returned ``{section: [arrays]}`` (a
list per section, so a snapshot could not address one weight), the engine's weights view returned
``{layer.name: array}`` (one entry per *layer*, so a two-weight layer lost one of them), and the
saved ``.h5`` used Keras's own variable paths.

The path grammar
================

``"{role}/{index}/{parameter name}"`` -- e.g. ``nn/0/kernel``, ``preprocessing/3/alpha_g``.

* the **role** is the stable contract role (``nn``, ``preprocessing``, ``sumrule``, ``photon``),
  not the backend's top-level layer name;
* the **index** counts the weight's position *within its role and replica*, in the order the
  layers were built.  It is what makes the path replica-independent: the replicas of a fit are
  built by the same generator from the same settings, so replica 0's third kernel is replica 1's
  third kernel -- and it is also the single-replica model's third kernel, which is what lets a
  one-replica file be broadcast into a multi-replica fit (the legacy ``load:`` semantics);
* the **weight name** is the framework's own (``kernel``, ``bias``, ``alpha_g``): the part that is
  stable across replicas *and* readable in a manifest.

Three storage layouts look the same through this grammar:

* **stacked slot**: replicas that are separate sub-models (``all_NNs`` holds ``NN_0``, ``NN_1``,
  ...), each weight belonging to exactly one replica and stored whole;
* **replica axis**: one array whose leading axis is the replica (the Keras
  ``preprocessing_factor/alpha_g`` weight is stored at ``preprocessing/3/alpha_g`` and has shape
  ``(n_replicas, 1)``), where a replica reads ``w[i:i+1]`` and writes the same slice;
  the slice keeps the axis so a one-replica graph and a many-replica graph have identical array
  shapes per path -- which is what makes a file written by one loadable into the other;
* **shared**: a weight with no replica structure at all (e.g. sum-rule integrals): every replica
  reads and writes the same value, and it is stored once.

Everything here is numpy at the boundary: the only framework-specific operations are "walk a
graph's layers" and "assign one variable".
"""

from __future__ import annotations

import json
import re

import numpy as np

from n3fit.backends.base import ROLES
from n3fit.backends.keras_backend.roles import ROLE_LAYER_NAMES

# Persist role names, never Keras' internal top-level layer names.  The Keras adapter owns the
# translation from roles to layer names; inverting that table here keeps this serialization
# backend-neutral at the contract boundary.
_SECTION_TO_ROLE = {layer_name: role for role, layer_name in ROLE_LAYER_NAMES.items()}

# The sub-models that hold one replica each; the legacy convention, kept because
# ``MetaModel.is_stacked_single_replicas`` and the photon code both read it.
REPLICA_SLOT = re.compile(r"^(?P<prefix>NN)_(?P<index>\d+)$")

# Attributes a layer may keep the replica count of its weights under.  ``Preprocessing`` (and the
# ``MultiInitializer`` layers) store it as ``num_replicas``; the others are the names newer layer
# types use.  A variable on its own cannot tell "2 replicas in a (2, 1) array" from "a 2-row
# matrix", so the count is always read from the layer that built it.
_REPLICA_COUNT_ATTRIBUTES = ("num_replicas", "n_replicas", "replica_axis_size")


class WeightSlot:
    """One weight of one graph: where it lives and how one replica reads/writes it.

    * ``replica`` is the replica index for a slot weight (a ``NN_<i>`` sub-model), ``None``
      otherwise;
    * ``axis`` is the number of replicas carried by the weight's leading axis (replica-axis
      weights), or ``None`` for slot/shared weights.
    """

    __slots__ = ("path", "variable", "replica", "axis", "role", "index", "layer")

    def __init__(self, path, variable, replica, axis, role, index, layer):
        self.path = path
        self.variable = variable
        self.replica = replica
        self.axis = axis
        self.role = role
        self.index = index
        self.layer = layer

    def read(self, replica=0):
        """The numpy value this slot holds for ``replica`` (slicing the replica axis if any)."""
        value = np.asarray(self.variable)
        if self.axis is None:
            # a slot weight *is* this replica's (whatever else is in the graph), and a shared
            # weight is every replica's
            return value
        self._check_replica(replica)
        # keep the axis: one-replica and many-replica graphs then have the same array shape per
        # path, so a file written by one loads into the other
        return value[replica : replica + 1]

    def prepare(self, value, replica=0):
        """Validate and convert a value without mutating the variable."""
        value = np.asarray(value)
        full = np.asarray(self.variable)
        if self.axis is None:
            expected_shape = full.shape
        else:
            self._check_replica(replica)
            expected_shape = full[replica : replica + 1].shape
            if value.shape != expected_shape:
                if value.shape == full[replica].shape:
                    value = value.reshape(expected_shape)
                else:
                    raise ValueError(
                        f"the weight {self.path!r} has shape {expected_shape} for one replica, "
                        f"got {tuple(value.shape)}"
                    )
        if value.shape != expected_shape:
            raise ValueError(
                f"the weight {self.path!r} has shape {expected_shape}, got {tuple(value.shape)}"
            )
        try:
            return np.asarray(value, dtype=full.dtype)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"the weight {self.path!r} cannot be converted to dtype {full.dtype}"
            ) from err

    def _assign_prepared(self, value, replica=0):
        """Assign a value already validated by :meth:`prepare`."""
        if self.axis is None:
            self.variable.assign(value)
            return
        self._check_replica(replica)
        full = np.array(np.asarray(self.variable), copy=True)
        full[replica : replica + 1] = value
        self.variable.assign(full)

    def write(self, value, replica=0):
        """Set the weight in place from ``value`` (the inverse of :meth:`read`)."""
        self._assign_prepared(self.prepare(value, replica), replica)

    def _check_replica(self, replica):
        if replica < 0 or replica >= self.axis:
            raise IndexError(
                f"the weight {self.path!r} carries {self.axis} replicas, got replica {replica}"
            )

    @property
    def trainable(self):
        return bool(getattr(self.variable, "trainable", True))

    def __repr__(self):
        kind = f"slot={self.replica}" if self.replica is not None else (
            f"axis={self.axis}" if self.axis is not None else "shared"
        )
        return f"<WeightSlot {self.path} {kind}>"


def _replica_count(layer):
    """How many replicas the layer's own weights carry (``None`` = no replica axis)."""
    for attribute in _REPLICA_COUNT_ATTRIBUTES:
        value = getattr(layer, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _own_weights(layer):
    """The weights a layer owns *directly*, excluding those of its sub-models.

    Keras's ``layer.weights`` recurses into sub-layers (``all_NNs.weights`` lists every ``NN_i``
    weight), so the own ones are ``layer.weights`` minus every sub-layer's, by identity.
    """
    weights = list(getattr(layer, "weights", ()) or ())
    sublayers = list(getattr(layer, "layers", ()) or ())
    if not sublayers:
        return weights
    foreign = {id(weight) for sub in sublayers for weight in getattr(sub, "weights", ()) or ()}
    return [weight for weight in weights if id(weight) not in foreign]


def _weight_name(variable):
    name = getattr(variable, "name", None) or getattr(variable, "path", "")
    return str(name).split("/")[-1].split(":")[0]


def _role_for_section(section):
    """Translate a weight-bearing Keras section to its closed contract role."""
    try:
        role = _SECTION_TO_ROLE[section]
    except KeyError:
        raise ValueError(
            f"weight-bearing Keras section {section!r} has no contract role mapping; "
            f"known sections are {sorted(_SECTION_TO_ROLE)}"
        ) from None
    if role not in ROLES:
        raise ValueError(f"the mapped role {role!r} is not part of the contract vocabulary")
    return role


def _walk(layer, section, replica=None):
    """Yield ``(layer, variable, replica)`` for every own weight in ``layer``'s subtree.

    ``replica`` is set when the walk crosses a ``NN_<i>`` slot and inherited by its subtree.
    """
    name = getattr(layer, "name", None) or type(layer).__name__
    match = REPLICA_SLOT.match(name)
    if match and replica is None:
        replica = int(match.group("index"))
    for variable in _own_weights(layer):
        yield layer, variable, replica
    for sub in list(getattr(layer, "layers", ()) or ()):
        yield from _walk(sub, section, replica)


def weight_slots(graph, replica=0):
    """Every weight of ``graph`` that belongs to ``replica``, as :class:`WeightSlot` objects.

    The path prefix is the contract role for each top-level weight-bearing section (the adapter's
    ``ROLE_LAYER_NAMES`` table supplies the Keras-to-role translation).  Within a role, ordering is
    depth-first and follows weight creation order; the index is therefore independent of replica
    identity. ``replica`` selects the replica whose *slot* weights are returned; replica-axis and
    shared weights are returned for every replica, so a single-replica graph and a many-replica
    graph have identical path sets.
    """
    slots = []

    def collect(layer, variable, slot_replica, role):
        axis = None
        if slot_replica is None:
            axis = _replica_count(layer)
            shape = tuple(np.shape(variable))
            if axis is not None and (not shape or shape[0] != axis):
                axis = None  # the layer stacks replicas but this weight does not carry the axis
        slots.append(
            WeightSlot(
                path=None,  # filled below, once the per-(section, replica) numbering is known
                variable=variable,
                replica=slot_replica,
                axis=axis,
                role=role,
                index=0,
                layer=layer,
            )
        )

    graph_weights = _own_weights(graph)
    if graph_weights:
        graph_name = getattr(graph, "name", None) or type(graph).__name__
        graph_role = _role_for_section(graph_name)
        for variable in graph_weights:
            collect(graph, variable, None, graph_role)

    for child in list(getattr(graph, "layers", ()) or ()):
        keras_section = getattr(child, "name", None) or type(child).__name__
        for layer, variable, slot_replica in _walk(child, keras_section):
            if slot_replica is not None and slot_replica != replica:
                continue  # this sub-model builds another replica
            role = _role_for_section(keras_section)
            collect(layer, variable, slot_replica, role)

    # Number the weights *within a role and a replica* so that the path does not depend on the
    # order in which the walk found the weights of other replicas: replica 0's third kernel,
    # replica 1's third kernel and the single-replica model's third kernel are all ``.../2/...``.
    counters = {}
    for slot in slots:
        key = (slot.role, slot.replica)
        slot.index = counters.get(key, 0)
        counters[key] = slot.index + 1
        slot.path = f"{slot.role}/{slot.index}/{_weight_name(slot.variable)}"
    return slots


def weight_map(graph, replica=0, *, trainable_only=False):
    """``{path: numpy array}`` for one replica of ``graph`` (the D7 mapping)."""
    slots = weight_slots(graph, replica=replica)
    if trainable_only:
        slots = [slot for slot in slots if slot.trainable]
    return {slot.path: slot.read(replica) for slot in slots}


def _prepare_weight_map(graph, values, replica=0, *, strict=True):
    """Validate paths, shapes, and conversions before any variable is changed."""
    slots = {slot.path: slot for slot in weight_slots(graph, replica=replica)}
    provided = set(values)
    known = set(slots)
    if provided - known:
        raise ValueError(
            f"this graph has no weight {sorted(provided - known)[:3]}... "
            f"(it has {len(known)}: {sorted(known)[:3]}...)"
        )
    if strict and known - provided:
        raise ValueError(
            f"the weight map is missing {len(known - provided)} of this graph's {len(known)} "
            f"weights, e.g. {sorted(known - provided)[:3]}"
        )
    prepared = {path: slots[path].prepare(value, replica) for path, value in values.items()}
    return slots, prepared


def validate_weight_map(graph, values, replica=0, *, strict=True):
    """Check that ``values`` can be assigned to one replica, without mutating the graph."""
    _prepare_weight_map(graph, values, replica=replica, strict=strict)


def assign_weight_map(graph, values, replica=0, *, strict=True):
    """Set one replica's weights from ``values`` (the inverse of :meth:`weight_map`).

    Paths, shapes, and dtype conversions are checked before the first assignment. If an unexpected
    framework assignment error still occurs, every variable attempted by this call is restored to
    its original value before the error is re-raised. ``strict=False`` allows a known subset (used
    when changing trainable parameters while preserving non-trainable weights); unknown keys are
    always rejected.
    """
    slots, prepared = _prepare_weight_map(graph, values, replica=replica, strict=strict)
    original = {
        path: np.array(slots[path].read(replica), copy=True)
        for path in prepared
    }
    attempted = []
    try:
        for path, value in prepared.items():
            attempted.append(path)
            slots[path]._assign_prepared(value, replica)
    except Exception as err:
        rollback_errors = []
        for path in reversed(attempted):
            try:
                slots[path]._assign_prepared(original[path], replica)
            except Exception as rollback_error:  # pragma: no cover - framework failure path
                rollback_errors.append((path, rollback_error))
        if rollback_errors:  # pragma: no cover - framework failure path
            details = "; ".join(f"{path}: {failure}" for path, failure in rollback_errors)
            raise RuntimeError(f"{err}; weight rollback also failed ({details})") from err
        raise


def weight_shapes(graph, replica=0):
    """``{path: shape}`` -- what a manifest records so a mismatch can be diagnosed."""
    return {slot.path: tuple(slot.read(replica).shape) for slot in weight_slots(graph, replica)}


def parameter_layout(graph, replica=0):
    """The paths of the *trainable* weights, in the order ``theta`` concatenates them (P7)."""
    return [slot.path for slot in weight_slots(graph, replica=replica) if slot.trainable]


def parameter_vector(graph, replica=0):
    """``theta``: the trainable weights flattened in layout order (one replica)."""
    layout = parameter_layout(graph, replica=replica)
    values = weight_map(graph, replica=replica, trainable_only=True)
    return np.concatenate([np.asarray(values[path]).ravel() for path in layout])


def set_parameter_vector(graph, theta, replica=0):
    """The inverse of :meth:`parameter_vector` (used by parameter-space optimizers)."""
    layout = parameter_layout(graph, replica=replica)
    values = weight_map(graph, replica=replica, trainable_only=True)
    theta = np.asarray(theta)
    offset = 0
    for path in layout:
        shape = np.asarray(values[path]).shape
        size = int(np.prod(shape))
        chunk = theta[offset : offset + size]
        if chunk.size != size:
            raise ValueError(
                f"theta is too short: {offset + size} needed for {path!r}, got {theta.size}"
            )
        values[path] = chunk.reshape(shape)
        offset += size
    if offset != theta.size:
        raise ValueError(f"theta has {theta.size} entries, this graph has {offset} parameters")
    assign_weight_map(graph, values, replica, strict=False)


# --------------------------------------------------------------------------------------
# The on-disk schema (D8): one replica per file, npz, with a manifest inside
# --------------------------------------------------------------------------------------
# The legacy ``.weights.h5`` is retired wholesale: current Keras cannot load the nested-format
# files the old fits wrote (it returns without error and leaves the model unchanged), so there is
# nothing to be compatible *with* -- a new schema, as D8 decided.  A file holds the weight map of
# exactly one replica (which is what the per-replica fit folders, ``load:`` and
# ``load_weights_from_fit`` all exchange), the arrays keyed by their path, plus a JSON manifest
# that records what the file describes so a mismatch is diagnosed instead of half-loaded.

FORMAT_NAME = "n3fit-weights/2"
MANIFEST_KEY = "__manifest__"


def make_manifest(values, *, n_replicas_graph=None, replica=None):
    """What the file records about itself (shapes included, so a mismatch can be diagnosed)."""
    return {
        "format": FORMAT_NAME,
        "n_replicas_file": 1,
        "n_replicas_graph": None if n_replicas_graph is None else int(n_replicas_graph),
        "replica": None if replica is None else int(replica),
        "shapes": {path: list(np.shape(value)) for path, value in values.items()},
    }


def save_weight_file(path, values, *, manifest=None):
    """Write one replica's weight map to ``path`` (npz, arrays keyed by path + the manifest)."""
    if manifest is None:
        manifest = make_manifest(values)
    payload = {MANIFEST_KEY: np.array(json.dumps(manifest, sort_keys=True))}
    payload.update({path_key: np.asarray(value) for path_key, value in values.items()})
    np.savez_compressed(path, **payload)


def load_weight_file(path):
    """Read a weight file back as ``(values, manifest)``.

    Every check that can fail *here* does, naming the file: a wrong format, a manifest whose
    arrays do not match its shapes, and a manifest with no arrays at all are all errors -- the
    only silent outcome of loading is "the weights of one replica".
    """
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as err:  # not an npz at all (a legacy h5 lands here)
        raise ValueError(
            f"{path} is not a {FORMAT_NAME!r} (npz) file. Legacy '.weights.h5' files are "
            "retired (the current framework cannot read them); re-save the weights from the "
            "fit that produced them."
        ) from err
    with archive:
        if MANIFEST_KEY not in archive:
            raise ValueError(
                f"{path} has no manifest ({MANIFEST_KEY!r}): not an {FORMAT_NAME!r} file"
            )
        manifest = json.loads(np.asarray(archive[MANIFEST_KEY]).item())
        if not isinstance(manifest, dict):
            raise ValueError(f"{path}: the weight manifest must be a JSON object")
        if manifest.get("format") != FORMAT_NAME:
            raise ValueError(
                f"{path} declares format {manifest.get('format')!r}, expected {FORMAT_NAME!r}"
            )
        replicas_in_file = manifest.get("n_replicas_file")
        if (
            not isinstance(replicas_in_file, int)
            or isinstance(replicas_in_file, bool)
            or replicas_in_file != 1
        ):
            raise ValueError(
                f"{path}: the weight file must describe exactly one replica, "
                f"the manifest says {replicas_in_file!r}"
            )
        values = {}
        for key in archive.files:
            if key != MANIFEST_KEY:
                values[key] = np.array(archive[key])
    shapes = manifest.get("shapes")
    if not isinstance(shapes, dict):
        raise ValueError(f"{path}: the manifest must contain a weight-shapes mapping")
    if set(values) != set(shapes):
        raise ValueError(
            f"{path}: manifest lists {len(shapes)} weights but the file has {len(values)}"
        )
    for key, value in values.items():
        declared_shape = shapes[key]
        if not isinstance(declared_shape, list) or any(
            type(dimension) is not int or dimension < 0 for dimension in declared_shape
        ):
            raise ValueError(f"{path}: the manifest has an invalid shape for weight {key!r}")
        if tuple(value.shape) != tuple(declared_shape):
            raise ValueError(
                f"{path}: weight {key!r} has shape {tuple(value.shape)}, "
                f"the manifest says {tuple(declared_shape)}"
            )
    if not values:
        raise ValueError(f"{path}: the manifest describes no weights")
    return values, manifest
