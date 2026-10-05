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

``"{section}/{index}/{weight name}"`` -- e.g. ``all_NNs/0/kernel``, ``preprocessing_factor/3/alpha_g``.

* the **section** is the top-level layer that owns the weight (``all_NNs``, ``preprocessing_factor``,
  ``impose_msr``, ...): the thing n3fit talks about by role (``ROLE_NN`` → ``all_NNs``);
* the **index** counts the weight's position *within its section and its replica*, in the order the
  layers were built.  It is what makes the path replica-independent: the replicas of a fit are
  built by the same generator from the same settings, so replica 0's third kernel is replica 1's
  third kernel -- and it is also the single-replica model's third kernel, which is what lets a
  one-replica file be broadcast into a multi-replica fit (the legacy ``load:`` semantics);
* the **weight name** is the framework's own (``kernel``, ``bias``, ``alpha_g``): the part that is
  stable across replicas *and* readable in a manifest.

Three storage layouts look the same through this grammar:

* **stacked slot**: replicas that are separate sub-models (``all_NNs`` holds ``NN_0``, ``NN_1``,
  ...), each weight belonging to exactly one replica and stored whole;
* **replica axis**: one array whose leading axis is the replica (``preprocessing_factor/alpha_g``
  has shape ``(n_replicas, 1)``), where a replica reads ``w[i:i+1]`` and writes the same slice;
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

    __slots__ = ("path", "variable", "replica", "axis", "section", "index", "layer")

    def __init__(self, path, variable, replica, axis, section, index, layer):
        self.path = path
        self.variable = variable
        self.replica = replica
        self.axis = axis
        self.section = section
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

    def write(self, value, replica=0):
        """Set the weight in place from ``value`` (the inverse of :meth:`read`)."""
        value = np.asarray(value)
        if self.axis is None:
            self.variable.assign(value)
            return
        self._check_replica(replica)
        full = np.asarray(self.variable)
        if value.shape != full[replica : replica + 1].shape:
            if value.shape == np.shape(full[replica]):
                value = value.reshape(full[replica : replica + 1].shape)
            else:
                raise ValueError(
                    f"the weight {self.path!r} has shape {full[replica : replica + 1].shape} for "
                    f"one replica, got {tuple(value.shape)}"
                )
        full = np.array(full)
        full[replica : replica + 1] = value
        self.variable.assign(full)

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

    The sections are the graph's own top-level layers (``all_NNs``, ``preprocessing_factor``,
    ...); the order is the graph's own (depth-first, and within a layer the creation order), which
    is what makes the index term of the path stable.  ``replica`` selects the replica whose *slot*
    weights are returned; replica-axis and shared weights are returned for every replica, so a
    single-replica graph and a many-replica graph have identical path sets.
    """
    slots = []

    def collect(layer, variable, slot_replica, section):
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
                section=section,
                index=0,
                layer=layer,
            )
        )

    for variable in _own_weights(graph):
        collect(graph, variable, None, getattr(graph, "name", None) or "model")
    for child in list(getattr(graph, "layers", ()) or ()):
        section = getattr(child, "name", None) or type(child).__name__
        for layer, variable, slot_replica in _walk(child, section):
            if slot_replica is not None and slot_replica != replica:
                continue  # this sub-model builds another replica
            collect(layer, variable, slot_replica, section)

    # Number the weights *within a section and a replica* so that the path does not depend on the
    # order in which the walk found the weights of other replicas: replica 0's third kernel,
    # replica 1's third kernel and the single-replica model's third kernel are all ``.../2/...``.
    counters = {}
    for slot in slots:
        key = (slot.section, slot.replica)
        slot.index = counters.get(key, 0)
        counters[key] = slot.index + 1
        slot.path = f"{slot.section}/{slot.index}/{_weight_name(slot.variable)}"
    return slots


def weight_map(graph, replica=0, *, trainable_only=False):
    """``{path: numpy array}`` for one replica of ``graph`` (the D7 mapping)."""
    slots = weight_slots(graph, replica=replica)
    if trainable_only:
        slots = [slot for slot in slots if slot.trainable]
    return {slot.path: slot.read(replica) for slot in slots}


def assign_weight_map(graph, values, replica=0, *, strict=True):
    """Set the weights of one replica of ``graph`` from ``values`` (the inverse of :meth:`weight_map`).

    ``strict`` compares the *sets* of paths: a mapping that does not describe this graph is an
    error naming both sides, which turns "I loaded the wrong file" into a message instead of a
    silently half-updated model.  With ``strict=False`` the mapping is applied key by key and an
    unknown key is an error of the same kind (there is no use for a partial load).
    """
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
    for path, value in values.items():
        if path in slots:
            slots[path].write(value, replica)


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
    assign_weight_map(graph, values, replica)


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
        if manifest.get("format") != FORMAT_NAME:
            raise ValueError(
                f"{path} declares format {manifest.get('format')!r}, expected {FORMAT_NAME!r}"
            )
        values = {}
        for key in archive.files:
            if key != MANIFEST_KEY:
                values[key] = np.array(archive[key])
    shapes = manifest.get("shapes") or {}
    if set(values) != set(shapes):
        raise ValueError(
            f"{path}: manifest lists {len(shapes)} weights but the file has {len(values)}"
        )
    for key, value in values.items():
        if tuple(value.shape) != tuple(shapes[key]):
            raise ValueError(
                f"{path}: weight {key!r} has shape {tuple(value.shape)}, "
                f"the manifest says {tuple(shapes[key])}"
            )
    if not values:
        raise ValueError(f"{path}: the manifest describes no weights")
    return values, manifest
