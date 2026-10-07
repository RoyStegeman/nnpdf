"""The on-disk weight schema (D8): one replica per file, npz, with a manifest inside.

Factored out of ``keras_backend/weights.py`` in P6 so that every backend reads and writes
*the same* ``n3fit-weights/2`` files: a file holds the weight map of exactly one replica
(which is what the per-replica fit folders, ``load:`` and ``load_weights_from_fit`` all
exchange), the arrays keyed by their path, plus a JSON manifest that records what the file
describes so a mismatch is diagnosed instead of half-loaded.

This module is framework-free (numpy + stdlib only): it is the file format, not a walk over
a graph.  Each backend keeps its own params<->path-map walker and calls into here for the
bytes.  The legacy ``.weights.h5`` is retired wholesale (see the P5 proposal §1): a legacy
file is refused with a pointer, never half-loaded.
"""

import json

import numpy as np

__all__ = [
    "FORMAT_NAME",
    "MANIFEST_KEY",
    "make_manifest",
    "save_weight_file",
    "load_weight_file",
]

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
