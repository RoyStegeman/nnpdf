"""Filename conventions for n3fit's per-replica weight files.

The runcard ``save`` value predates the current on-disk format and may be bare (``weights``),
carry a legacy suffix (``weights.h5`` / ``weights.weights.h5``), or already use an npz suffix.
All of those names resolve to one canonical file name: ``<stem>.weights.npz``.  This small,
framework-free helper is shared by the n3fit writer and validphys' ``load_weights_from_fit``
resolver so they cannot drift apart.
"""

from pathlib import PurePath


_WEIGHT_SUFFIXES = (".weights.npz", ".weights.h5", ".npz", ".h5")


def n3fit_weights_filename(save_value):
    """Return the canonical per-replica weight filename for a runcard ``save`` value.

    Parent path components are preserved.  The suffix matching order is significant: the
    compound suffixes must be removed before their shorter ``.npz`` / ``.h5`` endings.
    """
    value = str(save_value)
    if not value:
        raise ValueError("the n3fit weight filename must not be empty")

    path = PurePath(value)
    name = path.name
    stem = name
    for suffix in _WEIGHT_SUFFIXES:
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            break

    if not stem:
        raise ValueError(f"the n3fit weight filename {value!r} has no basename")
    return str(path.with_name(stem + ".weights.npz"))
