"""
P5: the weight store -- one replica's weights as ``{path: numpy array}``, on disk and in memory.

The store is what a fit writes to disk, what a hook snapshots, and what ``Backend.save``/``load``
exchange.  These tests pin its three promises:

* the mapping is replica-independent: every replica of a fit -- and the single-replica model the
  generator also builds -- has the *same* paths with the *same* shapes, so a file written by one
  loads into any of them;
* loading is all or nothing: a file that does not describe the graph is an error naming the file,
  never a half-updated model;
* the file format is self-describing: it carries a manifest, and a legacy ``.h5`` is refused with
  a pointer instead of the framework's silence.
"""

import numpy as np
import pytest

pytest.importorskip("keras")

from n3fit.backends.keras_backend import weights as store  # noqa: E402
from n3fit.backends.keras_backend.backend import KerasBackend  # noqa: E402
from n3fit.backends.keras_backend.roles import KerasRoleEnsemble  # noqa: E402
from n3fit.backends.base import GROUP_TRAINING, ROLES, role_of  # noqa: E402
from n3fit.model_gen import ReplicaSettings, generate_pdf_model  # noqa: E402

FLAV_INFO = [
    {"fl": fl, "largex": [1, 3], "smallx": [1, 2]}
    for fl in ["sng", "g", "v", "v3", "v8", "t3", "t8", "t15"]
]


def _settings(seed0, n_replicas):
    return [
        ReplicaSettings(
            nodes=[6, 5, 8],
            activations=["sigmoid", "tanh", "linear"],
            initializer="glorot_normal",
            seed=seed0 + i,
        )
        for i in range(n_replicas)
    ]


def _heterogeneous_settings():
    """Different hidden widths model the per-replica architectures used by hyperopt."""
    return [
        ReplicaSettings(
            nodes=nodes,
            activations=["sigmoid", "tanh", "linear"],
            initializer="glorot_normal",
            seed=100 + i,
        )
        for i, nodes in enumerate(([6, 5, 8], [6, 7, 8]))
    ]


@pytest.fixture(scope="module")
def two_replica_model():
    return generate_pdf_model(_settings(10, 2), flav_info=FLAV_INFO, fitbasis="EVOL")


@pytest.fixture(scope="module")
def one_replica_model():
    return generate_pdf_model(_settings(77, 1), flav_info=FLAV_INFO, fitbasis="EVOL")


def test_the_path_set_is_replica_independent(two_replica_model):
    """Same paths, same shapes, for every replica -- the paths name the *weight*, not the replica."""
    maps = [store.weight_shapes(two_replica_model, replica=i) for i in range(2)]
    assert maps[0] == maps[1]
    assert set(maps[0]) == set(store.weight_map(two_replica_model, 0))
    assert all(role_of(path) in ROLES for path in maps[0])
    assert "nn/0/kernel" in maps[0]
    assert any(path.startswith("preprocessing/") for path in maps[0])


def test_single_and_multi_replica_graphs_have_the_same_layout(two_replica_model, one_replica_model):
    """... so a file written by a one-replica model loads into a many-replica fit and back."""
    assert store.weight_shapes(one_replica_model, 0) == store.weight_shapes(two_replica_model, 0)


def test_the_read_write_round_trip_is_exact(two_replica_model):
    snapshot = {k: v.copy() for k, v in store.weight_map(two_replica_model, 0).items()}
    # perturb, then restore: the restore must land exactly where the snapshot was taken
    store.assign_weight_map(
        two_replica_model, {k: v * 1.5 for k, v in snapshot.items()}, replica=0
    )
    store.assign_weight_map(two_replica_model, snapshot, replica=0)
    again = store.weight_map(two_replica_model, 0)
    assert all(np.array_equal(snapshot[k], again[k]) for k in snapshot)


def test_one_replica_can_be_copied_into_another(two_replica_model):
    """The slot/axis dispatch writes only the addressed replica."""
    replica_zero = {k: v.copy() for k, v in store.weight_map(two_replica_model, 0).items()}
    replica_one = {k: v.copy() for k, v in store.weight_map(two_replica_model, 1).items()}
    store.assign_weight_map(two_replica_model, replica_zero, replica=1)
    assert all(
        np.array_equal(replica_zero[k], store.weight_map(two_replica_model, 1)[k])
        for k in replica_zero
    )
    # ... and the donor replica is untouched
    assert all(
        np.array_equal(replica_zero[k], store.weight_map(two_replica_model, 0)[k])
        for k in replica_zero
    )
    store.assign_weight_map(two_replica_model, replica_one, replica=1)  # restore


def test_a_map_that_does_not_describe_the_graph_is_an_error(two_replica_model):
    values = store.weight_map(two_replica_model, 0)
    with pytest.raises(ValueError, match="no weight"):
        store.assign_weight_map(
            two_replica_model, {**values, "bogus/0/kernel": np.zeros(1)}, replica=0
        )
    short = dict(list(values.items())[:-2])
    with pytest.raises(ValueError, match="missing"):
        store.assign_weight_map(two_replica_model, short, replica=0)


def test_the_file_round_trip(tmp_path, two_replica_model):
    values = store.weight_map(two_replica_model, 0)
    path = tmp_path / "w.weights.npz"
    store.save_weight_file(path, values)
    read_back, manifest = store.load_weight_file(path)
    assert set(read_back) == set(values)
    assert all(np.array_equal(read_back[k], values[k]) for k in values)
    assert manifest["format"] == store.FORMAT_NAME
    assert manifest["shapes"] == {k: list(np.shape(v)) for k, v in values.items()}


def test_a_legacy_h5_is_refused_with_a_pointer(tmp_path):
    legacy = tmp_path / "weights.weights.h5"
    legacy.write_bytes(b"\x89HDF\r\n\x1a\n not really, but the extension is the point")
    with pytest.raises(ValueError, match="retired"):
        store.load_weight_file(legacy)


def test_a_tampered_file_is_refused_whole(tmp_path, two_replica_model):
    values = store.weight_map(two_replica_model, 0)
    path = tmp_path / "tampered.weights.npz"
    store.save_weight_file(path, values)
    # remove one array from the zip: the manifest no longer matches the file
    import zipfile, shutil

    victim = next(k for k in values)
    fixed = tmp_path / "fixed.weights.npz"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(fixed, "w") as dst:
        for item in src.infolist():
            if item.filename != victim + ".npy":
                dst.writestr(item, src.read(item.filename))
    shutil.move(fixed, path)
    with pytest.raises(ValueError, match="manifest lists"):
        store.load_weight_file(path)


# --------------------------------------------------------------------------------------
# Backend.save / Backend.load
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def backend():
    return KerasBackend()


def _ensemble_of(model):
    return KerasRoleEnsemble({GROUP_TRAINING: model}, weights_graph=model)


def test_save_writes_one_replica_and_load_broadcasts_it(tmp_path, backend, two_replica_model):
    """The ``load:`` semantics: one file, and every replica starts from it."""
    single = store.weight_map(two_replica_model, 0)
    path = tmp_path / "warmstart.weights.npz"
    backend.save(two_replica_model, path)  # a raw graph saves replica 0
    read_back, manifest = store.load_weight_file(path)
    assert set(read_back) == set(single)
    assert manifest["n_replicas_graph"] == 2

    ensemble = _ensemble_of(two_replica_model)
    # perturb both replicas, then broadcast the file back into all of them
    perturbation = [
        {k: v * 1.25 for k, v in store.weight_map(two_replica_model, i).items()} for i in range(2)
    ]
    for i, values in enumerate(perturbation):
        store.assign_weight_map(two_replica_model, values, replica=i)
    backend.load(ensemble, path)
    for i in range(2):
        assert all(
            np.array_equal(single[k], store.weight_map(two_replica_model, i)[k]) for k in single
        )


def test_load_into_one_replica_only(tmp_path, backend, two_replica_model):
    """The ``load_weights_from_fit`` semantics: each file fills its own replica."""
    replica_one = {k: v.copy() for k, v in store.weight_map(two_replica_model, 1).items()}
    replica_zero = {k: v.copy() for k, v in store.weight_map(two_replica_model, 0).items()}
    path = tmp_path / "replica1.weights.npz"
    store.save_weight_file(path, replica_one)

    ensemble = _ensemble_of(two_replica_model)
    backend.load(ensemble, path, replica=0)  # replica 1's values into replica 0
    assert all(
        np.array_equal(replica_one[k], store.weight_map(two_replica_model, 0)[k])
        for k in replica_one
    )
    assert all(
        np.array_equal(replica_one[k], store.weight_map(two_replica_model, 1)[k])
        for k in replica_one
    )
    store.assign_weight_map(two_replica_model, replica_zero, replica=0)  # restore


def test_broadcast_prevalidates_every_replica_before_mutating(tmp_path, backend):
    """An incompatible later replica must not leave earlier replicas loaded."""
    model = generate_pdf_model(_heterogeneous_settings(), flav_info=FLAV_INFO, fitbasis="EVOL")
    ensemble = _ensemble_of(model)
    before = [
        {key: value.copy() for key, value in store.weight_map(model, replica=i).items()}
        for i in range(2)
    ]
    path = tmp_path / "heterogeneous.weights.npz"
    store.save_weight_file(path, before[0])

    with pytest.raises(ValueError, match="shape"):
        backend.load(ensemble, path)

    after = [store.weight_map(model, replica=i) for i in range(2)]
    for expected, actual in zip(before, after):
        assert all(np.array_equal(expected[key], actual[key]) for key in expected)


def test_saving_a_multi_replica_ensemble_is_an_error(tmp_path, backend, two_replica_model):
    ensemble = _ensemble_of(two_replica_model)
    with pytest.raises(ValueError, match="one replica"):
        backend.save(ensemble, tmp_path / "nope.weights.npz")


def test_loading_into_a_replica_the_ensemble_does_not_have(tmp_path, backend, two_replica_model):
    values = store.weight_map(two_replica_model, 0)
    path = tmp_path / "w.weights.npz"
    store.save_weight_file(path, values)
    with pytest.raises(ValueError, match="replicas"):
        backend.load(_ensemble_of(two_replica_model), path, replica=5)


def test_the_parameter_vector_round_trips(two_replica_model):
    """``theta`` for parameter-space optimizers (P7 groundwork, layout fixed here)."""
    theta = store.parameter_vector(two_replica_model, 0)
    store.set_parameter_vector(two_replica_model, 0.5 * theta, replica=0)
    assert np.allclose(store.parameter_vector(two_replica_model, 0), 0.5 * theta)
    store.set_parameter_vector(two_replica_model, theta, replica=0)
    assert np.allclose(store.parameter_vector(two_replica_model, 0), theta)
