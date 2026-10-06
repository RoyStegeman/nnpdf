"""The Keras weight-store mechanics that do not require an installed framework."""

import json

import numpy as np
import pytest

from n3fit.backends.base import ROLES, role_of
from n3fit.backends.keras_backend import weights as store
from validphys.n3fit_weights import n3fit_weights_filename


class FakeVariable:
    def __init__(self, name, value, trainable=True):
        self.name = name
        self.value = np.asarray(value).copy()
        self.trainable = trainable

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)

    def assign(self, value):
        value = np.asarray(value)
        if value.shape != self.value.shape:
            raise ValueError(f"shape mismatch: expected {self.value.shape}, got {value.shape}")
        self.value[...] = value


class FakeLayer:
    def __init__(self, name, weights=(), layers=(), **attributes):
        self.name = name
        self._own_weights = list(weights)
        self.layers = list(layers)
        for name, value in attributes.items():
            setattr(self, name, value)

    @property
    def weights(self):
        return self._own_weights + [weight for layer in self.layers for weight in layer.weights]


class FakeGraph(FakeLayer):
    pass


class FailOnceVariable(FakeVariable):
    """An assignment that fails once, so the store's rollback path can be exercised."""

    def __init__(self, name, value):
        super().__init__(name, value)
        self.fail_next_assignment = True

    def assign(self, value):
        if self.fail_next_assignment:
            self.fail_next_assignment = False
            raise RuntimeError("injected assignment failure")
        super().assign(value)


@pytest.mark.parametrize(
    ("save_value", "expected"),
    [
        ("weights", "weights.weights.npz"),
        ("weights.h5", "weights.weights.npz"),
        ("weights.weights.h5", "weights.weights.npz"),
        ("weights.npz", "weights.weights.npz"),
        ("weights.weights.npz", "weights.weights.npz"),
        ("checkpoints/weights.h5", "checkpoints/weights.weights.npz"),
    ],
)
def test_save_name_variants_share_one_canonical_weight_filename(save_value, expected):
    assert n3fit_weights_filename(save_value) == expected


def test_weight_store_paths_use_contract_roles_not_keras_layer_names():
    nn = FakeLayer(
        "all_NNs",
        layers=[
            FakeLayer("NN_0", [FakeVariable("kernel:0", [[1.0, 2.0]])]),
            FakeLayer("NN_1", [FakeVariable("kernel:0", [[3.0, 4.0]])]),
        ],
    )
    preprocessing = FakeLayer(
        "preprocessing_factor",
        [FakeVariable("alpha_up:0", [[0.1], [0.2]])],
        num_replicas=2,
    )
    graph = FakeGraph("PDFs", layers=[nn, preprocessing])

    first = store.weight_map(graph, replica=0)
    second = store.weight_map(graph, replica=1)
    assert set(first) == {"nn/0/kernel", "preprocessing/0/alpha_up"}
    assert set(first) == set(second)
    assert all(role_of(path) in ROLES for path in first)
    assert np.array_equal(first["nn/0/kernel"], [[1.0, 2.0]])
    assert np.array_equal(second["nn/0/kernel"], [[3.0, 4.0]])


def test_unknown_weight_bearing_section_fails_instead_of_leaking_a_layer_name():
    unknown = FakeLayer("implementation_detail", [FakeVariable("kernel", [1.0])])
    graph = FakeGraph("PDFs", layers=[unknown])
    with pytest.raises(ValueError, match="no contract role mapping"):
        store.weight_map(graph)


def test_shape_mismatch_is_detected_before_any_weight_is_changed():
    first = FakeVariable("kernel:0", [1.0])
    second = FakeVariable("bias:0", [2.0, 3.0])
    graph = FakeGraph("all_NNs", [first, second])
    original_first, original_second = first.value.copy(), second.value.copy()
    values = store.weight_map(graph)
    values["nn/0/kernel"] = np.array([9.0])
    values["nn/1/bias"] = np.array([8.0, 7.0, 6.0])

    with pytest.raises(ValueError, match="shape"):
        store.assign_weight_map(graph, values)

    assert np.array_equal(first.value, original_first)
    assert np.array_equal(second.value, original_second)


def test_parameter_vector_updates_trainables_without_requiring_nontrainable_weights():
    trainable = FakeVariable("kernel:0", [1.0, 2.0])
    frozen = FakeVariable("mask:0", [0.0, 1.0], trainable=False)
    graph = FakeGraph("all_NNs", [trainable, frozen])
    theta = store.parameter_vector(graph)

    store.set_parameter_vector(graph, theta * 3.0)

    assert np.array_equal(trainable.value, [3.0, 6.0])
    assert np.array_equal(frozen.value, [0.0, 1.0])


def test_unexpected_assignment_failure_rolls_back_all_attempted_weights():
    trainable = FakeVariable("kernel:0", [1.0])
    fails_once = FailOnceVariable("bias:0", [2.0])
    graph = FakeGraph("all_NNs", [trainable, fails_once])
    before = store.weight_map(graph)
    values = {path: value + 10.0 for path, value in before.items()}

    with pytest.raises(RuntimeError, match="injected assignment failure"):
        store.assign_weight_map(graph, values)

    after = store.weight_map(graph)
    assert all(np.array_equal(before[path], after[path]) for path in before)


def test_npz_round_trip_preserves_the_manifest_and_arrays(tmp_path):
    graph = FakeGraph("all_NNs", [FakeVariable("kernel:0", [[1.0, 2.0]])])
    values = store.weight_map(graph)
    path = tmp_path / "weights.weights.npz"

    store.save_weight_file(path, values)
    loaded, manifest = store.load_weight_file(path)

    assert manifest["format"] == "n3fit-weights/2"
    assert manifest["n_replicas_file"] == 1
    assert manifest["shapes"] == {key: list(value.shape) for key, value in values.items()}
    assert all(np.array_equal(loaded[key], values[key]) for key in values)


def test_npz_loader_rejects_a_manifest_that_claims_multiple_replicas(tmp_path):
    graph = FakeGraph("all_NNs", [FakeVariable("kernel:0", [1.0])])
    path = tmp_path / "bad_manifest.weights.npz"
    store.save_weight_file(path, store.weight_map(graph))
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key]) for key in archive.files}
    manifest = json.loads(payload[store.MANIFEST_KEY].item())
    manifest["n_replicas_file"] = 2
    payload[store.MANIFEST_KEY] = np.array(json.dumps(manifest))
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="exactly one replica"):
        store.load_weight_file(path)
