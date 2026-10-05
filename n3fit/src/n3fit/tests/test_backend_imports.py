"""
The "import ban": n3fit must not reach into the deep-learning framework outside the
backend implementation.

The rule enforced here (see the design documents, §"governance"):

* files under ``n3fit/backends/<name>_backend/`` **may** import a framework: that is what
  they are for;
* everything else -- including the backend *facade* (``n3fit/backends/__init__.py``),
  the contract (``backends/base.py``) and the registry (``backends/registry.py``) --
  may **not**;
* leaks must be listed in ``KNOWN_LEAKS`` with the phase that removes them.  That list is
  **empty as of P1**, so from here on any framework import outside a backend implementation
  is a failure, full stop: ``n3fit`` can no longer reach the framework except through
  ``n3fit.backends.base``.  (A stale entry -- one that has been fixed -- is reported rather
  than failed, so removing a leak never breaks the build.)

Why bother: this is the contract that makes "backend-agnostic" verifiable rather than
aspirational.  Without it, the next feature added under time pressure will import keras
inside ``n3fit/`` and the second backend becomes impossible to write.
"""

import ast
from pathlib import Path

# Frameworks and framework-adjacent packages that must stay inside the backends
BANNED = {"keras", "tensorflow", "torch", "jax", "optax", "flax", "tf_keras"}

# Files allowed to import a banned module, with the phase that fixes them.
# This is deliberately explicit: it is a to-do list, not a courtesy.
#
# P1 emptied it: ``checks.py`` now asks ``capabilities.optimizers``/``initializers``/
# ``supports_tensorboard``, ``io/writer.py`` asks ``backend.version_info()``, and
# ``hyper_optimization/rewards.py`` asks ``capabilities.requires_eager_workaround`` and
# ``state.set_eager()``.  New entries require a phase that removes them.
KNOWN_LEAKS = {}

# The tests themselves may import frameworks (they test them / skip on them)
ALLOWED_PATHS = ("tests/",)


def n3fit_source_root():
    """``.../n3fit/src/n3fit``, derived from this file's location."""
    return Path(__file__).resolve().parents[1]


def imported_frameworks(path):
    """All banned top-level modules imported by a python file (any nesting level)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module] if node.module else []
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top in BANNED:
                found.add(top)
    return found


def is_backend_implementation(path, root):
    """Whether the file lives in ``backends/<name>_backend/`` (where frameworks are fine)."""
    relative = path.relative_to(root)
    parts = relative.parts
    return len(parts) > 2 and parts[0] == "backends" and parts[1].endswith("_backend")


def collect_leaks():
    """``{relative_path: {framework: ...}}`` for every file that violates the ban."""
    root = n3fit_source_root()
    leaks = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if is_backend_implementation(path, root):
            continue
        if str(relative).replace("\\", "/").startswith(ALLOWED_PATHS):
            continue
        for framework in imported_frameworks(path):
            key = "/".join(relative.parts)
            leaks.setdefault(key, {})[framework] = True
    return leaks


def test_no_new_framework_imports_outside_the_backends():
    leaks = collect_leaks()
    new_leaks = {
        path: sorted(frameworks) for path, frameworks in leaks.items() if path not in KNOWN_LEAKS
    }
    assert not new_leaks, (
        "New imports of a deep-learning framework outside n3fit/backends/<name>_backend/.\n"
        "n3fit code must go through the backend contract (n3fit.backends.base). If the\n"
        "import is legitimate, it belongs in the backend implementation; if it is a\n"
        "transitional state, add it to KNOWN_LEAKS with the phase that removes it.\n"
        f"Offending files: {new_leaks}"
    )


def test_known_leaks_are_still_accurate():
    """Report (not fail on) entries that have already been fixed, to keep the list honest."""
    leaks = collect_leaks()
    stale = []
    for path, frameworks in KNOWN_LEAKS.items():
        found = leaks.get(path, {})
        if not found:
            stale.append(
                f"{path}: file no longer imports any framework -- remove it from KNOWN_LEAKS"
            )
        elif extra := sorted(set(found) - set(frameworks)):
            stale.append(f"{path}: also imports {extra} -- update KNOWN_LEAKS")
    if stale:  # pragma: no cover - depends on the state of the migration
        print("KNOWN_LEAKS is out of date:\n  " + "\n  ".join(stale))


def test_the_backends_package_is_reachable_without_a_framework():
    """The facade, the contract and the registry are framework-free *by inspection*.

    (The runtime version of this check, with an import blocker, lives in
    ``backend_conformance/test_registry.py``.)
    """
    root = n3fit_source_root()
    for name in ("backends/__init__.py", "backends/base.py", "backends/registry.py"):
        path = root / name
        assert path.exists(), f"{name} is missing"
        assert not imported_frameworks(path), f"{name} imports a deep-learning framework"

# --------------------------------------------------------------------------------------------
# The legacy-name ratchet
# --------------------------------------------------------------------------------------------
# The import ban above stops n3fit from *importing a framework*.  This stops it from growing
# new dependencies on the legacy Keras-flavoured names, which is the other half of the same
# boundary: those names (`MetaModel`, `operations`, `Input`, ...) are the accidental API the
# contract replaces in P2, and every one of them is a place a second backend would have to
# pretend to be Keras.
#
# The table below is the P2 work list, taken from the code as of P1 (it is the same list as
# the call-site table in the design documents).  A new entry fails; an entry that disappears
# is reported, so the table keeps describing reality without breaking the build.
LEGACY_NAME_USERS = {
    # P2 removed ``operations`` from rewards.py (the override function is plain numpy and the
    # adapter converts it); P3 removed ``MetaModel`` too, by moving the diagnostic graph the
    # ``*_exp`` block used to build into the adapter (``KerasModelView.prediction_before``).
    # rewards.py now imports nothing legacy at all.

    "layers/DIS.py": {"operations"},
    # P4: the layer that *names* a graph output.  It is the same kind of thing as the rest of
    # ``layers/`` -- Keras layers by construction, which is why the whole directory is listed here
    # -- and it is the one that the contract needs from it: the engine routes a term to its
    # prediction by name, and in Keras a name on a tensor comes from the operation that made it.
    "layers/output.py": {"MetaLayer"},
    "layers/DY.py": {"operations"},
    "layers/__init__.py": {"MetaLayer"},
    "layers/losses.py": {"MetaLayer", "operations"},
    "layers/mask.py": {"MetaLayer", "operations"},
    "layers/msr_normalization.py": {"MetaLayer", "operations"},
    "layers/observable.py": {"MetaLayer", "operations"},
    "layers/preprocessing.py": {"MetaLayer", "MultiInitializer", "constraints", "operations"},
    "layers/rotations.py": {"MetaLayer", "operations"},
    "layers/x_operations.py": {"MetaLayer", "operations"},
    "model_gen.py": {
        "Input",
        "Lambda",
        "MetaLayer",
        "MetaModel",
        "NN_LAYER_ALL_REPLICAS",
        "NN_PREFIX",
        "PREPROCESSING_LAYER_ALL_REPLICAS",
        "base_layer_selector",
        "operations",
        "regularizer_selector",
    },
    # P2 removed ``NN_LAYER_ALL_REPLICAS`` (the summary chain now goes through the role-aware
    # ``summary()``); the rest is P4 training-loop machinery.
    "model_trainer.py": {"MetaModel", "callbacks", "clear_backend_state", "operations"},
    "msr.py": {"Input", "Lambda", "MetaModel", "operations"},
    # (P2 removed the "vpinterface.py" entry: it addressed the photon and the preprocessing
    # factors by role instead of by layer name and now imports nothing legacy at all.)

}

# Names the facade exports that are part of the *new* interface, not the legacy surface.
NEW_INTERFACE = {
    "Backend",
    "Capabilities",
    "FitResult",
    "Objective",
    "ObjectiveSpec",
    "ParametrizationSpec",
    "ShapeSpec",
    "OptimizerSpec",
    "ROLE_NN",
    "ROLE_OBJECTIVE",
    "ROLE_PHOTON",
    "ROLE_PREPROCESSING",
    "ROLE_REFERENCE",
    "ROLE_SUMRULE",
    "ROLES",
    "GROUP_TRAINING",
    "GROUP_VALIDATION",
    "GROUP_EXPERIMENTAL",
    "GROUP_POSITIVITY",
    "GROUP_INTEGRABILITY",
    "OPTIMIZATION_GROUPS",
    "REPORT_GROUPS",
    "ObjectiveGroup",
    "available_backends",
    "get_backend",
    "importable_backends",
    "register_backend",
    "selected_backend_name",
}


def collect_legacy_name_users():
    """``{relative file: {legacy names imported from n3fit.backends}}`` for all of n3fit."""
    root = n3fit_source_root()
    users = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if is_backend_implementation(path, root):
            continue
        if str(relative).replace("\\", "/").startswith(ALLOWED_PATHS):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "n3fit.backends":
                names.update(alias.name for alias in node.names if alias.name not in NEW_INTERFACE)
        if names:
            users["/".join(relative.parts)] = names
    return users


def test_no_new_legacy_backend_names():
    users = collect_legacy_name_users()
    new = {}
    for path, names in users.items():
        if path not in LEGACY_NAME_USERS:
            new[path] = sorted(names)
            continue
        extra = sorted(names - LEGACY_NAME_USERS[path])
        if extra:
            new[path] = extra
    assert not new, (
        "New uses of the legacy Keras-flavoured backend names. The contract"
        "(n3fit/backends/base.py) replaces them: MetaModel/MetaLayer/callbacks -> the Backend"
        "object, operations -> backend.ops, Input/Lambda/... -> the spec-driven builders."
        "If the use is temporary, add it to LEGACY_NAME_USERS with the phase that removes it."
        f"\nOffending: {new}"
    )


def test_no_direct_backend_implementation_imports():
    """Only ``n3fit.backends`` (facade) and its registry may reach into a backend implementation.

    ``from n3fit.backends.keras_backend... import ...`` anywhere else means the caller has
    chosen a backend by hand: the same thing the import ban forbids for frameworks, one level
    up.
    """
    root = n3fit_source_root()
    offenders = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if is_backend_implementation(path, root):
            continue
        if str(relative).replace("\\", "/").startswith(ALLOWED_PATHS):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "n3fit.backends."
            ):
                if not node.module.endswith(("registry", "base")):
                    modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("n3fit.backends.")
                    and not alias.name.endswith(("registry", "base"))
                )
        if modules:
            offenders["/".join(relative.parts)] = sorted(modules)
    assert not offenders, (
        f"These files import a backend implementation directly instead of going through"
        f"n3fit.backends: {offenders}"
    )


# --------------------------------------------------------------------------------------------
# The method-level ratchet (P2b)
# --------------------------------------------------------------------------------------------
# The import ban stops `from n3fit.backends import MetaModel`; this stops *calling the methods
# of a graph object* from outside the backend.  P2 moved the role-based access, P2b moved
# evaluation and the replica split; what is left is listed below with the phase that removes it.
#
# Names are matched on the parsed AST, not by grep: comments and docstrings mention them
# constantly (they are how this code explains itself) and only real attribute access is a
# migration item.  Tests are not scanned -- they exercise the legacy generators deliberately
# (e.g. `tests/test_multireplica.py` checks the replica weight layout that P4 replaces).
BACKEND_METHOD_NAMES = {
    "add_covmat",
    "compile",
    "compute_losses",
    "get_layer",
    "get_layer_re",
    "get_replica_weights",
    "get_weight_by_name",
    "load_identical_replicas",
    "load_weights",
    "perform_fit",
    "predict",
    "register_photon",
    "reset_layer_weights_to",
    "save_weights",
    "set_replica_weights",
    "set_replica_weights_from_file",
    "split_replicas",
    "update_mask",
    # P5 retired the h5 weight store: fits write and read their replicas through
    # ``Backend.save``/``Backend.load`` (the ``n3fit-weights/2`` files), and the four names above
    # that it replaced must not come back.
}

# file -> the names it still uses, with the phase that removes them.
#
# **P4 emptied this table.**  ``compile``/``perform_fit``/``compute_losses`` are gone from n3fit
# (the engine compiles and drives the graph, and evaluation is ``Optimizer.evaluate``), the
# stopping snapshot went with them (``Optimizer.weights``/``Ensemble.set_weights``), and the k-fold
# diagnostic that used to reach for the Keras adapter's term lookup with it (Q15 deleted
# ``fit_future_tests``).  It stays as a *table* -- with no entries -- because the check it drives is
# the thing that keeps them out: a new ``model.get_layer(...)`` in n3fit fails this test.
#
# ``reset_layer_weights_to`` is in ``BACKEND_METHOD_NAMES`` with no users either: P3 moved the
# k-fold reset to ``Objective.set_scalar`` (the multiplier is a scalar of the term, not a weight to
# poke), and the method itself must not come back -- it is exactly the "address a graph by layer
# name" pattern the role/objective APIs replace, and it was silently wrong anyway.
BACKEND_METHOD_USERS = {}


def _backend_method_calls(path):
    """The names from ``BACKEND_METHOD_NAMES`` used as attribute accesses in ``path``."""
    tree = ast.parse(path.read_text())
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def collect_backend_method_users():
    n3fit_root = Path(__file__).resolve().parents[1]
    users = {}
    for path in sorted(n3fit_root.rglob("*.py")):
        relative = path.relative_to(n3fit_root)
        if str(relative).startswith(("backends/", "tests/")) or "__pycache__" in str(relative):
            continue
        names = _backend_method_calls(path) & BACKEND_METHOD_NAMES
        if names:
            users["/".join(relative.parts)] = names
    return users


def test_no_new_backend_method_calls_outside_the_backend():
    """A new ``model.get_layer(...)``/``.predict(...)`` in n3fit is a regression: the contract
    has replacements (``view``, ``ensemble``, ``weights``, ``override``, ``__call__``)."""
    users = collect_backend_method_users()
    new = {
        path: sorted(names - BACKEND_METHOD_USERS.get(path, set()))
        for path, names in users.items()
        if names - BACKEND_METHOD_USERS.get(path, set())
    }
    assert not new, (
        f"these files call backend-object methods the contract replaces: {new}\n"
        f"Use the contract (Backend.view / Backend.ensemble / Model.weights / Model.override /"
        f" Model.__call__) or record the new site in BACKEND_METHOD_USERS with its phase."
    )


def test_backend_method_table_is_still_accurate():
    """The table must keep describing reality: a site that migrates leaves the table."""
    users = collect_backend_method_users()
    stale = []
    for path, recorded in BACKEND_METHOD_USERS.items():
        missing = sorted(recorded - users.get(path, set()))
        if missing:
            stale.append(f"{path}: no longer uses {missing} -- update BACKEND_METHOD_USERS")
    if stale:  # pragma: no cover - depends on the state of the migration
        print("BACKEND_METHOD_USERS is out of date:\n  " + "\n  ".join(stale))


def test_legacy_name_table_is_still_accurate():
    """``LEGACY_NAME_USERS`` must describe the current state of the migration (P2's work list)."""
    users = collect_legacy_name_users()
    stale = []
    for path, recorded in LEGACY_NAME_USERS.items():
        found = users.get(path, set())
        if not found:
            stale.append(f"{path}: no longer imports any legacy name -- remove it from the table")
        elif missing := sorted(recorded - found):
            stale.append(f"{path}: no longer imports {missing} -- update the table")
    if stale:  # pragma: no cover - depends on the state of the migration
        print("LEGACY_NAME_USERS is out of date:\n  " + "\n  ".join(stale))
