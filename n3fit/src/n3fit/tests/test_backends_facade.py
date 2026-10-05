"""
Tests of the backend facade (``n3fit/backends/__init__.py``).

The facade has to do two things at once during the migration:

1. expose the *new* interface (registry, contract) without importing a deep-learning
   framework, and
2. keep every *legacy* Keras-flavoured name working (``MetaModel``, ``operations``, ...),
   importing the framework lazily on first use.

The second part is tested here without installing a framework, by substituting the import
machinery: what is under test is the resolution mechanism, not Keras.
"""

import ast
import importlib
import sys
import types
from pathlib import Path

import pytest

import n3fit.backends as facade


@pytest.fixture
def restore_facade():
    """Undo the caching done by the lazy resolver, so tests cannot leak into each other."""
    before = set(globals())
    module_globals = vars(facade)
    snapshot = dict(module_globals)
    yield
    module_globals.clear()
    module_globals.update(snapshot)
    assert before  # silence linters about the unused-but-intentional variable


def test_every_legacy_usage_in_the_tree_is_exposed():
    """Every name the rest of n3fit imports from the facade must be resolvable.

    This is the safety net of the P0 rewrite: the facade moved from eager to lazy imports,
    so a name that is missing from the mapping would break at *use* time, deep inside a fit,
    rather than at import time.
    """
    root = Path(__file__).resolve().parents[1]
    requested = set()
    for path in root.rglob("*.py"):
        if path == Path(__file__).resolve():
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "n3fit.backends":
                requested.update(alias.name for alias in node.names)

    assert requested, "the scan found no imports of n3fit.backends: is the path wrong?"
    missing = sorted(name for name in requested if name not in getattr(facade, "__all__", []))
    assert not missing, (
        f"{missing} are imported from n3fit.backends by other n3fit modules but are not "
        f"exported by the facade"
    )


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        facade.this_is_not_a_backend_name


def test_legacy_names_are_resolved_lazily(monkeypatch, restore_facade):
    """A legacy name is imported on first access, cached, and reported once."""
    fake_keras_module = types.ModuleType("n3fit.backends.keras_backend.MetaModel")
    fake_keras_module.MetaModel = "the-fake-MetaModel"  # type: ignore[attr-defined]
    imported = []

    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "n3fit.backends.keras_backend.MetaModel":
            imported.append(name)
            return fake_keras_module
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.delitem(vars(facade), "MetaModel", raising=False)

    assert facade.MetaModel == "the-fake-MetaModel"
    assert imported == ["n3fit.backends.keras_backend.MetaModel"]

    # second access must not import again
    assert facade.MetaModel == "the-fake-MetaModel"
    assert imported == ["n3fit.backends.keras_backend.MetaModel"]


def test_facade_is_importable_without_a_framework():
    """Importing the facade must not have pulled a framework in (in this very process)."""
    banned = {"keras", "tensorflow", "torch", "jax"}
    loaded = banned & set(sys.modules)
    if loaded:
        pytest.skip(
            f"the framework(s) {sorted(loaded)} are already imported in this process, "
            f"so this check cannot say anything (see "
            f"backend_conformance/test_registry.py for the subprocess version)"
        )
    assert not (banned & set(sys.modules))

def module_level_bindings(tree):
    """Names bound at module level, descending into if/try/with but not into functions.

    ``internal_state.py`` defines ``set_eager`` inside an ``if K.backend() == ...`` block,
    so a check that only looked at the top-level statement list would miss it.
    """
    names = set()
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        # descend into compound statements, but not into function/class bodies
        for field in ("body", "orelse", "finalbody"):
            stack.extend(getattr(node, field, []) or [])
        stack.extend(getattr(node, "handlers", []) or [])
    return names


def test_legacy_mapping_points_at_things_that_exist():
    """Every entry of the lazy map must name a real module (and attribute) in the tree.

    This is a static check on purpose: a wrong entry would only fail at *use* time inside a
    fit, and this suite cannot import the framework to find out.  It already caught one
    mistake -- ``callbacks``/``constraints``/``operations`` are submodules of a package with
    an empty ``__init__``, so they must be resolved as modules and not as attributes: the
    historical ``from n3fit.backends.keras_backend import operations`` works through the
    import system, whereas ``getattr(<that package>, "operations")`` does not.
    """
    root = Path(__file__).resolve().parents[1]
    for name, (module_name, attribute) in facade._LEGACY.items():
        relative = Path(*module_name.split(".")[1:])
        module_path = root / relative.with_suffix(".py")
        package_init = root / relative / "__init__.py"

        if attribute is None:
            assert module_path.exists(), (
                f"legacy name {name!r} is declared as 'the module itself' ({module_name!r}), "
                f"which must therefore be a module file; {module_path.relative_to(root)} does "
                f"not exist"
            )
            continue

        assert module_path.exists() or package_init.exists(), (
            f"legacy name {name!r} points at {module_name!r}, which does not exist in the tree"
        )
        source = module_path if module_path.exists() else package_init
        if attribute not in module_level_bindings(ast.parse(source.read_text())):
            hint = ""
            if package_init.exists() and (root / relative / f"{attribute}.py").exists():
                hint = (
                    f"  ({module_name} is a package with an empty __init__ and {attribute!r} is "
                    f"one of its submodules: map it to \"{module_name}.{attribute}\" with "
                    f"attribute=None instead)"
                )
            raise AssertionError(
                f"legacy name {name!r} resolves to {module_name}.{attribute}, but "
                f"{source.relative_to(root)} does not bind {attribute!r} at module level.{hint}"
            )


def test_legacy_names_are_a_subset_of_the_original_exports():
    """The lazy map must not *invent* names: it replaces a fixed list of eager imports."""
    # the eager imports of the historical facade, transcribed
    original = {
        "callbacks", "constraints", "operations", "MetaLayer", "MetaModel", "NN_PREFIX",
        "NN_LAYER_ALL_REPLICAS", "PREPROCESSING_LAYER_ALL_REPLICAS", "Concatenate", "Input",
        "Lambda", "base_layer_selector", "regularizer_selector", "clear_backend_state",
        "get_physical_gpus", "set_eager", "set_initial_state", "MultiInitializer",
    }
    invented = sorted(set(facade._LEGACY) - original)
    missing = sorted(original - set(facade._LEGACY))
    assert not invented, f"the lazy map exports names the Keras facade never had: {invented}"
    assert not missing, f"the lazy map is missing names the Keras facade used to export: {missing}"
