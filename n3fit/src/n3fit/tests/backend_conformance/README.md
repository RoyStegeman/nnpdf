# Backend conformance suite

This suite is the *specification* of the backend contract (`n3fit/backends/base.py`) in
executable form: any backend that passes it can be handed to n3fit.  It is not a unit-test
suite for the Keras backend -- the Keras backend has its own tests (`tests/test_modelgen.py`,
`tests/test_losses.py`, ...).

## How it is organised

| File | What it checks | Runs without a framework? |
| --- | --- | --- |
| `test_ops.py` | `Backend.ops`: dense, einsum, multiplication, logs, division, flattening, tensor products, sums, NaNs, derivatives. Seeded from the old `tests/test_backend.py`. | no (needs a backend) |
| `test_capabilities.py` | `Backend.capabilities`: registries have the contracted structure, everything declared is actually available, and the Keras declarations agree with the legacy dictionaries they were derived from (anti-drift). | partly |
| `test_registry.py` | discovery, selection (argument / `N3FIT_BACKEND` / `KERAS_BACKEND`), registration, error messages, and that the package imports with all frameworks blocked. | **yes** |
| `../test_backend_imports.py` | the import ban: no framework import outside `backends/<name>_backend/`, with the current leaks listed as `KNOWN_LEAKS` (a to-do list that is expected to shrink). | **yes** |
| `../test_backends_facade.py` | the facade keeps every legacy name the rest of n3fit imports, resolves them lazily, and stays importable without a framework. | **yes** |
| `test_model_roles.py` | the P2 access API against the reference implementation in `testing_backends.py`: the role vocabulary, `weights(role)`, `bound_inputs`, `bind_input` (and that the graph still evaluates), `override` + `freeze`. | **yes** |
| `test_keras_role_view.py` | the Keras adapter (`keras_backend/roles.py`) against a fake graph: the role→layer-name table, the path keys of the preprocessing weights, the photon rebind, and that overriding a missing role raises. | **yes** |
| `test_central_value.py` | the real `rewards._set_central_value` end to end on the double: which grid it reads, the function it installs, the freezing. | **yes** |
| `test_model_ensemble.py` | P2b: `Model.__call__` (numpy in/out, shapes preserved) and `Ensemble` (`__iter__`/`__getitem__`/`__len__`, slicing, per-replica weight maps) against the reference implementation. | **yes** |
| `test_optimizer_run.py` | P4: `Optimizer.run` reproduces the loop it replaces.  Runs one small synthetic fit (three role graphs, per-replica masks, chi2 + positivity + integrability, a Lagrange schedule, early stopping with a best-weight restore) and compares it with the recording of the *pre-P4* loop in `p4_golden.json`.  Slow: two 30-step fits, ~30 s. | no (needs a backend) |
| `p4_golden.py` | The synthetic problem behind that test (and the script that recorded the fixture from a pre-P4 checkout).  Also documents the measured float32 floor the tolerances come from. | no (needs a backend) |
| `test_flip.py` | P4 acceptance, part two: `ModelTrainer`'s own assembly.  The real `ObservableWrapper` (as called by `_model_generation`) produces `(prediction, term)` pairs; the three role graphs carry one named output per term, all names distinct (A6); the engine runs a short fit through the real `StoppingHook` and `LagrangeHook`, and `POS_val` is *not* scaled while `POS` is. | no (needs a backend) |
| `testing_backends.py` | `NumpyDoubleBackend`: a numpy-only backend used as a test double, plus `NumpyModelView` (the reference `Model`). | **yes** |

The framework-free files are the point of P0: they can run in the minimal CI job (no
TensorFlow), which is what makes "backend-agnostic" checkable at all.

## Running it

```console
# everything, in an environment where a framework is installed
pytest n3fit/src/n3fit/tests/backend_conformance n3fit/src/n3fit/tests/test_backend_imports.py

# the part that must work with no framework at all
pytest n3fit/src/n3fit/tests/backend_conformance n3fit/src/n3fit/tests/test_backend_imports.py \
       n3fit/src/n3fit/tests/test_backends_facade.py
```

## Adding a backend

1. Implement `Backend` (see `base.py`; it is a `Protocol`, so structural conformance is
   enough -- no inheritance required).
2. Declare its `Capabilities`, including `uses_validation_stopping` per optimizer (D13) and the
   `derivatives` it supports (D12).
3. Register it (`register_backend`) or add it to `_BUILTIN_BACKENDS` in
   `n3fit/backends/registry.py`.
4. Run this suite with `N3FIT_BACKEND=<name>`.  Anything that fails is either a bug in the
   backend or a gap in the contract -- both worth knowing before a fit is attempted.

## Status

* **P4** — the engine (`Optimizer.run`), the hooks, and **the flip**: n3fit assembles the terms and
  the role ensemble and asks the backend to run the loop.  `p4_golden.py`+`test_optimizer_run.py`
  check the engine against the recorded pre-P4 loop (`test_flip.py` checks the trainer's own
  assembly).  The legacy loop — `MetaModel.perform_fit`/`compile`/`compute_losses`, the three Keras
  callbacks, the legacy `Stopping` — is deleted, and `p4_golden.run_legacy` says so rather than
  failing obscurely.
* **P0** — the suite exists, is parametrized over backends, and the framework-free half runs
  with no framework installed.
* **P1** — `KNOWN_LEAKS` (in `../test_backend_imports.py`) is **empty**: n3fit can no longer
  import a framework except through the contract.  The `Ops` vocabulary is reconciled in
  `keras_backend/ops.py`, and `test_ops.py` now runs unchanged against both the Keras backend
  and the numpy reference implementation.
* **P2** — role-based access is implemented and consumed: the six call sites that reached into
  the graph by name now go through `Backend.view(graph)`, `vpinterface.py` has no legacy import
  left, and `LEGACY_NAME_USERS` records the smaller remainder.  `weights(role=...)` implements
  `preprocessing`; the other roles raise until P4 gives them the per-replica layout.
* **P2b** — the last two Keras accessors are gone: `model.predict(...)` is `Model.__call__` and
  `pdf_model.split_replicas()` is `Backend.ensemble(graph)`, which splits a stacked graph in the
  backend.  `../test_backend_imports.py` now also carries a **method-level ratchet**
  (`BACKEND_METHOD_USERS`): the remaining sites are listed with the phase that removes them.

## What this suite deliberately does *not* do (yet)

* **No numerical golden fits.**  Reproducing the Keras regressions is P6 (`tests/regressions`
  and `extra_tests/regression_fits` are the reference); until then the conformance suite is
  about the *interface*, not about bit-for-bit agreement.
* **No performance gate.**  The `monitor_every=None` fast path (D11) needs a compiled backend
  to be worth measuring; it belongs with the second backend in P6.
* **No coverage of the staged members.**  `Backend.parametrization/optimizer/objective/...` raise
  `NotImplementedError` naming their phase until P3/P4 implement them; the tests for those
  arrive with the implementations.
* **No enumeration of the remaining legacy-method sites.**  They are listed in
  `../test_backend_imports.py` (`BACKEND_METHOD_USERS`, P3/P4), but the suite does not check that
  a file *listed* there still uses the backend objects the way the table assumes.
* **No coverage of the un-implemented roles.**  `weights(role=...)` for `nn`, `objective` and
  `sumrule` raises by design (P4 defines the per-replica layout); only the *behaviour* of the
  three outcomes -- unknown / not implemented / empty -- is tested.
