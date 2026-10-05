"""
P4: ``Optimizer.run`` reproduces the loop it replaces, fit for fit.

This is the P4 acceptance test.  ``p4_golden.py`` builds one small synthetic fit that exercises
everything the training loop does (three role graphs, per-replica masks, chi2 + positivity +
integrability, a Lagrange schedule, cross-validation stopping with a best-weight snapshot and
restore) and records what the *pre-P4* loop produced -- ``MetaModel.perform_fit`` plus the Keras
callbacks, run against a pristine checkout.  Here the same problem is solved by the engine and
compared with the recording.

What is checked, and how tightly, is in the fixture's docstring (``p4_golden.py``): the run is a
float32 fit, so the two implementations cannot agree bit for bit -- per step they differ by a single
ulp in one weight, and this toy's dynamics stretch that to 1.6e-4 relative in the worst weight entry
after 30 steps.  Everything the fit *decides* on is much tighter than that, and is asserted
separately so a regression says which of the two moved.

The fixture is environment-pinned: the recorded trajectory is chaotic in the initializer stream, so
its numbers are only comparable under the exact keras/jax/framework stack that recorded them (the
``env`` block in ``p4_golden.json``).  Under any other stack the comparisons below are skipped with
a pointer to how to record a new fixture -- the engine itself is still covered by the rest of this
suite, which is stack-independent.
"""

import json
import pathlib

import numpy as np
import pytest

from n3fit.tests.backend_conformance.p4_golden import CASES, _flatten, environment, run_contract

FIXTURE = pathlib.Path(__file__).parent / "p4_golden.json"

# Measured on the fixture, see the module docstring: the scalars are float32 sums of the same
# terms computed in a different order; the weights additionally carry 30 steps of that difference.
RTOL_SCALARS = 1e-4
RTOL_WEIGHTS = 1e-3
ATOL = 1e-6
RTOL_EXACT = 1e-6  # the multipliers: updated ten times at most, compared as the legacy did


@pytest.fixture(scope="module", autouse=True)
def _keras_only():
    """The reference implementation is the Keras loop, so this test needs a backend installed."""
    pytest.importorskip("keras")


@pytest.fixture(scope="module")
def fixture_file():
    with open(FIXTURE) as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def reference(fixture_file):
    return fixture_file["cases"]


@pytest.fixture(scope="module")
def _recorded_environment(fixture_file):
    """The fixture records one float32 trajectory of a chaotic toy (see ``p4_golden.py``).

    The trajectory is fixed by the random stream of the initializers, which moves with every
    keras/jax version and with the framework Keras runs on -- so the numbers are only comparable
    in the environment that recorded them.  In any other environment the engine is still checked
    everywhere else in this suite; only the comparison against *these* numbers is out of scope.
    """
    recorded = fixture_file.get("env")
    if recorded is None:  # pre-P4.1 fixture without an environment block
        return
    current = environment()
    diffs = {k: (recorded.get(k), current.get(k)) for k in recorded if recorded.get(k) != current.get(k)}
    if diffs:
        pytest.skip(
            "p4_golden.json was recorded in a different environment (the fit it records is"
            f" chaotic in the initializer stream): {diffs}.  Run 'python p4_golden.py legacy"
            " --out p4_golden.json' on a pre-P4 checkout of this environment to record a new one."
        )


@pytest.fixture(scope="module")
def produced(_recorded_environment):
    """The engine's answer for each case (one fit per case, shared by the tests)."""
    return {name: run_contract(case) for name, case in CASES.items()}


def test_the_reference_records_both_cases(reference):
    """The fixture is the pre-P4 loop's output: it should describe the two paths it was made for."""
    assert set(reference) == set(CASES)
    assert reference["main"]["stop_epoch"] == 30  # ran out of steps
    assert reference["stopping"]["stop_epoch"] == 4  # stopped early
    assert reference["stopping"]["e_best_chi2"] == [0, 0]  # ... and restored the best weights


@pytest.mark.parametrize("case", sorted(CASES))
def test_stop_epoch(reference, produced, case):
    """How many steps the fit took -- the stopping decision itself."""
    assert produced[case]["stop_epoch"] == reference[case]["stop_epoch"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_best_epochs_and_positivity(reference, produced, case):
    """Per-replica best epoch and the positivity verdict: discrete, so equal or wrong."""
    assert produced[case]["e_best_chi2"] == reference[case]["e_best_chi2"]
    assert produced[case]["positivity_statuses"] == reference[case]["positivity_statuses"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_multipliers(reference, produced, case):
    """The Lagrange multipliers: the schedule fired at the same steps with the same values."""
    want = np.array(_flatten(reference[case]["multipliers"]))
    got = np.array(_flatten(produced[case]["multipliers"]))
    assert np.allclose(want, got, rtol=RTOL_EXACT, atol=ATOL), f"{want} vs {got}"


@pytest.mark.parametrize("case", sorted(CASES))
def test_monitored_steps(reference, produced, case):
    """The engine evaluated the same number of times as the loop it replaces."""
    assert len(produced[case]["states"]) == len(reference[case]["states"])
    assert sorted(produced[case]["parsed"], key=int) == sorted(reference[case]["parsed"], key=int)


@pytest.mark.parametrize("case", sorted(CASES))
def test_per_step_losses(reference, produced, case):
    """The per-step training losses: what the stopping rule and the logs are made of."""
    want = np.array([_flatten(step["losses"]) for step in reference[case]["states"]])
    got = np.array([_flatten(step["losses"]) for step in produced[case]["states"]])
    assert want.shape == got.shape
    assert np.allclose(want, got, rtol=RTOL_SCALARS, atol=ATOL), (
        f"worst step: {np.argmax(np.abs(want - got).max(axis=1))}"
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_validation_chi2_and_loss(reference, produced, case):
    """Per-replica validation chi2 and loss: the only numbers the stopping rule compares."""
    for key in ("vl_chi2", "vl_loss"):
        steps = sorted(reference[case]["parsed"], key=int)
        want = np.array([reference[case]["parsed"][step][key] for step in steps])
        got = np.array([produced[case]["parsed"][step][key] for step in steps])
        assert want.shape == got.shape
        assert np.allclose(want, got, rtol=RTOL_SCALARS, atol=ATOL), f"{key}: {want} vs {got}"


@pytest.mark.parametrize("case", sorted(CASES))
def test_final_weights(reference, produced, case):
    """The weights the fit ends with -- including the restore of the best ones when it stops."""
    want = np.array(_flatten(reference[case]["weights"]))
    got = np.array(_flatten(produced[case]["weights"]))
    assert want.shape == got.shape
    assert np.allclose(
        want, got, rtol=RTOL_WEIGHTS, atol=ATOL
    ), f"worst relative difference {np.max(np.abs(want - got) / np.abs(want)):.3e}"
