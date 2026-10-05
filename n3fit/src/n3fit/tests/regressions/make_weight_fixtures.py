#!/usr/bin/env python
"""
Regenerate the regression weight fixtures (``weights_*.weights.npz``).

The fixtures hold the *initial* weights each regression fit starts from, captured from the real
pipeline: this script runs ``vp-setupfit`` + ``n3fit`` with the runcard's ``load:`` pointed at a
placeholder and intercepts ``Backend.load`` right before training to snapshot the freshly
initialised weights (then aborts the run).  Capturing at the load point -- instead of rebuilding
the model here -- guarantees the fixture's layout is exactly the fit's own, for every runcard.

Because the capture *is* the fit's own initialisation, a fit started from the fixture reproduces
the fit's seeded trajectory bit for bit, and the recorded json regressions stay valid: loading
the fixture is a no-op in value, and the legacy h5 fixtures were already inert anyway -- current
Keras cannot read them and used to ignore them silently (the P5 store refuses them loudly).

Usage (from ``n3fit/src``, in an environment where ``validphys`` and ``n3fit`` import)::

    KERAS_BACKEND=jax python n3fit/tests/regressions/make_weight_fixtures.py [runcard.yml:replica ...]

with no arguments every job below is regenerated.  The framework chosen by ``KERAS_BACKEND``
does not matter for correctness (the fixture stores plain numpy arrays), only for which backend
does the initialising.
"""

import contextlib
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# runcard -> (fixture base name, replicas), as test_fit.py's _auxiliary_performfit copies them
JOBS = {
    "quickcard.yml": ("weights", (1, 3)),
    "quickcard_pol.yml": ("weights_pol", (1, 3)),
    "quickcard_qed.yml": ("weights_qed", (1, 3)),
}
PLACEHOLDER = "placeholder.weights.npz"


def _parse_jobs(argv):
    if not argv:
        return [(runcard, base, replica) for runcard, (base, replicas) in JOBS.items() for replica in replicas]
    jobs = []
    for spec in argv:
        runcard, _, replica = spec.partition(":")
        if runcard not in JOBS or not replica.isdigit():
            raise SystemExit(f"unknown job {spec!r}; expected e.g. 'quickcard.yml:1'")
        jobs.append((runcard, JOBS[runcard][0], int(replica)))
    return jobs


def capture_one(runcard, base, replica):
    """Run the pipeline for ``runcard`` replica ``replica`` and snapshot the initial weights."""
    from n3fit.backends.keras_backend import weights as store
    from n3fit.backends.keras_backend.backend import KerasBackend

    out = HERE / f"{base}_{replica}.weights.npz"

    def _capture(self, ensemble, path, replica=None):  # pylint: disable=unused-argument
        values = store.weight_map(ensemble._weights_graph, replica=0)
        manifest = store.make_manifest(values, n_replicas_graph=len(ensemble), replica=0)
        store.save_weight_file(out, values, manifest=manifest)
        print(f"[fixture] captured {out.name}: {len(values)} weights")
        raise SystemExit(0)

    KerasBackend.load = _capture

    workdir = HERE / f".fixture_build_{base}_{replica}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    text = (HERE / runcard).read_text()
    patched, n_subs = re.subn(r"^load:\s*.*$", f'load: "{PLACEHOLDER}"', text, flags=re.M)
    if n_subs != 1:
        raise SystemExit(f"{runcard}: expected one 'load:' line, found {n_subs}")
    (workdir / runcard).write_text(patched)
    (workdir / PLACEHOLDER).write_bytes(b"")

    from n3fit.scripts import n3fit_exec, vp_setupfit

    old_argv, old_cwd = sys.argv, os.getcwd()
    os.chdir(workdir)
    try:
        for argv, entry in (
            (["vp-setupfit", runcard], vp_setupfit.main),
            (["n3fit", runcard, str(replica)], n3fit_exec.main),
        ):
            sys.argv = argv
            try:
                entry()
                code = 0
            except SystemExit as exit_:  # expected: the capture aborts the fit right at load
                code = exit_.code or 0
            if code != 0:
                raise SystemExit(f"{' '.join(argv)} failed (exit {code})")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        shutil.rmtree(workdir, ignore_errors=True)

    if not out.exists():
        raise SystemExit(f"{runcard} replica {replica}: the load point was never reached")
    print(f"[fixture] {out}")


def main(argv):
    os.environ.setdefault("KERAS_BACKEND", "jax")
    for runcard, base, replica in _parse_jobs(argv):
        capture_one(runcard, base, replica)


if __name__ == "__main__":
    main(sys.argv[1:])
