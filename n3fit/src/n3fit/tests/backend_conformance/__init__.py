"""
Conformance suite for n3fit backends.

The idea: every backend must satisfy the *same* numerical checks, so that a new backend is
"done" when it passes this suite rather than when someone eyeballs it.  The suite is
parametrized over the registered backends (see ``conftest.py``), so adding a backend to the
registry automatically adds it here.

This directory supersedes ``n3fit/tests/test_backend.py``, which was the same set of
numerical checks but hard-wired to the Keras backend.
"""
