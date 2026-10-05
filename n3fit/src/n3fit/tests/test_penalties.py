"""
Test the penalties for n3fit hyperopt
"""

from types import SimpleNamespace

from n3fit.hyper_optimization.penalties import integrability, patience, saturation
from n3fit.model_gen import ReplicaSettings, generate_pdf_model


def test_saturation():
    """Check that the saturation penalty runs and returns a float"""
    fake_fl = [
        {"fl": i, "largex": [0, 1], "smallx": [1, 2]}
        for i in ["u", "ubar", "d", "dbar", "c", "g", "s", "sbar"]
    ]
    rps = [ReplicaSettings(nodes=[8], activations=["linear"], seed=0)]
    pdf_model = generate_pdf_model(rps, flav_info=fake_fl, fitbasis="FLAVOUR")
    assert isinstance(saturation(pdf_model, 5)[0], float)


def test_patience():
    """Check that the patience penalty runs and returns a float"""
    # The attributes are the read side of ``n3fit.stopping.FitRecord`` (P4): the same four the
    # legacy ``Stopping`` object exposed, with the budget renamed ``total_steps`` -- the engine
    # counts optimizer steps, and ``run`` is handed ``steps``.
    fake_stopping = SimpleNamespace(
        e_best_chi2=1000, stopping_patience=500, total_steps=5000, vl_chi2=2.42
    )
    res = patience(stopping_object=fake_stopping, alpha=1e-4)
    assert isinstance(res, float)


def test_integrability_numbers():
    """Check that the integrability penalty runs and returns a float"""
    fake_fl = [
        {"fl": i, "largex": [0, 1], "smallx": [1, 2]}
        for i in ["u", "ubar", "d", "dbar", "c", "g", "s", "sbar"]
    ]
    rps = [ReplicaSettings(nodes=[8], activations=["linear"], seed=0)]
    pdf_model = generate_pdf_model(rps, flav_info=fake_fl, fitbasis="FLAVOUR")
    assert isinstance(integrability(pdf_model), float)
