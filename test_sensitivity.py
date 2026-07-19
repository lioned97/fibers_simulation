"""
Minimal assert-based self-check for sensitivity.py. Run directly: py test_sensitivity.py
Small ray counts (~2000) so the whole suite finishes in well under two minutes.
Not a framework suite -- just enough to fail loudly if the FoM math regresses.
"""
import numpy as np

from sensitivity import sensitivity_fom, calibrate_b_auto, MM_CFG, C0, DNU0

RAYS = 2000                      # fast; the FoM shapes converge long before nen
GAP = 30.0                       # a representative, non-degenerate MM gap


def test_mm_sanity_all_depths_positive_and_contrast_bounded():
    r = sensitivity_fom(MM_CFG, GAP, num_rays=RAYS)          # b_auto=None -> calibrated
    assert np.all(np.isfinite(r['psi'])) and np.all(r['psi'] > 0.0)
    assert np.isfinite(r['psi_robust']) and r['psi_robust'] > 0.0
    assert np.all(r['S'] > 0.0)
    assert np.all((r['C_eff'] > 0.0) & (r['C_eff'] <= C0))


def test_more_background_worsens_sensitivity():
    b = calibrate_b_auto(MM_CFG, num_rays=RAYS)
    lo = sensitivity_fom(MM_CFG, GAP, b_auto=b, num_rays=RAYS)['psi']
    hi = sensitivity_fom(MM_CFG, GAP, b_auto=2.0 * b, num_rays=RAYS)['psi']
    assert np.all(hi > lo), "doubling b_auto must strictly raise Psi (worse)"


def test_photon_limited_limit_is_exact():
    r = sensitivity_fom(MM_CFG, GAP, b_auto=0.0, b0=0.0, num_rays=RAYS)
    assert np.all(r['B'] == 0.0)
    assert np.all(r['C_eff'] == C0), "no background -> C_eff must equal C0 exactly"
    assert np.allclose(r['psi'], DNU0 / (C0 * np.sqrt(r['sbar'])), rtol=0, atol=0)


def test_gradient_broadening_worsens_sensitivity():
    base = sensitivity_fom(MM_CFG, GAP, b_auto=0.0, k_grad=0.0, num_rays=RAYS)
    grad = sensitivity_fom(MM_CFG, GAP, b_auto=0.0, k_grad=1.0e4, num_rays=RAYS)
    assert np.all(grad['fwhm_sig'] > 0.0)
    assert grad['psi_robust'] > base['psi_robust']
    assert np.all(grad['psi'] > base['psi'])


def test_deterministic_repeat():
    a = sensitivity_fom(MM_CFG, GAP, b_auto=0.0, num_rays=RAYS)['psi']
    b = sensitivity_fom(MM_CFG, GAP, b_auto=0.0, num_rays=RAYS)['psi']
    assert np.array_equal(a, b), "fixed seeds -> identical Psi on repeat"


if __name__ == "__main__":
    import time
    t0 = time.time()
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} checks passed in {time.time() - t0:.0f} s.")
