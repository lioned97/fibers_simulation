"""
Minimal assert-based self-check for physics.py. Run directly: py test_physics.py
Not a framework suite — just enough to fail loudly if the ray-tracing math regresses.
"""
import numpy as np
from physics import (
    diamond_sellmeier, compute_coupling, run_ray_tracing, sample_ray_directions,
    angle_to_boresight, fiber_mode_params, lens_dome_mesh, GEOMETRIC_MODEL, MODE_OVERLAP_MODEL,
)


def test_diamond_sellmeier_known_points():
    assert abs(diamond_sellmeier(0.589) - 2.4173) < 1e-3
    assert abs(diamond_sellmeier(0.637) - 2.4118) < 1e-3


def test_normal_incidence_fresnel_matches_known_828pct():
    # Single straight-down ray, on-axis fiber directly below -> normal incidence.
    emitters = np.array([[0.0, 0.0, -10.0]])
    V0 = np.array([[0.0, 0.0, 1.0]])
    W0 = np.array([1.0])
    fibers = [{'x': 0.0, 'y': 0.0, 'd_core': 1000.0, 'na': 1.0}]  # huge core+NA: accept everything
    res = run_ray_tracing(emitters, V0, W0, n_dia=2.417, n_med=1.0, z_fiber=5.0,
                          fibers=fibers, coupling_model=GEOMETRIC_MODEL)
    # eff = 0.5 * T (the 0.5 accounts for the upper-hemisphere-only sampling convention)
    T = res['fiber_stats'][0]['avg_efficiency'] / 0.5
    assert abs(T - 0.828) < 0.01, f"expected T~=0.828 at normal incidence, got {T}"


def test_tir_beyond_critical_angle():
    # ~30 deg in diamond exceeds the ~24.4 deg diamond-air critical angle -> TIR.
    theta = np.radians(30.0)
    V0 = np.array([[np.sin(theta), 0.0, np.cos(theta)]])
    W0 = np.array([1.0])
    emitters = np.array([[0.0, 0.0, -10.0]])
    fibers = [{'x': 0.0, 'y': 0.0, 'd_core': 1000.0, 'na': 1.0}]
    res = run_ray_tracing(emitters, V0, W0, n_dia=2.417, n_med=1.0, z_fiber=5.0, fibers=fibers)
    assert bool(res['tir_mask'][0, 0]) is True
    assert res['fiber_stats'][0]['avg_efficiency'] == 0.0


def test_angle_to_boresight_matches_global_z_case():
    # boresight=(0,0,1) must exactly reproduce the plain global-Z angle test.
    rng = np.random.default_rng(0)
    V1 = rng.normal(size=(50, 3))
    V1 /= np.linalg.norm(V1, axis=-1, keepdims=True)
    sin2_global = 1.0 - V1[:, 2] ** 2
    sin2_boresight = angle_to_boresight(V1, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(sin2_global, sin2_boresight)


def test_angle_to_boresight_zero_for_aligned_ray():
    v = np.array([[0.6, 0.8, 0.0]])  # arbitrary unit vector
    assert angle_to_boresight(v, np.array([0.6, 0.8, 0.0]))[0] < 1e-12
    assert abs(angle_to_boresight(v, np.array([-0.8, 0.6, 0.0]))[0] - 1.0) < 1e-12  # perpendicular


def test_tilted_channel_favors_rays_aimed_at_it_over_straight_ones():
    # A ray tilted 5 degrees off Z should couple better into a channel whose
    # boresight is tilted the same 5 degrees than into a straight-up (Z) one.
    theta = np.radians(5.0)
    v1 = np.array([[np.sin(theta), 0.0, np.cos(theta)]])
    tir = np.array([[False]])
    w = np.array([[1.0]])
    dist_sq = np.array([[0.0]])
    w0, na_mode = 5.0, 0.05

    sin2_tilted = angle_to_boresight(v1, np.array([np.sin(theta), 0.0, np.cos(theta)]))
    sin2_straight = angle_to_boresight(v1, np.array([0.0, 0.0, 1.0]))

    c_tilted = compute_coupling(dist_sq, sin2_tilted, tir, 0.0, 0.0, 1.0, MODE_OVERLAP_MODEL, w0, na_mode)
    c_straight = compute_coupling(dist_sq, sin2_straight, tir, 0.0, 0.0, 1.0, MODE_OVERLAP_MODEL, w0, na_mode)
    assert c_tilted[0, 0] > c_straight[0, 0]


def test_run_ray_tracing_accepts_empty_fiber_list():
    emitters = np.array([[0.0, 0.0, -10.0]])
    V0, W0 = sample_ray_directions(200, "Isotropic", "Isotropic")
    res = run_ray_tracing(emitters, V0, W0, n_dia=2.417, n_med=1.0, z_fiber=50.0, fibers=[])
    assert res['fiber_stats'] == []
    assert res['X_f'].shape == (1, 200)


def test_run_ray_tracing_boresight_fiber_matches_manual_calc():
    # A fiber dict with 'boresight' should give the exact same result as
    # manually computing angle_to_boresight + compute_coupling outside.
    emitters = np.array([[0.0, 0.0, -80.0]])
    V0, W0 = sample_ray_directions(2000, "NV Symmetry Axis", "Ensemble (4-axis average)")
    boresight = np.array([0.3, 0.1, 0.95])
    boresight = boresight / np.linalg.norm(boresight)
    w0, lam_nm, n_med = 17.5, 700.0, 1.0

    fibers = [{'x': 5.0, 'y': -2.0, 'd_core': 35.0, 'na': 0.0, 'boresight': boresight, 'w0': w0}]
    res = run_ray_tracing(emitters, V0, W0, n_dia=2.41, n_med=n_med, z_fiber=250.0,
                          fibers=fibers, coupling_model=MODE_OVERLAP_MODEL, lambda_nm=lam_nm)
    via_dict = res['fiber_stats'][0]['avg_efficiency']

    # Manual reference calculation using the bare building blocks.
    res0 = run_ray_tracing(emitters, V0, W0, n_dia=2.41, n_med=n_med, z_fiber=250.0,
                           fibers=[], coupling_model=MODE_OVERLAP_MODEL, lambda_nm=lam_nm)
    V1, Xf, Yf = res0['V1'][0], res0['X_f'][0], res0['Y_f'][0]
    tir, W = res0['tir_mask'][0], res0['weights'][0]
    na_mode = (lam_nm / 1000.0) / (np.pi * n_med * w0)
    sin2_local = angle_to_boresight(V1, boresight)
    dist_sq = (Xf - 5.0) ** 2 + (Yf - (-2.0)) ** 2
    coupling = compute_coupling(dist_sq, sin2_local, tir, 0.0, 0.0, n_med, MODE_OVERLAP_MODEL, w0, na_mode)
    manual = 0.5 * np.sum(W * coupling) / len(V0)

    assert abs(via_dict - manual) < 1e-15


def test_run_ray_tracing_w0_override_scales_na_mode_with_wavelength():
    # na_mode = lambda / (pi * n_med * w0): doubling lambda must double na_mode,
    # so the override can't silently bake in a single wavelength's value.
    emitters = np.array([[0.0, 0.0, -50.0]])
    V0, W0 = sample_ray_directions(500, "Isotropic", "Isotropic")
    fibers = [{'x': 0.0, 'y': 0.0, 'd_core': 35.0, 'na': 0.0, 'w0': 17.5}]

    eff_1x = run_ray_tracing(emitters, V0, W0, 2.41, 1.0, 250.0, fibers,
                             MODE_OVERLAP_MODEL, lambda_nm=650.0)['fiber_stats'][0]['avg_efficiency']
    eff_2x = run_ray_tracing(emitters, V0, W0, 2.41, 1.0, 250.0, fibers,
                             MODE_OVERLAP_MODEL, lambda_nm=1300.0)['fiber_stats'][0]['avg_efficiency']
    # A larger na_mode (longer lambda) accepts a wider angular range -> higher or equal efficiency.
    assert eff_2x >= eff_1x


def test_run_ray_tracing_mode_overlap_without_override_matches_fiber_mode_params():
    # No 'w0' key -> must fall back to deriving w0/na_mode from fiber_mode_params,
    # exactly as it did before boresight/w0 overrides existed (no regression
    # for every existing SM/MM fiber dict, which never sets 'w0').
    emitters = np.array([[0.0, 0.0, -30.0]])
    V0, W0 = sample_ray_directions(300, "Isotropic", "Isotropic")
    d_core, na, lam_nm, n_med, z_fiber = 4.0, 0.12, 700.0, 1.0, 5.0
    fibers = [{'x': 0.0, 'y': 0.0, 'd_core': d_core, 'na': na}]
    res = run_ray_tracing(emitters, V0, W0, 2.41, n_med, z_fiber, fibers, MODE_OVERLAP_MODEL, lam_nm)
    via_fibers_list = res['fiber_stats'][0]['avg_efficiency']

    _, _, w0_expected, na_mode_expected = fiber_mode_params(d_core, na, lam_nm, n_med)
    res0 = run_ray_tracing(emitters, V0, W0, 2.41, n_med, z_fiber, fibers=[],
                           coupling_model=MODE_OVERLAP_MODEL, lambda_nm=lam_nm)
    tir, W = res0['tir_mask'][0], res0['weights'][0]
    dist_sq = (res0['X_f'][0] - 0.0) ** 2 + (res0['Y_f'][0] - 0.0) ** 2
    sin2_global = 1.0 - res0['V1'][0][:, 2] ** 2
    coupling = compute_coupling(dist_sq, sin2_global, tir, 0.0, 0.0, n_med,
                                MODE_OVERLAP_MODEL, w0_expected, na_mode_expected)
    manual = 0.5 * np.sum(W * coupling) / len(V0)

    assert abs(via_fibers_list - manual) < 1e-15


def test_lens_dome_mesh_apex_follows_boresight():
    # The dome's apex (theta=0 in its own local frame) must land at
    # center + height*boresight, for ANY tilt direction -- that's the whole
    # point of rotating the dome to align with the boresight.
    center = np.array([5.0, -3.0, 250.0])
    radius, height = 17.5, 6.0
    for boresight in [np.array([0.0, 0.0, 1.0]),
                      np.array([0.3, 0.1, 0.95]) / np.linalg.norm([0.3, 0.1, 0.95]),
                      np.array([-0.4, 0.2, 0.9]) / np.linalg.norm([-0.4, 0.2, 0.9])]:
        X, Y, Z = lens_dome_mesh(center, boresight, radius, height)
        # theta=0 is the first column (n_theta axis) for every phi row -> all identical, the apex.
        apex = np.array([X[0, 0], Y[0, 0], Z[0, 0]])
        expected = center + height * boresight
        assert np.allclose(apex, expected, atol=1e-9), f"{apex} != {expected}"


def test_tir_rays_with_tilted_boresight_give_zero_coupling_not_nan():
    # TIR rays are stored with a NON-unit V1 (norm = n*sin(theta1) > 1); against
    # a strongly tilted boresight, 1-cos^2 went negative -> exp overflow -> NaN.
    # Regression: a steep ray past the critical angle + a 28deg-tilted lensed
    # channel must yield exactly 0 coupling, never NaN/inf.
    theta = np.radians(40.0)                       # well past ~24.4deg critical
    V0 = np.array([[np.sin(theta), 0.0, np.cos(theta)]])
    W0 = np.array([1.0])
    emitters = np.array([[0.0, 0.0, -85.0]])
    b = np.array([np.sin(np.radians(28.0)), 0.0, np.cos(np.radians(28.0))])
    fibers = [{'x': 17.0, 'y': 0.0, 'd_core': 35.0, 'na': 0.0, 'boresight': b, 'w0': 17.5}]
    res = run_ray_tracing(emitters, V0, W0, 2.406, 1.0, 0.0, fibers,
                          MODE_OVERLAP_MODEL, 700.0)
    eff = res['fiber_stats'][0]['avg_efficiency']
    assert np.isfinite(eff) and eff == 0.0, f"expected 0, got {eff}"


def test_lens_dome_mesh_bulges_opposite_the_given_direction_argument():
    # Regression lock for the app.py calling convention: 'boresight' (as used
    # everywhere else in this codebase) points AWAY from the diamond, toward
    # the fiber side. The printed lens physically bulges the other way, so
    # callers must pass -boresight to lens_dome_mesh. Encode that: the apex
    # must land on the opposite side of the base plane from +boresight.
    center = np.array([17.0, 0.0, 250.0])
    boresight_away_from_diamond = np.array([0.057, 0.0, 0.998])  # typical MCF value, mostly +Z
    X, Y, Z = lens_dome_mesh(center, -boresight_away_from_diamond, radius=17.5, height=6.0)
    apex_z = Z[0, 0]
    assert apex_z < center[2], (
        f"apex_z={apex_z} should be below the base plane ({center[2]}) -- toward the diamond, "
        "not toward the fiber. If this fails, check the sign passed to lens_dome_mesh in app.py."
    )


def test_lens_dome_mesh_handles_antiparallel_boresight():
    # boresight = -Z is the degenerate case the cross-product rotation can't
    # handle directly; must not crash and must still point the apex downward.
    center = np.array([0.0, 0.0, 0.0])
    X, Y, Z = lens_dome_mesh(center, np.array([0.0, 0.0, -1.0]), radius=10.0, height=3.0)
    assert np.isfinite(X).all() and np.isfinite(Y).all() and np.isfinite(Z).all()
    assert abs(Z[0, 0] - (-3.0)) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} checks passed.")
