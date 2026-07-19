"""
Assert-based self-check for lens_design.py.  Run directly: py test_lens_design.py
Mirrors test_physics.py: no framework, small ray counts, fails loudly.
Covers requirements 1,2,3,5 (Part A) and 6,7,8,9,10 (Part B).
"""
import numpy as np

from physics import trapz
from paper_figures import (
    N_DIA_RED, N_DIA_GREEN, LAM_RED, LAM_GREEN,
    exc_rate, escape_cone_quadrature, eta_per_emitter, CONFIGS,
)
from lens_design import (
    A_EXC, exc_beam_params, exc_intensity, exc_rate_ideal,
    make_ideal_channels, eta_ideal_channels,
)


def _plane_power(z_depth, gap, knobs, n_w=12.0, n_pts=6000):
    """Numerically integrate I(r) 2 pi r dr over a plane."""
    bp = exc_beam_params(z_depth, gap, knobs)
    r_max = n_w * (bp['w_used'] + 0.5 * bp['L_edof'] * bp['theta_dia_max'] + 1.0)
    r = np.linspace(0.0, r_max, n_pts)
    I = exc_intensity(r, z_depth, gap, knobs)
    return trapz(I * 2.0 * np.pi * r, r), bp['P_delivered']


# --------------------------- Part A ---------------------------
def test_energy_conservation_gaussian_every_plane():
    # Req. 1: plain Gaussian focus conserves P exactly at every depth plane.
    knobs = dict(w_focus=3.0, z_focus=85.0, L_edof=0.0)
    for z in (-40.0, -85.0, -120.0):
        got, P = _plane_power(z, 50.0, knobs)
        assert abs(got - P) / P < 1e-3, f"z={z}: {got} vs {P}"


def test_energy_conservation_edof_within_2pct():
    # Req. 1 (EDOF): core + pedestal must conserve the plane power to <2%.
    knobs = dict(w_focus=1.5, z_focus=85.0, L_edof=40.0)
    for z in (-70.0, -85.0, -100.0):
        got, P = _plane_power(z, 50.0, knobs)
        assert abs(got - P) / P < 0.02, f"z={z}: {got} vs {P}"


def test_aperture_bound_clamps_convergence_angle():
    # Req. 2: an over-tight requested waist cannot exceed the aperture-set
    # convergence angle; the returned beam's angle must respect the ceiling.
    knobs = dict(w_focus=0.05, z_focus=85.0, L_edof=0.0)
    bp = exc_beam_params(-85.0, 50.0, knobs)
    assert bp['clamped']
    theta_beam = (LAM_GREEN * 1e-3) / (np.pi * N_DIA_GREEN * bp['w_used'])
    assert theta_beam <= bp['theta_dia_max'] * (1.0 + 1e-6), (
        f"beam angle {theta_beam} > aperture max {bp['theta_dia_max']}")


def test_diffraction_bound_floor_on_waist():
    # Req. 3: used waist is never below the diffraction floor w_diff, and a
    # sub-floor request is clamped exactly up to it.
    knobs = dict(w_focus=1e-3, z_focus=85.0, L_edof=0.0)
    bp = exc_beam_params(-85.0, 50.0, knobs)
    assert bp['w_used'] >= bp['w_diff'] * (1.0 - 1e-9)
    assert abs(bp['w_used'] - bp['w_diff']) < 1e-9
    # A generous request is left untouched (no clamp).
    bp2 = exc_beam_params(-85.0, 50.0, dict(w_focus=5.0, z_focus=85.0, L_edof=0.0))
    assert not bp2['clamped'] and abs(bp2['w_used'] - 5.0) < 1e-12


def test_focus_beats_collimated_mcf_on_axis():
    # Req. 5: a focus at the layer beats the current collimated MCF excitation
    # in on-axis peak intensity (saturation) for the same delivered power.
    knobs = dict(w_focus=1.0, z_focus=85.0, L_edof=0.0)
    _, s_ideal = exc_rate_ideal(0.0, -85.0, 50.0, knobs)
    _, s_mcf = exc_rate("MCF", 0.0, 0.0, -85.0, 50.0)
    assert s_ideal > s_mcf, f"ideal s={s_ideal} !> MCF s={s_mcf}"


# --------------------------- Part B ---------------------------
def test_etendue_floor_on_waist():
    # Req. 6: waist * theta_acc >= lam/(pi n); a sub-floor request is clamped up.
    ch = make_ideal_channels(waist_at_layer=0.05, aim_depth=85.0, gap=50.0)[0]
    floor = (LAM_RED * 1e-3) / (np.pi * N_DIA_RED)
    assert ch['clamped']
    assert ch['waist'] * ch['theta_acc'] >= floor * (1.0 - 1e-9), (
        f"{ch['waist'] * ch['theta_acc']} < {floor}")


def test_aperture_ceiling_on_theta_acc():
    # Req. 7: theta_acc never exceeds the physical lens aperture half-angle.
    ch = make_ideal_channels(waist_at_layer=17.5, aim_depth=85.0, gap=50.0)[0]
    assert ch['theta_acc'] <= ch['theta_ap_dia'] * (1.0 + 1e-12)


def test_per_channel_efficiency_upper_bound():
    # Req. 8: a single channel's eta for an emitter AT the aim point cannot
    # exceed the lens solid-angle fraction (x1.5 margin).
    V0, W0 = escape_cone_quadrature(16384, N_DIA_RED, seed=7)
    channels = make_ideal_channels(waist_at_layer=17.5, aim_depth=85.0, gap=50.0)
    emitter = np.array([[0.0, 0.0, -85.0]])            # at the aim point
    eta = eta_ideal_channels(emitter, V0, W0, channels[:1], 50.0)[0]
    theta_ap = channels[0]['theta_ap_dia']
    bound = 1.5 * (np.pi * theta_ap ** 2) / (4.0 * np.pi)
    assert eta <= bound, f"per-channel eta {eta} > bound {bound}"


def test_ideal_channels_beat_as_built_mcf():
    # Req. 9 (the premise of the redesign): re-aimable ideal channels aimed at
    # the NV depth beat the as-built fixed-aim MCF at the same gap.  Reported
    # honestly -- if this fails the model is NOT fudged.
    V0, W0 = escape_cone_quadrature(32768, N_DIA_RED)
    emitter = np.array([[0.0, 0.0, -85.0]])
    channels = make_ideal_channels(waist_at_layer=17.5, aim_depth=85.0, gap=50.0)
    eta_ideal = eta_ideal_channels(emitter, V0, W0, channels, 50.0)[0]
    fibs = CONFIGS[2]['fibers'](50.0)                  # MCF fixed aim, as-built
    eta_asbuilt = eta_per_emitter(emitter, V0, W0, fibs, 50.0, CONFIGS[2]['model'])[0]
    print(f"    [req9] eta_ideal={eta_ideal:.4e}  eta_asbuilt={eta_asbuilt:.4e}"
          f"  ratio={eta_ideal / eta_asbuilt:.2f}")
    assert eta_ideal > eta_asbuilt, (
        f"ideal {eta_ideal:.4e} did NOT beat as-built {eta_asbuilt:.4e}")


def test_determinism_repeat_identical():
    # Req. 10: fixed seeds -> bit-identical repeats.
    channels = make_ideal_channels(waist_at_layer=17.5, aim_depth=85.0, gap=50.0)
    emitter = np.array([[0.0, 0.0, -85.0]])
    V0, W0 = escape_cone_quadrature(8192, N_DIA_RED, seed=3)
    a = eta_ideal_channels(emitter, V0, W0, channels, 50.0)
    V0b, W0b = escape_cone_quadrature(8192, N_DIA_RED, seed=3)
    b = eta_ideal_channels(emitter, V0b, W0b, channels, 50.0)
    assert np.array_equal(a, b)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} checks passed.")
