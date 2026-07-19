"""
lens_design.py -- idealized, physics-constrained replacement optics for the
NV-diamond MCF probe redesign study.  We do NOT model a fabricable lens; we
model its optical ACTION (a focus, a converging acceptance channel) subject to
the same hard physical limits a real lens obeys: energy conservation, the
aperture-set convergence angle, the diffraction floor, and the etendue floor.

Geometry (shared with physics.py / paper_figures.py): diamond surface at z=0,
NVs at negative z (depth d -> z=-d), lenses at positive z (gap g above surface).
All lengths in micrometres, angles in radians.

------------------------------------------------------------------ PART A
Idealized EXCITATION profile: a green focus placed at an arbitrary depth, with
an optional extended depth of focus (EDOF, axicon/Bessel-like).

  exc_rate_ideal(r, z_depth, gap, knobs) -> (rate photons/s into 4pi, max s)
  knobs = dict(w_focus [um], z_focus [um depth], L_edof [um, 0 = plain focus]).

Delivered power (stated):  P_delivered = P_GREEN * T_GREEN_IN * T_LENS_EXC,
with T_LENS_EXC = 1.0 (idealized AR-coated excitation lens).  T_GREEN_IN is the
green Fresnel loss entering the diamond, kept from paper_figures.

Excitation aperture (stated):  A_EXC = 35 um effective radius (the central
excitation pupil; the STL fit MCF_CENTRAL_R=74.91 is the outer cap, the clear
converging pupil is ~half that -- we take 35 um, the paper's side pitch).

Air-side standoff to the focus (the hint's "gap + 35.2 + n-scaled depth"):
  L_air = gap + MCF_SIDE_TIP_OFFSET + z_focus / N_DIA_GREEN.

EDOF power-fraction formula (documented, requirement 4):
  A plain Gaussian focus holds its w_focus core over ~2*z_R (twice the Rayleigh
  range z_R = pi n w_focus^2 / lambda).  An axicon spreads the SAME aperture
  power over a chosen axial line L_edof.  Conserving the aperture power over a
  longer line means the central lobe keeps only the fraction of the axial
  budget a plain focus would have occupied:
        P_frac = min(1, 2 * z_R / L_edof).
  The complementary power (1 - P_frac) is placed in a wide pedestal ring
  (Gaussian of waist w_ped = w_core + (L_edof/2)*theta_dia_max), so the plane
  integral conserves BY CONSTRUCTION -- core + pedestal = P at every plane.
  (L_edof=0 -> plain Gaussian focus, P_frac=1, no pedestal.)

------------------------------------------------------------------ PART B
Idealized focused COLLECTION channels.  The facet-plane Gaussian overlap in
physics.compute_coupling cannot represent a converging channel (it has no
position-angle correlation), so acceptance is evaluated in the DIAMOND-side
conjugate space, per ray, BEFORE refraction:

  make_ideal_channels(waist_at_layer, aim_depth, gap, ...) -> [channel dicts]
  eta_ideal_channels(emitters, V0, W0, channels, gap, ...) -> per-emitter eta
                                                              (summed over channels)

Each channel aims at the SAME on-axis point A = (0,0,-aim_depth).  Its chief
direction u (in diamond, pointing from A toward the lens) is the exact Snell
solve reused from paper_figures.mcf_fibers' re-aim.  Acceptance per ray:
  spatial  = exp(-2 d^2 / w^2),  d = closest approach of ray line to A
  angular  = exp(-2 sin^2(theta) / theta_acc^2),  theta = angle(V0, u)
  times the ray's Fresnel/TIR weight (0 for TIR) from run_ray_tracing.
  eta = 0.5 * sum(weights * acceptance) / n_rays  (mirrors eta_per_emitter;
  works with escape_cone_quadrature because the cone factor is inside W0).

theta_acc aperture approximation (documented, requirement 7): the lens of clear
radius a_lens sits a distance standoff = gap + MCF_SIDE_TIP_OFFSET from the
surface, so it subtends a half-angle a_lens/standoff in air; Snell-compressed
into the diamond (small angle) that is (a_lens/standoff)/n_dia.  We set
theta_acc to exactly this ceiling.
"""
import numpy as np

from physics import run_ray_tracing, MODE_OVERLAP_MODEL
from paper_figures import (
    N_MED, LAM_RED, LAM_GREEN, N_DIA_RED, N_DIA_GREEN,
    P_GREEN_MW, I_SAT, R_SAT, T_GREEN_IN,
    LENS_R, W0_LENS, MCF_SIDE_TIP_OFFSET,
    bisect, escape_cone_quadrature,
)

# Stated idealized-optics choices (see module docstring).
A_EXC = 35.0        # um, effective excitation pupil radius
T_LENS_EXC = 1.0    # idealized AR-coated excitation lens transmission


# ============================== PART A ==============================
def exc_beam_params(z_depth, gap, knobs):
    """Resolve knobs into an honest, physically-clamped beam at plane z_depth.

    Returns a dict exposing the requested vs. used waist and the two limits
    (aperture-set convergence angle, diffraction floor) so the clamp is
    inspectable -- 'design honesty', requirements 2 & 3.
    """
    w_req = float(knobs['w_focus'])
    z_focus = float(knobs['z_focus'])
    L_edof = float(knobs.get('L_edof', 0.0))
    lam = LAM_GREEN * 1e-3                       # um, in vacuum
    n = N_DIA_GREEN

    # Aperture-set maximum convergence half-angle (req. 2).
    L_air = gap + MCF_SIDE_TIP_OFFSET + z_focus / n
    theta_air_max = A_EXC / L_air
    theta_dia_max = theta_air_max / n            # small-angle Snell into diamond

    # Diffraction floor on the waist for that angle (req. 3).
    w_diff = lam / (np.pi * n * theta_dia_max)
    w_used = max(w_req, w_diff)                  # clamp UP, never grant a tighter focus
    clamped = w_used > w_req * (1.0 + 1e-12)

    z_R = np.pi * n * w_used * w_used / lam       # Rayleigh range in diamond
    P_delivered = P_GREEN_MW * T_GREEN_IN * T_LENS_EXC
    return dict(w_focus_req=w_req, w_used=w_used, w_diff=w_diff,
                theta_air_max=theta_air_max, theta_dia_max=theta_dia_max,
                z_R=z_R, z_focus=z_focus, L_edof=L_edof,
                P_delivered=P_delivered, clamped=clamped)


def exc_intensity(r, z_depth, gap, knobs):
    """Green intensity I(r) [mW/um^2] at radius r on plane z_depth (depth, um).

    Conserves the plane power P_delivered exactly for the Gaussian case, and by
    construction (core + pedestal) inside the EDOF slab.
    """
    bp = exc_beam_params(z_depth, gap, knobs)
    r2 = np.asarray(r, dtype=float) ** 2
    P, w0, zR, zf, L = (bp['P_delivered'], bp['w_used'], bp['z_R'],
                        bp['z_focus'], bp['L_edof'])

    def gauss(w):
        return 2.0 * P / (np.pi * w * w) * np.exp(-2.0 * r2 / (w * w))

    dz = abs(z_depth - zf)
    if L > 0.0 and dz <= 0.5 * L:                 # inside the EDOF line focus
        Pf = min(1.0, 2.0 * zR / L)
        w_ped = w0 + 0.5 * L * bp['theta_dia_max']
        return Pf * gauss(w0) + (1.0 - Pf) * gauss(w_ped)
    dz_eff = dz - 0.5 * L if L > 0.0 else dz      # plain / defocused Gaussian
    w = w0 * np.sqrt(1.0 + (dz_eff / zR) ** 2)
    return gauss(w)


def exc_rate_ideal(r, z_depth, gap, knobs):
    """Per-NV emission rate (photons/s into 4pi) and peak saturation s.

    s = I / I_SAT, R = R_SAT * s/(1+s) -- identical photophysics to
    paper_figures.exc_rate.  Returns (rate, float(max s)).
    """
    I = exc_intensity(r, z_depth, gap, knobs)
    s = I / I_SAT
    return R_SAT * s / (1.0 + s), float(np.max(s))


# ============================== PART B ==============================
def make_ideal_channels(waist_at_layer, aim_depth, gap, n_ch=6,
                        lens_r=LENS_R, a_lens=W0_LENS,
                        n_dia=N_DIA_RED, lam_nm=LAM_RED):
    """Build n_ch converging collection channels, all aimed at (0,0,-aim_depth).

    theta_acc is the aperture ceiling (a_lens/standoff, Snell-compressed);
    the waist is clamped UP to the etendue floor lam/(pi n theta_acc) if the
    request violates it (requirements 6 & 7).  The clamped values are exposed
    on each channel dict.
    """
    standoff = gap + MCF_SIDE_TIP_OFFSET
    lam = lam_nm * 1e-3

    # Aperture-limited angular half-width (req. 7) and etendue floor (req. 6).
    theta_ap_dia = (a_lens / standoff) / n_dia
    theta_acc = theta_ap_dia
    w_floor = lam / (np.pi * n_dia * theta_acc)
    w_used = max(float(waist_at_layer), w_floor)
    clamped = w_used > float(waist_at_layer) * (1.0 + 1e-12)

    crit = np.arcsin(1.0 / n_dia)
    aim = np.array([0.0, 0.0, -float(aim_depth)])
    channels = []
    for k in range(n_ch):
        a = k * 2.0 * np.pi / n_ch
        # Exact Snell aim of the chief ray from A to the lens centre.
        f = lambda td: (aim_depth * np.tan(td)
                        + standoff * np.tan(np.arcsin(n_dia * np.sin(td))) - lens_r)
        theta_d = bisect(f, 1e-9, crit - 1e-6)
        u = np.array([np.cos(a) * np.sin(theta_d),
                      np.sin(a) * np.sin(theta_d), np.cos(theta_d)])
        channels.append(dict(
            center=np.array([lens_r * np.cos(a), lens_r * np.sin(a), standoff]),
            aim=aim, u=u, theta_d=theta_d, theta_acc=theta_acc,
            theta_ap_dia=theta_ap_dia, waist=w_used, waist_req=float(waist_at_layer),
            clamped=clamped))
    return channels


def eta_ideal_channels(emitters, V0, W0, channels, gap,
                       lam_nm=LAM_RED, n_dia=N_DIA_RED):
    """Per-emitter collection efficiency, summed over the converging channels.

    Acceptance is computed in-diamond on the unit ray lines (emitter, V0) before
    refraction; the ray's Fresnel/TIR weight comes from run_ray_tracing (0 for
    TIR).  emitters (n,3), V0 (m,3) unit, W0 (m,).
    """
    emitters = np.atleast_2d(np.asarray(emitters, dtype=float))
    V0 = np.asarray(V0, dtype=float)
    raw = run_ray_tracing(emitters, V0, W0, n_dia, N_MED, gap, [],
                          MODE_OVERLAP_MODEL, lam_nm)
    W = raw['weights']                                   # (n, m), 0 for TIR
    n_r = len(V0)

    total = np.zeros(len(emitters))
    for ch in channels:
        AP = ch['aim'][None, :] - emitters               # (n,3)
        ap2 = np.sum(AP * AP, axis=1)                    # (n,)
        proj = AP @ V0.T                                 # (n,m); V0 unit
        d2 = np.maximum(ap2[:, None] - proj * proj, 0.0)  # closest-approach^2
        spatial = np.exp(-2.0 * d2 / (ch['waist'] ** 2))
        sin2 = np.maximum(1.0 - (V0 @ ch['u']) ** 2, 0.0)  # (m,)
        angular = np.exp(-2.0 * sin2 / (ch['theta_acc'] ** 2))
        accept = spatial * angular[None, :]
        total += 0.5 * np.sum(W * accept, axis=1) / n_r
    return total
