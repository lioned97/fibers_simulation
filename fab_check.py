"""fab_check.py -- fabrication-reality pass on the idealized MCF-lens winner.

Stress-tests the Phase-3 idealized optimum against five literature-defensible
two-photon-polymerization (2PP / IP-S) limits C1-C5 (justified in FAB_CHECK.md):
we clamp the winner's knobs to what a printed lens can actually deliver,
rescore with phase3_optimize.ideal_fom at 16384 rays (shared calibrated
b_auto), and report vs the MM / SM baselines.

The idealized winner (with-bg Psi_robust = 389.8 @ 16384 rays):
  gap = 0 (tip in contact), 6 side channels aimed at (0,0,-85 um) waist 4 um,
  excitation w_focus = 1 um @ z_focus = 85 um, L_edof = 20 um.

numpy + stdlib only.  Runtime ~10-15 min (calibration + 3 scenarios x 2 FoM
calls at 16384 rays).  Run:  python fab_check.py
"""
import time

import numpy as np

from lens_design import exc_beam_params, make_ideal_channels, A_EXC
from phase3_optimize import ideal_fom, PSI_MM_BG, PSI_SM_BG
from sensitivity import calibrate_b_auto
from paper_figures import (
    MCF_SIDE_TIP_OFFSET, N_DIA_GREEN, N_DIA_RED, LAM_GREEN, LAM_RED,
)

# ----------------------------- constraints -----------------------------
NA_ANGLE_CAP   = 0.6      # C1: max printable air-side half-angle (rad); sin~0.56 NA
SURFACE_DERATE = 1.2      # C5: printed waist >= 1.2 x diffraction/etendue floor (~l/10 form error)
NUM_RAYS       = 16384    # study's high-fidelity ray count
PSI_IDEAL      = 389.8    # Phase-3 idealized winner, with-background, 16384 rays

# ----------------------------- winner knobs ----------------------------
WIN_WFOCUS, WIN_ZFOCUS, WIN_LEDOF = 1.0, 85.0, 20.0   # excitation
WIN_WAIST,  WIN_AIM               = 4.0, 85.0          # collection

# C2 standoff scenarios (name, as-built lens->diamond standoff in um)
SCENARIOS = [("scaffold", 250.0), ("short scaffold", 50.0), ("contact", 0.0)]

LAM_G = LAM_GREEN * 1e-3   # um
LAM_R = LAM_RED   * 1e-3   # um


def clamp_excitation(gap):
    """Fab-clamp the excitation focus: C1 caps the air-side convergence angle,
    C5 derates the diffraction floor by 1.2x.  Returns achieved w_focus + the
    ideal diffraction floor (for the sanity assert) and the C1 angle."""
    knobs0 = dict(w_focus=WIN_WFOCUS, z_focus=WIN_ZFOCUS, L_edof=WIN_LEDOF)
    bp = exc_beam_params(WIN_ZFOCUS, gap, knobs0)          # ideal floors + geometry
    theta_air = min(bp['theta_air_max'], NA_ANGLE_CAP)     # C1
    theta_dia = theta_air / N_DIA_GREEN
    w_diff_cap = LAM_G / (np.pi * N_DIA_GREEN * theta_dia)  # diffraction floor at capped angle
    w_ach = max(WIN_WFOCUS, SURFACE_DERATE * w_diff_cap)    # C5 + request
    return dict(w_ach=w_ach, w_diff_ideal=bp['w_diff'],
                theta_air=theta_air, theta_air_max=bp['theta_air_max'])


def clamp_collection(gap):
    """Fab-clamp the side-channel waist: same C1 cap (air-side) + C5 derate on
    the etendue floor.  theta_acc geometry comes from make_ideal_channels."""
    ch = make_ideal_channels(WIN_WAIST, WIN_AIM, gap)[0]
    theta_air_acc = ch['theta_ap_dia'] * N_DIA_RED          # air-side = a_lens/standoff
    theta_air = min(theta_air_acc, NA_ANGLE_CAP)            # C1
    theta_acc = theta_air / N_DIA_RED
    w_floor_cap = LAM_R / (np.pi * N_DIA_RED * theta_acc)   # etendue floor at capped angle
    w_floor_ideal = LAM_R / (np.pi * N_DIA_RED * ch['theta_acc'])
    w_ach = max(WIN_WAIST, SURFACE_DERATE * w_floor_cap)    # C5 + request
    return dict(w_ach=w_ach, w_floor_ideal=w_floor_ideal,
                theta_air=theta_air, theta_air_max=theta_air_acc)


def axicon_budget(w_ach_exc, theta_air_exc):
    """C3: an axicon extends the focus beyond its natural Gaussian depth-of-focus
    2*z_R up to L_edof.  Extra air-side cone angle is zero while 2*z_R >= L_edof
    (the Gaussian already covers the line); otherwise a full-pupil axicon making
    the residual line length L_resid needs cone ~ arctan(A_EXC/(L_resid_air)).
    Total air-side angle (focus + axicon) must stay under C1's 0.6 rad."""
    z_R = np.pi * N_DIA_GREEN * w_ach_exc * w_ach_exc / LAM_G
    natural = 2.0 * z_R
    if WIN_LEDOF <= natural:
        beta_air = 0.0                                     # no axicon required
    else:
        L_resid_air = (WIN_LEDOF - natural) / N_DIA_GREEN  # diamond line -> air length
        beta_air = np.arctan(A_EXC / L_resid_air)          # full-pupil axicon cone
    total = theta_air_exc + beta_air
    return dict(z_R=z_R, beta_air=beta_air, total=total, ok=total <= NA_ANGLE_CAP)


def main():
    t0 = time.time()
    b_auto = calibrate_b_auto(num_rays=NUM_RAYS)
    print(f"shared b_auto = {b_auto:.4g}  (same pipeline as Phase 1)\n")

    rows = []
    for name, standoff in SCENARIOS:
        # req 1: lens_design adds MCF_SIDE_TIP_OFFSET internally; pass gap =
        # standoff - offset (clamped >= 0).  "contact" gap=0 => the ~35.2 um tip
        # offset IS the true standoff, so the reachable minimum is not 0.
        gap = max(0.0, standoff - MCF_SIDE_TIP_OFFSET)
        eff_standoff = gap + MCF_SIDE_TIP_OFFSET

        ex = clamp_excitation(gap)
        co = clamp_collection(gap)
        ax = axicon_budget(ex['w_ach'], ex['theta_air'])

        # (a) never grant a tighter focus than physics: clamped >= ideal floor.
        assert ex['w_ach'] >= ex['w_diff_ideal'] - 1e-9, "exc waist below ideal floor"
        assert co['w_ach'] >= co['w_floor_ideal'] - 1e-9, "coll waist below ideal floor"

        exc_knobs = dict(w_focus=ex['w_ach'], z_focus=WIN_ZFOCUS, L_edof=WIN_LEDOF)
        f_bg  = ideal_fom(gap, exc_knobs, WIN_AIM, co['w_ach'], b_auto, NUM_RAYS)
        f_ph  = ideal_fom(gap, exc_knobs, WIN_AIM, co['w_ach'], 0.0,    NUM_RAYS)

        row = dict(name=name, eff=eff_standoff, gap=gap,
                   w_exc=ex['w_ach'], w_coll=co['w_ach'],
                   th_exc=ex['theta_air'], th_coll=co['theta_air'],
                   th_exc_max=ex['theta_air_max'], th_coll_max=co['theta_air_max'],
                   ax=ax, psi_bg=f_bg['psi_robust'], psi_ph=f_ph['psi_robust'])
        rows.append(row)
        print(f"[{name}] eff standoff {eff_standoff:6.1f} um  "
              f"w_exc {ex['w_ach']:.3f} (floor {ex['w_diff_ideal']:.3f})  "
              f"w_coll {co['w_ach']:.3f} (floor {co['w_floor_ideal']:.3f})  "
              f"Psi_bg {f_bg['psi_robust']:.4g}  [{time.time()-t0:.0f}s]",
              flush=True)
        # C3 report
        print(f"         C3 axicon: z_R={ax['z_R']:.1f} 2z_R={2*ax['z_R']:.1f} "
              f">= L_edof={WIN_LEDOF:.0f}? extra cone={ax['beta_air']:.3f} rad, "
              f"total air-angle={ax['total']:.3f} <= {NA_ANGLE_CAP} -> {ax['ok']}",
              flush=True)

    # (b) fab cannot beat ideal at the matching gap=0 (contact) scenario.
    contact = next(r for r in rows if r['gap'] == 0.0)
    assert contact['psi_bg'] >= PSI_IDEAL * 0.99, \
        f"fab contact Psi {contact['psi_bg']:.4g} beats ideal {PSI_IDEAL} (impossible)"

    # ----------------------------- results table -----------------------------
    print("\n" + "=" * 118)
    hdr = (f"{'scenario':<15} {'eff standoff':>12} {'w_exc':>7} {'w_coll':>7} "
           f"{'th_exc/th_coll (cap0.6)':>24} {'Psi_bg':>10} {'vs MM':>7} {'vs SM':>7}")
    print(hdr)
    print("-" * 118)
    print(f"{'IDEALIZED':<15} {'35.2 (g=0)':>12} {WIN_WFOCUS:>7.2f} {WIN_WAIST:>7.2f} "
          f"{'—':>24} {PSI_IDEAL:>10.4g} {PSI_MM_BG/PSI_IDEAL:>6.1f}x {PSI_SM_BG/PSI_IDEAL:>6.1f}x")
    for r in rows:
        th = f"{r['th_exc']:.3f}/{r['th_coll']:.3f}"
        print(f"{r['name']:<15} {r['eff']:>12.1f} {r['w_exc']:>7.2f} {r['w_coll']:>7.2f} "
              f"{th:>24} {r['psi_bg']:>10.4g} "
              f"{PSI_MM_BG/r['psi_bg']:>6.1f}x {PSI_SM_BG/r['psi_bg']:>6.1f}x")
    print("=" * 118)
    print("photon-limited (b_auto=0) Psi_robust for decomposition:")
    for r in rows:
        print(f"  {r['name']:<15} Psi_ph = {r['psi_ph']:.4g}")
    print(f"\nreferences: MM {PSI_MM_BG:.4g}, SM {PSI_SM_BG:.4g}  (>1x = fab MCF better)")
    print(f"[total {time.time()-t0:.0f} s]")


if __name__ == "__main__":
    main()
