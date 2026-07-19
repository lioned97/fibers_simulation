"""
phase3_optimize.py -- Phase 3 of the MCF-redesign study: minimax-optimize the
idealized lens knobs (lens_design.py) against the sensitivity FoM
(sensitivity.py) and score the winner vs the Phase-1 SM/MM baselines.

Knobs: collection (gap, aim_depth, waist_at_layer) + excitation
(w_focus, z_focus, L_edof).  Two-stage coarse grid (collection first, then
excitation -- they couple only through the background term), then a
high-fidelity rescore of the winner with the photon-limited decomposition.

Run: python phase3_optimize.py   (~10-20 min; tracer-bound)
"""
import time

import numpy as np

from physics import trapz
from paper_figures import RHO, N_DIA_RED
from sensitivity import (
    DEPTHS, T_LAYER, C0, DNU0, N_R, N_R_BG, BG_PLANES,
    width_half_max, calibrate_b_auto,
)
from lens_design import (
    exc_rate_ideal, exc_beam_params, make_ideal_channels, eta_ideal_channels,
)
from paper_figures import escape_cone_quadrature

NUM_RAYS_SEARCH = 4096
NUM_RAYS_FINAL = 16384
RAY_BUDGET = 750_000          # same chunking rule as eta_per_emitter
N_AZ = 6                      # azimuth midpoints of one 60-degree channel period

# Phase-1 references (same NUM_RAYS_FINAL, same calibrated b_auto pipeline):
PSI_MM_BG, PSI_SM_BG = 7.847e4, 5.09e4


def eta_ideal_chunked(emitters, V0, W0, channels, gap):
    block = max(1, RAY_BUDGET // len(V0))
    return np.concatenate([
        eta_ideal_channels(emitters[i:i + block], V0, W0, channels, gap)
        for i in range(0, len(emitters), block)])


def ideal_collection_profile(channels, gap, depth, r, num_rays, seed=17):
    """Azimuthally averaged eta(r): 6 midpoints of one 60-degree period."""
    az = (np.arange(N_AZ) + 0.5) * (np.pi / 3.0) / N_AZ
    rr, aa = np.meshgrid(r, az, indexing="ij")
    em = np.column_stack([(rr * np.cos(aa)).ravel(), (rr * np.sin(aa)).ravel(),
                          np.full(rr.size, -depth)])
    V0, W0 = escape_cone_quadrature(num_rays, N_DIA_RED, seed=seed)
    return eta_ideal_chunked(em, V0, W0, channels, gap).reshape(len(r), N_AZ).mean(axis=1)


def _r_grid(exc_knobs, gap, depth, n_r):
    bp = exc_beam_params(depth, gap, exc_knobs)
    dz = abs(depth - bp['z_focus'])
    dz_eff = max(0.0, dz - 0.5 * bp['L_edof'])
    w_plane = (bp['w_used'] + 0.5 * bp['L_edof'] * bp['theta_dia_max']
               if dz <= 0.5 * bp['L_edof'] else
               bp['w_used'] * np.sqrt(1.0 + (dz_eff / bp['z_R']) ** 2))
    r_max = max(45.0, 3.0 * w_plane)
    return r_max * np.linspace(0.0, 1.0, n_r) ** 1.5


def ideal_fom(gap, exc_knobs, aim_depth, waist, b_auto, num_rays,
              depths=DEPTHS, b0=0.0):
    """sensitivity_fom equivalent for the ideal probe (same formulas)."""
    channels = make_ideal_channels(waist, aim_depth, gap)
    # background: whole-path autofluorescence, reduced fidelity (same recipe)
    B = b0
    if b_auto != 0.0:
        n_bg = max(1, num_rays // 4)
        dzp = BG_PLANES[1] - BG_PLANES[0]
        b_int = 0.0
        for zp in BG_PLANES:
            r = _r_grid(exc_knobs, gap, zp, N_R_BG)
            R_exc = exc_rate_ideal(r, zp, gap, exc_knobs)[0]
            eta = ideal_collection_profile(channels, gap, zp, r, n_bg, seed=29)
            b_int += trapz(R_exc * eta * 2.0 * np.pi * r, r) * dzp
        B = b_auto * b_int + b0

    n = len(depths)
    out = dict(psi=np.empty(n), C_eff=np.empty(n), S=np.empty(n),
               sbar=np.empty(n), fwhm_sig=np.empty(n), B=np.full(n, B))
    for i, z in enumerate(depths):
        r = _r_grid(exc_knobs, gap, z, N_R)
        R_exc = exc_rate_ideal(r, z, gap, exc_knobs)[0]
        eta = ideal_collection_profile(channels, gap, z, r, num_rays)
        s = RHO * R_exc * eta
        w = 2.0 * np.pi * r
        flux = trapz(s * w, r)
        out['S'][i] = T_LAYER * flux
        out['sbar'][i] = trapz(s * s * w, r) / flux if flux > 0 else 0.0
        out['fwhm_sig'][i] = width_half_max(r, s)
        out['C_eff'][i] = C0 * out['S'][i] / (out['S'][i] + B) if (out['S'][i] + B) > 0 else 0.0
        out['psi'][i] = (DNU0 / (out['C_eff'][i] * np.sqrt(out['sbar'][i]))
                         if out['C_eff'][i] > 0 and out['sbar'][i] > 0 else np.inf)
    out['psi_robust'] = float(np.max(out['psi']))
    return out


if __name__ == "__main__":
    t0 = time.time()
    b_auto = calibrate_b_auto(num_rays=NUM_RAYS_FINAL)
    print(f"shared b_auto: {b_auto:.4g}  (same calibration pipeline as Phase 1)")

    # ---- stage 1: collection knobs, fixed reasonable excitation ----
    exc0 = dict(w_focus=4.0, z_focus=85.0, L_edof=10.0)
    best = (np.inf, None)
    print("\nstage 1 -- collection (gap, aim_depth, waist), excitation fixed "
          f"{exc0}, {NUM_RAYS_SEARCH} rays:")
    for gap in (0.0, 10.0, 50.0):
        for aim in (82.0, 85.0, 88.0):
            for waist in (2.0, 4.0, 6.0, 10.0, 15.0):
                f = ideal_fom(gap, exc0, aim, waist, b_auto, NUM_RAYS_SEARCH)
                tag = f"g={gap:4.0f} aim={aim:4.0f} w={waist:4.1f} -> Psi_rob={f['psi_robust']:.4g}"
                if f['psi_robust'] < best[0]:
                    best = (f['psi_robust'], (gap, aim, waist))
                    tag += "  *best*"
                print("  " + tag, flush=True)
    gap, aim, waist = best[1]

    # ---- stage 2: excitation knobs at the best collection ----
    print(f"\nstage 2 -- excitation (w_focus, L_edof), collection fixed "
          f"g={gap} aim={aim} w={waist}:")
    best2 = (np.inf, None)
    for w_focus in (1.0, 2.0, 4.0, 8.0, 15.0):
        for L_edof in (0.0, 10.0, 20.0, 30.0):
            exc = dict(w_focus=w_focus, z_focus=85.0, L_edof=L_edof)
            f = ideal_fom(gap, exc, aim, waist, b_auto, NUM_RAYS_SEARCH)
            tag = (f"wf={w_focus:4.1f} L={L_edof:4.0f} -> Psi_rob={f['psi_robust']:.4g}")
            if f['psi_robust'] < best2[0]:
                best2 = (f['psi_robust'], exc)
                tag += "  *best*"
            print("  " + tag, flush=True)
    exc = best2[1]

    # ---- stage 3: high-fidelity rescore + decomposition ----
    print(f"\nstage 3 -- winner at {NUM_RAYS_FINAL} rays: gap={gap} aim={aim} "
          f"waist={waist} exc={exc}")
    for tag, ba in (("photon-limited (b=0)", 0.0), ("with background", b_auto)):
        f = ideal_fom(gap, exc, aim, waist, ba, NUM_RAYS_FINAL)
        print(f"\n  [{tag}]  Psi_robust = {f['psi_robust']:.4g}")
        for i, z in enumerate(DEPTHS):
            print(f"    z={z:3.0f}: S={f['S'][i]:.4g}  C_eff={f['C_eff'][i]:.4f}  "
                  f"sbar={f['sbar'][i]:.4g}  FWHM={f['fwhm_sig'][i]:.1f}  Psi={f['psi'][i]:.4g}")
    psi_final = f['psi_robust']
    print(f"\n==> ideal MCF vs MM: {PSI_MM_BG / psi_final:.2f}x   vs SM: "
          f"{PSI_SM_BG / psi_final:.2f}x   (>1 = ideal MCF better)")
    print(f"[{time.time()-t0:.0f} s]")
