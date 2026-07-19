"""
phase1_baseline.py -- Phase 1 of the MCF-redesign study: baseline sensitivity
FoM Psi(z) for SM / MM / MCF (as built) over the 80-90 um NV layer.

Prints, per probe: optimal gap, per-depth S, C_eff, sbar, FWHM, Psi, and
Psi_robust -- twice: photon-limited (b_auto=0) and with the calibrated shared
background.  The photon-limited column isolates how much of the MM gap is raw
signal vs. how much the background/contrast lever can move.

Run: python phase1_baseline.py   (a few minutes; tracer-bound)
"""
import time

import numpy as np

from paper_figures import CONFIGS
from sensitivity import sensitivity_fom, probe_optimum, calibrate_b_auto, DEPTHS

# ponytail: 16384 cone-quadrature rays everywhere -- equal-solid-angle grid, so
# profile ratios are already smooth; bump to cfg defaults only if Phase 3 margins
# come out within noise.
NUM_RAYS = 16384

# "MCF re-aimed" is bit-identical to fixed aim under the printed-union trace
# (see resolution_at_layer.py) -- skip the duplicate.
PROBES = [c for c in CONFIGS if c['name'] != "MCF re-aimed"]


def report(cfg, gap, fom, tag):
    print(f"\n  [{tag}]  Psi_robust = {fom['psi_robust']:.4g}")
    hdr = f"    {'z (um)':>7} | {'S (ph/s)':>11} | {'C_eff':>7} | {'sbar':>10} | {'FWHM (um)':>9} | {'Psi':>10}"
    print(hdr + "\n    " + "-" * (len(hdr) - 4))
    for i, z in enumerate(DEPTHS):
        print(f"    {z:7.0f} | {fom['S'][i]:11.4g} | {fom['C_eff'][i]:7.4f} | "
              f"{fom['sbar'][i]:10.4g} | {fom['fwhm_sig'][i]:9.1f} | {fom['psi'][i]:10.4g}")


if __name__ == "__main__":
    t0 = time.time()
    b_auto = calibrate_b_auto(num_rays=NUM_RAYS)
    print(f"shared b_auto (calibrated on MM, S/(S+B)=0.5 @ 85 um): {b_auto:.4g}")

    rows = []
    for cfg in PROBES:
        g = probe_optimum(cfg, num_rays=NUM_RAYS)
        print(f"\n=== {cfg['name']}  (optimal gap g = {g:.0f} um) ===")
        fom_pl = sensitivity_fom(cfg, g, b_auto=0.0, num_rays=NUM_RAYS)
        report(cfg, g, fom_pl, "photon-limited (b=0)")
        fom_bg = sensitivity_fom(cfg, g, b_auto=b_auto, num_rays=NUM_RAYS)
        report(cfg, g, fom_bg, "with background")
        rows.append((cfg['name'], g, fom_pl['psi_robust'], fom_bg['psi_robust']))

    print(f"\n{'probe':>14} | {'g (um)':>6} | {'Psi_rob (b=0)':>13} | {'Psi_rob (bg)':>13} | {'vs MM (bg)':>10}")
    print("-" * 68)
    mm_bg = next(r[3] for r in rows if r[0] == "MM")
    for name, g, ppl, pbg in rows:
        print(f"{name:>14} | {g:6.0f} | {ppl:13.4g} | {pbg:13.4g} | {mm_bg / pbg:10.2f}x")
    print(f"\n(vs MM > 1x means better sensitivity than MM)  [{time.time()-t0:.0f} s]")
