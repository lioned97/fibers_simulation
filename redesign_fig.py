"""
redesign_fig.py -- Figure 7 of the MCF-redesign study: volume-normalized ODMR
sensitivity Psi(z) over the 80-90 um NV layer for the four probes (SM, MM,
MCF as-built, redesigned MCF), recomputed fresh (nothing hard-coded).

Panel (a): Psi(z) vs depth, log y, one line+marker per probe, Psi_robust
annotated.  Panel (b): sorted horizontal bar chart of Psi_robust (log x) with
the 12x equal-footprint caveat as a footnote.

Reuses the study's own pipeline verbatim -- probe_optimum + sensitivity_fom for
the physical probes (phase1_baseline.py recipe), ideal_fom for the redesign
winner (phase3_optimize.py) -- all at the shared calibrated b_auto.  Run:
    python redesign_fig.py     (~5-10 min, tracer-bound)
"""
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_figures import CONFIGS, save, OUT, INK, INK2
from sensitivity import sensitivity_fom, probe_optimum, calibrate_b_auto, DEPTHS
from phase3_optimize import ideal_fom

NUM_RAYS = 16384
REDESIGN_COLOR = "#d7263d"

# Phase-3 winning knobs (idealized redesign): gap 0, 6 side channels aimed at
# (0,0,-85 um), 4 um waist at the layer; green focus w=1 um at z=85 with a
# 20 um extended depth of focus.
REDESIGN = dict(gap=0.0, aim_depth=85.0, waist=4.0,
                exc_knobs=dict(w_focus=1.0, z_focus=85.0, L_edof=20.0))

# Quoted study results (Psi_robust, with background) for the match check.
QUOTED = {"SM": 5.09e4, "MM": 7.847e4, "MCF fixed aim": 9.922e4,
          "redesigned MCF": 389.8}


def compute():
    """Fresh recompute -> ordered list of (label, color, psi_per_depth, psi_rob)."""
    b_auto = calibrate_b_auto(num_rays=NUM_RAYS)
    print(f"shared calibrated b_auto: {b_auto:.4g}\n", flush=True)

    out = []
    for cfg in CONFIGS:
        if cfg['name'] == "MCF re-aimed":          # bit-identical duplicate of fixed aim
            continue
        g = probe_optimum(cfg, num_rays=NUM_RAYS)
        f = sensitivity_fom(cfg, g, b_auto=b_auto, num_rays=NUM_RAYS)
        print(f"  {cfg['name']:>14} (g={g:3.0f}): "
              f"Psi_rob={f['psi_robust']:.4g}", flush=True)
        out.append((cfg['name'], cfg['color'], np.asarray(f['psi']),
                    f['psi_robust']))

    f = ideal_fom(REDESIGN['gap'], REDESIGN['exc_knobs'], REDESIGN['aim_depth'],
                  REDESIGN['waist'], b_auto, NUM_RAYS)
    print(f"  {'redesigned MCF':>14} (g={REDESIGN['gap']:3.0f}): "
          f"Psi_rob={f['psi_robust']:.4g}", flush=True)
    out.append(("redesigned MCF", REDESIGN_COLOR, np.asarray(f['psi']),
                f['psi_robust']))
    return out


def make_figure(rows):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.2),
                                   gridspec_kw=dict(wspace=0.32))

    # ---- (a) Psi(z) vs depth ----
    z = np.asarray(DEPTHS)
    for label, color, psi, psi_rob in rows:
        axa.plot(z, psi, color=color, marker="o", markersize=4.5, lw=2.0,
                 label=label)
        axa.annotate(f"{psi_rob:.3g}", xy=(z[-1], psi[-1]),
                     xytext=(4, 0), textcoords="offset points",
                     fontsize=7.5, color=color, va="center")
    axa.set_yscale("log")
    axa.set_xticks(z)
    axa.set_xlabel(r"NV depth $z$ ($\mu$m)")
    axa.set_ylabel(r"Sensitivity $\Psi(z)$  (lower = better)")
    axa.set_title("(a) Sensitivity vs depth", loc="left", fontsize=9.5, color=INK2)
    axa.legend(fontsize=7.8)

    # ---- (b) sorted Psi_robust bars ----
    srt = sorted(rows, key=lambda r: r[3])          # best (lowest) first
    y = np.arange(len(srt))
    axb.barh(y, [r[3] for r in srt], color=[r[1] for r in srt], height=0.62)
    axb.set_yticks(y)
    axb.set_yticklabels([r[0] for r in srt], fontsize=8.5)
    axb.set_xscale("log")
    axb.set_xlabel(r"$\Psi_{\mathrm{robust}}$ (worst over 80-90 $\mu$m)")
    axb.set_title("(b) Robust sensitivity", loc="left", fontsize=9.5, color=INK2)
    for yi, r in zip(y, srt):
        axb.text(r[3] * 1.15, yi, f"{r[3]:.3g}", va="center", fontsize=7.8,
                 color=INK)
    axb.set_xlim(right=axb.get_xlim()[1] * 3.0)     # room for the labels

    plt.figtext(0.5, -0.02,
                "Equal-footprint control (excitation $w_{\\mathrm{focus}}=15\\,\\mu$m, "
                "MM-like): redesign $\\Psi\\approx6.6\\times10^{3}$, i.e. $\\sim$12$\\times$ "
                "better than MM -- the conservative claim.",
                ha="center", fontsize=7.6, color=INK2)
    save(fig, "fig7_redesign_psi")


def check(rows):
    print("\n  recomputed vs quoted (Psi_robust, with background):")
    print(f"    {'probe':>14} | {'recomputed':>11} | {'quoted':>9} | {'dev %':>7}")
    print("    " + "-" * 50)
    bad = []
    for label, _, _, psi_rob in rows:
        q = QUOTED[label]
        dev = 100.0 * (psi_rob - q) / q
        flag = "  <-- >10%!" if abs(dev) > 10.0 else ""
        print(f"    {label:>14} | {psi_rob:11.4g} | {q:9.4g} | {dev:+7.1f}{flag}")
        if abs(dev) > 10.0:
            bad.append((label, psi_rob, q, dev))
    return bad


if __name__ == "__main__":
    t0 = time.time()
    rows = compute()
    make_figure(rows)
    bad = check(rows)
    print(f"\n  figures -> {OUT}\\fig7_redesign_psi.png (+ .pdf)")
    if bad:
        print("\n  *** MISMATCH > 10% -- NOT adjusting anything, reporting as-is:")
        for label, got, q, dev in bad:
            print(f"      {label}: got {got:.4g}, quoted {q:.4g} ({dev:+.1f}%)")
    else:
        print("\n  all four match the quoted study results within ~few % (MC tolerance).")
    print(f"\n[{time.time()-t0:.0f} s]")
