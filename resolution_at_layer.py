"""
resolution_at_layer.py -- lateral resolution of each fiber probe AT the NV
layer (80 / 85 / 90 um below the diamond's fiber-facing surface).

Run `py resolution_at_layer.py`; writes figures/fig5_resolution_at_layer.png/.pdf
and prints a FWHM + d50 table. No UI, no CLI args.

Everything physical is imported from paper_figures.py so this can never drift
from the paper's model: the STL-calibrated MCF side-lens trace, the collimated
central-lens green beam (Gaussian-q through the measured R = 74.91 um cap --
i.e. the residual divergence of the collimated beam IS the MCF excitation
width here), the escape-cone quadrature, and each probe's configuration.

Per probe, evaluated at that probe's own optimal gap (argmax of the single-NV
into-fiber efficiency over the paper's 0-200 um sweep):
  column 1  excitation profile  R_exc(r)  at each depth (green, saturable),
  column 2  collection profile  eta(r)    azimuthally averaged (for the MCF,
            emitter azimuths uniform over one 60-degree lens period make the
            single-traced-lens x6 symmetry reduction an exact azimuthal mean),
  column 3  detected-signal profile R_exc(r) * eta(r) -- the probe's actual
            sensing point-spread function on the layer.
Each curve is normalized to its own peak (widths are the message; absolute
levels live in fig1). FWHM = outermost half-maximum diameter (well defined
even for ring-shaped MCF collection); d50 = diameter enclosing half the
azimuthally integrated signal (2*pi*r weighting).
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from paper_figures import (
    CONFIGS, GAPS, NV_DEPTH, N_DIA_RED, INK2, OUT,
    exc_rate, exc_width, eta_per_emitter, escape_cone_quadrature, save,
)

DEPTHS = (80.0, 85.0, 90.0)                 # um below the facing surface
DEPTH_LS = {80.0: ":", 85.0: "-", 90.0: "--"}
N_R, N_AZ = 120, 6                           # radial grid x azimuths per period

# The current printed-IP-S-union collection trace derives its acceptance from
# the fixed printed geometry alone and ignores the per-config boresight, so
# "MCF fixed aim" and "MCF re-aimed" are bit-identical here (verified).
# Keep one and label it plainly to avoid duplicate rows in a paper figure.
RES_CONFIGS = [c for c in CONFIGS if c['name'] != "MCF re-aimed"]
LABELS = {"MCF fixed aim": "MCF"}


def probe_optimum(cfg):
    """Optimal gap = argmax single-NV efficiency, same sweep as paper fig1."""
    V0, W0 = escape_cone_quadrature(cfg['n1'], N_DIA_RED, seed=17)
    single = np.array([[0.0, 0.0, -NV_DEPTH]])
    eta = [eta_per_emitter(single, V0, W0, cfg['fibers'](g), g, cfg['model'])[0]
           for g in GAPS]
    return float(GAPS[int(np.argmax(eta))])


def collection_profile(cfg, g, depth, r):
    """Azimuthally averaged eta(r) for single NVs on a ring grid at `depth`.

    Rays: cfg['nen'] (the fidelity the paper itself uses for ensemble
    estimates) -- these profiles are normalized shapes, so the n1-level ray
    count would buy nothing but minutes. Azimuths: SM/MM are axisymmetric,
    one azimuth is exact; the MCF needs the 6 midpoints of one 60-degree lens
    period, whose mean over the single traced lens x6 equals the exact
    azimuthal average of the 6-lens sum.
    """
    n_az = N_AZ if cfg['exc'] == "MCF" else 1
    az = (np.arange(n_az) + 0.5) * (np.pi / 3.0) / n_az
    rr, aa = np.meshgrid(r, az, indexing="ij")
    em = np.column_stack([(rr * np.cos(aa)).ravel(), (rr * np.sin(aa)).ravel(),
                          np.full(rr.size, -depth)])
    V0, W0 = escape_cone_quadrature(cfg['nen'], N_DIA_RED, seed=17)
    eta = eta_per_emitter(em, V0, W0, cfg['fibers'](g), g, cfg['model'])
    return eta.reshape(N_R, n_az).mean(axis=1)


def width_half_max(r, p):
    """Outermost half-maximum DIAMETER of a radial profile."""
    h = p.max() / 2.0
    above = np.nonzero(p >= h)[0]
    if len(above) == 0:
        return np.nan
    i = above[-1]
    if i == len(r) - 1:
        return 2.0 * r[-1]                   # not resolved inside the grid
    r_half = r[i] + (r[i + 1] - r[i]) * (p[i] - h) / (p[i] - p[i + 1])
    return 2.0 * r_half


def d50(r, p):
    """Diameter enclosing 50% of the azimuthally integrated signal."""
    dc = 0.5 * (p[1:] * r[1:] + p[:-1] * r[:-1]) * np.diff(r)
    c = np.concatenate([[0.0], np.cumsum(dc)])
    if c[-1] <= 0:
        return np.nan
    return 2.0 * float(np.interp(0.5 * c[-1], c, r))


if __name__ == "__main__":
    import time
    t_start = time.time()
    os.makedirs(OUT, exist_ok=True)
    fig, axs = plt.subplots(len(RES_CONFIGS), 3, figsize=(7.2, 2.0 * len(RES_CONFIGS)),
                            sharey=True)
    col_titles = ["(a) excitation", "(b) collection (az. avg.)",
                  "(c) detected signal (a$\\times$b)"]
    rows_out = []

    for irow, cfg in enumerate(RES_CONFIGS):
        name = LABELS.get(cfg['name'], cfg['name'])
        g = probe_optimum(cfg)
        print(f"  {name}: optimal gap g = {g:.0f} um", flush=True)

        # radial grid sized to the widest thing this probe does at this gap
        r_max = 0.0
        for d in DEPTHS:
            r_max = max(r_max, 3.0 * float(np.atleast_1d(exc_width(cfg['exc'], d, g))[0]))
        r_max = max(r_max, 45.0)
        # Power-law grid: ~0.1 um steps near the axis so a focused conjugate
        # (re-aimed MCF collection can be a ~1 um spike) is actually resolved,
        # while still reaching r_max. All metrics below handle nonuniform r.
        r = r_max * np.linspace(0.0, 1.0, N_R) ** 1.5

        for d in DEPTHS:
            print(f"    depth {d:.0f} um ...", flush=True)
            R_exc, _ = exc_rate(cfg['exc'], r, 0.0, -d, g)
            eta = collection_profile(cfg, g, d, r)
            sig = R_exc * eta
            assert np.all(np.isfinite(sig)) and sig.max() > 0
            for icol, p in enumerate((R_exc, eta, sig)):
                axs[irow, icol].plot(r, p / p.max(), color=cfg['color'],
                                     ls=DEPTH_LS[d], lw=1.8)
            rows_out.append((name, g, d,
                             width_half_max(r, R_exc), width_half_max(r, sig),
                             d50(r, sig)))

        for icol in range(3):
            ax = axs[irow, icol]
            ax.set_xlim(0, r_max)
            ax.set_ylim(0, 1.06)
            if irow == 0:
                ax.set_title(col_titles[icol], fontsize=9.5, loc="left", color=INK2)
            if irow == len(RES_CONFIGS) - 1:
                ax.set_xlabel("r ($\\mu$m)")
        axs[irow, 0].set_ylabel(f"{name}\n(g = {g:.0f} $\\mu$m)\nnormalized",
                                fontsize=8.5)
        fw = next(x[4] for x in rows_out if x[0] == name and x[2] == 85.0)
        axs[irow, 2].annotate(f"FWHM$_{{85}}$ = {fw:.1f} $\\mu$m",
                              xy=(0.96, 0.86), xycoords="axes fraction",
                              ha="right", fontsize=8, color=INK2)

    handles = [plt.Line2D([], [], color="#52514e", ls=DEPTH_LS[d], lw=1.8,
                          label=f"depth {d:.0f} $\\mu$m") for d in DEPTHS]
    fig.legend(handles=handles, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.015), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig5_resolution_at_layer")

    print(f"\nfigure written to {OUT}  ({time.time()-t_start:.0f} s)")
    hdr = (f"{'probe':>14} | {'g (um)':>6} | {'depth':>5} | "
           f"{'FWHM exc (um)':>13} | {'FWHM signal (um)':>16} | {'d50 signal (um)':>15}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for name, g, d, fw_e, fw_s, d5 in rows_out:
        print(f"{name:>14} | {g:6.0f} | {d:5.0f} | {fw_e:13.1f} | {fw_s:16.1f} | {d5:15.1f}")
    print("\nMCF excitation uses the collimated central-lens beam (Gaussian-q through"
          "\nthe measured R = 74.91 um cap): its width is set by the residual divergence"
          "\nof the collimated beam, which is why it barely changes across 80-90 um,"
          "\nunlike the NA-diverging SM/MM cones. Collection profiles are azimuthal"
          "\naverages; the MCF's can be ring-shaped, so FWHM is the outermost"
          "\nhalf-maximum diameter and d50 the half-signal-enclosing diameter.")
