"""Where the photons go: the as-built tip against the new design.

Run ``py figure_photon_budget.py`` (``--coarse`` for a fast preview).
Writes figures/comparison_collection/photon_budget.png|pdf.

Two things a collection-efficiency number does not show on its own:

  (a,b)  the collection probability across the NV layer for each tip, on one
         shared colour scale, with the green excitation spot outlined.  The
         question is not just how much light a tip collects but whether it
         collects it from where the green is actually making photons.
  (c)    the fate of a fixed batch of emitted photons, so the comparison is a
         count rather than a ratio.

Geometry is read from the exported designs; every number is retraced here with
the corrected per-wavelength escape ceiling, so this does not depend on the
metrics stored alongside those designs.
"""
import json
import os

import numpy as np

from compare_collection import _quadrature, _surfaces_of
from compare_probes import AS_BUILT_GAPS, as_built_surfaces
from lens_design import (I_SAT, MCF_FULL_NA, MCF_IPS_N, P_GREEN_MW,
                         RED_DESIGN_NM, R_SAT, W_MODE, _field_axis,
                         _ray_density, beam_stats, diamond_sellmeier,
                         escape_ceiling, replicated_surfaces, trace_full_na)
from method_export import METHODS_DIRNAME, headline_label, list_methods

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = os.path.join(HERE, "figures", METHODS_DIRNAME)
OUT = os.path.join(HERE, "figures", "comparison_collection")
BATCH = 10_000                      # photons emitted, for the budget panel

GREY, BLUE, GREEN, INK, MUTED = "#8a94a1", "#2f5fc4", "#17803d", "#131920", "#52514e"


def fields(central, side, coarse):
    """Collection probability and excitation rate across the NV layer."""
    depths, red_lam, red_w, grid_n, ray_grid = _quadrature(coarse)
    union = replicated_surfaces(central, side)
    central_trace = trace_full_na(central, union, 532.0, depths, ray_grid)
    stats = beam_stats(central_trace, 532.0, depths)
    side_traces = []
    for lam, weight in zip(red_lam, red_w):
        trace = trace_full_na(side, union, float(lam), depths, ray_grid)
        side_traces.append((float(lam), float(weight), trace))
        stats = stats + beam_stats(trace, float(lam), depths)
    axis = _field_axis(stats, grid_n)
    dx = axis[1]-axis[0]
    dz = float(depths[-1]-depths[0])/(len(depths)-1)

    mid = len(depths)//2
    collection_mid = np.zeros((len(axis), len(axis)))
    emitted = collected = 0.0
    for iz in range(len(depths)):
        intensity = P_GREEN_MW*_ray_density(central_trace, iz, axis, 532.0)
        saturation = intensity/I_SAT
        excitation = R_SAT*saturation/(1.0+saturation)
        collection = np.zeros_like(excitation)
        for lam, weight, trace in side_traces:
            n_dia = float(diamond_sellmeier(lam/1000.0))
            theta_ips = np.arcsin(MCF_FULL_NA/MCF_IPS_N)
            solid_angle = 2*np.pi*(1.0-np.cos(theta_ips))
            core_area = 0.5*np.pi*W_MODE*W_MODE
            acceptance = (core_area*MCF_IPS_N**2*solid_angle /
                          (4*np.pi*n_dia*n_dia))
            contribution = acceptance*_ray_density(
                trace, iz, axis, lam, np.arange(6)*np.pi/3.0)
            collection += weight*np.minimum(contribution, escape_ceiling(lam))
        volume = dx*dx*dz*(0.5 if iz in (0, len(depths)-1) else 1.0)
        emitted += float(excitation.sum())*volume
        collected += float((excitation*collection).sum())*volume
        if iz == mid:
            collection_mid = collection
            excitation_mid = excitation
    return dict(axis=axis, collection=collection_mid, excitation=excitation_mid,
                efficiency=collected/max(emitted, 1e-30))


def as_built_best(coarse):
    best = None
    for gap in AS_BUILT_GAPS:
        try:
            row = fields(*as_built_surfaces(float(gap)), coarse=coarse)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or row["efficiency"] > best[1]["efficiency"]:
            best = (float(gap), row)
    if best is None:
        raise RuntimeError("the as-built tip produced no valid evaluation")
    return best


def main(coarse=False):
    os.makedirs(OUT, exist_ok=True)
    designs = list_methods(METHODS)
    if not designs:
        raise SystemExit(f"no exported designs in {METHODS}")

    gap, old = as_built_best(coarse)
    # the exported winner, not list_methods[0]: the summaries sort by the
    # search-grid sensitivity, and that ordering disagrees with the final
    # re-score that actually chose the design
    chosen = headline_label(os.path.dirname(METHODS))
    new_design = next(d for d in designs if d["label"] == chosen)
    new = fields(*_surfaces_of(new_design), coarse=coarse)
    ceiling = escape_ceiling(RED_DESIGN_NM)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig = plt.figure(figsize=(13.4, 4.6))
    # the colour bar gets its own slim column, so it cannot eat into the
    # budget panel's axis labels
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 0.055, 1.05],
                            wspace=0.30)

    top = max(new["collection"].max(), old["collection"].max(), 1e-12)
    floor = top/1e4
    panels = [(fig.add_subplot(grid[0, 0]), old, f"(a) as-built tip, {gap:.0f} um gap"),
              (fig.add_subplot(grid[0, 1]), new, f"(b) new design: {new_design['label']}")]
    for axis_, data, title in panels:
        span = 30.0
        image = axis_.pcolormesh(data["axis"], data["axis"],
                                 np.maximum(data["collection"], floor),
                                 norm=LogNorm(vmin=floor, vmax=top),
                                 cmap="magma", shading="auto")
        peak_excitation = data["excitation"].max()
        if peak_excitation > 0:
            axis_.contour(data["axis"], data["axis"],
                          data["excitation"]/peak_excitation, levels=[0.5],
                          colors=GREEN, linewidths=1.6, linestyles="--")
        axis_.set_xlim(-span, span)
        axis_.set_ylim(-span, span)
        axis_.set_aspect("equal")
        axis_.set_xlabel("x (um)")
        axis_.set_title(title, loc="left", fontsize=10)
        axis_.annotate(f"collects {100*data['efficiency']:.4f}%",
                       xy=(0.03, 0.04), xycoords="axes fraction", fontsize=10,
                       color="white", fontweight="bold")
    panels[0][0].set_ylabel("y (um)")
    bar = fig.colorbar(image, cax=fig.add_subplot(grid[0, 2]))
    bar.set_label("collection probability per photon", fontsize=8.5)

    # ---- (c) the fate of a fixed batch of photons
    axis_budget = fig.add_subplot(grid[0, 3])
    old_caught = BATCH*old["efficiency"]
    new_caught = BATCH*new["efficiency"]
    escaping = BATCH*ceiling
    labels = ["emitted", "can leave\nthe diamond", "as-built\ncollects",
              "new design\ncollects"]
    values = [BATCH, escaping, old_caught, new_caught]
    colours = [GREY, "#c9ced6", GREY, BLUE]
    bars = axis_budget.bar(labels, values, color=colours, width=0.62)
    for bar_, value in zip(bars, values):
        text = f"{value:,.0f}" if value >= 10 else f"{value:.2f}"
        axis_budget.annotate(text, (bar_.get_x()+bar_.get_width()/2, value),
                             ha="center", va="bottom", fontsize=9.5,
                             fontweight="bold" if bar_ is bars[-1] else "normal",
                             color=BLUE if bar_ is bars[-1] else INK)
    axis_budget.set_yscale("log")
    axis_budget.set_ylim(max(old_caught*0.25, 1e-2), BATCH*6)
    axis_budget.set_ylabel(f"photons, out of {BATCH:,} emitted")
    axis_budget.set_title("(c) where a batch of photons ends up", loc="left",
                          fontsize=10)
    axis_budget.spines[["top", "right"]].set_visible(False)
    axis_budget.grid(axis="y", color="#e1e0d9", lw=0.6)
    axis_budget.set_axisbelow(True)
    axis_budget.annotate(f"x{new_caught/max(old_caught, 1e-12):.0f}",
                         xy=(3, new_caught), xytext=(2.5, new_caught*8),
                         fontsize=13, color=BLUE, fontweight="bold",
                         ha="center")

    fig.text(0.5, -0.02,
             "Dashed green outline: half-maximum of the 532 nm excitation spot. The new tip "
             "concentrates its collection onto exactly that spot,\nwhich is what turns a small "
             "etendue into a large photon count. Collection is held to the escape-cone ceiling "
             "at every wavelength.",
             fontsize=8.0, color=MUTED, ha="center")
    path = os.path.join(OUT, "photon_budget.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"as-built tip ({gap:.0f} um gap): {100*old['efficiency']:.5f}%  "
          f"-> {old_caught:.2f} photons per {BATCH:,}")
    print(f"new design {new_design['label']}: {100*new['efficiency']:.5f}%  "
          f"-> {new_caught:,.0f} photons per {BATCH:,}")
    print(f"difference: x{new_caught/max(old_caught, 1e-12):.0f} more photons "
          "collected from the same emission")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    import sys
    main(coarse="--coarse" in sys.argv)
