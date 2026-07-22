"""The photon budget drawn as rays: SM, MM, the as-built MCF tip, the new design.

Run ``py figure_ray_budget.py`` (``--coarse`` for a fast preview).
Writes figures/comparison_collection/ray_budget.png|pdf.

Every panel holds the same NV source fixed -- one emitter on axis at 85 um in
the 80-90 um layer -- and draws the rays that can carry its light into a
collection core.  They are traced from the core outward because that is the
direction the tracer runs; by reciprocity they are the paths the photons take.
The last panel turns the four collection efficiencies into a count out of one
batch of emitted photons, against the escape-cone ceiling.

SM and MM are bare fibers with no printed cap, so they cannot be written as
surfaces in the design model.  Their efficiencies come from the
characterisation model in paper_figures.py -- a different tracer -- and are
drawn hatched wherever they appear.
"""
import json
import os

import numpy as np

from compare_collection import (OUT, _surfaces_of, build, multimode_reference)
from compare_probes import as_built_surfaces
from lens_design import (RED_DESIGN_NM, _union_boundary_arrays,
                         diamond_sellmeier, escape_ceiling,
                         replicated_surfaces, trace_full_na)
from method_export import METHODS_DIRNAME, headline_label, list_methods

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = os.path.join(HERE, "figures", METHODS_DIRNAME)
BATCH = 10_000                    # photons emitted, for the budget panel
NV_Z = -85.0                      # the one emitter every panel is drawn for
SPAN = 72.0                       # half-width of every ray panel, um
FLOOR = -96.0                     # how deep into the diamond the panels look

RED, GREEN, GREY, BLUE, INK, MUTED = ("#c0392b", "#17803d", "#8a94a1",
                                      "#2f5fc4", "#131920", "#52514e")
DIAMOND, LAYER, IPS, GLASS = "#cfe6f2", "#9dc6dc", "#e8d8b7", "#dfe3e8"


def bare_fiber_rays(ax, core_um, na, top):
    """A bare fiber at contact: its NA cone, refracted into the diamond.

    Only the diamond-side segment is drawn.  Inside the fiber the light is
    guided, so a straight line over 200 um of core would leave the core and
    show a ray the fiber never carries.
    """
    n_dia = float(diamond_sellmeier(RED_DESIGN_NM/1000.0))
    radius = core_um/2.0
    theta = float(np.arcsin(na))
    ax.add_patch(_rect(-62.5, 0.0, 125.0, top, GLASS, "125 um cladding"))
    ax.add_patch(_rect(-radius, 0.0, core_um, top, "#f2e4c4",
                       f"{core_um:.0f} um core"))
    for x0 in np.linspace(-radius, radius, 7):
        for angle in np.linspace(-theta, theta, 5):
            inside = float(np.arcsin(np.sin(angle)/n_dia))
            ax.plot([x0, x0-FLOOR*np.tan(inside)], [0.0, FLOOR],
                    color=RED, lw=0.6, alpha=0.34, zorder=2)
    ax.annotate(f"core {core_um:.0f} um, NA {na}", xy=(0.5, 0.985),
                xycoords="axes fraction", ha="center", va="top", fontsize=7.6,
                color=MUTED)


def mcf_rays(ax, central, side):
    """A printed seven-core tip: the cap profile and the side cores' cones."""
    union = replicated_surfaces(central, side)
    xs = np.linspace(-SPAN, SPAN, 400)
    boundary = _union_boundary_arrays(
        union, central["base_z"], xs, np.zeros_like(xs))[0]
    finite = np.isfinite(boundary)
    ax.fill_between(xs[finite], boundary[finite], central["base_z"],
                    color=IPS, zorder=1, label="printed IP-S")
    ax.plot(xs[finite], boundary[finite], color="#875a13", lw=1.3, zorder=3)
    for surface, colour, name in ((union[1], RED, "collection cone"),
                                  (union[4], RED, None)):
        trace = trace_full_na(surface, union, RED_DESIGN_NM, depths=[NV_Z*-1.0],
                              n_grid=25)
        near = trace["valid"] & (np.abs(trace["points"][:, 1]) < 2.5)
        rays = np.flatnonzero(near)[:110]
        weights = trace["weight"][rays]
        strongest = float(weights.max()) if len(weights) else 1.0
        for ray, weight in zip(rays, weights):
            share = weight/strongest if strongest > 0.0 else 0.0
            ax.plot([trace["origins"][ray, 0], trace["points"][ray, 0],
                     trace["air_surface"][ray, 0], trace["hits"][0, ray, 0]],
                    [trace["origins"][ray, 2], trace["points"][ray, 2],
                     trace["air_surface"][ray, 2], trace["hits"][0, ray, 2]],
                    color=colour, lw=0.4+0.9*share, alpha=0.07+0.5*share,
                    zorder=2)
        if name:
            ax.plot([], [], color=colour, lw=1.5, label=name)


def _rect(x, y, width, height, colour, label=None):
    import matplotlib.patches as patches
    return patches.Rectangle((x, y), width, height, facecolor=colour,
                             edgecolor="none", zorder=1, label=label)


def probes(coarse):
    """The four probes: label, efficiency, how to draw it, and its top edge."""
    designs = list_methods(METHODS)
    if not designs:
        raise SystemExit(f"no exported designs in {METHODS}")
    report = _report(coarse)
    # the design the project chose, not whichever pairing tops the eta
    # sort -- the top four are tied and the order moves with the grid
    label = headline_label(os.path.join(HERE, "figures"))
    best = next(r for r in report["designs"] if r["label"] == label)
    design = next(d for d in designs if d["label"] == best["label"])
    central, side = _surfaces_of(design)
    old_central, old_side = as_built_surfaces(report["as_built"]["gap_um"])

    # one vertical scale for every panel, so the NV sits at the same height in
    # all four and the printed tips can be compared against the bare faces
    top = max(old_central["base_z"], central["base_z"])+8.0
    rows = [dict(name="SM\nbare fiber", eta=multimode_reference("SM")["collection_efficiency"],
                 note="characterisation model", top=top,
                 draw=lambda ax: bare_fiber_rays(ax, 4.0, 0.12, top)),
            dict(name="MM\nbare fiber", eta=report["multimode"]["collection_efficiency"],
                 note="characterisation model", top=top,
                 draw=lambda ax: bare_fiber_rays(ax, 50.0, 0.22, top)),
            dict(name="MCF as-built\n{:.0f} um gap".format(report["as_built"]["gap_um"]),
                 eta=report["as_built"]["collection_efficiency"], note=None,
                 top=top, draw=lambda ax: mcf_rays(ax, old_central, old_side)),
            dict(name=f"MCF new design\n{best['label']}", eta=best["collection_efficiency"],
                 note=None, top=top,
                 draw=lambda ax: mcf_rays(ax, central, side))]
    ceiling = escape_ceiling(RED_DESIGN_NM)
    for row in rows:
        # the bug this model already had once: nothing may collect more light
        # than can leave the diamond at all.
        assert row["eta"] <= ceiling, f"{row['name']}: {row['eta']} over ceiling"
    return rows, ceiling


def _report(coarse):
    """The collection comparison, recomputed only if it has not been written."""
    path = os.path.join(OUT, "collection_comparison.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return build(coarse)


def main(coarse=False):
    os.makedirs(OUT, exist_ok=True)
    rows, ceiling = probes(coarse)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16.2, 4.8))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.25], wspace=0.28)
    letters = "abcd"
    for index, row in enumerate(rows):
        ax = fig.add_subplot(grid[0, index])
        ax.axhspan(FLOOR, 0.0, color=DIAMOND, zorder=0, label="diamond")
        ax.axhspan(-90.0, -80.0, color=LAYER, zorder=0, label="NV layer")
        row["draw"](ax)
        ax.plot(0.0, NV_Z, marker="*", ms=13, color=GREEN, zorder=5,
                markeredgecolor="white", markeredgewidth=0.6,
                label="the NV, 85 um deep")
        ax.set_xlim(-SPAN, SPAN)
        ax.set_ylim(FLOOR, row["top"])
        ax.set_xlabel("x (um)")
        ax.set_title(f"({letters[index]}) {row['name']}", loc="left", fontsize=9.5)
        ax.annotate(f"{BATCH*row['eta']:,.1f} of {BATCH:,}\ncollected",
                    xy=(0.03, 0.03), xycoords="axes fraction", fontsize=9,
                    color=BLUE if index == 3 else INK,
                    fontweight="bold" if index == 3 else "normal")
        if row["note"]:
            ax.annotate(row["note"], xy=(0.5, 0.945), xycoords="axes fraction",
                        fontsize=7.2, color=MUTED, ha="center", style="italic")
        if index == 0:
            ax.set_ylabel("z from the diamond surface (um)")
        if index == 3:
            ax.legend(fontsize=6.8, loc="upper right", framealpha=0.9)

    # ---- (e) the same four numbers as a count out of one batch
    ax = fig.add_subplot(grid[0, 4])
    labels = ["emitted", "escape\nceiling", "SM", "MM", "as-built",
              "new design"]
    values = [BATCH, BATCH*ceiling] + [BATCH*r["eta"] for r in rows]
    colours = [GREY, "#c9ced6", GREY, GREY, GREY, BLUE]
    bars = ax.bar(labels, values, color=colours, width=0.62)
    for bar, row in zip(bars[2:], rows):
        if row["note"]:
            bar.set_hatch("///")
            bar.set_edgecolor("white")
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:,.0f}" if value >= 10 else f"{value:.2f}",
                    (bar.get_x()+bar.get_width()/2, value), ha="center",
                    va="bottom", fontsize=9,
                    color=BLUE if bar is bars[-1] else INK,
                    fontweight="bold" if bar is bars[-1] else "normal")
    ax.set_yscale("log")
    ax.set_ylim(max(min(values)*0.3, 1e-2), BATCH*6)
    ax.tick_params(axis="x", labelrotation=30, labelsize=8.5)
    ax.set_ylabel(f"photons, out of {BATCH:,} emitted")
    ax.set_title("(e) where one batch of photons ends up", loc="left",
                 fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    gain = rows[3]["eta"]/max(rows[2]["eta"], 1e-30)
    ax.annotate(f"x{gain:.0f} vs as-built", xy=(5, values[-1]),
                xytext=(4.4, values[-1]*9), fontsize=11, color=BLUE,
                fontweight="bold", ha="center")

    fig.text(0.5, -0.12,
             "All four panels are drawn for the same NV and the same 80-90 um layer; rays run from the "
             "collection core outward, which by reciprocity is the path a photon takes to reach it.\n"
             "Hatched bars (SM, MM) are bare fibers taken from the characterisation model, a different "
             "tracer, so only the two MCF tips are a like-for-like ratio. Collection is held to the "
             "escape-cone ceiling at every wavelength.",
             fontsize=8.0, color=MUTED, ha="center")
    path = os.path.join(OUT, "ray_budget.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"ceiling: {100*ceiling:.4f}%  ({BATCH*ceiling:,.0f} of {BATCH:,})")
    for row in rows:
        print(f"{row['name'].replace(chr(10), ' '):<28}"
              f"{100*row['eta']:>10.5f}%{BATCH*row['eta']:>12,.2f}"
              f"   {row['note'] or ''}")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    import sys
    main(coarse="--coarse" in sys.argv)
