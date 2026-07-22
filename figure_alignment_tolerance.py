"""How much misalignment each MCF tip tolerates: standoff and diamond tilt.

Run ``py figure_alignment_tolerance.py`` (``--coarse`` for a fast preview).
Writes figures/comparison_collection/alignment_tolerance.png|pdf and .json.

Two ways a probe loses light on the bench that a single efficiency number
hides: it sits at the wrong distance from the diamond, or its end face is not
parallel to the diamond surface.  Both tips are swept through the same
standoffs and the same tilts with the same tracer, and each is plotted against
its own peak, so the panels read as shape -- how fast the signal falls off --
rather than level.

The level is not close: at its own optimum the new design collects about 200x
what the as-built tip does (see compare_collection.py, which is where the exact
figure quoted on the plot comes from).  A flatter curve here is tolerance, not
performance.

Signal at every point is the traced 532 nm excitation times the six-core
collection, so an NV only counts where the green reaches it and a core can see
it; both are re-traced at every standoff and every tilt.  The lower panels give
the participation-ratio volume of that product, which is the volume the green
and the red actually share.

Tilt is applied to the diamond plane about the fiber y-axis, so rotational
symmetry breaks and all six side cores are traced separately -- which is why
the tilt sweep costs about six times a standoff point.
"""
import json
import os

import numpy as np

from compare_collection import OUT, _surfaces_of
from compare_probes import as_built_surfaces
from lens_design import alignment_sweep
from method_export import METHODS_DIRNAME, headline_label, list_methods

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = os.path.join(HERE, "figures", METHODS_DIRNAME)

# The new design's standoff optimum is a spike about 2 um wide, so the sweep is
# linear where that spike lives and geometric out to the as-built tip's long
# tail.  A pure geomspace stepped straight over the peak.
GAPS = np.unique(np.concatenate([np.arange(5.0, 41.0, 1.0),
                                 np.geomspace(44.0, 500.0, 12)]))
ANGLES = np.array([-10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0,
                   0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])
COARSE_GAPS = np.unique(np.concatenate([np.arange(5.0, 41.0, 3.0),
                                        np.geomspace(50.0, 500.0, 5)]))
COARSE_ANGLES = np.array([-8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0])
GRID_N = 241        # converged; see compare_collection._quadrature

GREY, BLUE, INK, MUTED = "#8a94a1", "#2f5fc4", "#131920", "#52514e"


def half_width(x, y, fraction=0.5):
    """Where the curve falls to ``fraction`` of its peak, either side of it.

    Linear interpolation between samples; an edge that never falls that far
    comes back as the end of the swept range, so a window reported at 5 or 500
    um means the sweep ran out, not that the tolerance did.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    peak = int(np.argmax(y))
    level = fraction*y[peak]
    edges = []
    for step in (-1, +1):
        index = peak
        while 0 <= index+step < len(y) and y[index+step] >= level:
            index += step
        if index+step < 0 or index+step >= len(y):
            edges.append(float(x[index]))       # sweep ended first
            continue
        x0, x1, y0, y1 = x[index], x[index+step], y[index], y[index+step]
        edges.append(float(x0 + (x1-x0)*(y0-level)/max(y0-y1, 1e-30)))
    return min(edges), max(edges)


def retained(x, y, at):
    """Signal left at +/-``at``, as a fraction of the peak (mean of both sides)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    both = [float(np.interp(sign*at, x, y)) for sign in (-1, +1)]
    return float(np.mean(both))/max(float(y.max()), 1e-30)


def sweep(central, side, coarse):
    gaps, angles = ((COARSE_GAPS, COARSE_ANGLES) if coarse else (GAPS, ANGLES))
    data = alignment_sweep(dict(central=central, side=side), gaps, angles,
                           grid_n=121 if coarse else GRID_N)
    for key in ("gap", "angle"):
        rows = data[key]
        top = max(row["model_fiber_photons_s"] for row in rows)
        for row in rows:
            row["relative"] = row["model_fiber_photons_s"]/max(top, 1e-30)
    return data


def build(coarse=False):
    designs = list_methods(METHODS)
    if not designs:
        raise SystemExit(f"no exported designs in {METHODS}")
    with open(os.path.join(OUT, "collection_comparison.json"),
              encoding="utf-8") as fh:
        report = json.load(fh)
    # the design the project chose, not whichever pairing tops the eta
    # sort -- the top four are tied and the order moves with the grid
    chosen = headline_label(os.path.join(HERE, "figures"))
    best = next(r for r in report["designs"] if r["label"] == chosen)
    design = next(d for d in designs if d["label"] == best["label"])

    probes = []
    for label, surfaces, eta in (
            ("MCF as-built", as_built_surfaces(report["as_built"]["gap_um"]),
             report["as_built"]["collection_efficiency"]),
            (f"MCF new design ({best['label']})", _surfaces_of(design),
             best["collection_efficiency"])):
        data = sweep(*surfaces, coarse=coarse)
        gap_x = [row["gap_um"] for row in data["gap"]]
        gap_y = [row["relative"] for row in data["gap"]]
        angle_x = [row["angle_deg"] for row in data["angle"]]
        angle_y = [row["relative"] for row in data["angle"]]
        probes.append(dict(label=label, eta_at_reference=eta, sweep=data,
                           best_gap_um=data["best_gap_um"],
                           gap_window_um=half_width(gap_x, gap_y),
                           tilt_window_deg=half_width(angle_x, angle_y),
                           tilt_at_2deg=retained(angle_x, angle_y, 2.0),
                           tilt_at_5deg=retained(angle_x, angle_y, 5.0)))
    return dict(
        definition="collected photon rate, each tip against its own peak; the "
                   "tilt sweep is taken at that tip's own best standoff",
        gain_vs_as_built=best["gain_vs_as_built"],
        note=f"shape only -- at its own optimum the new design collects about "
             f"{best['gain_vs_as_built']:.0f}x what the as-built tip does",
        probes=probes)


def write_figure(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0), sharex="col")
    (ax_gap, ax_tilt), (ax_gap_v, ax_tilt_v) = axes
    for probe, colour, marker in zip(report["probes"], (GREY, BLUE), ("o", "s")):
        data = probe["sweep"]
        ax_gap.plot([r["gap_um"] for r in data["gap"]],
                    [100*r["relative"] for r in data["gap"]],
                    color=colour, marker=marker, ms=3.5, lw=1.6,
                    label=probe["label"])
        ax_tilt.plot([r["angle_deg"] for r in data["angle"]],
                     [100*r["relative"] for r in data["angle"]],
                     color=colour, marker=marker, ms=3.5, lw=1.6,
                     label=f"{probe['label']}, at {probe['best_gap_um']:.0f} um")
        # the volume the green and the red actually share: participation ratio
        # of their product, so it has no threshold to jump across
        ax_gap_v.plot([r["gap_um"] for r in data["gap"]],
                      [r["effective_volume_um3"] for r in data["gap"]],
                      color=colour, marker=marker, ms=3.5, lw=1.6)
        ax_tilt_v.plot([r["angle_deg"] for r in data["angle"]],
                       [r["effective_volume_um3"] for r in data["angle"]],
                       color=colour, marker=marker, ms=3.5, lw=1.6)
    # one stacked, colour-keyed block per panel: the windows collided when they
    # were drawn against their own curves
    for axis_, lines in (
            (ax_gap, [(f"half-signal {p['gap_window_um'][0]:.1f} - "
                       f"{p['gap_window_um'][1]:.1f} um "
                       f"(best {p['best_gap_um']:.0f})") for p in report["probes"]]),
            (ax_tilt, [(f"keeps {100*p['tilt_at_2deg']:.0f}% at +/-2 deg, "
                        f"{100*p['tilt_at_5deg']:.0f}% at +/-5 deg")
                       for p in report["probes"]])):
        for index, (line, colour) in enumerate(zip(lines, (GREY, BLUE))):
            axis_.annotate(line, xy=(0.02, 0.965-0.065*index),
                           xycoords="axes fraction", fontsize=8.4,
                           color=colour, fontweight="bold")

    ax_gap.set_xscale("log")
    ax_gap.set_title("(a) distance from the diamond", loc="left", fontsize=10.5)
    ax_tilt.set_title("(b) angle against the diamond, each at its own best "
                      "standoff", loc="left", fontsize=10.5)
    ax_gap_v.set_xlabel("standoff from the diamond (um)")
    ax_tilt_v.set_xlabel("diamond tilt (deg)")
    ax_gap_v.set_title("(c) volume the green and the red share", loc="left",
                       fontsize=10.5)
    ax_tilt_v.set_title("(d) same, against tilt", loc="left", fontsize=10.5)
    for axis_ in (ax_gap, ax_tilt):
        axis_.axhline(50.0, color="#c0392b", lw=1.0, ls="--")
        axis_.set_ylabel("collected signal, % of that tip's own peak")
        axis_.set_ylim(0, 118)
        axis_.legend(fontsize=8, loc="lower center", framealpha=0.9)
    for axis_ in (ax_gap_v, ax_tilt_v):
        axis_.set_yscale("log")
        axis_.set_ylabel("green $\\cap$ red sensing volume (um$^3$)")
    for axis_ in axes.ravel():
        axis_.spines[["top", "right"]].set_visible(False)
        axis_.grid(color="#e1e0d9", lw=0.6)
        axis_.set_axisbelow(True)

    fig.text(0.01, 0.005,
             "Signal is the product of the traced 532 nm excitation and the six-core collection at every point of the 80-90 um layer, so an NV counts only where the green\n"
             "reaches it and a core can see it; both are re-traced at every standoff and every tilt. (a,b) are normalised to each tip's own peak and compare how fast a tip\n"
             f"loses light, not how much it collects -- at its own optimum the new design collects about {report['gain_vs_as_built']:.0f}x the as-built tip. Dashed line is half signal. (c,d) give the\n"
             "participation-ratio volume of that product, threshold-free so it stays smooth off-optimum; growing means a blurrier probe, not a better one. Tilt rotates\n"
             "the diamond plane about the fiber y-axis, with all six side cores traced separately.\n"
             "The few-percent ripple on the as-built curve reproduces under grid refinement, so it is the model's ray optics --\n"
             "hard apertures and no wave averaging -- rather than numerical noise.",
             fontsize=7.4, color=MUTED)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def main(coarse=False):
    os.makedirs(OUT, exist_ok=True)
    report = build(coarse)
    with open(os.path.join(OUT, "alignment_tolerance.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    write_figure(report, os.path.join(OUT, "alignment_tolerance.png"))

    for probe in report["probes"]:
        low, high = probe["gap_window_um"]
        print(f"{probe['label']}")
        print(f"   best standoff      {probe['best_gap_um']:.1f} um")
        print(f"   half-signal gap    {low:.1f} - {high:.1f} um")
        print(f"   tilt               {100*probe['tilt_at_2deg']:.0f}% at +/-2 deg, "
              f"{100*probe['tilt_at_5deg']:.0f}% at +/-5 deg")
        peak = max(probe["sweep"]["gap"],
                   key=lambda row: row["model_fiber_photons_s"])
        print(f"   at that standoff   eta {100*peak['collection_efficiency']:.5f}%, "
              f"green-red volume {peak['effective_volume_um3']:.1f} um3")
    print(f"\n{report['note']}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    import sys
    main(coarse="--coarse" in sys.argv)
