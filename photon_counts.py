"""How many photons each probe makes, and how many it catches.

Run ``py photon_counts.py``.  Writes figures/comparison_collection/photon_counts.*

Collection efficiency alone hides half the story: a probe that pumps a large
volume weakly and one that pumps a small volume hard can share an efficiency and
differ by orders of magnitude in counts.  So this reports both ends of the
budget for every probe -- what the 532 nm delivery actually creates, and what
comes back down a core:

    created    integral of RHO * R_exc over the 80-90 um layer, photons/s into
               4 pi, where R_exc = R_SAT * s/(1+s) and s = I_green/I_SAT, so the
               green intensity per unit volume sets it and it saturates
    collected  the same integral weighted by the collection probability
    eta        collected / created

Resolution is the second-moment FWHM of the collected signal.  It is driven by
the green: an NV only signals where it is pumped, so the excitation spot sets
the size of the sensing region unless the collection optics are tighter still.
Both are reported against the green spot so that can be read off directly.

SM and MM are bare fibers taken from the characterisation model in
paper_figures.py -- a different tracer -- and are marked as such.  Only the two
MCF tips are a like-for-like ratio.
"""
import json
import os

import numpy as np

from compare_collection import OUT, _surfaces_of, multimode_reference
from compare_probes import as_built_surfaces
from lens_design import (P_GREEN_MW, SEARCH_DEPTHS, SEARCH_RED_LAM,
                         SEARCH_RED_W, evaluate_design)
from method_export import METHODS_DIRNAME, headline_label, list_methods

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")
METHODS = os.path.join(FIGURES, METHODS_DIRNAME)
GRID_N = 241          # converged; see compare_collection._quadrature
# A Gaussian's 1/e^2 diameter is FWHM/sqrt(2 ln 2)... per radius, so 2/1.1774
# across.  beam_stats reports FWHM, exc_width reports a 1/e^2 radius; without
# this the two tracers' spot sizes would be quietly on different conventions.
FWHM_TO_1E2_DIAMETER = 2.0/np.sqrt(2.0*np.log(2.0))

GREY, BLUE, GREEN, MUTED = "#8a94a1", "#2f5fc4", "#17803d", "#52514e"


def mcf_row(label, central, side, note=None):
    """One printed tip, through the design tracer."""
    result = evaluate_design(central, side, grid_n=GRID_N, depths=SEARCH_DEPTHS,
                             red_lam=SEARCH_RED_LAM, red_w=SEARCH_RED_W,
                             ray_grid=31)
    middle = result["central_stats"][len(result["central_stats"])//2]
    return dict(label=label, note=note,
                created_photons_s=result["model_excited_rate_s"],
                collected_photons_s=result["model_fiber_photons_s"],
                collection_efficiency=result["collection_efficiency"],
                green_spot_diameter_um=FWHM_TO_1E2_DIAMETER*middle["fwhm"],
                resolution_um=result["resolution_um"],
                effective_volume_um3=result["effective_volume_um3"])


def bare_row(label, name):
    """One bare fiber, through the characterisation tracer."""
    row = multimode_reference(name)
    row.update(label=label, note="characterisation model",
               effective_volume_um3=float("nan"))
    return row


def build():
    designs = list_methods(METHODS)
    if not designs:
        raise SystemExit(f"no exported designs in {METHODS}")
    chosen = headline_label(FIGURES)
    design = next(d for d in designs if d["label"] == chosen)

    with open(os.path.join(OUT, "collection_comparison.json"),
              encoding="utf-8") as fh:
        gap = json.load(fh)["as_built"]["gap_um"]

    rows = [bare_row("SM bare fiber", "SM"),
            bare_row("MM bare fiber", "MM"),
            mcf_row(f"MCF as-built ({gap:.0f} um gap)", *as_built_surfaces(gap)),
            mcf_row(f"MCF new design ({chosen})", *_surfaces_of(design))]
    reference = rows[2]["collected_photons_s"]
    for row in rows:
        row["vs_as_built"] = row["collected_photons_s"]/max(reference, 1e-30)
    return dict(
        green_power_mw=P_GREEN_MW,
        definition="created = integral of RHO*R_exc over the 80-90 um layer "
                   "(photons/s into 4pi, saturated in the green intensity); "
                   "collected = the same integral weighted by collection "
                   "probability; resolution = second-moment FWHM of the "
                   "collected signal",
        caveat="SM and MM come from a different tracer; only the two MCF tips "
               "are a like-for-like ratio",
        rows=rows)


def write_figure(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]
    names = [r["label"].split(" (")[0].replace(" ", "\n", 1) for r in rows]
    hatch = ["///" if r["note"] else None for r in rows]
    colours = [GREY, GREY, GREY, BLUE]
    positions = np.arange(len(rows))

    fig, (ax, ax_eta, ax_res) = plt.subplots(1, 3, figsize=(14.6, 4.6))

    width = 0.38
    created = [r["created_photons_s"] for r in rows]
    collected = [r["collected_photons_s"] for r in rows]
    for offset, values, colour, label in ((-width/2, created, GREEN, "created by the green"),
                                          (+width/2, collected, BLUE, "collected")):
        bars = ax.bar(positions+offset, values, width, color=colour, label=label)
        for bar, value, mark in zip(bars, values, hatch):
            if mark:
                bar.set_hatch(mark)
                bar.set_edgecolor("white")
            ax.annotate(f"{value:.1e}", (bar.get_x()+bar.get_width()/2, value),
                        ha="center", va="bottom", fontsize=7.2, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(top=max(created)*60)
    ax.set_ylabel("photons/s over the NV layer")
    ax.set_title(f"(a) made vs caught, {report['green_power_mw']:.0f} mW green",
                 loc="left", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")

    values = [100*r["collection_efficiency"] for r in rows]
    bars = ax_eta.bar(positions, values, 0.6, color=colours)
    for bar, value, row, mark in zip(bars, values, rows, hatch):
        if mark:
            bar.set_hatch(mark)
            bar.set_edgecolor("white")
        ax_eta.annotate(f"{value:.4g}%\nx{row['vs_as_built']:.3g} counts",
                        (bar.get_x()+bar.get_width()/2, value), ha="center",
                        va="bottom", fontsize=8)
    ax_eta.set_yscale("log")
    ax_eta.set_ylim(top=max(values)*30)
    ax_eta.set_ylabel("collection efficiency (%)")
    ax_eta.set_title("(b) efficiency, and counts against the as-built tip",
                     loc="left", fontsize=10.5)

    for row, colour, marker in zip(rows, colours, ("o", "s", "^", "D")):
        ax_res.plot(row["green_spot_diameter_um"], row["resolution_um"],
                    marker=marker, ms=9, color=colour, ls="none",
                    label=row["label"].split(" (")[0])
        ax_res.annotate(f"  {row['resolution_um']:.2f} um",
                        (row["green_spot_diameter_um"], row["resolution_um"]),
                        fontsize=8, color=colour, va="center")
    span = np.array([0.7*min(r["green_spot_diameter_um"] for r in rows),
                     1.4*max(r["green_spot_diameter_um"] for r in rows)])
    ax_res.plot(span, span, color=MUTED, lw=1.0, ls="--")
    ax_res.annotate("resolution = green spot", (span[1], span[1]), fontsize=7.6,
                    color=MUTED, ha="right", va="bottom")
    ax_res.set_xscale("log")
    ax_res.set_yscale("log")
    ax_res.set_xlabel("green spot at the NV layer, 1/e$^2$ diameter (um)")
    ax_res.set_ylabel("resolution, FWHM of the collected signal (um)")
    ax_res.set_title("(c) the green spot sets the resolution", loc="left",
                     fontsize=10.5)
    ax_res.legend(fontsize=8, loc="upper left")

    for axis_ in (ax, ax_eta):
        axis_.set_xticks(positions)
        axis_.set_xticklabels(names, fontsize=8.5)
    for axis_ in (ax, ax_eta, ax_res):
        axis_.spines[["top", "right"]].set_visible(False)
        axis_.grid(axis="y", color="#e1e0d9", lw=0.6)
        axis_.set_axisbelow(True)

    fig.text(0.01, 0.01,
             "Created counts only NVs the green actually reaches, saturated: R_exc = R_SAT*s/(1+s) with s = I_green/I_SAT, integrated over the 80-90 um layer. "
             "Hatched bars\n(SM, MM) are bare fibers from the characterisation model, a different tracer, so only the two MCF tips are a like-for-like ratio. "
             "In (c) a probe on the dashed\nline is limited by its own excitation spot; below it the collection optics are tighter than the green and the resolution "
             "is set by the overlap of the two.",
             fontsize=7.4, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    report = build()
    with open(os.path.join(OUT, "photon_counts.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    write_figure(report, os.path.join(OUT, "photon_counts.png"))

    header = (f"{'probe':<34}{'created 1/s':>13}{'collected 1/s':>15}"
              f"{'eta %':>10}{'x counts':>10}{'green um':>10}{'res um':>9}")
    print(f"green delivered: {report['green_power_mw']:.0f} mW\n")
    print(header + "\n" + "-"*len(header))
    for row in report["rows"]:
        print(f"{row['label']:<34}{row['created_photons_s']:>13.3e}"
              f"{row['collected_photons_s']:>15.3e}"
              f"{100*row['collection_efficiency']:>10.5f}"
              f"{row['vs_as_built']:>10.3g}"
              f"{row['green_spot_diameter_um']:>10.2f}"
              f"{row['resolution_um']:>9.2f}")
    print(f"\n{report['caveat']}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
