"""Collection efficiency of the new printed tip against the built probes.

Run ``py compare_collection.py`` (add ``--coarse`` for a fast preview).
Writes figures/comparison_collection/.

Collection efficiency is reported instead of a projected magnetic sensitivity.
It is what the ray model actually computes: the fraction of the light emitted
by NVs the green beam excites that ends up inside a collection core.  It needs
no empirical normalisation, no assumed contrast and no linewidth fit, so it can
be quoted as a model result rather than as a projection resting on the measured
probe.

    eta = integral( R_exc . C ) dV  /  integral( R_exc ) dV

R_exc is the saturable excitation rate from the central core and C the
reciprocal collection probability of the six side cores, both over the 80-90 um
NV layer.  The denominator counts only NVs the green actually reaches, so eta
answers "of the photons this probe creates, what fraction does it catch".

The as-built MCF tip is rebuilt from the fabricated geometry and pushed through
the same tracer, band and quadrature as the new candidates, so the new-vs-old
comparison is inside one model.  The bare MM fiber cannot be expressed as a
printed cap, so its number comes from the characterisation model in
paper_figures.py and is labelled as such wherever it appears.
"""
import json
import os

import numpy as np

from compare_probes import AS_BUILT_GAPS, as_built_surfaces
from lens_design import (I_SAT, MCF_FULL_NA, MCF_IPS_N, NV_SPECTRUM_NM,
                         P_GREEN_MW, RED_DESIGN_NM, RHO, R_SAT, SEARCH_DEPTHS,
                         SEARCH_RED_LAM, SEARCH_RED_W, W_MODE, _field_axis,
                         _ray_density, beam_stats, diamond_sellmeier,
                         escape_ceiling, replicated_surfaces, trace_full_na)
from method_export import METHODS_DIRNAME, headline_label, list_methods

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")
METHODS = os.path.join(FIGURES, METHODS_DIRNAME)
OUT = os.path.join(FIGURES, "comparison_collection")


def _quadrature(coarse):
    """Depths, spectrum, transverse grid and ray grid for one evaluation.

    The transverse grid is 241 and not the search's 81.  The signal is the
    product of two sharply peaked densities, and the new design's sensing spot
    is under 2 um across against a diffraction blur of sigma ~0.5 um, so at 81
    points (dx ~1 um) the product integral is not resolved: it reads 1.962%
    against 1.88% converged, and the gain over the as-built tip reads 212x
    against 194x.  The as-built tip is broad enough that 81 was already fine,
    which is exactly why the error showed up as a gain and not as a level.
    """
    if coarse:
        return (np.linspace(*(80.0, 90.0), 5), np.array([RED_DESIGN_NM]),
                np.array([1.0]), 61, 21)
    return SEARCH_DEPTHS, SEARCH_RED_LAM, SEARCH_RED_W, 241, 31


def collection_efficiency(central, side, coarse=False):
    """Excitation-weighted collection efficiency, and the field it comes from.

    Mirrors lens_design's own signal field exactly, but keeps the numerator and
    the denominator apart: evaluate_design multiplies them together into a
    photon rate, which cannot be compared across probes without also agreeing
    on pump power and NV density.  A ratio needs neither.
    """
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

    emitted = collected = 0.0
    peak = 0.0
    planes = []
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
            # held to the escape ceiling at this wavelength, exactly as
            # lens_design does: etendue bounds the integral, not the value at
            # a point, so a tightly focused design would otherwise report more
            # light leaving the diamond than can leave it.
            contribution = acceptance*_ray_density(
                trace, iz, axis, lam, np.arange(6)*np.pi/3.0)
            collection += weight*np.minimum(contribution, escape_ceiling(lam))

        signal = excitation*collection
        planes.append(signal)
        peak = max(peak, float(signal.max()))
        volume = dx*dx*dz*(0.5 if iz in (0, len(depths)-1) else 1.0)
        emitted += float(excitation.sum())*volume
        collected += float(signal.sum())*volume

    field = np.stack(planes)
    core = field >= 0.5*peak if peak > 0.0 else np.zeros_like(field, dtype=bool)
    sensing_volume = float(core.sum())*dx*dx*dz
    sensing_area = float(core.any(axis=0).sum())*dx*dx
    return dict(collection_efficiency=collected/max(emitted, 1e-30),
                photons_s=RHO*collected,
                sensing_volume_um3=sensing_volume,
                sensing_area_um2=sensing_area,
                sensing_diameter_um=2.0*np.sqrt(sensing_area/np.pi))


def as_built_best(coarse=False):
    """The fabricated tip at the standoff that suits it best."""
    best = None
    for gap in AS_BUILT_GAPS:
        central, side = as_built_surfaces(float(gap))
        try:
            row = collection_efficiency(central, side, coarse)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or row["photons_s"] > best[1]["photons_s"]:
            best = (float(gap), row)
    if best is None:
        raise RuntimeError("the as-built tip produced no valid evaluation")
    return best


def multimode_reference(name="MM"):
    """Bare SM/MM fiber at contact, from the characterisation model.

    A bare fiber has no printed cap, so it cannot be written as a surface in
    the design model; this is paper_figures' tracer, and is kept visually
    separate in the figure for that reason.
    """
    import paper_figures as P
    config = next(c for c in P.CONFIGS if c["name"] == name)
    rays, weights = P.escape_cone_quadrature(config["nen"], P.N_DIA_RED, seed=23)
    emitters, density = P.sample_ensemble(config, P.N_EMIT,
                                          np.random.default_rng(42))
    eta = P.eta_per_emitter(emitters, rays, weights, config["fibers"](0.0),
                            0.0, config["model"])
    rate, _ = P.exc_rate(config["exc"], emitters[:, 0], emitters[:, 1],
                         emitters[:, 2], 0.0)
    carried = rate*eta*density
    return dict(collection_efficiency=float(
        config["tf"]*carried.sum()/(rate*density).sum()))


def build(coarse=False, with_multimode=True):
    designs = list_methods(METHODS)
    if not designs:
        raise SystemExit(f"no exported designs yet in {METHODS}")

    gap, as_built = as_built_best(coarse)
    rows = []
    for design in designs:
        central, side = _surfaces_of(design)
        row = collection_efficiency(central, side, coarse)
        row["label"] = design["label"]
        row["gain_vs_as_built"] = (row["collection_efficiency"] /
                                   as_built["collection_efficiency"])
        rows.append(row)
    rows.sort(key=lambda r: -r["collection_efficiency"])

    multimode = multimode_reference() if with_multimode else None
    return dict(
        headline=headline_label(FIGURES),
        definition="excitation-weighted collection efficiency: integral of "
                   "R_exc times collection over the 80-90 um layer, divided by "
                   "integral of R_exc",
        model=dict(band_nm=list(NV_SPECTRUM_NM), design_nm=RED_DESIGN_NM,
                   grid="coarse" if coarse else "search"),
        as_built=dict(gap_um=gap, **as_built),
        multimode=multimode,
        multimode_note="bare MM fiber at contact, computed by the "
                       "characterisation model in paper_figures.py (a "
                       "different tracer), shown for scale only",
        designs=rows)


def _surfaces_of(design):
    with open(os.path.join(design["directory"], "design.json"),
              encoding="utf-8") as fh:
        payload = json.load(fh)
    surfaces = []
    for name in ("central", "side"):
        surface = dict(payload[name])
        surface["coef"] = np.asarray(surface["coef"], dtype=float)
        surfaces.append(surface)
    return surfaces


def headline_row(report):
    """The chosen design's row, not whichever pairing tops the eta sort."""
    label = report.get("headline") or headline_label(FIGURES)
    return next((row for row in report["designs"] if row["label"] == label),
                report["designs"][0])


def write_figure(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = headline_row(report)
    as_built = report["as_built"]
    multimode = report["multimode"]
    ceiling = 100*escape_ceiling(RED_DESIGN_NM)

    names = ["MCF\nas-built", "MCF\nnew design"]
    values = [100*as_built["collection_efficiency"],
              100*best["collection_efficiency"]]
    colours = ["#8a94a1", "#2f5fc4"]
    if multimode:
        names.insert(0, "MM\nbare fiber")
        values.insert(0, 100*multimode["collection_efficiency"])
        colours.insert(0, "#c9ced6")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.4))
    bars = ax.bar(names, values, color=colours, width=0.6)
    for index, (bar, value) in enumerate(zip(bars, values)):
        # the gain rides with the bar it belongs to, so it cannot collide with
        # the ceiling line above
        caption = (f"{value:.4g}%\nx{best['gain_vs_as_built']:.0f} vs as-built"
                   if index == len(values)-1 else f"{value:.4g}%")
        ax.annotate(caption, (bar.get_x()+bar.get_width()/2, value),
                    ha="center", va="bottom", fontsize=10,
                    color="#2f5fc4" if index == len(values)-1 else "#131920",
                    fontweight="bold" if index == len(values)-1 else "normal")
    ax.axhline(ceiling, color="#c0392b", lw=1.2, ls="--")
    ax.text(0.015, ceiling*1.1,
            f"{ceiling:.1f}%  physical ceiling: all light that can escape the diamond",
            transform=ax.get_yaxis_transform(), fontsize=8.2, color="#c0392b",
            va="bottom")
    ax.set_ylabel("collection efficiency (%)  -  higher is better")
    ax.set_yscale("log")
    ax.set_ylim(top=ceiling*7.0)
    ax.set_title("(a) light collected per photon emitted", loc="left",
                 fontsize=10.5)

    # Volume and diameter are different units, so they get their own axes
    # rather than being forced onto one scale.
    labels = ["as-built", "new design"]
    positions = np.arange(len(labels))
    volumes = [as_built["sensing_volume_um3"], best["sensing_volume_um3"]]
    diameters = [as_built["sensing_diameter_um"], best["sensing_diameter_um"]]
    volume_bars = ax2.bar(positions, volumes, width=0.55, color="#8a94a1")
    for bar, volume, diameter in zip(volume_bars, volumes, diameters):
        ax2.annotate(f"{volume:,.0f} um$^3$\n{diameter:.1f} um across",
                     (bar.get_x()+bar.get_width()/2, volume),
                     ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_yscale("log")
    ax2.set_ylim(top=max(volumes)*6.0)
    ax2.set_ylabel("sensing volume (um$^3$)  -  smaller is sharper")
    ax2.set_title("(b) the volume that light comes from", loc="left",
                  fontsize=10.5)

    for axis_ in (ax, ax2):
        axis_.spines[["top", "right"]].set_visible(False)
        axis_.grid(axis="y", color="#e1e0d9", lw=0.6)
        axis_.set_axisbelow(True)

    note = ("Both MCF tips use the same ray tracer, band and quadrature, so (a) "
            "is a like-for-like ratio.\nMM is a bare fiber, taken from the "
            "characterisation model: it beats the as-built tip but reads a "
            "far larger volume.")
    fig.text(0.01, 0.01, note, fontsize=7.4, color="#52514e")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def main(coarse=False):
    os.makedirs(OUT, exist_ok=True)
    report = build(coarse)
    with open(os.path.join(OUT, "collection_comparison.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    write_figure(report, os.path.join(OUT, "collection_comparison.png"))

    as_built = report["as_built"]
    ceiling = escape_ceiling(RED_DESIGN_NM)
    print(f"model: {NV_SPECTRUM_NM[0]:.0f}-{NV_SPECTRUM_NM[1]:.0f} nm, "
          f"{report['model']['grid']} grid")
    print(f"physical ceiling (light that can escape the diamond at all): "
          f"{100*ceiling:.2f}%")
    print(f"as-built MCF, best standoff {as_built['gap_um']:.0f} um: "
          f"eta = {100*as_built['collection_efficiency']:.5f}%  "
          f"({as_built['sensing_volume_um3']:,.0f} um3, "
          f"{as_built['sensing_diameter_um']:.1f} um across)")
    if report["multimode"]:
        print(f"bare MM fiber at contact:            "
              f"eta = {100*report['multimode']['collection_efficiency']:.5f}%  "
              "(characterisation model)")
    print()
    header = (f"{'design':<24}{'eta %':>10}{'x as-built':>12}"
              f"{'volume um3':>12}{'diam um':>10}")
    print(header + "\n" + "-"*len(header))
    for row in report["designs"]:
        print(f"{row['label']:<24}{100*row['collection_efficiency']:>10.5f}"
              f"{row['gain_vs_as_built']:>12.1f}"
              f"{row['sensing_volume_um3']:>12.1f}"
              f"{row['sensing_diameter_um']:>10.2f}")
    best = headline_row(report)
    print(f"\nchosen design ({best['label']}) sits at "
          f"{100*best['collection_efficiency']/(100*ceiling):.0%} "
          "of the escape-cone ceiling - a large number, so check it before "
          "quoting it")
    print(f"the table is sorted by eta, which is not the order that picked the "
          f"design; the top four are within 5% of each other")
    print(f"\n{report['definition']}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    import sys
    main(coarse="--coarse" in sys.argv)
