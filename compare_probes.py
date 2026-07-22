"""How much better is the new printed design than the probes we already have?

Run ``py compare_probes.py``.  Writes figures/comparison/.

The honest comparison is a ratio taken inside one model.  Absolute modelled
photon rates carry a large systematic -- the empirical normalisation in
lens_design is ~11x, which is the model disagreeing with the experiment, not a
detector efficiency -- but that systematic largely cancels between two designs
evaluated by the same tracer with the same band, grid and quadrature.  So the
as-built MCF tip is rebuilt here in lens_design's own surface language and put
through the same ``evaluate_design`` as the new candidates, instead of being
compared against the legacy number in paper_figures.py that was produced by a
different tracer.

Three tiers are reported and never blended:

  measured    what the built probes actually did on the bench (sensitivity.py)
  modelled    this model's photon rate, for ratios only
  projected   measured MCF sensitivity carried across by the modelled ratio

The projection assumes the model misestimates the new tip by the same factor it
misestimates the as-built tip.  That is the whole basis of the claim, so it is
printed with the result rather than buried.
"""
import json
import os

import numpy as np

from lens_design import (MCF_IPS_N, NV_SPECTRUM_NM, RED_DESIGN_NM,
                         SEARCH_DEPTHS, SEARCH_RED_LAM, SEARCH_RED_W,
                         evaluate_design, surface_limits)
from method_export import METHODS_DIRNAME, list_methods
from sensitivity import MEASURED, calibrated_sensitivity, linewidth_from_resolution

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")
METHODS = os.path.join(FIGURES, METHODS_DIRNAME)
OUT = os.path.join(FIGURES, "comparison")

# The fabricated tip, measured directly off the supplied STLs (Cylinder.STL,
# Side lenses.STL, Central lens.STL) rather than taken from paper_figures'
# constants, which sized every cap at 17.5 um.  Each printed side lens is one
# wedge running from the central pillar at r=17.5 out to the fibre edge at
# r=63, with its high point at the inner rim; fitting sag = u^2/(2R) along the
# lobe axis gives R_radial = 39.8 um, and sweeping across the lobe at the core
# radius gives R_tangential = 1139 um.  The old 17.5 um pupil stopped at the
# core, so the outer half of each collection cone met flat polymer instead of
# lens.  Heights are unchanged: the STL puts the side vertex at z = gap+35.23
# and the central vertex at z = gap, exactly as the pillar heights say.
AS_BUILT = dict(core_pitch=35.0, side_inner_r=17.5, side_outer_r=63.0,
                side_radius_radial=39.82, side_radius_tangential=1139.27,
                central_radius=74.91, central_aperture=17.5,
                red_lens_height=158.815, green_lens_height=194.041)
AS_BUILT_GAPS = np.arange(10.0, 210.0, 10.0)


def _cap(role, gap, aperture, centre_r, core_r, apex, base_z,
         radius_radial, radius_tangential, vertex_at_inner_rim=False):
    """One printed cap as a lens_design surface.

    sag = u^2/(2*Rr) + v^2/(2*Rt) in the cap's own frame, and lens_design
    writes sag as a*(p0*x + p1*x^2/2 + p2*y^2/2) with x = u/a, so p1 = a/Rr,
    p2 = a/Rt.

    The printed side wedge is not centred on its vertex -- it climbs from the
    fibre edge to a high point at the inner rim -- so its sag is written about
    x = -1.  d(sag)/dx = p0 + p1*x vanishes there when p0 = p1, and the shift
    puts the vertex back at zero sag so ``apex`` still means the lowest point
    of the cap, which is what the clearance checks assume.
    """
    coef = np.zeros(8)
    coef[1] = aperture/radius_radial
    coef[2] = aperture/radius_tangential
    shift = 0.0
    if vertex_at_inner_rim:
        coef[0] = coef[1]
        shift = 0.5*aperture*coef[1]
    return dict(family="as_built", role=role, center_r=float(centre_r),
                angle=0.0, core_r=float(core_r), apex=float(apex),
                base_z=float(base_z), aperture=float(aperture),
                coef=coef, shift=float(shift))


def as_built_surfaces(gap):
    """The fabricated seven-core tip, standing ``gap`` above the diamond.

    The side caps sit higher than the central one by the difference in printed
    pillar height, and each spans r = 17.5 to 63 um with its vertex on the
    inner rim, so the core at r = 35 looks through the middle of its own lens
    and the surface tilt steers that cone toward the axis -- which is what the
    printed wedge does in the real device.
    """
    base_z = gap + AS_BUILT["green_lens_height"]
    side_offset = AS_BUILT["green_lens_height"] - AS_BUILT["red_lens_height"]
    inner, outer = AS_BUILT["side_inner_r"], AS_BUILT["side_outer_r"]
    central = _cap("central", gap, AS_BUILT["central_aperture"], 0.0, 0.0,
                   gap, base_z, AS_BUILT["central_radius"],
                   AS_BUILT["central_radius"])
    side = _cap("side", gap, 0.5*(outer-inner), 0.5*(outer+inner),
                AS_BUILT["core_pitch"], gap+side_offset, base_z,
                AS_BUILT["side_radius_radial"], AS_BUILT["side_radius_tangential"],
                vertex_at_inner_rim=True)
    return central, side


def evaluate(central, side, coarse=False):
    """Same tracer, band, depths and quadrature for every probe compared."""
    if coarse:
        return evaluate_design(central, side, grid_n=41,
                               depths=np.linspace(*(80.0, 90.0), 3),
                               red_lam=np.array([RED_DESIGN_NM]),
                               red_w=np.array([1.0]), ray_grid=17)
    # 241, not the search's 81: see compare_collection._quadrature -- a sub-2-um
    # sensing spot is not resolved on a ~1 um mesh, and the error lands on the
    # ratio because only the new design's spot is that small.
    return evaluate_design(central, side, grid_n=241, depths=SEARCH_DEPTHS,
                           red_lam=SEARCH_RED_LAM, red_w=SEARCH_RED_W,
                           ray_grid=31)


def as_built_reference(coarse=False, gaps=AS_BUILT_GAPS):
    """Best-case as-built MCF: its own optimal standoff, not an arbitrary one.

    Handing the baseline its best gap keeps the comparison from flattering the
    new design by holding the old one at a spacing it was never used at.
    """
    best = None
    for gap in gaps:
        central, side = as_built_surfaces(float(gap))
        try:
            result = evaluate(central, side, coarse)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or result["model_fiber_photons_s"] > best[1]["model_fiber_photons_s"]:
            best = (float(gap), result)
    if best is None:
        raise RuntimeError("the as-built tip produced no valid evaluation")
    return best


def project(photon_ratio, resolution_um, anchor="MCF"):
    """Carry a measured sensitivity across by a modelled photon ratio.

    photon_ratio is new/anchor from this model.  Sensitivity scales as the
    linewidth over the root of the photon rate at fixed contrast, which is what
    calibrated_sensitivity encodes against the measured MCF.
    """
    reference = MEASURED[anchor]
    fwhm = linewidth_from_resolution(resolution_um)
    return calibrated_sensitivity(reference["cps"]*photon_ratio,
                                  reference["contrast"], fwhm)


def build_report(coarse=False):
    designs = list_methods(METHODS)
    if not designs:
        raise SystemExit(f"no exported designs yet in {METHODS}")

    gap, reference = as_built_reference(coarse)
    reference_photons = reference["model_fiber_photons_s"]

    rows = []
    for design in designs:
        ratio = design["photons_s"]/reference_photons
        rows.append(dict(
            label=design["label"],
            photons_s=design["photons_s"],
            resolution_um=design["resolution_um"],
            overlap_volume_um3=design.get("overlap_volume_um3", float("nan")),
            photon_gain_vs_as_built=ratio,
            projected_nt=project(ratio, design["resolution_um"]),
        ))
    rows.sort(key=lambda row: row["projected_nt"])

    report = dict(
        model=dict(band_nm=list(NV_SPECTRUM_NM), design_nm=RED_DESIGN_NM,
                   ips_index=MCF_IPS_N, grid="coarse" if coarse else "search"),
        as_built_reference=dict(
            gap_um=gap, model_photons_s=reference_photons,
            model_resolution_um=reference["resolution_um"],
            note="rebuilt from the fabricated STL geometry and run through the "
                 "same evaluate_design as the candidates, so the ratios below "
                 "are within one model"),
        measured=MEASURED,
        designs=rows,
        assumption="projected sensitivities assume this model misestimates a "
                   "new tip by the same factor it misestimates the as-built "
                   "tip; only the ratio is modelled, the anchor is measured")
    return report


def write_figure(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = report["designs"][0]
    names = ["SM\n(measured)", "MCF as-built\n(measured)", "MM\n(measured)",
             f"new: {best['label']}\n(projected)"]
    values = [MEASURED["SM"]["sensitivity_nt"], MEASURED["MCF"]["sensitivity_nt"],
              MEASURED["MM"]["sensitivity_nt"], best["projected_nt"]]
    colours = ["#8a94a1", "#8a94a1", "#8a94a1", "#2f5fc4"]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bars = ax.bar(names, values, color=colours, width=0.62)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.3g}", (bar.get_x()+bar.get_width()/2, value),
                    ha="center", va="bottom", fontsize=9.5)
    ax.set_ylabel("magnetic sensitivity (nT/$\\sqrt{Hz}$)  -  lower is better")
    ax.set_yscale("log")
    ax.set_title("Built probes (measured) vs the new printed tip (projected)",
                 loc="left", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01,
             "Projection carries the measured MCF sensitivity across by the "
             "modelled photon ratio;\nonly that ratio is modelled, and both "
             "sides of it come from the same tracer.",
             fontsize=7.4, color="#52514e")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def main(coarse=False):
    os.makedirs(OUT, exist_ok=True)
    report = build_report(coarse)
    with open(os.path.join(OUT, "comparison.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    write_figure(report, os.path.join(OUT, "sensitivity_comparison.png"))

    reference = report["as_built_reference"]
    print(f"model: {NV_SPECTRUM_NM[0]:.0f}-{NV_SPECTRUM_NM[1]:.0f} nm, "
          f"design {RED_DESIGN_NM:.1f} nm, "
          f"{'coarse' if coarse else 'search'} grid")
    print(f"as-built MCF rebuilt in this model: best gap {reference['gap_um']:.0f} um, "
          f"{reference['model_photons_s']:.4g} photons/s\n")
    header = (f"{'design':<24}{'photons/s':>12}{'x as-built':>12}"
              f"{'res um':>9}{'proj nT':>10}{'vs MM':>9}")
    print(header + "\n" + "-"*len(header))
    measured_mm = MEASURED["MM"]["sensitivity_nt"]
    for row in report["designs"]:
        print(f"{row['label']:<24}{row['photons_s']:>12.4g}"
              f"{row['photon_gain_vs_as_built']:>12.1f}"
              f"{row['resolution_um']:>9.3f}{row['projected_nt']:>10.3f}"
              f"{measured_mm/row['projected_nt']:>8.1f}x")
    print(f"\nmeasured on the bench: SM {MEASURED['SM']['sensitivity_nt']:.0f}, "
          f"MCF {MEASURED['MCF']['sensitivity_nt']:.0f}, "
          f"MM {MEASURED['MM']['sensitivity_nt']:.0f} nT/sqrt(Hz)")
    print(report["assumption"])
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    import sys
    main(coarse="--coarse" in sys.argv)
