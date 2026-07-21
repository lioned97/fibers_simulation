"""Per-method artifacts for the phase-3 lens search.

The search compares sixteen central/side lens-family pairings and previously
kept only the overall winner.  Every pairing is a real candidate design, so
each one is written to its own folder here: geometry, both STLs, a ray-trace
picture and a flat summary.  The folders are written as the search proceeds,
so an interrupted run still leaves everything it finished, and phase3_gui.py
reads them so the choice of design stays with the user rather than being
decided by one scalar.

Layout, under ``figures/methods``::

    freeform__biconic/
        design.json                 full geometry + result
        summary.json                flat numbers for the browser
        raytrace.png                meridional trace + end-face layout
        lens_one_side.stl           central + one side lens
        lens_full_seven_core.stl    complete printed tip
"""
import json
import os
import time

import numpy as np

from lens_design import (_union_boundary_arrays, design_parameters,
                         lit_radius, replicated_surfaces, surface_limits,
                         trace_full_na, write_binary_stl, write_design_json)

METHODS_DIRNAME = "methods"
SUMMARY_NAME = "summary.json"


def method_slug(label):
    """'freeform + biconic' -> 'freeform__biconic' (a safe folder name)."""
    return label.replace(" + ", "__").replace(" ", "_")


def _raytrace_figure(central, side, label, path):
    """Meridional ray trace plus the end-face cap layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    union = replicated_surfaces(central, side)
    fig, (ax, ax_face) = plt.subplots(
        1, 2, figsize=(11.0, 4.4), gridspec_kw=dict(width_ratios=[1.65, 1.0]))

    reach = max(side["center_r"]+side["aperture"], central["aperture"])*1.25
    xs = np.linspace(-reach, reach, 400)
    boundary = _union_boundary_arrays(
        union, central["base_z"], xs, np.zeros_like(xs))[0]
    finite = np.isfinite(boundary)
    ax.fill_between(xs[finite], boundary[finite], central["base_z"],
                    color="#e8d8b7", zorder=1, label="printed IP-S")
    ax.plot(xs[finite], boundary[finite], color="#875a13", lw=1.4, zorder=3)
    ax.axhspan(-100.0, 0.0, color="#cfe6f2", zorder=0, label="diamond")
    ax.axhspan(-90.0, -80.0, color="#9dc6dc", zorder=0, label="NV layer")

    for surface, lam, colour, name in (
            (central, 532.0, "#17803d", "532 nm excitation"),
            (union[1], 750.0, "#c0392b", "750 nm collection")):
        trace = trace_full_na(surface, union, lam, depths=[85.0], n_grid=25)
        near_plane = trace["valid"] & (np.abs(trace["points"][:, 1]) < 2.5)
        rays = np.flatnonzero(near_plane)[:110]
        # Shade each ray by the power it carries.  A core is launched from 13
        # points across its mode field, and drawing them equally made one core
        # look like several sources of equal strength when the outermost pair
        # carries under 0.1% of the light.
        weights = trace["weight"][rays]
        strongest = float(weights.max()) if len(weights) else 1.0
        for ray, weight in zip(rays, weights):
            share = weight/strongest if strongest > 0.0 else 0.0
            ax.plot([trace["origins"][ray, 0], trace["points"][ray, 0],
                     trace["air_surface"][ray, 0], trace["hits"][0, ray, 0]],
                    [trace["origins"][ray, 2], trace["points"][ray, 2],
                     trace["air_surface"][ray, 2], trace["hits"][0, ray, 2]],
                    color=colour, lw=0.4+0.9*share,
                    alpha=0.06+0.5*share, zorder=2)
        ax.plot([], [], color=colour, lw=1.6, label=name)

    ax.set_xlim(-reach, reach)
    ax.set_ylim(-100.0, central["base_z"]+8.0)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z from diamond surface (um)")
    ax.set_title(f"{label}  -  meridional ray trace", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    angles = np.linspace(0.0, 2*np.pi, 180)
    for k in range(6):
        centre = side["center_r"]*np.array(
            [np.cos(k*np.pi/3.0), np.sin(k*np.pi/3.0)])
        ax_face.plot(centre[0]+side["aperture"]*np.cos(angles),
                     centre[1]+side["aperture"]*np.sin(angles),
                     color="#c0392b", lw=1.2)
        lit = lit_radius(side)
        core = 35.0*np.array([np.cos(k*np.pi/3.0), np.sin(k*np.pi/3.0)])
        ax_face.plot(core[0]+lit*np.cos(angles), core[1]+lit*np.sin(angles),
                     color="#c0392b", lw=0.7, ls=":")
        ax_face.plot(*core, marker="o", color="#c0392b", ms=3.5)
    ax_face.plot(central["aperture"]*np.cos(angles),
                 central["aperture"]*np.sin(angles), color="#17803d", lw=1.2)
    ax_face.plot(lit_radius(central)*np.cos(angles),
                 lit_radius(central)*np.sin(angles),
                 color="#17803d", lw=0.7, ls=":")
    ax_face.plot(0, 0, marker="s", color="#17803d", ms=4.5)
    ax_face.set_aspect("equal")
    ax_face.set_xlabel("x (um)")
    ax_face.set_ylabel("y (um)")
    ax_face.set_title("end face: caps (solid) and lit spots (dotted)",
                      loc="left", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def export_method(label, central, side, result, root):
    """Write one family pairing's best design.  Returns its folder."""
    directory = os.path.join(root, method_slug(label))
    os.makedirs(directory, exist_ok=True)
    design = dict(central=central, side=side, result=result,
                  parameters=design_parameters(central, side))
    write_design_json(design, os.path.join(directory, "design.json"))
    write_binary_stl(design, os.path.join(directory, "lens_one_side.stl"))
    write_binary_stl(design, os.path.join(directory, "lens_full_seven_core.stl"),
                     all_sides=True)

    picture = os.path.join(directory, "raytrace.png")
    try:
        _raytrace_figure(central, side, label, picture)
    except Exception as exc:                      # a picture must never lose a design
        print(f"  ({label}: ray-trace picture failed, {type(exc).__name__}: {exc})",
              flush=True)
        picture = None

    parameters = design_parameters(central, side)
    summary = dict(
        label=label, slug=method_slug(label),
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        sensitivity_nt=float(result["raw_model_sensitivity_nt"]),
        normalized_sensitivity_nt=float(
            result["comparison_normalized_sensitivity_nt"]),
        resolution_um=float(result["resolution_um"]),
        fwhm_mhz=float(result["fwhm_mhz"]),
        photons_s=float(result["model_fiber_photons_s"]),
        # the green-and-red shared sensing volume, and its shape
        overlap_volume_um3=float(result.get("overlap_volume_um3", float("nan"))),
        overlap_area_um2=float(result.get("overlap_area_um2", float("nan"))),
        overlap_depth_um=float(result.get("overlap_depth_um", float("nan"))),
        max_saturation=float(result.get("max_saturation", float("nan"))),
        clamped_signal_fraction=float(
            result.get("clamped_signal_fraction", float("nan"))),
        central_tir_fraction=float(result.get("central_tir_fraction", float("nan"))),
        side_tir_fraction=float(result.get("side_tir_fraction", float("nan"))),
        central_max_slope=float(surface_limits(central)["max_slope"]),
        side_max_slope=float(surface_limits(side)["max_slope"]),
        parameters={k: v for k, v in parameters.items()
                    if not isinstance(v, dict)},
        raytrace_png=os.path.basename(picture) if picture else None,
        one_side_stl="lens_one_side.stl",
        full_stl="lens_full_seven_core.stl")
    with open(os.path.join(directory, SUMMARY_NAME), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return directory


def list_methods(root):
    """Every exported method under ``root``, best sensitivity first."""
    if not root or not os.path.isdir(root):
        return []
    found = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, SUMMARY_NAME)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                summary = json.load(fh)
        except (ValueError, OSError):
            continue
        summary["directory"] = os.path.join(root, name)
        found.append(summary)
    return sorted(found, key=lambda row: row.get("sensitivity_nt", float("inf")))
