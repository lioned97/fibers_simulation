"""Re-evaluate every exported design with the corrected collection ceiling.

Run ``py recompute_corrected.py``.  Rewrites figures/methods/*/summary.json and
design.json, and refreshes the top-level winner, choosing it on the corrected
objective.

Why this exists: the collection probability was built as etendue/(4 pi) times a
ray density.  That conserves etendue in the integral but bounds nothing at a
point, so a design that focused tightly could report a per-point collection
probability several times the escape cone -- more light leaving the diamond
than can leave it.  The only clamp sat at 1.0, about 28x looser than physics.
lens_design now holds every wavelength to its own escape ceiling; this script
brings the already-exported designs onto that model rather than leaving a
folder of numbers produced by the old one.

The geometry is untouched.  Only the numbers attached to it change, and the
pre-correction values are preserved under *_unclamped so nothing is lost.
"""
import json
import os
import shutil
import time

import numpy as np

from lens_design import (RED_DESIGN_NM, SEARCH_DEPTHS, SEARCH_RED_LAM,
                         SEARCH_RED_W, escape_ceiling, evaluate_design,
                         surface_limits, write_binary_stl, write_design_json)
from method_export import METHODS_DIRNAME, list_methods

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")
METHODS = os.path.join(FIGURES, METHODS_DIRNAME)


def surfaces_of(directory):
    with open(os.path.join(directory, "design.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    out = []
    for name in ("central", "side"):
        surface = dict(payload[name])
        surface["coef"] = np.asarray(surface["coef"], dtype=float)
        out.append(surface)
    return out[0], out[1], payload


def main():
    designs = list_methods(METHODS)
    if not designs:
        raise SystemExit(f"nothing to recompute in {METHODS}")
    ceiling = escape_ceiling(RED_DESIGN_NM)
    print(f"escape ceiling at {RED_DESIGN_NM:.1f} nm: {100*ceiling:.4f}%")
    print(f"recomputing {len(designs)} exported design(s) on the corrected "
          "model\n")

    header = (f"{'design':<24}{'eta before':>12}{'eta after':>11}"
              f"{'photons before':>16}{'photons after':>15}")
    print(header + "\n" + "-"*len(header))

    rebuilt = []
    for design in designs:
        directory = design["directory"]
        central, side, payload = surfaces_of(directory)
        result = evaluate_design(central, side, grid_n=81, depths=SEARCH_DEPTHS,
                                 red_lam=SEARCH_RED_LAM, red_w=SEARCH_RED_W,
                                 ray_grid=31)

        with open(os.path.join(directory, "summary.json"), encoding="utf-8") as fh:
            summary = json.load(fh)
        before_photons = summary.get("photons_s", float("nan"))

        # keep what the old model said, plainly labelled
        summary["unclamped"] = {k: summary.get(k) for k in
                                ("sensitivity_nt", "photons_s", "resolution_um",
                                 "overlap_volume_um3", "overlap_area_um2")}
        summary["unclamped"]["note"] = ("produced before the per-point escape "
                                        "ceiling was enforced; kept for "
                                        "traceability, do not quote")
        summary.update(
            sensitivity_nt=float(result["raw_model_sensitivity_nt"]),
            normalized_sensitivity_nt=float(
                result["comparison_normalized_sensitivity_nt"]),
            resolution_um=float(result["resolution_um"]),
            fwhm_mhz=float(result["fwhm_mhz"]),
            photons_s=float(result["model_fiber_photons_s"]),
            overlap_volume_um3=float(result.get("overlap_volume_um3", float("nan"))),
            overlap_area_um2=float(result.get("overlap_area_um2", float("nan"))),
            overlap_depth_um=float(result.get("overlap_depth_um", float("nan"))),
            max_saturation=float(result.get("max_saturation", float("nan"))),
            clamped_signal_fraction=float(
                result.get("clamped_signal_fraction", float("nan"))),
            central_tir_fraction=float(result.get("central_tir_fraction", float("nan"))),
            side_tir_fraction=float(result.get("side_tir_fraction", float("nan"))),
            escape_ceiling=float(ceiling),
            recomputed=time.strftime("%Y-%m-%d %H:%M:%S"),
            model_note="per-wavelength escape ceiling enforced on the "
                       "collection probability")
        with open(os.path.join(directory, "summary.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        payload["result"] = {k: v for k, v in result.items()
                             if k not in ("central_stats", "side_stats")}
        with open(os.path.join(directory, "design.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        print(f"{design['label']:<24}"
              f"{summary['unclamped']['sensitivity_nt']:>12.4g}"
              f"{summary['sensitivity_nt']:>11.4g}"
              f"{before_photons:>16.4g}{summary['photons_s']:>15.4g}")
        rebuilt.append((result["raw_model_sensitivity_nt"], design["label"],
                        central, side, result))

    # the winner is whichever design is best under the corrected objective,
    # which need not be the one the old model picked
    rebuilt.sort(key=lambda row: row[0])
    merit, label, central, side, result = rebuilt[0]
    design = dict(central=central, side=side, result=result)
    for name, target in (("design", "mcf_freeform_design.json"),):
        path = os.path.join(FIGURES, target)
        if os.path.exists(path):
            shutil.copyfile(path, path.replace(".json", "_unclamped.json"))
    write_design_json(design, os.path.join(FIGURES, "mcf_freeform_design.json"))
    write_binary_stl(design, os.path.join(FIGURES,
                                          "mcf_freeform_central_one_side.stl"))
    write_binary_stl(design, os.path.join(FIGURES,
                                          "mcf_freeform_full_seven_core.stl"),
                     all_sides=True)

    print(f"\ncorrected winner: {label} at {merit:.4g} nT/sqrt(Hz)")
    limits = {n: surface_limits(s) for n, s in (("central", central), ("side", side))}
    for name, lim in limits.items():
        print(f"  {name:<8} max slope {lim['max_slope']:.2f}, "
              f"print height {lim['print_height']:.1f} um")
    print(f"  clamped signal fraction {100*result['clamped_signal_fraction']:.1f}% "
          "- how much of the signal sits at the escape ceiling")
    print("\ntop-level design JSON and STLs refreshed; the previous ones are "
          "kept as *_unclamped.json")
    print("now safe to run:  py paper_figures.py  /  py compare_collection.py  "
          "/  py compare_probes.py")


if __name__ == "__main__":
    main()
