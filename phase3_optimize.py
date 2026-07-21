"""Search physical lens families and export one-side and full seven-core STLs."""
import os

from lens_design import (COMPARISON_NORMALIZATION, MCF_FULL_NA, design_parameters,
                         search_design, surface_limits,
                         validate_design, write_binary_stl, write_design_json)
from paper_figures import OUT


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Phase 3 ray model: uniformly filled full NA={MCF_FULL_NA:g} "
          "for the central core and all six side cores", flush=True)
    # Written after every family pair; an interrupted run resumes here instead
    # of repeating completed work.  Delete it to force a fresh search.
    checkpoint = os.path.join(OUT, "phase3_checkpoint.json")
    print(f"checkpoint: {checkpoint}", flush=True)
    design = search_design(checkpoint=checkpoint)
    validate_design(design)
    json_path = os.path.join(OUT, "mcf_freeform_design.json")
    stl_path = os.path.join(OUT, "mcf_freeform_central_one_side.stl")
    full_stl_path = os.path.join(OUT, "mcf_freeform_full_seven_core.stl")
    write_design_json(design, json_path)
    ntri = write_binary_stl(design, stl_path)
    nfull = write_binary_stl(design, full_stl_path, all_sides=True)
    from redesign_fig import (full_3d_figure, full_3d_interactive, full_na_figure,
                              overlap_volume_outputs)
    overlap = overlap_volume_outputs(design)
    full_3d_figure(design, overlap)
    full_3d_interactive(design, overlap)
    full_na_figure(design, overlap=overlap)
    p = design_parameters(design["central"], design["side"])
    print("optimized geometry:")
    for key in ("central_lens_type", "side_lens_type", "air_gap_um",
                "central_side_overlap_um", "side_side_overlap_um",
                "side_core_offset_um", "central_height_um", "side_height_um"):
        print(f"  {key}: {p[key]}")
    for name in ("central", "side"):
        s = design[name]; lim = surface_limits(s)
        print(f"{name}: {s['family']}, aperture={s['aperture']:.1f} um, "
              f"radial centre={s['center_r']:.1f} um, apex z={s['apex']:.1f} um, "
              f"height={lim['print_height']:.1f} um, max slope={lim['max_slope']:.3f}")
        print("  shape parameters:", s.get("shape_parameters", {}))
        print("  coefficients:", " ".join(f"{x:.8g}" for x in s['coef']))
    r = design['result']
    print(f"fiber={r['model_fiber_photons_s']:.6g} photons/s, "
          f"comparison-normalized={r['comparison_normalized_cps']:.6g} cps, "
          f"C={100*r['contrast']:.3f}%, "
          f"FWHM={r['fwhm_mhz']:.4g} MHz, resolution={r['resolution_um']:.4g} um, "
          f"raw sensitivity={r['raw_model_sensitivity_nt']:.4g}, "
          f"normalized sensitivity={r['comparison_normalized_sensitivity_nt']:.4g} "
          "nT/sqrt(Hz)")
    print(f"model regime: max I/I_sat={r['max_saturation']:.3g}, "
          f"collection clamped over {100*r['clamped_signal_fraction']:.2f}% "
          "of the signal")
    print(f"NOTE the normalized figures carry an empirical "
          f"{COMPARISON_NORMALIZATION['factor']:.1f}x model-to-experiment factor "
          "({}); quote the raw model sensitivity unless that factor is "
          "being discussed.".format(COMPARISON_NORMALIZATION['interpretation']))
    print(f"{json_path}\n{stl_path} ({ntri} triangles)\n"
          f"{full_stl_path} ({nfull} triangles)")


if __name__ == "__main__":
    main()
