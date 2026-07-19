"""Search physical lens families and export one-side and full seven-core STLs."""
import os

from lens_design import (search_design, validate_design, write_binary_stl,
                         write_design_json, surface_limits)
from paper_figures import OUT


def main():
    os.makedirs(OUT, exist_ok=True)
    design = search_design()
    validate_design(design)
    json_path = os.path.join(OUT, "mcf_freeform_design.json")
    stl_path = os.path.join(OUT, "mcf_freeform_central_one_side.stl")
    full_stl_path = os.path.join(OUT, "mcf_freeform_full_seven_core.stl")
    write_design_json(design, json_path)
    ntri = write_binary_stl(design, stl_path)
    nfull = write_binary_stl(design, full_stl_path, all_sides=True)
    for name in ("central", "side"):
        s = design[name]; lim = surface_limits(s)
        print(f"{name}: {s['family']}, aperture={s['aperture']:.1f} um, "
              f"radial centre={s['center_r']:.1f} um, apex z={s['apex']:.1f} um, "
              f"height={lim['print_height']:.1f} um, max slope={lim['max_slope']:.3f}")
        print("  coefficients:", " ".join(f"{x:.8g}" for x in s['coef']))
    r = design['result']
    print(f"fiber={r['model_fiber_photons_s']:.6g} photons/s, "
          f"comparison-normalized={r['comparison_normalized_cps']:.6g} cps, "
          f"C={100*r['contrast']:.3f}%, "
          f"FWHM={r['fwhm_mhz']:.4g} MHz, resolution={r['resolution_um']:.4g} um, "
          f"raw sensitivity={r['raw_model_sensitivity_nt']:.4g}, "
          f"normalized sensitivity={r['comparison_normalized_sensitivity_nt']:.4g} "
          "nT/sqrt(Hz)")
    print(f"{json_path}\n{stl_path} ({ntri} triangles)\n"
          f"{full_stl_path} ({nfull} triangles)")


if __name__ == "__main__":
    main()
