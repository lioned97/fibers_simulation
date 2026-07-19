"""Numerical and Snell-law checks for the final MCF lens design."""
import json
import os

import numpy as np

from lens_design import (DEPTHS, MCF_IPS_N, N_AIR, RED_LAM, RED_W,
                         evaluate_design, replicated_surfaces, trace_mode)
from paper_figures import OUT
from physics import diamond_sellmeier, nv_emission_spectrum


def load_surfaces():
    with open(os.path.join(OUT, "mcf_freeform_design.json"), encoding="utf-8") as fh:
        raw = json.load(fh)
    for name in ("central", "side"):
        raw[name]["coef"] = np.asarray(raw[name]["coef"], dtype=float)
    return raw["central"], raw["side"]


def spectral_rule(points):
    wavelength = np.linspace(650.0, 850.0, points)
    weight = nv_emission_spectrum(wavelength)
    weight[[0, -1]] *= 0.5
    return wavelength, weight/weight.sum()


def compact(result):
    keys = ("model_fiber_photons_s", "comparison_normalized_cps",
            "resolution_um", "raw_model_sensitivity_nt",
            "comparison_normalized_sensitivity_nt")
    return {key: result[key] for key in keys}


def snell_residual(trace, n1, n2, normal):
    valid = trace["valid"]
    incident = trace["incident"] if n1 == MCF_IPS_N else trace["air"]
    outgoing = trace["air"] if n1 == MCF_IPS_N else trace["diamond"]
    normal = np.broadcast_to(normal, incident.shape)
    ci = np.abs(np.sum(incident*normal, axis=1))
    ct = np.abs(np.sum(outgoing*normal, axis=1))
    residual = np.abs(n1*np.sqrt(np.maximum(0.0, 1.0-ci*ci)) -
                      n2*np.sqrt(np.maximum(0.0, 1.0-ct*ct)))
    return float(np.max(residual[valid]))


def main():
    central, side = load_surfaces()
    lam17, weight17 = spectral_rule(17)
    depth17 = np.linspace(80.0, 90.0, 17)
    cases = {
        "final_33_depth_33_spectrum_grid161": evaluate_design(
            central, side, grid_n=161),
        "grid121": evaluate_design(central, side, grid_n=121),
        "depth17": evaluate_design(central, side, grid_n=121, depths=depth17),
        "spectrum17": evaluate_design(central, side, grid_n=121,
                                      red_lam=lam17, red_w=weight17),
    }
    final_rate = cases["final_33_depth_33_spectrum_grid161"]["model_fiber_photons_s"]
    payload = {
        "cases": {name: compact(result) for name, result in cases.items()},
        "relative_photon_difference_from_final": {
            name: result["model_fiber_photons_s"]/final_rate-1.0
            for name, result in cases.items()
        },
    }

    union = replicated_surfaces(central, side)
    snell = {}
    for name, surface, wavelength in (("central_532_nm", central, 532.0),
                                      ("side_750_nm", side, 750.0)):
        trace = trace_mode(surface, union, wavelength, depths=[85.0], n_grid=41)
        snell[name] = {
            "polymer_air_max_n_sin_theta_residual": snell_residual(
                trace, MCF_IPS_N, N_AIR, trace["surface_normal"]),
            "air_diamond_max_n_sin_theta_residual": snell_residual(
                trace, N_AIR, float(diamond_sellmeier(wavelength/1000.0)),
                trace["diamond_normal"]),
        }
    payload["snell_law"] = snell
    payload["quadrature"] = {
        "depth_points": len(DEPTHS), "spectral_points": len(RED_LAM),
        "spectral_weight_sum": float(RED_W.sum()),
        "rule": "normalized trapezoidal",
    }
    path = os.path.join(OUT, "mcf_numerical_validation.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    assert max(abs(x) for x in payload["relative_photon_difference_from_final"].values()) < 0.01
    assert max(value for row in snell.values() for value in row.values()) < 1e-12
    print(path)
    print(json.dumps(payload["relative_photon_difference_from_final"], indent=2))
    print(json.dumps(snell, indent=2))


if __name__ == "__main__":
    main()
