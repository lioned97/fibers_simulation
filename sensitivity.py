"""Experiment-calibrated shot-noise sensitivity for the MCF lens redesign."""
import numpy as np

MEASURED = {
    "SM": dict(od=3.0, observed_kcps=2000.0, cps=2.0e9,
               snr=1.83, contrast=0.0659, fwhm_mhz=3.69, sensitivity_nt=100.0),
    "MM": dict(od=4.0, observed_kcps=2000.0, cps=2.0e10,
               snr=4.20, contrast=0.0693, fwhm_mhz=4.52, sensitivity_nt=29.0),
    "MCF": dict(od=3.0, observed_kcps=2000.0, cps=2.0e9,
                snr=1.98, contrast=0.0810, fwhm_mhz=3.44, sensitivity_nt=103.0),
}


def corrected_cps(observed_kcps, optical_density):
    return float(observed_kcps)*1e3*10.0**float(optical_density)


def linewidth_from_resolution(resolution_um):
    """Resolution/MW model fitted to the supplied SM/MM/MCF measurements."""
    resolution = np.array([3.5, 49.0, 1.5])
    linewidth = np.array([3.69, 4.52, 3.44])
    a, b = np.linalg.lstsq(np.column_stack([np.ones(3), resolution**2]),
                           linewidth**2, rcond=None)[0]
    return float(np.sqrt(max(0.0, a+b*float(resolution_um)**2)))


def calibrated_sensitivity(cps, contrast, fwhm_mhz):
    """nT/sqrt(Hz), preserving the measured MCF duty cycle and detector chain."""
    ref = MEASURED["MCF"]
    return (ref["sensitivity_nt"]*float(fwhm_mhz)/ref["fwhm_mhz"]
            * ref["contrast"]/float(contrast)*np.sqrt(ref["cps"]/float(cps)))


def design_metrics(cps, resolution_um, contrast=MEASURED["MCF"]["contrast"]):
    fwhm = linewidth_from_resolution(resolution_um)
    return dict(cps=float(cps), contrast=float(contrast), fwhm_mhz=fwhm,
                sensitivity_nt=calibrated_sensitivity(cps, contrast, fwhm))


if __name__ == "__main__":
    assert corrected_cps(2000, 3) == 2e9
    assert corrected_cps(2000, 4) == 2e10
    assert abs(calibrated_sensitivity(2e9, 0.081, 3.44)-103.0) < 1e-12
    print("PASS: ND correction and MCF sensitivity calibration")
