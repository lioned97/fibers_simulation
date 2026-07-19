"""Small assert-based check for experiment-calibrated sensitivity."""
from sensitivity import (MEASURED, calibrated_sensitivity, corrected_cps,
                         linewidth_from_resolution)


if __name__ == '__main__':
    assert corrected_cps(2000, 3) == 2e9
    assert corrected_cps(2000, 4) == 2e10
    m = MEASURED['MCF']
    assert abs(calibrated_sensitivity(m['cps'], m['contrast'], m['fwhm_mhz'])
               - m['sensitivity_nt']) < 1e-12
    assert linewidth_from_resolution(49) > linewidth_from_resolution(1.5)
    assert calibrated_sensitivity(4*m['cps'], m['contrast'], m['fwhm_mhz']) == 51.5
    print('PASS: ND correction, linewidth fit, and shot-noise scaling')
