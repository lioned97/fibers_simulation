"""Print the authoritative measured baselines used by the lens optimizer."""
from sensitivity import MEASURED, corrected_cps


if __name__ == "__main__":
    print(f"{'fiber':>5} {'OD':>4} {'corrected cps':>15} {'C (%)':>8} "
          f"{'FWHM (MHz)':>11} {'eta (nT/sqrtHz)':>16}")
    for name, r in MEASURED.items():
        cps = corrected_cps(r['observed_kcps'], r['od'])
        assert cps == r['cps']
        print(f"{name:>5} {r['od']:4.1f} {cps:15.4g} {100*r['contrast']:8.2f} "
              f"{r['fwhm_mhz']:11.2f} {r['sensitivity_nt']:16.1f}")
