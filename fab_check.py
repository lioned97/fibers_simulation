"""Check print volume, surface smoothness, spectral throughput, and STL integrity."""
import json
import os
import struct

import numpy as np

from lens_design import (MAX_PRINT, RED_LAM, beam_stats, replicated_surfaces,
                         surface_limits, trace_mode, validate_design)
from paper_figures import OUT


def load_design(path=os.path.join(OUT, "mcf_freeform_design.json")):
    from lens_design import evaluate_design
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    central, side = raw['central'], raw['side']
    central['coef'] = np.asarray(central['coef']); side['coef'] = np.asarray(side['coef'])
    return dict(central=central, side=side,
                result=evaluate_design(central, side, grid_n=121),
                method_comparison=raw.get('method_comparison', {}))


def main():
    design = load_design()
    validate_design(design)
    union = replicated_surfaces(design['central'], design['side'])
    print("surface checks")
    for name in ('central', 'side'):
        lim = surface_limits(design[name])
        assert lim['print_height'] <= MAX_PRINT
        print(name, lim)
    print("side-lens spectral throughput")
    for lam in RED_LAM:
        tr = trace_mode(design['side'], union, lam, depths=[85.0], n_grid=31)
        assert tr['throughput'] > 0
        print(f"  {lam:.0f} nm: T={tr['throughput']:.4f}, "
              f"FWHM@85={beam_stats(tr, lam, depths=[85.0])[0]['fwhm']:.4f} um")
    for name in ("mcf_freeform_central_one_side.stl",
                 "mcf_freeform_full_seven_core.stl"):
        stl = os.path.join(OUT, name)
        with open(stl, 'rb') as fh:
            header = fh.read(84)
        n = struct.unpack_from('<I', header, 80)[0]
        assert os.path.getsize(stl) == 84+50*n and n > 0
        print(f"{name}: structurally complete, {n} triangles")


if __name__ == '__main__':
    main()
