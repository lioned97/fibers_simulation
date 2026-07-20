"""Small assert-based check for the physical freeform lens model."""
import io
import os
import struct
import tempfile
from contextlib import redirect_stdout

import numpy as np

from lens_design import (COMPARISON_NORMALIZATION, DEPTHS, MCF_MFD, RED_LAM,
                         RED_W, SEARCH_DEPTHS, SEARCH_RED_LAM, SEARCH_RED_W,
                         PRINT_X_UM, PRINT_Y_UM, PRINT_Z_UM, _ProgressBar,
                         _ray_density, _search_objective, _search_spec,
                         _search_surfaces, alignment_sweep, beam_stats,
                         design_parameters, evaluate_design, fit_surface,
                         replicated_surfaces, surface_limits, trace_mode,
                         validate_design, write_binary_stl)
from physics import nv_emission_spectrum


def main():
    central = fit_surface('asphere', 'central', 5, 200, 20)
    side = fit_surface('freeform', 'side', 5, 200, 17.5,
                       center_r=35, side_offset=35)
    union = replicated_surfaces(central, side)
    assert MCF_MFD == 10.0
    assert (PRINT_X_UM, PRINT_Y_UM, PRINT_Z_UM) == (300.0, 300.0, 300.0)
    assert len(DEPTHS) == len(RED_LAM) == 33
    assert len(SEARCH_DEPTHS) == len(SEARCH_RED_LAM) == 9
    assert np.isclose(RED_W.sum(), 1.0) and np.isclose(SEARCH_RED_W.sum(), 1.0)
    raw_spectrum = nv_emission_spectrum(RED_LAM)
    assert np.isclose(RED_W[0]/raw_spectrum[0],
                      0.5*RED_W[1]/raw_spectrum[1])
    assert COMPARISON_NORMALIZATION['factor'] > 1.0
    for surface, lam in ((central, 532.0), (side, 750.0)):
        limits = surface_limits(surface)
        assert limits['print_height'] <= 300 and np.isfinite(limits['max_slope'])
        tr = trace_mode(surface, union, lam, n_grid=25)
        assert tr['throughput'] > 0 and np.any(tr['valid'])
        assert beam_stats(tr, lam)[2]['fwhm'] > 0
    density_axis = np.linspace(-150.0, 150.0, 601)
    density = _ray_density(tr, 2, density_axis, 750.0,
                           np.arange(6)*np.pi/3.0)
    assert np.isclose(density.sum()*(density_axis[1]-density_axis[0])**2,
                      6.0*tr['throughput'], rtol=2e-3)
    from redesign_fig import _incident_start
    ray = np.flatnonzero(tr['valid'])[0]
    start = _incident_start(side, tr, ray)
    drawn_incident = tr['points'][ray]-start
    drawn_incident /= np.linalg.norm(drawn_incident)
    assert np.isclose(start[2], side['base_z'])
    assert np.allclose(drawn_incident, tr['incident'][ray])
    result = evaluate_design(central, side, grid_n=81)
    design = dict(central=central, side=side, result=result)
    assert validate_design(design)
    _, _, x0, _ = _search_spec('quadratic', 'quadratic')
    searched = _search_surfaces(x0, 'quadratic', 'quadratic')
    p = design_parameters(*searched)
    assert p['central_side_overlap_um'] == 3.0
    assert p['side_core_offset_um'] == 0.0
    bad = x0.copy(); bad[:2] = [295, 295]
    assert np.isinf(_search_objective(
        bad, 'quadratic', 'quadratic', 'full'))
    progress_output = io.StringIO()
    with redirect_stdout(progress_output):
        progress = _ProgressBar('test', 2)
        progress(None, None); progress(None, None); progress.finish()
    assert '100%' in progress_output.getvalue()
    tilted = trace_mode(side, union, 750.0, depths=[85.0], n_grid=25, tilt_deg=3.0)
    a = np.deg2rad(3.0); normal = np.array([np.sin(a), 0.0, -np.cos(a)])
    assert np.max(np.abs(tilted['air_surface'][tilted['valid']] @ normal)) < 1e-10
    assert np.max(np.abs(tilted['hits'][0, tilted['valid']] @ normal-85.0)) < 1e-10
    sweep = alignment_sweep(design, [5.0, 50.0], [-3.0, 0.0, 3.0], grid_n=61)
    assert len(sweep['gap']) == 2 and len(sweep['angle']) == 3
    assert all(row['model_fiber_photons_s'] > 0
               for row in sweep['gap']+sweep['angle'])
    assert np.isclose(sweep['angle'][0]['model_fiber_photons_s'],
                      sweep['angle'][2]['model_fiber_photons_s'], rtol=1e-8)
    with tempfile.TemporaryDirectory() as td:
        for full in (False, True):
            path = os.path.join(td, f'design_{full}.stl')
            n = write_binary_stl(design, path, nr=15, nt=36, all_sides=full)
            with open(path, 'rb') as fh:
                header = fh.read(84)
                raw = np.fromfile(fh, dtype=np.dtype([
                    ('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attr', '<u2')]),
                    count=n)
            assert struct.unpack_from('<I', header, 80)[0] == n
            assert os.path.getsize(path) == 84+50*n
            vertices = raw['vertices']
            assert np.all(np.linalg.norm(np.cross(vertices[:, 1]-vertices[:, 0],
                                                   vertices[:, 2]-vertices[:, 0]), axis=1) > 0)
            _, inverse = np.unique(vertices.reshape(-1, 3), axis=0, return_inverse=True)
            faces = inverse.reshape(-1, 3)
            edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]],
                                       faces[:, [2, 0]])), axis=1)
            assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
    print('PASS: physical surfaces, rays, sensitivity, and watertight STL')


if __name__ == '__main__':
    main()
