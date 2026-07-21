"""Small assert-based check for the physical freeform lens model."""
import io
import os
import struct
import tempfile
from contextlib import redirect_stdout

import numpy as np

from lens_design import (COMPARISON_NORMALIZATION, DEPTHS, MCF_FULL_NA,
                         MCF_IPS_N, MCF_MFD, PRINT_X_UM, PRINT_Y_UM,
                         PRINT_Z_UM, RED_LAM, RED_W, SEARCH_DEPTHS,
                         SEARCH_RED_LAM, SEARCH_RED_W, _ProgressBar,
                         _ray_density, _search_objective, _search_spec,
                         _search_surfaces, alignment_sweep, beam_stats,
                         combined_overlap_volume, design_parameters,
                         evaluate_design, fit_surface, replicated_surfaces,
                         surface_limits, trace_full_na, trace_mode,
                         trace_na_cone, validate_design, write_binary_stl)
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
        converged = trace_mode(surface, union, lam, n_grid=41)
        assert np.isclose(tr['throughput'], converged['throughput'], rtol=0.03)
        assert converged['throughput'] <= 1.0+1e-3
        assert beam_stats(tr, lam)[2]['fwhm'] > 0
    density_axis = np.linspace(-150.0, 150.0, 601)
    density = _ray_density(tr, 2, density_axis, 750.0,
                           np.arange(6)*np.pi/3.0)
    assert np.isclose(density.sum()*(density_axis[1]-density_axis[0])**2,
                      6.0*tr['throughput'], rtol=2e-3)
    full_na = trace_full_na(central, union, 532.0, depths=[85.0], n_grid=17)
    assert full_na['ray_model'] == 'uniform_full_na'
    assert np.isclose(full_na['theta_max'], np.arcsin(MCF_FULL_NA/MCF_IPS_N))
    assert 0.0 < full_na['throughput'] <= 1.0+1e-12
    assert np.allclose(full_na['hits'][0, full_na['valid'], 2], -85.0)
    full_na_density = _ray_density(
        full_na, 0, density_axis, 532.0, np.array([0.0]))
    assert np.isclose(
        full_na_density.sum()*(density_axis[1]-density_axis[0])**2,
        full_na['throughput'], rtol=2e-3)
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
    small_depths = np.array([80.0, 85.0, 90.0])
    small_lam = np.array([650.0, 750.0, 850.0])
    small_w = np.full(3, 1/3)
    small_result = evaluate_design(
        central, side, grid_n=31, depths=small_depths,
        red_lam=small_lam, red_w=small_w, ray_grid=21)
    overlap = combined_overlap_volume(
        central, side, grid_n=31, depths=small_depths,
        red_lam=small_lam, red_w=small_w, ray_grid=21)
    assert overlap['signal_density'].shape == (3, 31, 31)
    assert np.isclose(overlap['relative_signal'].max(), 1.0)
    assert np.isclose(overlap['summary']['integrated_signal_photons_s'],
                      small_result['model_fiber_photons_s'])
    assert 80.0 <= -overlap['summary']['centroid_xyz_um'][2] <= 90.0
    assert overlap['summary']['half_max_volume_um3'] > 0.0
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
    tilted = trace_full_na(
        side, union, 750.0, depths=[85.0], n_grid=25, tilt_deg=3.0)
    a = np.deg2rad(3.0); normal = np.array([np.sin(a), 0.0, -np.cos(a)])
    assert np.max(np.abs(tilted['air_surface'][tilted['valid']] @ normal)) < 1e-10
    assert np.max(np.abs(tilted['hits'][0, tilted['valid']] @ normal-85.0)) < 1e-10
    na_cone = trace_na_cone(side, union, 750.0, fiber_na=0.22,
                            depth=85.0, n_theta=3, n_phi=8)
    assert len(na_cone['paths']) > 0
    assert np.allclose(na_cone['paths'][:, 0, 2], side['base_z'])
    assert np.allclose(na_cone['paths'][:, -1, 2], -85.0)
    assert np.max(na_cone['theta']) <= np.arcsin(0.22/MCF_IPS_N)+1e-12
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
