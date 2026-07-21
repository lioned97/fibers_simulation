"""Publication figures and 3-D ray trace for the physical seven-core design."""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from fab_check import load_design
from lens_design import (MCF_FULL_NA, MEASURED, alignment_sweep,
                         combined_overlap_volume, design_at_gap, evaluate_design,
                         replicated_surfaces, surface_z, trace_full_na,
                         trace_na_cone)
from paper_figures import INK, INK2, OUT, save


def _incident_start(surface, trace, index):
    """Back-project the traced polymer direction to the fibre base plane."""
    point = trace['points'][index]
    direction = trace['incident'][index]
    scale = (surface['base_z']-point[2])/direction[2]
    return point + scale*direction


def _ray_panel(ax, surface, union, lam, title, color):
    tr = trace_full_na(surface, union, lam, depths=[85.0], n_grid=31)
    ids = np.flatnonzero(tr['valid'] & (tr['weight'] >= 1e-3*tr['weight'].max()) &
                         (np.abs(tr['points'][:, 1]) < surface['aperture']/10.0))
    if len(ids) > 17:
        ids = ids[np.linspace(0, len(ids)-1, 17, dtype=int)]
    core_x = surface['core_r']
    for i in ids:
        start = _incident_start(surface, tr, i)
        ax.plot([start[0], tr['points'][i, 0], tr['air_surface'][i, 0],
                 tr['hits'][0, i, 0]],
                [start[2], tr['points'][i, 2], 0.0,
                 tr['hits'][0, i, 2]], color=color, lw=0.65, alpha=0.72)
    x = np.linspace(-65, 75, 900)
    support = surface['base_z']-8.0
    z = np.full_like(x, support)
    for s in union:
        z = np.minimum(z, surface_z(s, x, np.zeros_like(x)))
    ax.plot(x, z, color=INK, lw=1.3, label="exposed polymer union")
    ax.axhline(0, color="#6aaec2", lw=1.2)
    ax.axhspan(-90, -80, color="#d7cc45", alpha=0.25)
    ax.scatter([core_x], [surface['base_z']], marker="s", s=26, color="#2a78d6")
    ax.set_xlim(-65, 75); ax.set_ylim(-98, surface['base_z']+8)
    ax.set_xlabel("x ($\\mu$m)"); ax.set_ylabel("z ($\\mu$m)")
    ax.set_title(title, loc="left", fontsize=9.5, color=INK2)


def ray_figure(design):
    c, s = design['central'], design['side']
    union = replicated_surfaces(c, s)
    fig = plt.figure(figsize=(11.0, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.05], wspace=0.30)
    ax3 = fig.add_subplot(gs[0], projection='3d')
    axc, axs = fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    x = np.linspace(-58, 70, 120); y = np.linspace(-52, 52, 100)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    zz = np.full_like(xx, c['base_z']-8.0)
    for surf in (c, s):
        zz = np.minimum(zz, surface_z(surf, xx, yy))
    ax3.plot_surface(xx, yy, zz, cmap='viridis', linewidth=0, antialiased=True,
                     rcount=60, ccount=70, alpha=0.92)
    ax3.set_xlabel("x ($\\mu$m)"); ax3.set_ylabel("y ($\\mu$m)")
    ax3.set_zlabel("z ($\\mu$m)")
    ax3.set_title("(a) Exported central + one-side union", loc='left',
                  fontsize=9.5, color=INK2)
    ax3.view_init(elev=24, azim=-58); ax3.set_box_aspect((1.2, 1, 0.85))

    _ray_panel(axc, c, union, 532.0, "(b) Central excitation, 532 nm", "#1baf7a")
    _ray_panel(axs, s, union, 750.0,
               "(c) Side-core reciprocal collection, 750 nm", "#d7263d")
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.16, top=0.93)
    save(fig, "fig8_physical_freeform_raytrace")


def _full_union_grid(design, n=161):
    union = replicated_surfaces(design['central'], design['side'])
    radius = min(145.0, max(60.0, design['central']['aperture']+6.0,
                            design['side']['center_r']+design['side']['aperture']+6.0))
    q = np.linspace(-radius, radius, n)
    x, y = np.meshgrid(q, q, indexing='xy')
    z = np.full_like(x, design['central']['base_z']-8.0)
    for surface in union:
        z = np.minimum(z, surface_z(surface, x, y))
    z[x*x+y*y > radius*radius] = np.nan
    return union, x, y, z, radius


def _paths(surface, union, lam, count, fiber_na=MCF_FULL_NA):
    """True 3-D core/surface/air/diamond Snell-law polylines."""
    tr = trace_full_na(surface, union, lam, depths=[85.0], n_grid=31,
                       fiber_na=fiber_na)
    ids = np.flatnonzero(
        tr['valid'] & (tr['weight'] >= 1e-3*tr['weight'].max()))
    if len(ids) > count:
        ids = ids[np.linspace(0, len(ids)-1, count, dtype=int)]
    return [np.vstack([_incident_start(surface, tr, i), tr['points'][i],
                       tr['air_surface'][i],
                       tr['hits'][0, i]]) for i in ids]


def _full_paths(design, union, fiber_na=MCF_FULL_NA):
    central = _paths(design['central'], union, 532.0, 17, fiber_na)
    sides = [path for surface in union[1:]
             for path in _paths(surface, union, 750.0, 9, fiber_na)]
    return central, sides


def _overlap_points(overlap, threshold=0.5, max_points=6000):
    field = overlap['relative_signal']
    iz, iy, ix = np.where(field >= threshold)
    if len(ix) > max_points:
        keep = np.linspace(0, len(ix)-1, max_points, dtype=int)
        iz, iy, ix = iz[keep], iy[keep], ix[keep]
    axis, depths = overlap['axis_um'], overlap['depth_um']
    return axis[ix], axis[iy], -depths[iz], field[iz, iy, ix]


def _static_arrows(ax, paths, color, max_arrows):
    stride = max(1, int(np.ceil(len(paths)/max_arrows)))
    chosen = paths[::stride]
    origin = np.array([path[2] + 0.38*(path[3]-path[2]) for path in chosen])
    direction = np.array([path[3]-path[2] for path in chosen])
    direction /= np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1e-15)
    ax.quiver(*origin.T, *direction.T, length=8.0, normalize=False,
              arrow_length_ratio=0.35, color=color, alpha=0.75, linewidth=0.65)


def overlap_volume_outputs(design):
    """Calculate and save the exact 80--90 um field used by the score."""
    print('Calculating combined excitation x six-core collection volume...', flush=True)
    overlap = combined_overlap_volume(design['central'], design['side'], grid_n=101)
    with open(os.path.join(OUT, 'mcf_combined_overlap_volume.json'),
              'w', encoding='utf-8') as fh:
        json.dump(overlap['summary'], fh, indent=2)
    np.savez_compressed(
        os.path.join(OUT, 'mcf_combined_overlap_volume.npz'),
        x_um=overlap['axis_um'], y_um=overlap['axis_um'],
        depth_um=overlap['depth_um'],
        signal_density_photons_s_um3=overlap['signal_density'],
        relative_signal=overlap['relative_signal'])
    return overlap


def full_3d_figure(design, overlap=None):
    """Full core-to-diamond ray paths plus the scored overlap volume."""
    union, x, y, z, radius = _full_union_grid(design)
    central, sides = _full_paths(design, union)
    overlap = overlap or combined_overlap_volume(design['central'], design['side'])
    ox, oy, oz, ov = _overlap_points(overlap)
    fig = plt.figure(figsize=(12.0, 6.2))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis', linewidth=0, alpha=0.80,
                    rcount=90, ccount=90)
    plane = np.array([[-radius, radius], [-radius, radius]])
    px, py = np.meshgrid(plane[0], plane[1])
    ax.plot_surface(px, py, np.zeros_like(px), color='#6aaec2', alpha=0.16)
    ax.plot_surface(px, py, np.full_like(px, -85.0), color='#d7cc45', alpha=0.20)
    for i, path in enumerate(central):
        ax.plot(*path.T, color='#1baf7a', lw=0.75, alpha=0.72,
                label='central fiber -> diamond (532 nm)' if i == 0 else None)
    for i, path in enumerate(sides):
        ax.plot(*path.T, color='#d7263d', lw=0.65, alpha=0.58,
                label='six side fibers -> diamond (750 nm shown)' if i == 0 else None)
    _static_arrows(ax, central, '#1baf7a', 6)
    _static_arrows(ax, sides, '#d7263d', 18)
    ax.scatter(ox, oy, oz, c=ov, cmap='plasma', vmin=0.5, vmax=1.0,
               s=13, alpha=0.32, linewidths=0,
               label='combined excitation x collection volume')
    cores = np.array([[0.0, 0.0, design['central']['base_z']]] +
                     [[35*np.cos(k*np.pi/3), 35*np.sin(k*np.pi/3),
                       design['central']['base_z']] for k in range(6)])
    ax.scatter(cores[:, 0], cores[:, 1], cores[:, 2], marker='s', s=18,
               color='#2a78d6', label='seven fiber cores')
    ax.set(xlabel='x ($\\mu$m)', ylabel='y ($\\mu$m)', zlabel='z ($\\mu$m)',
           xlim=(-radius, radius), ylim=(-radius, radius),
           zlim=(-95, design['central']['base_z']+8))
    ax.set_title('(a) Rays propagate from fibers into diamond',
                  loc='left', fontsize=10, color=INK2)
    ax.view_init(elev=23, azim=-54)
    ax.set_box_aspect((1, 1, 1.55))
    ax.legend(loc='upper left', fontsize=7)

    zoom = fig.add_subplot(122, projection='3d')
    for path in central:
        zoom.plot(*path[2:].T, color='#1baf7a', lw=0.55, alpha=0.20)
    for path in sides:
        zoom.plot(*path[2:].T, color='#d7263d', lw=0.50, alpha=0.12)
    zoom.scatter(ox, oy, oz, c=ov, cmap='plasma', vmin=0.5, vmax=1.0,
                 s=22, alpha=0.38, linewidths=0)
    zoom_limit = max(3.0, 4.0*max(overlap['summary']['rms_xyz_um'][:2]))
    zoom.set(xlabel='x ($\\mu$m)', ylabel='y ($\\mu$m)', zlabel='z ($\\mu$m)',
             xlim=(-zoom_limit, zoom_limit), ylim=(-zoom_limit, zoom_limit),
             zlim=(-90, -80))
    zoom.set_box_aspect((1, 1, 1.1))
    zoom.view_init(elev=22, azim=-52)
    summary = overlap['summary']
    zoom.set_title('(b) Scored common volume in the 80-90 $\\mu$m NV layer',
                   loc='left', fontsize=10, color=INK2)
    zoom.text2D(0.02, 0.91,
                f"50%-peak volume = {summary['half_max_volume_um3']:.3g} $\\mu$m$^3$\n"
                f"weighted depth = {-summary['centroid_xyz_um'][2]:.3f} $\\mu$m",
                transform=zoom.transAxes, fontsize=8, color=INK2)
    fig.suptitle('Central 532-nm excitation x summed six-core 650-850-nm collection',
                 fontsize=11, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.04, top=0.90, wspace=0.02)
    save(fig, 'fig10_full_mcf_3d_raytrace')


def full_3d_interactive(design, overlap=None):
    """Rotatable rays and the exact combined volume used by the score."""
    import plotly.graph_objects as go

    union, x, y, z, radius = _full_union_grid(design, 121)
    central, sides = _full_paths(design, union)
    overlap = overlap or combined_overlap_volume(design['central'], design['side'])
    fig = go.Figure(go.Surface(x=x, y=y, z=z, name='monolithic IP-S union',
                               colorscale='Viridis', opacity=0.78,
                               showscale=False, hoverinfo='skip'))
    p = np.array([-radius, radius])
    px, py = np.meshgrid(p, p)
    for depth, name, color, opacity in ((0.0, 'diamond surface', '#6aaec2', 0.16),
                                         (-85.0, '3 ppm NV layer', '#d7cc45', 0.20)):
        fig.add_trace(go.Surface(x=px, y=py, z=np.full_like(px, depth), name=name,
                                 colorscale=[[0, color], [1, color]], opacity=opacity,
                                 showscale=False, hoverinfo='skip'))

    depth_grid, y_grid, x_grid = np.meshgrid(
        -overlap['depth_um'], overlap['axis_um'], overlap['axis_um'], indexing='ij')
    fig.add_trace(go.Isosurface(
        x=x_grid.ravel(), y=y_grid.ravel(), z=depth_grid.ravel(),
        value=overlap['relative_signal'].ravel(), isomin=0.25, isomax=0.85,
        surface_count=4, opacity=0.22, caps=dict(x_show=False, y_show=False,
                                                z_show=False),
        colorscale='Plasma', showscale=False,
        name='combined excitation x collection volume',
        hovertemplate='relative combined signal: %{value:.3f}<extra></extra>'))

    def add_paths(paths, name, color, width):
        xx, yy, zz = [], [], []
        for path in paths:
            xx += path[:, 0].tolist()+[None]
            yy += path[:, 1].tolist()+[None]
            zz += path[:, 2].tolist()+[None]
        fig.add_trace(go.Scatter3d(x=xx, y=yy, z=zz, mode='lines', name=name,
                                   line=dict(color=color, width=width), hoverinfo='skip'))

    add_paths(central, 'central fiber -> diamond (532 nm)', '#1baf7a', 3)
    add_paths(sides, 'six side fibers -> diamond (750 nm shown)', '#d7263d', 2)

    def add_arrows(paths, name, color, max_arrows):
        stride = max(1, int(np.ceil(len(paths)/max_arrows)))
        chosen = paths[::stride]
        origin = np.array([path[2] + 0.38*(path[3]-path[2]) for path in chosen])
        direction = np.array([path[3]-path[2] for path in chosen])
        direction /= np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1e-15)
        fig.add_trace(go.Cone(
            x=origin[:, 0], y=origin[:, 1], z=origin[:, 2],
            u=direction[:, 0], v=direction[:, 1], w=direction[:, 2],
            anchor='tail', sizemode='absolute', sizeref=7.0,
            colorscale=[[0, color], [1, color]], showscale=False,
            opacity=0.75, name=name, hoverinfo='skip', showlegend=False))

    add_arrows(central, 'central direction', '#1baf7a', 6)
    add_arrows(sides, 'side direction', '#d7263d', 18)
    angles = np.arange(6)*np.pi/3
    cores = np.vstack(([0.0, 0.0], 35*np.column_stack([np.cos(angles), np.sin(angles)])))
    fig.add_trace(go.Scatter3d(x=cores[:, 0], y=cores[:, 1],
                               z=np.full(7, design['central']['base_z']),
                               mode='markers', name='seven fiber cores',
                               marker=dict(size=4, color='#2a78d6', symbol='square')))
    positions = [('Top left', 0.01, 0.99, 'left', 'top'),
                 ('Top right', 0.99, 0.99, 'right', 'top'),
                 ('Bottom left', 0.01, 0.01, 'left', 'bottom'),
                 ('Bottom right', 0.99, 0.01, 'right', 'bottom')]
    zoom = max(3.0, 4.0*max(overlap['summary']['rms_xyz_um'][:2]))
    fig.update_layout(
        title=('Fiber-to-diamond rays and scored 80-90 um overlap volume: '
               '532-nm excitation x six-core 650-850-nm collection'),
        template='plotly_white', height=780, margin=dict(l=0, r=0, t=90, b=0),
        scene=dict(xaxis_title='x (µm)', yaxis_title='y (µm)', zaxis_title='z (µm)',
                   aspectmode='manual', aspectratio=dict(x=1, y=1, z=1.6),
                   xaxis=dict(range=[-radius, radius]), yaxis=dict(range=[-radius, radius]),
                   zaxis=dict(range=[-95, design['central']['base_z']+8]),
                   camera=dict(eye=dict(x=1.35, y=1.35, z=1.0))),
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top',
                    bgcolor='rgba(255,255,255,0.78)'),
        updatemenus=[
            dict(type='buttons', direction='right', x=0.01, y=1.10,
                 buttons=[dict(label=label, method='relayout',
                               args=[{'legend': dict(x=lx, y=ly,
                                                     xanchor=xa, yanchor=ya)}])
                          for label, lx, ly, xa, ya in positions]),
            dict(type='buttons', direction='right', x=0.56, y=1.10,
                 buttons=[
                     dict(label='Full probe', method='relayout', args=[{
                         'scene.xaxis.range': [-radius, radius],
                         'scene.yaxis.range': [-radius, radius],
                         'scene.zaxis.range': [-95, design['central']['base_z']+8]}]),
                     dict(label='Zoom overlap volume', method='relayout', args=[{
                         'scene.xaxis.range': [-zoom, zoom],
                         'scene.yaxis.range': [-zoom, zoom],
                         'scene.zaxis.range': [-90, -80]}])])],
        annotations=[dict(
            x=0.5, y=0.01, xref='paper', yref='paper', showarrow=False,
            text=(f"50%-peak volume: {overlap['summary']['half_max_volume_um3']:.3g} um^3; "
                  f"weighted depth: {-overlap['summary']['centroid_xyz_um'][2]:.3f} um"),
            font=dict(size=11, color=INK2))])
    path = os.path.join(OUT, 'fig10_full_mcf_3d_raytrace_interactive.html')
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                   config={'responsive': True, 'displaylogo': False, 'scrollZoom': True})
    return path


def full_na_figure(design, fiber_na=MCF_FULL_NA, overlap=None):
    """Show the full-NA boundary beside the weighted Phase 3 quadrature."""
    union, x, y, z, radius = _full_union_grid(design, 121)
    central_cone = trace_na_cone(
        design['central'], union, 532.0, fiber_na=fiber_na,
        n_theta=5, n_phi=18)
    side_cones = [trace_na_cone(
        surface, union, 750.0, fiber_na=fiber_na, n_theta=4, n_phi=12)
        for surface in union[1:]]
    central_mode, side_modes = _full_paths(design, union, fiber_na)
    overlap = overlap or combined_overlap_volume(
        design['central'], design['side'], grid_n=81)

    fig = plt.figure(figsize=(15.0, 5.5))
    ax3 = fig.add_subplot(131, projection='3d')
    ax3.plot_surface(x, y, z, cmap='viridis', linewidth=0, alpha=0.34,
                     rcount=65, ccount=65)

    def cone_paths(ax, cone, color, projected=False):
        for path, theta in zip(cone['paths'], cone['theta']):
            boundary = np.isclose(theta, cone['theta_max'])
            if projected:
                if np.max(np.abs(path[:, 1])) > 1e-6:
                    continue
                ax.plot(path[:, 0], path[:, 2], color=color,
                        lw=0.75 if boundary else 0.35,
                        alpha=0.34 if boundary else 0.08)
            else:
                ax.plot(*path.T, color=color, lw=0.55 if boundary else 0.25,
                        alpha=0.20 if boundary else 0.035)

    cone_paths(ax3, central_cone, '#1baf7a')
    for cone in side_cones:
        cone_paths(ax3, cone, '#d7263d')
    for path in central_mode:
        ax3.plot(*path.T, color='#087f5b', lw=1.1, alpha=0.85)
    for path in side_modes:
        ax3.plot(*path.T, color='#b5162d', lw=0.75, alpha=0.52)
    ax3.set(xlabel='x ($\\mu$m)', ylabel='y ($\\mu$m)', zlabel='z ($\\mu$m)',
            xlim=(-radius, radius), ylim=(-radius, radius),
            zlim=(-95, design['central']['base_z']+8))
    ax3.set_box_aspect((1, 1, 1.45)); ax3.view_init(elev=22, azim=-55)
    ax3.set_title('(a) Full-NA envelopes of all seven cores', loc='left',
                  fontsize=9.5, color=INK2)

    ax_section = fig.add_subplot(132)
    cone_paths(ax_section, central_cone, '#1baf7a', projected=True)
    cone_paths(ax_section, side_cones[0], '#d7263d', projected=True)
    for path in central_mode:
        if np.max(np.abs(path[:, 1])) < 1.0:
            ax_section.plot(path[:, 0], path[:, 2], color='#087f5b', lw=1.0)
    first_side_mode = _paths(union[1], union, 750.0, 21, fiber_na)
    for path in first_side_mode:
        if np.max(np.abs(path[:, 1])) < 1.0:
            ax_section.plot(path[:, 0], path[:, 2], color='#b5162d', lw=0.9)
    section_x = np.linspace(-48.0, 78.0, 1600)
    section_z = np.full_like(section_x, design['central']['base_z']-8.0)
    for candidate in union:
        section_z = np.minimum(section_z, surface_z(candidate, section_x,
                                                     np.zeros_like(section_x)))
    ax_section.plot(section_x, section_z, color=INK, lw=1.2)
    ax_section.axhline(0.0, color='#6aaec2', lw=1.0)
    ax_section.axhspan(-90.0, -80.0, color='#d7cc45', alpha=0.16)
    ax_section.scatter([0.0, 35.0], [design['central']['base_z']]*2,
                       marker='s', s=24, color='#2a78d6', zorder=5)
    theta_ips = np.degrees(np.arcsin(fiber_na/1.52))
    ax_section.text(0.02, 0.02,
                    f'NA={fiber_na:g}: polymer half-angle {theta_ips:.2f} deg',
                    transform=ax_section.transAxes, fontsize=8, color=INK2)
    ax_section.set(xlim=(-48, 78), ylim=(-95, design['central']['base_z']+8),
                   xlabel='x ($\\mu$m)', ylabel='z ($\\mu$m)')
    ax_section.set_title('(b) Central core + one side core section', loc='left',
                         fontsize=9.5, color=INK2)

    ax_hit = fig.add_subplot(133)
    central_hits = central_cone['paths'][:, -1, :2]
    side_hits = np.vstack([cone['paths'][:, -1, :2] for cone in side_cones])
    ax_hit.scatter(side_hits[:, 0], side_hits[:, 1], s=8, facecolors='none',
                   edgecolors='#d7263d', alpha=0.30,
                   label='six side full-NA rays')
    ax_hit.scatter(central_hits[:, 0], central_hits[:, 1], s=10,
                   facecolors='none', edgecolors='#1baf7a', alpha=0.55,
                   label='central full-NA rays')
    side_mode_hits = np.array([path[-1, :2] for path in side_modes])
    central_mode_hits = np.array([path[-1, :2] for path in central_mode])
    ax_hit.scatter(side_mode_hits[:, 0], side_mode_hits[:, 1], s=15,
                   marker='x', color='#b5162d', alpha=0.75,
                   label='weighted side full-NA samples')
    ax_hit.scatter(central_mode_hits[:, 0], central_mode_hits[:, 1], s=18,
                   marker='+', color='#087f5b', alpha=0.9,
                   label='weighted central full-NA samples')
    iz = int(np.argmin(np.abs(overlap['depth_um']-85.0)))
    axis = overlap['axis_um']
    ax_hit.contour(axis, axis, overlap['relative_signal'][iz], levels=[0.5],
                   colors=['#7c3aed'], linewidths=1.6)
    hit_limit = max(5.0, min(140.0, 1.05*np.max(np.abs(
        np.vstack([central_hits, side_hits])))))
    ax_hit.set(xlim=(-hit_limit, hit_limit), ylim=(-hit_limit, hit_limit),
               xlabel='x at 85 $\\mu$m depth ($\\mu$m)',
               ylabel='y at 85 $\\mu$m depth ($\\mu$m)', aspect='equal')
    ax_hit.set_title('(c) Full-NA landing positions and scored overlap', loc='left',
                     fontsize=9.5, color=INK2)
    ax_hit.legend(loc='upper right', fontsize=6.5)
    ax_hit.text(0.02, 0.02, 'purple contour: 50% combined signal',
                transform=ax_hit.transAxes, fontsize=8, color=INK2)

    fig.suptitle('Phase 3 model: 10-um Gaussian core area x uniformly filled full-NA angles',
                 fontsize=10.5, color=INK2)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.13, top=0.87, wspace=0.28)
    save(fig, 'fig13_full_na_core_envelopes')


def fabrication_blueprint(design):
    """Dimensioned top and section views for reproducing the exported STL."""
    central, side = design['central'], design['side']
    union = replicated_surfaces(central, side)
    base_z = central['base_z']
    support_h = 8.0
    radius = min(145.0, max(60.0, central['aperture']+6.0,
                            side['center_r']+side['aperture']+6.0))
    fig, (axt, axs) = plt.subplots(1, 2, figsize=(10.2, 4.6),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.35],
                                                    wspace=0.30))

    axt.add_patch(Circle((0, 0), radius, facecolor='#d8dde1',
                         edgecolor=INK, lw=1.2, alpha=0.48))
    axt.add_patch(Circle((0, 0), central['aperture'], facecolor='#1baf7a',
                         edgecolor='#106b4c', lw=1.2, alpha=0.25))
    for k in range(6):
        angle = k*np.pi/3.0
        center = side['center_r']*np.array([np.cos(angle), np.sin(angle)])
        axt.add_patch(Circle(center, side['aperture'], facecolor='#d78500',
                             edgecolor='#9a5d00', lw=1.0, alpha=0.20))
        axt.plot(center[0], center[1], 's', ms=4.5, color='#2a78d6')
    axt.plot(0, 0, 's', ms=4.5, color='#2a78d6')
    axt.annotate('', (-radius, -57), (radius, -57),
                 arrowprops=dict(arrowstyle='<->', lw=0.9, color=INK2))
    axt.text(0, -54, f'Ø{2*radius:.0f} support', ha='center', va='bottom', fontsize=8)
    axt.annotate('', (0, 0), (side['center_r'], 0),
                 arrowprops=dict(arrowstyle='<->', lw=0.9, color=INK2))
    axt.text(side['center_r']/2, 3, f"{side['center_r']:.0f} core radius",
             ha='center', fontsize=7.5)
    axt.text(0, 24, f"central Ø{2*central['aperture']:.0f}", ha='center', fontsize=8)
    axt.text(42, 34, f"side Ø{2*side['aperture']:.0f}\n6 × 60°", ha='left', fontsize=8)
    axt.set(xlim=(-72, 72), ylim=(-67, 72), xlabel='fiber x (µm)',
            ylabel='fiber y (µm)', aspect='equal')
    axt.set_title('(a) Fiber-facet layout', loc='left', fontsize=9.5, color=INK2)

    x = np.linspace(-radius, radius, 2401)
    z_model = np.full_like(x, base_z-support_h)
    for surface in union:
        z_model = np.minimum(z_model, surface_z(surface, x, np.zeros_like(x)))
    height = base_z-z_model
    axs.fill_between(x, 0, height, color='#4c9a8a', alpha=0.56)
    axs.plot(x, height, color='#176b60', lw=1.25)
    axs.axhline(0, color=INK, lw=1.0)
    axs.axhline(125, color='#6aaec2', lw=1.2)
    axs.axhspan(125, 130, color='#6aaec2', alpha=0.12)
    axs.text(63, -1.0, 'fiber facet', fontsize=8, ha='right', va='top')
    axs.text(-63, 126.5, 'diamond surface', fontsize=8, va='bottom')
    axs.annotate('', (0, 0), (0, 120),
                 arrowprops=dict(arrowstyle='<->', lw=0.9, color=INK2))
    axs.text(2.5, 60, '120 central height', rotation=90, va='center', fontsize=8)
    axs.annotate('', (7, 120), (7, 125),
                 arrowprops=dict(arrowstyle='<->', lw=0.9, color='#d7263d'))
    axs.text(9.5, 122.5, '5 gap', color='#a51c30', va='center', fontsize=8)
    axs.annotate('', (-61, 0), (-61, support_h),
                 arrowprops=dict(arrowstyle='<->', lw=0.9, color=INK2))
    axs.text(-56.5, 10.5, '8 support', va='bottom', fontsize=7.5)
    axs.set(xlim=(-68, 68), ylim=(-4, 131), xlabel='fiber x (µm)',
            ylabel='height outward from fiber facet (µm)')
    axs.set_title('(b) Monolithic x-z section through two side cores',
                  loc='left', fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.15, top=0.92)
    save(fig, 'fig12_mcf_fabrication_blueprint')


def comparison_figure(design):
    methods = design.get('method_comparison', {})
    result = design['result']
    fig, (axm, axb) = plt.subplots(1, 2, figsize=(10.0, 4.6),
                                   gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.42))
    key = 'comparison_normalized_sensitivity_nt'
    rows = sorted(methods.items(), key=lambda q: q[1][key])
    y = np.arange(len(rows))
    axm.barh(y, [r[1][key] for r in rows], color="#7b8790")
    axm.set_yticks(y); axm.set_yticklabels([r[0] for r in rows], fontsize=7.2)
    axm.invert_yaxis(); axm.set_xlabel("Comparison-normalized sensitivity (nT/$\\sqrt{Hz}$)")
    axm.set_title("(a) Physical surface-method screening", loc='left', fontsize=9.5, color=INK2)

    labels = ['SM measured', 'MM measured', 'MCF measured', 'new MCF']
    values = [MEASURED['SM']['sensitivity_nt'], MEASURED['MM']['sensitivity_nt'],
              MEASURED['MCF']['sensitivity_nt'],
              result['comparison_normalized_sensitivity_nt']]
    colors = ['#2a78d6', '#1baf7a', '#d78500', '#d7263d']
    bars = axb.bar(labels, values, color=colors, width=0.68)
    axb.set_yscale('log'); axb.set_ylabel("Sensitivity (nT/$\\sqrt{Hz}$), lower is better")
    axb.tick_params(axis='x', rotation=25)
    axb.set_title("(b) Comparison-normalized result", loc='left', fontsize=9.5, color=INK2)
    for bar, value in zip(bars, values):
        axb.text(bar.get_x()+bar.get_width()/2, value*1.08, f"{value:.3g}",
                 ha='center', va='bottom', fontsize=8)
    fig.text(0.77, 0.02,
             f"new MCF: {result['model_fiber_photons_s']:.3g} fiber photons/s; "
             f"{result['comparison_normalized_cps']:.3g} normalized cps; "
             f"C={100*result['contrast']:.2f}%; FWHM={result['fwhm_mhz']:.3g} MHz; "
             f"optical FWHM={result['resolution_um']:.3g} $\\mu$m",
             ha='center', va='bottom', fontsize=7.2, color=INK2)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.30, top=0.91)
    save(fig, "fig9_lens_methods_and_sensitivity")


def alignment_figure(design, given_angle_deg=0.0):
    """Fixed-tip Z optimum and arbitrary diamond/fiber plane-angle penalty."""
    gaps = np.geomspace(5.0, 500.0, 27)
    span = max(10.0, 1.2*abs(float(given_angle_deg)))
    angles = np.unique(np.r_[np.linspace(-span, span, 25), float(given_angle_deg), 0.0])
    data = alignment_sweep(design, gaps, angles, grid_n=81)
    shifted = design_at_gap(design, data['best_gap_um'])
    given = evaluate_design(shifted['central'], shifted['side'], grid_n=101,
                            tilt_deg=float(given_angle_deg))
    data['given'] = {k: given[k] for k in (
        'tilt_deg', 'model_fiber_photons_s', 'comparison_normalized_cps',
        'raw_model_sensitivity_nt', 'comparison_normalized_sensitivity_nt',
        'resolution_um')}
    data['tilt_axis'] = 'diamond plane rotated about fiber y-axis through z=0'
    with open(os.path.join(OUT, 'mcf_gap_angle_sweep.json'), 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2)

    gap = np.array([r['gap_um'] for r in data['gap']])
    gap_cps = np.array([r['model_fiber_photons_s'] for r in data['gap']])
    angle = np.array([r['angle_deg'] for r in data['angle']])
    angle_cps = np.array([r['model_fiber_photons_s'] for r in data['angle']])
    zero_cps = angle_cps[np.argmin(np.abs(angle))]
    fig, (axg, axa) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axg.semilogx(gap, gap_cps/gap_cps.max(), '-o', ms=3, color='#2a78d6')
    axg.axvline(data['best_gap_um'], color='#d7263d', lw=1, ls='--')
    axg.annotate(f"optimum = {data['best_gap_um']:.2f} µm",
                 (data['best_gap_um'], 1.0), xytext=(8, -18),
                 textcoords='offset points', fontsize=8)
    axg.set(xlabel='Central-lens–diamond gap (µm)',
            ylabel='Relative fiber fluorescence', ylim=(0, 1.05))
    axg.set_title('(a) Fixed printed tip: Z-distance sweep', loc='left',
                  fontsize=9.5, color=INK2)

    axa.plot(angle, angle_cps/zero_cps, '-o', ms=3, color='#d78500')
    axa.axvline(0, color=INK, lw=0.8)
    given_sweep = min(data['angle'],
                      key=lambda row: abs(row['angle_deg']-given_angle_deg))
    relative = given_sweep['model_fiber_photons_s']/zero_cps
    axa.scatter([given_angle_deg], [relative], s=35, color='#d7263d', zorder=4)
    axa.annotate(f"{given_angle_deg:g}°: {100*relative:.1f}% photons\n"
                 f"ηnorm = {given['comparison_normalized_sensitivity_nt']:.2f} nT/√Hz",
                 (given_angle_deg, relative), xytext=(8, -30),
                 textcoords='offset points', fontsize=8)
    axa.set(xlabel='Diamond–fiber plane angle about y (degrees)',
            ylabel='Fluorescence relative to parallel planes')
    axa.set_title('(b) Angular misalignment at optimum Z', loc='left',
                  fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.90, wspace=0.28)
    save(fig, 'fig11_gap_and_angle_tolerance')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--angle-deg', type=float, default=0.0,
                        help='measured diamond/fiber plane angle about y')
    parser.add_argument('--fiber-na', type=float, default=MCF_FULL_NA,
                        help='geometric full-NA envelope used only in figure 13')
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = load_design()
    ray_figure(d)
    comparison_figure(d)
    overlap = overlap_volume_outputs(d)
    full_3d_figure(d, overlap)
    full_3d_interactive(d, overlap)
    full_na_figure(d, args.fiber_na, overlap)
    fabrication_blueprint(d)
    alignment = alignment_figure(d, args.angle_deg)
    print(f"figures written to {OUT}")
    print(f"best fixed-tip gap: {alignment['best_gap_um']:.3f} um; "
          f"reported tilt: {args.angle_deg:g} deg")
