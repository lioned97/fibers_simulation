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
from lens_design import (MEASURED, alignment_sweep, design_at_gap, evaluate_design,
                         replicated_surfaces, surface_z, trace_mode)
from paper_figures import INK, INK2, OUT, save


def _ray_panel(ax, surface, union, lam, title, color):
    tr = trace_mode(surface, union, lam, depths=[85.0], n_grid=31)
    ids = np.flatnonzero(tr['valid'] & (np.abs(tr['points'][:, 1])
                                        < surface['aperture']/10.0))
    if len(ids) > 17:
        ids = ids[np.linspace(0, len(ids)-1, 17, dtype=int)]
    core_x = surface['core_r']
    for i in ids:
        ax.plot([core_x, tr['points'][i, 0], tr['air_surface'][i, 0],
                 tr['hits'][0, i, 0]],
                [surface['base_z'], tr['points'][i, 2], 0.0,
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


def _paths(surface, union, lam, count):
    """True 3-D core/surface/air/diamond Snell-law polylines."""
    tr = trace_mode(surface, union, lam, depths=[85.0], n_grid=31)
    ids = np.flatnonzero(tr['valid'] & (tr['weight'] > 0))
    if len(ids) > count:
        ids = ids[np.linspace(0, len(ids)-1, count, dtype=int)]
    a = surface.get('angle', 0.0)
    core = np.array([surface['core_r']*np.cos(a),
                     surface['core_r']*np.sin(a), surface['base_z']])
    return [np.vstack([core, tr['points'][i], tr['air_surface'][i],
                       tr['hits'][0, i]]) for i in ids]


def _full_paths(design, union):
    central = _paths(design['central'], union, 532.0, 17)
    sides = [path for surface in union[1:]
             for path in _paths(surface, union, 750.0, 9)]
    return central, sides


def full_3d_figure(design):
    """Static publication view of the complete union and full 3-D rays."""
    union, x, y, z, radius = _full_union_grid(design)
    central, sides = _full_paths(design, union)
    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis', linewidth=0, alpha=0.80,
                    rcount=90, ccount=90)
    plane = np.array([[-radius, radius], [-radius, radius]])
    px, py = np.meshgrid(plane[0], plane[1])
    ax.plot_surface(px, py, np.zeros_like(px), color='#6aaec2', alpha=0.16)
    ax.plot_surface(px, py, np.full_like(px, -85.0), color='#d7cc45', alpha=0.20)
    for i, path in enumerate(central):
        ax.plot(*path.T, color='#1baf7a', lw=0.75, alpha=0.72,
                label='532 nm excitation' if i == 0 else None)
    for i, path in enumerate(sides):
        ax.plot(*path.T, color='#d7263d', lw=0.65, alpha=0.58,
                label='750 nm collected (reciprocal)' if i == 0 else None)
    cores = np.array([[0.0, 0.0, design['central']['base_z']]] +
                     [[35*np.cos(k*np.pi/3), 35*np.sin(k*np.pi/3),
                       design['central']['base_z']] for k in range(6)])
    ax.scatter(cores[:, 0], cores[:, 1], cores[:, 2], marker='s', s=18,
               color='#2a78d6', label='seven fiber cores')
    ax.set(xlabel='x ($\\mu$m)', ylabel='y ($\\mu$m)', zlabel='z ($\\mu$m)',
           xlim=(-radius, radius), ylim=(-radius, radius),
           zlim=(-95, design['central']['base_z']+8))
    ax.set_title('Complete monolithic MCF lens and 3-D Snell-law rays',
                 loc='left', fontsize=10, color=INK2)
    ax.view_init(elev=23, azim=-54)
    ax.set_box_aspect((1, 1, 1.55))
    ax.legend(loc='upper left', fontsize=7)
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.04, top=0.94)
    save(fig, 'fig10_full_mcf_3d_raytrace')


def full_3d_interactive(design):
    """Rotatable complete seven-lens union with the same traced 3-D rays."""
    import plotly.graph_objects as go

    union, x, y, z, radius = _full_union_grid(design, 121)
    central, sides = _full_paths(design, union)
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

    def add_paths(paths, name, color, width):
        xx, yy, zz = [], [], []
        for path in paths:
            xx += path[:, 0].tolist()+[None]
            yy += path[:, 1].tolist()+[None]
            zz += path[:, 2].tolist()+[None]
        fig.add_trace(go.Scatter3d(x=xx, y=yy, z=zz, mode='lines', name=name,
                                   line=dict(color=color, width=width), hoverinfo='skip'))

    add_paths(central, '532 nm excitation', '#1baf7a', 3)
    add_paths(sides, '750 nm collected (reciprocal)', '#d7263d', 2)
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
    fig.update_layout(
        title='Complete seven-lens MCF: rotatable 3-D Snell-law ray trace',
        template='plotly_white', height=780, margin=dict(l=0, r=0, t=90, b=0),
        scene=dict(xaxis_title='x (µm)', yaxis_title='y (µm)', zaxis_title='z (µm)',
                   aspectmode='manual', aspectratio=dict(x=1, y=1, z=1.6),
                   xaxis=dict(range=[-radius, radius]), yaxis=dict(range=[-radius, radius]),
                   zaxis=dict(range=[-95, design['central']['base_z']+8]),
                   camera=dict(eye=dict(x=1.35, y=1.35, z=1.0))),
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top',
                    bgcolor='rgba(255,255,255,0.78)'),
        updatemenus=[dict(type='buttons', direction='right', x=0.01, y=1.10,
                          buttons=[dict(label=label, method='relayout',
                                        args=[{'legend': dict(x=lx, y=ly,
                                                              xanchor=xa, yanchor=ya)}])
                                   for label, lx, ly, xa, ya in positions])])
    path = os.path.join(OUT, 'fig10_full_mcf_3d_raytrace_interactive.html')
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                   config={'responsive': True, 'displaylogo': False, 'scrollZoom': True})
    return path


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
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = load_design()
    ray_figure(d)
    comparison_figure(d)
    full_3d_figure(d)
    full_3d_interactive(d)
    fabrication_blueprint(d)
    alignment = alignment_figure(d, args.angle_deg)
    print(f"figures written to {OUT}")
    print(f"best fixed-tip gap: {alignment['best_gap_um']:.3f} um; "
          f"reported tilt: {args.angle_deg:g} deg")
