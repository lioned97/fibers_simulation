"""Physical freeform design for the printed seven-core NV-diamond probe.

The program designs one central 532-nm excitation lens and one 650--850-nm
side collection lens.  The side surface is rotated sixfold for optical scoring;
the exports contain both a one-side prototype and the complete seven-core
monolithic structure.  Surfaces are fitted to Snell-law normals, not ideal
acceptance functions.  Coordinates match paper_figures.py: diamond z=0, NVs z=-80..-90,
printed polymer/fibre z>0, and g is central-lens-to-diamond clearance.
"""
import json
import os
import struct
import sys
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import differential_evolution

from physics import diamond_sellmeier, nv_emission_spectrum
from paper_figures import (
    E_PHOT_RED, I_SAT, MCF_IPS_N, MCF_MFD, P_GREEN_MW,
    PPM, RHO, R_SAT,
)
from sensitivity import MEASURED, linewidth_from_resolution, calibrated_sensitivity

N_AIR = 1.0
CORE_R = 35.0
W_MODE = MCF_MFD / 2.0
PRINT_X_UM = PRINT_Y_UM = PRINT_Z_UM = 300.0
MAX_PRINT = PRINT_Z_UM  # compatibility name used by the fabrication scripts
NV_DEPTH_UM = (80.0, 90.0)
NV_SPECTRUM_NM = (650.0, 850.0)


def _spectral_quadrature(points):
    """Normalized trapezoidal quadrature of the NV spectrum over 650--850 nm."""
    wavelength = np.linspace(*NV_SPECTRUM_NM, int(points))
    weight = nv_emission_spectrum(wavelength)
    weight[[0, -1]] *= 0.5
    return wavelength, weight/weight.sum()


# Cheap, deterministic screening is followed by a converged final evaluation.
SEARCH_DEPTHS = np.linspace(*NV_DEPTH_UM, 9)
SEARCH_RED_LAM, SEARCH_RED_W = _spectral_quadrature(9)
DEPTHS = np.linspace(*NV_DEPTH_UM, 33)
RED_LAM, RED_W = _spectral_quadrature(33)

# The legacy model checkpoint is kept explicit because it is only an empirical
# comparison normalization.  Its factor exceeds one, so it must not be called a
# detector efficiency or used as a physical photons-to-counts conversion.
COMPARISON_NORMALIZATION = {
    "reference_probe": "measured original MCF",
    "reference_measured_corrected_cps": 2.0e9,
    "reference_model": "paper_figures.py legacy as-built MCF ensemble checkpoint",
    "reference_model_power_nw": 0.051756581935736595,
    "reference_photon_energy_j": float(E_PHOT_RED),
    "interpretation": "empirical model-to-experiment normalization; not detector efficiency",
}
COMPARISON_NORMALIZATION["reference_model_fiber_photons_s"] = (
    COMPARISON_NORMALIZATION["reference_model_power_nw"]*1e-9/E_PHOT_RED)
COMPARISON_NORMALIZATION["factor"] = (
    COMPARISON_NORMALIZATION["reference_measured_corrected_cps"] /
    COMPARISON_NORMALIZATION["reference_model_fiber_photons_s"])

def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-15)


def _fresnel_refract(v, normal, n1, n2):
    """Vector Snell refraction; normal points from n1 into n2."""
    v, normal = _unit(v), _unit(normal)
    ci = np.sum(v * normal, axis=-1)
    flip = ci < 0.0
    normal = np.where(flip[..., None], -normal, normal)
    ci = np.abs(ci)
    eta = n1 / n2
    st2 = eta * eta * np.maximum(0.0, 1.0 - ci * ci)
    ok = st2 < 1.0
    ct = np.sqrt(np.maximum(0.0, 1.0 - st2))
    out = eta * v + (ct - eta * ci)[..., None] * normal
    out = _unit(out)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct + 1e-15)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct + 1e-15)
    transmission = np.where(ok, 1.0 - 0.5 * (rs * rs + rp * rp), 0.0)
    return out, transmission, ok


def _disk(n=25):
    q = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(q, q, indexing="xy")
    keep = x * x + y * y <= 1.0
    return x[keep], y[keep], (2.0 / (n - 1)) ** 2


def _frame(surface):
    a = surface.get("angle", 0.0)
    er = np.array([np.cos(a), np.sin(a)])
    et = np.array([-np.sin(a), np.cos(a)])
    return er, et, surface["center_r"] * er


def _local(surface, x, y):
    er, et, center = _frame(surface)
    d = np.column_stack([np.asarray(x).ravel(), np.asarray(y).ravel()]) - center
    u, v = d @ er, d @ et
    return u.reshape(np.shape(x)), v.reshape(np.shape(y))


def _sag_grad(surface, u, v):
    """Polynomial sag and local slopes. Coefficients are dimensionless."""
    a = surface["aperture"]
    x, y = np.asarray(u) / a, np.asarray(v) / a
    p = surface["coef"]
    sag = a * (p[0] * x + 0.5 * p[1] * x*x + 0.5 * p[2] * y*y
               + p[3] * x**3 / 3.0 + p[4] * x*y*y
               + p[5] * x**4 / 4.0 + 0.5 * p[6] * x*x*y*y
               + p[7] * y**4 / 4.0) + surface.get("shift", 0.0)
    gx = p[0] + p[1]*x + p[3]*x*x + p[4]*y*y + p[5]*x**3 + p[6]*x*y*y
    gy = p[2]*y + 2.0*p[4]*x*y + p[6]*x*x*y + p[7]*y**3
    return sag, gx, gy


def surface_z(surface, x, y):
    u, v = _local(surface, x, y)
    sag = _sag_grad(surface, u, v)[0]
    return np.where(u*u + v*v <= surface["aperture"]**2,
                    surface["apex"] + sag, np.inf)


def _gaussian_input(surface, points, lam_nm):
    """Gaussian mode wavefront incident on the printed surface from its core."""
    lam = lam_nm * 1e-3
    h = surface["base_z"] - surface["apex"]
    zr = np.pi * MCF_IPS_N * W_MODE**2 / lam
    radius = h * (1.0 + (zr / max(h, 1e-9))**2)
    er, _, _ = _frame(surface)
    core_xy = surface["core_r"] * er
    virtual = np.array([core_xy[0], core_xy[1], surface["apex"] + radius])
    direction = _unit(points - virtual)
    width = W_MODE * np.sqrt(1.0 + (h / zr)**2)
    r2 = (points[:, 0] - core_xy[0])**2 + (points[:, 1] - core_xy[1])**2
    return direction, np.exp(-2.0 * r2 / width**2), width


def _desired_air(points, target_depth, n_dia):
    """Air directions from surface points to on-axis target through diamond."""
    rho = np.hypot(points[:, 0], points[:, 1])
    lo = np.zeros_like(rho)
    hi = np.full_like(rho, np.arcsin(1.0 / n_dia) - 1e-9)
    for _ in range(55):
        td = 0.5 * (lo + hi)
        ta = np.arcsin(np.clip(n_dia * np.sin(td), -1.0, 1.0))
        reached = target_depth * np.tan(td) + points[:, 2] * np.tan(ta)
        lo = np.where(reached < rho, td, lo)
        hi = np.where(reached >= rho, td, hi)
    td = 0.5 * (lo + hi)
    ta = np.arcsin(np.clip(n_dia * np.sin(td), -1.0, 1.0))
    ux = np.divide(-points[:, 0], rho, out=np.zeros_like(rho), where=rho > 0)
    uy = np.divide(-points[:, 1], rho, out=np.zeros_like(rho), where=rho > 0)
    return np.column_stack([ux*np.sin(ta), uy*np.sin(ta), -np.cos(ta)])


def _fit_matrix(family, x, y):
    """Return gradient design matrix and map solution to the 8 sag coefficients."""
    z = np.zeros_like(x)
    if family in ("quadratic", "spherical"):
        ax = np.column_stack([np.ones_like(x), x])
        ay = np.column_stack([z, y])
        def unpack(q): return np.array([q[0], q[1], q[1], 0, 0, 0, 0, 0])
    elif family == "asphere":
        r2 = x*x + y*y
        ax = np.column_stack([np.ones_like(x), x, x*r2])
        ay = np.column_stack([z, y, y*r2])
        def unpack(q): return np.array([q[0], q[1], q[1], 0, 0, q[2], q[2], q[2]])
    elif family == "biconic":
        r2 = x*x + y*y
        ax = np.column_stack([np.ones_like(x), x, z, x*r2])
        ay = np.column_stack([z, z, y, y*r2])
        def unpack(q): return np.array([q[0], q[1], q[2], 0, 0, q[3], q[3], q[3]])
    else:  # freeform: off-axis odd terms + independent radial/tangential power
        ax = np.column_stack([np.ones_like(x), x, z, x*x, y*y, x**3, x*y*y, z])
        ay = np.column_stack([z, z, y, z, 2*x*y, z, x*x*y, y**3])
        def unpack(q): return q
    return np.vstack([ax, ay]), unpack


def fit_surface(family, role, gap, base_height, aperture, center_r=0.0,
                side_offset=0.0, target_depth=85.0):
    """Fit a physical Snell surface for a core-to-target wavefront."""
    surface = dict(family=family, role=role, center_r=float(center_r), angle=0.0,
                   core_r=0.0 if role == "central" else CORE_R,
                   apex=float(gap + (side_offset if role == "side" else 0.0)),
                   base_z=float(gap + base_height), aperture=float(aperture),
                   coef=np.zeros(8), shift=0.0)
    lam = 532.0 if role == "central" else 750.0
    n_dia = float(diamond_sellmeier(lam / 1000.0))
    xx, yy, _ = _disk(27)
    u, v = aperture * xx, aperture * yy
    er, et, center = _frame(surface)
    xy = center + u[:, None]*er + v[:, None]*et

    for _ in range(4):
        sag = _sag_grad(surface, u, v)[0]
        points = np.column_stack([xy, surface["apex"] + sag])
        inc, weight, _ = _gaussian_input(surface, points, lam)
        desired = _desired_air(points, target_depth, n_dia)
        normal = _unit(MCF_IPS_N * inc - N_AIR * desired)
        normal = np.where((normal[:, 2] > 0)[:, None], -normal, normal)
        gxg, gyg = normal[:, 0] / -normal[:, 2], normal[:, 1] / -normal[:, 2]
        gx, gy = gxg*er[0] + gyg*er[1], gxg*et[0] + gyg*et[1]
        A, unpack = _fit_matrix(family, xx, yy)
        sw = np.sqrt(np.r_[weight, weight])
        q = np.linalg.lstsq(A * sw[:, None], np.r_[gx, gy] * sw, rcond=None)[0]
        surface["coef"] = unpack(q)
        dense_x, dense_y, _ = _disk(61)
        raw = _sag_grad(surface, aperture*dense_x, aperture*dense_y)[0]
        surface["shift"] -= min(0.0, float(raw.min()))
    return surface


def rotate_surface(surface, angle):
    out = dict(surface)
    out["angle"] = float(angle)
    return out


def replicated_surfaces(central, side):
    return [central] + [rotate_surface(side, k*np.pi/3.0) for k in range(6)]


def trace_mode(surface, all_surfaces, lam_nm, depths=SEARCH_DEPTHS, n_grid=31,
               tilt_deg=0.0):
    """Trace the core mode through the exposed part of one physical surface."""
    qx, qy, da = _disk(n_grid)
    a = surface["aperture"]
    u, v = a*qx, a*qy
    er, et, center = _frame(surface)
    xy = center + u[:, None]*er + v[:, None]*et
    sag, gx, gy = _sag_grad(surface, u, v)
    p = np.column_stack([xy, surface["apex"] + sag])
    z_self = p[:, 2]
    visible = np.ones(len(p), dtype=bool)
    for other in all_surfaces:
        if other is surface:
            continue
        visible &= surface_z(other, p[:, 0], p[:, 1]) >= z_self - 1e-7

    inc, mode, width = _gaussian_input(surface, p, lam_nm)
    grad = gx[:, None]*er + gy[:, None]*et
    normal = _unit(np.column_stack([grad, -np.ones(len(p))]))
    air, t1, ok1 = _fresnel_refract(inc, normal, MCF_IPS_N, N_AIR)
    tilt = np.deg2rad(float(tilt_deg))
    diamond_normal = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    denominator = air @ diamond_normal
    dt = -(p @ diamond_normal) / np.maximum(denominator, 1e-12)
    at_surface = p + dt[:, None]*air
    n_dia = float(diamond_sellmeier(lam_nm / 1000.0))
    dia, t2, ok2 = _fresnel_refract(
        air, np.tile(diamond_normal, (len(p), 1)), N_AIR, n_dia)
    valid = visible & ok1 & ok2 & (denominator > 0.0) & (dt >= 0.0)
    area = da*a*a
    norm_power = 0.5*np.pi*width*width
    weight = mode * area / norm_power * t1 * t2 * valid
    hits = []
    for d in np.atleast_1d(depths):
        td = (float(d) - at_surface @ diamond_normal) / np.maximum(
            dia @ diamond_normal, 1e-12)
        hits.append(at_surface + td[:, None]*dia)
    return dict(points=p, air_surface=at_surface, hits=np.asarray(hits),
                weight=weight, throughput=float(weight.sum()), valid=valid,
                incident=inc, air=air, diamond=dia, mode_width=width,
                surface_normal=normal, diamond_normal=diamond_normal,
                tilt_deg=float(tilt_deg))


def beam_stats(trace, lam_nm, depths=SEARCH_DEPTHS):
    """Weighted Gaussian-equivalent footprint, including the diffraction floor."""
    w = trace["weight"]
    total = max(w.sum(), 1e-30)
    rows = []
    n_dia = float(diamond_sellmeier(lam_nm / 1000.0))
    tilt = np.deg2rad(float(trace.get("tilt_deg", 0.0)))
    tangent_x = np.array([np.cos(tilt), 0.0, np.sin(tilt)])
    tangent_y = np.array([0.0, 1.0, 0.0])
    diamond_normal = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    angle = np.arccos(np.clip(trace["diamond"] @ diamond_normal, -1.0, 1.0))
    theta = max(float(np.percentile(angle[trace["valid"]], 90)) if np.any(trace["valid"]) else 0.0,
                1e-4)
    w_diff = (lam_nm*1e-3) / (np.pi*n_dia*np.sin(theta))
    for d, h in zip(np.atleast_1d(depths), trace["hits"]):
        xy = np.column_stack([h @ tangent_x, h @ tangent_y])
        mean = (w[:, None]*xy).sum(axis=0)/total
        q = xy - mean
        cov = (q.T*w)@q/total + np.eye(2)*(w_diff/2.0)**2
        eig = np.linalg.eigvalsh(cov)
        fwhm = 2.0*np.sqrt(2.0*np.log(2.0))*np.sqrt(np.mean(eig))
        rows.append(dict(depth=float(d), mean=mean, cov=cov, fwhm=float(fwhm),
                         throughput=trace["throughput"], w_diff=float(w_diff)))
    return rows


def _ray_density(trace, depth_index, axis, lam_nm, rotations=(0.0,)):
    """Deposit the actual weighted ray hits and apply only diffraction blur."""
    valid = trace["valid"] & (trace["weight"] > 0.0)
    if not np.any(valid):
        return np.zeros((len(axis), len(axis)))
    tilt = np.deg2rad(float(trace.get("tilt_deg", 0.0)))
    tangent_x = np.array([np.cos(tilt), 0.0, np.sin(tilt)])
    tangent_y = np.array([0.0, 1.0, 0.0])
    hits = trace["hits"][depth_index, valid]
    base_xy = np.column_stack([hits @ tangent_x, hits @ tangent_y])
    rotations = np.atleast_1d(rotations)
    xy = np.vstack([
        base_xy @ np.array([[np.cos(a), np.sin(a)],
                            [-np.sin(a), np.cos(a)]])
        for a in rotations
    ])

    dx = axis[1]-axis[0]
    qx = (xy[:, 0]-axis[0])/dx
    qy = (xy[:, 1]-axis[0])/dx
    ix, iy = np.floor(qx).astype(int), np.floor(qy).astype(int)
    fx, fy = qx-ix, qy-iy
    density = np.zeros((len(axis), len(axis)))
    weights = np.tile(trace["weight"][valid], len(rotations))
    for ox, oy, fraction in ((0, 0, (1-fx)*(1-fy)),
                             (1, 0, fx*(1-fy)),
                             (0, 1, (1-fx)*fy),
                             (1, 1, fx*fy)):
        xj, yj = ix+ox, iy+oy
        inside = ((xj >= 0) & (xj < len(axis)) &
                  (yj >= 0) & (yj < len(axis)))
        np.add.at(density, (yj[inside], xj[inside]),
                  weights[inside]*fraction[inside])

    n_dia = float(diamond_sellmeier(lam_nm/1000.0))
    diamond_normal = trace["diamond_normal"]
    angle = np.arccos(np.clip(trace["diamond"][valid] @ diamond_normal, -1.0, 1.0))
    theta = max(float(np.percentile(angle, 90)), 1e-4)
    diffraction_sigma = (lam_nm*1e-3)/(2*np.pi*n_dia*np.sin(theta))
    return gaussian_filter(
        density, diffraction_sigma/dx, mode="constant")/(dx*dx)


def surface_limits(surface):
    x, y, _ = _disk(81)
    sag, gx, gy = _sag_grad(surface, surface["aperture"]*x,
                            surface["aperture"]*y)
    return dict(min_z=float(surface["apex"] + sag.min()),
                max_z=float(surface["apex"] + sag.max()),
                max_slope=float(np.sqrt(gx*gx + gy*gy).max()),
                print_height=float(surface["base_z"] - surface["apex"] - sag.min()))


def candidate_merit(surface, role):
    """Cheap physical merit used only to shortlist surfaces."""
    if role == "central":
        tr = trace_mode(surface, [surface], 532.0)
        st = beam_stats(tr, 532.0)
        return np.mean([x["throughput"] / max(x["fwhm"]**2, 1e-12) for x in st])
    vals = []
    for lam, sw in zip(SEARCH_RED_LAM, SEARCH_RED_W):
        tr = trace_mode(surface, [surface], lam)
        st = beam_stats(tr, lam)
        vals.append(sw*np.mean([x["throughput"] / max(x["fwhm"]**2, 1e-12) for x in st]))
    return float(np.sum(vals))


def evaluate_design(central, side, grid_n=121, tilt_deg=0.0, depths=DEPTHS,
                    red_lam=RED_LAM, red_w=RED_W, ray_grid=35):
    """Full 3-D layer integration, including a diamond-plane tilt about y.

    Fresnel transmission is already present in ``trace_mode`` at both physical
    interfaces.  The excitation power is therefore multiplied only by the
    traced throughput here.
    """
    depths = np.asarray(depths, dtype=float)
    red_lam = np.asarray(red_lam, dtype=float)
    red_w = np.asarray(red_w, dtype=float)
    if len(depths) < 2 or len(red_lam) != len(red_w):
        raise ValueError("depth and spectral quadratures are inconsistent")
    union = replicated_surfaces(central, side)
    central_trace = trace_mode(central, union, 532.0, depths, ray_grid,
                               tilt_deg=tilt_deg)
    cstats = beam_stats(central_trace, 532.0, depths)
    sstats = []
    side_traces = []
    side_surfaces = [side] if abs(float(tilt_deg)) < 1e-12 else union[1:]
    for lam, sw in zip(red_lam, red_w):
        traces = [trace_mode(surface, union, lam, depths, ray_grid,
                             tilt_deg=tilt_deg) for surface in side_surfaces]
        rows = [beam_stats(trace, lam, depths) for trace in traces]
        sstats.append((lam, sw, rows))
        side_traces.append((lam, sw, traces))

    all_stats = cstats + [x for _, _, groups in sstats for rows in groups for x in rows]
    lim = max(5.0, max(np.max(np.abs(x["mean"])) + 5.0*x["fwhm"]
                       for x in all_stats))
    lim = min(lim, 140.0)
    axis = np.linspace(-lim, lim, grid_n)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    dx = axis[1]-axis[0]
    dz = float(depths[-1]-depths[0])/(len(depths)-1)
    photons = 0.0
    moments = np.zeros(6)  # W, Wx, Wy, Wxx, Wyy, Wxy

    for iz, depth in enumerate(depths):
        intensity = P_GREEN_MW*_ray_density(central_trace, iz, axis, 532.0)
        sat = intensity/I_SAT
        rate = R_SAT*sat/(1.0+sat)

        eta = np.zeros_like(x)
        for lam, sw, traces in side_traces:
            n = float(diamond_sellmeier(lam/1000.0))
            mode_area = (lam*1e-3)**2/(8*np.pi*n*n)
            if len(traces) == 1:
                density = _ray_density(
                    traces[0], iz, axis, lam, np.arange(6)*np.pi/3.0)
            else:
                density = sum(_ray_density(trace, iz, axis, lam)
                              for trace in traces)
            eta += sw*mode_area*density
        eta = np.minimum(eta, 1.0)
        sig = RHO*rate*eta
        volume = dx*dx*dz*(0.5 if iz in (0, len(depths)-1) else 1.0)
        photons += sig.sum()*volume
        ww = sig*volume
        moments += [ww.sum(), (ww*x).sum(), (ww*y).sum(),
                    (ww*x*x).sum(), (ww*y*y).sum(), (ww*x*y).sum()]

    if (not np.isfinite(photons) or photons <= 0.0 or
            not np.isfinite(moments[0]) or moments[0] <= 0.0):
        raise ValueError("design produces no finite collected signal")
    comparison_cps = photons*COMPARISON_NORMALIZATION["factor"]
    mx, my = moments[1]/moments[0], moments[2]/moments[0]
    vx = moments[3]/moments[0]-mx*mx
    vy = moments[4]/moments[0]-my*my
    resolution = 2*np.sqrt(2*np.log(2))*np.sqrt(0.5*(vx+vy))
    fwhm = linewidth_from_resolution(resolution)
    contrast = MEASURED["MCF"]["contrast"]  # separated pump/collection paths retained
    raw_sensitivity = calibrated_sensitivity(max(photons, 1e-30), contrast, fwhm)
    normalized_sensitivity = calibrated_sensitivity(
        max(comparison_cps, 1e-30), contrast, fwhm)
    return dict(model_fiber_photons_s=float(photons),
                comparison_normalized_cps=float(comparison_cps),
                contrast=float(contrast), fwhm_mhz=float(fwhm),
                resolution_um=float(resolution),
                raw_model_sensitivity_nt=float(raw_sensitivity),
                comparison_normalized_sensitivity_nt=float(normalized_sensitivity),
                tilt_deg=float(tilt_deg),
                central_stats=cstats, side_stats=sstats)


def design_at_gap(design, gap_um):
    """Move the unchanged printed tip so its central minimum is at gap_um."""
    gap = float(gap_um)
    if not 5.0 <= gap <= 500.0:
        raise ValueError("gap_um must be within 5..500")
    delta = gap-surface_limits(design["central"])["min_z"]
    shifted = {}
    for name in ("central", "side"):
        shifted[name] = dict(design[name])
        shifted[name]["apex"] += delta
        shifted[name]["base_z"] += delta
    return shifted


def alignment_sweep(design, gaps_um, angles_deg, grid_n=81):
    """Evaluate fixed-tip Z spacing, then diamond tilt at the best spacing."""
    gap_rows = []
    for gap in np.asarray(gaps_um, dtype=float):
        shifted = design_at_gap(design, gap)
        result = evaluate_design(shifted["central"], shifted["side"], grid_n=grid_n,
                                 depths=SEARCH_DEPTHS, red_lam=SEARCH_RED_LAM,
                                 red_w=SEARCH_RED_W, ray_grid=31)
        gap_rows.append(dict(
            gap_um=float(gap), model_fiber_photons_s=result["model_fiber_photons_s"],
            comparison_normalized_cps=result["comparison_normalized_cps"],
            raw_model_sensitivity_nt=result["raw_model_sensitivity_nt"],
            comparison_normalized_sensitivity_nt=(
                result["comparison_normalized_sensitivity_nt"])))
    best = max(gap_rows, key=lambda row: row["model_fiber_photons_s"])
    shifted = design_at_gap(design, best["gap_um"])
    angle_rows = []
    for angle in np.asarray(angles_deg, dtype=float):
        result = evaluate_design(shifted["central"], shifted["side"], grid_n=grid_n,
                                 tilt_deg=angle, depths=SEARCH_DEPTHS,
                                 red_lam=SEARCH_RED_LAM, red_w=SEARCH_RED_W,
                                 ray_grid=31)
        angle_rows.append(dict(angle_deg=float(angle),
                               model_fiber_photons_s=result["model_fiber_photons_s"],
                               comparison_normalized_cps=result["comparison_normalized_cps"],
                               raw_model_sensitivity_nt=result["raw_model_sensitivity_nt"],
                               comparison_normalized_sensitivity_nt=(
                                   result["comparison_normalized_sensitivity_nt"])))
    return dict(best_gap_um=best["gap_um"], gap=gap_rows, angle=angle_rows)


SEARCH_GEOMETRY_NAMES = (
    "air_gap_um", "central_height_um", "side_height_um",
    "central_aperture_um", "central_side_overlap_um", "side_core_offset_um",
)
SEARCH_GEOMETRY_BOUNDS = (
    (5, 295), (5, 295), (5, 295), (5, 150), (0, 290), (-30, 110),
)
FAMILY_SHAPE_PARAMETERS = {
    "quadratic": ("radius_scale",),
    "spherical": ("radius_scale",),
    "asphere": ("radius_scale", "asphere_scale"),
    "biconic": ("radius_x_scale", "radius_y_scale", "asphere_scale"),
    "freeform": ("radius_x_scale", "radius_y_scale", "odd_scale",
                 "quartic_scale"),
}
SHAPE_BOUNDS = {
    "tilt_scale": (0.35, 2.5),
    "radius_scale": (0.35, 2.5),
    "radius_x_scale": (0.35, 2.5),
    "radius_y_scale": (0.35, 2.5),
    "asphere_scale": (-1.0, 3.0),
    "odd_scale": (-1.0, 3.0),
    "quartic_scale": (-1.0, 3.0),
}


def _shape_names(family, role):
    names = FAMILY_SHAPE_PARAMETERS[family]
    return (("tilt_scale",) + names) if role == "side" else names


def _apply_shape_parameters(surface, values):
    """Apply family-specific factors around the fitted Snell-law surface."""
    surface = dict(surface)
    coef = np.array(surface["coef"], dtype=float, copy=True)
    values = dict(values)
    coef[0] *= values.get("tilt_scale", 1.0)
    if "radius_scale" in values:
        coef[1:3] /= values["radius_scale"]
    if "radius_x_scale" in values:
        coef[1] /= values["radius_x_scale"]
    if "radius_y_scale" in values:
        coef[2] /= values["radius_y_scale"]
    coef[3:5] *= values.get("odd_scale", 1.0)
    coef[5:8] *= values.get(
        "quartic_scale", values.get("asphere_scale", 1.0))
    surface["coef"] = coef

    # Keep apex as the actual lowest point after changing the polynomial.
    surface["shift"] = 0.0
    x, y, _ = _disk(81)
    raw = _sag_grad(surface, surface["aperture"]*x,
                    surface["aperture"]*y)[0]
    surface["shift"] = -float(raw.min())
    radius_x = (surface["aperture"]/coef[1]
                if abs(coef[1]) > 1e-12 else None)
    radius_y = (surface["aperture"]/coef[2]
                if abs(coef[2]) > 1e-12 else None)
    surface["shape_parameters"] = {
        **{name: float(value) for name, value in values.items()},
        "radius_x_um": None if radius_x is None else float(radius_x),
        "radius_y_um": None if radius_y is None else float(radius_y),
        "vertex_tilt_deg": float(np.degrees(np.arctan(coef[0]))),
    }
    return surface


def _search_spec(central_family, side_family):
    central_names = _shape_names(central_family, "central")
    side_names = _shape_names(side_family, "side")
    names = (SEARCH_GEOMETRY_NAMES +
             tuple(f"central_{name}" for name in central_names) +
             tuple(f"side_{name}" for name in side_names))
    bounds = (SEARCH_GEOMETRY_BOUNDS +
              tuple(SHAPE_BOUNDS[name] for name in central_names) +
              tuple(SHAPE_BOUNDS[name] for name in side_names))
    x0 = np.r_[25.0, 200.0, 165.0, 20.0, 3.0, 0.0,
                np.ones(len(central_names)+len(side_names))]
    integrality = np.r_[np.ones(len(SEARCH_GEOMETRY_NAMES), dtype=bool),
                        np.zeros(len(names)-len(SEARCH_GEOMETRY_NAMES), dtype=bool)]
    return names, bounds, x0, integrality


def _search_surfaces(parameters, central_family, side_family):
    names, bounds, _, _ = _search_spec(central_family, side_family)
    if len(parameters) != len(names) or not np.all(np.isfinite(parameters)):
        return None
    values = dict(zip(names, np.asarray(parameters, dtype=float)))
    for name in SEARCH_GEOMETRY_NAMES:
        values[name] = float(np.rint(values[name]))
    gap = values["air_gap_um"]
    central_height = values["central_height_um"]
    side_height = values["side_height_um"]
    central_aperture = values["central_aperture_um"]
    overlap = values["central_side_overlap_um"]
    center = CORE_R + values["side_core_offset_um"]
    side_aperture = center + overlap - central_aperture
    base_z = gap + central_height
    side_apex = base_z - side_height
    if (any(value < lower-1e-12 or value > upper+1e-12
            for value, (lower, upper) in zip(parameters, bounds)) or
            center <= 0.0 or side_aperture < 5.0 or
            central_aperture > PRINT_X_UM/2.0 or
            center + side_aperture > PRINT_X_UM/2.0 or
            base_z > PRINT_Z_UM or side_apex < 5.0):
        return None

    central = fit_surface(central_family, "central", gap, central_height,
                          central_aperture)
    side = fit_surface(
        side_family, "side", gap, central_height, side_aperture,
        center_r=center, side_offset=central_height-side_height)
    offset = len(SEARCH_GEOMETRY_NAMES)
    central_names = _shape_names(central_family, "central")
    central = _apply_shape_parameters(
        central, zip(central_names, parameters[offset:offset+len(central_names)]))
    offset += len(central_names)
    side_names = _shape_names(side_family, "side")
    side = _apply_shape_parameters(side, zip(side_names, parameters[offset:]))
    for surface in (central, side):
        limits = surface_limits(surface)
        if (limits["min_z"] < 5.0 or limits["max_z"] >= surface["base_z"] or
                limits["max_z"] > PRINT_Z_UM or
                limits["print_height"] > PRINT_Z_UM):
            return None
    return central, side


def design_parameters(central, side):
    """Return the independent geometry and derived overlap in report units."""
    return {
        "central_lens_type": central["family"],
        "side_lens_type": side["family"],
        "air_gap_um": float(central["apex"]),
        "central_side_overlap_um": float(
            central["aperture"] + side["aperture"] - side["center_r"]),
        "side_side_overlap_um": float(
            2.0*side["aperture"] - side["center_r"]),
        "side_core_offset_um": float(side["center_r"] - CORE_R),
        "central_height_um": float(central["base_z"] - central["apex"]),
        "side_height_um": float(side["base_z"] - side["apex"]),
        "central_aperture_um": float(central["aperture"]),
        "side_aperture_um": float(side["aperture"]),
        "central_shape": central.get("shape_parameters", {}),
        "side_shape": side.get("shape_parameters", {}),
    }


@lru_cache(maxsize=100_000)
def _search_objective_cached(parameters, central_family, side_family, stage):
    try:
        surfaces = _search_surfaces(parameters, central_family, side_family)
        if surfaces is None:
            return np.inf
        central, side = surfaces
        if stage == "full":
            result = evaluate_design(
                central, side, grid_n=81, depths=SEARCH_DEPTHS,
                red_lam=SEARCH_RED_LAM, red_w=SEARCH_RED_W, ray_grid=31)
            merit = result["raw_model_sensitivity_nt"]
            return merit if np.isfinite(merit) else np.inf
        cm = candidate_merit(central, "central")
        sm = candidate_merit(side, "side")
        union = replicated_surfaces(central, side)
        ct = trace_mode(central, union, 532.0, n_grid=25)
        st = trace_mode(side, union, 750.0, n_grid=25)
        merit = cm*sm*ct["throughput"]*st["throughput"]
        return -merit if np.isfinite(merit) and merit > 0.0 else np.inf
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return np.inf


def _search_objective(parameters, central_family, side_family, stage):
    geometry = tuple(int(x) for x in np.rint(
        parameters[:len(SEARCH_GEOMETRY_NAMES)]))
    shape = tuple(float(np.round(x, 5))
                  for x in parameters[len(SEARCH_GEOMETRY_NAMES):])
    key = geometry + shape
    return _search_objective_cached(key, central_family, side_family, stage)


def _coordinate_refine(parameters, central_family, side_family):
    """Finish at 1-um geometry and 0.02-factor shape resolution."""
    _, bounds, _, integrality = _search_spec(central_family, side_family)
    point = np.asarray(parameters, dtype=float).copy()
    point[integrality] = np.rint(point[integrality])
    merit = _search_objective(point, central_family, side_family, "full")
    for geometry_step, shape_step in ((10, 0.2), (5, 0.1), (2, 0.05), (1, 0.02)):
        while True:
            candidates = []
            for dimension, (lower, upper) in enumerate(bounds):
                step = geometry_step if integrality[dimension] else shape_step
                for direction in (-1, 1):
                    candidate = point.copy()
                    candidate[dimension] = np.clip(
                        candidate[dimension]+direction*step, lower, upper)
                    if integrality[dimension]:
                        candidate[dimension] = np.rint(candidate[dimension])
                    candidates.append(candidate)
            scored = [(_search_objective(candidate, central_family, side_family,
                                         "full"), candidate)
                      for candidate in candidates]
            next_merit, next_point = min(scored, key=lambda row: row[0])
            if next_merit >= merit:
                break
            merit, point = next_merit, next_point
    return merit, point


class _ProgressBar:
    """Terminal bar, with 10% checkpoints when output is redirected to a log."""

    def __init__(self, label, total, width=30):
        self.label, self.total, self.width = label, max(int(total), 1), width
        self.current = 0
        self.last_log_bucket = -1

    def __call__(self, _x, _convergence):
        self.current += 1
        self._write(False)
        return False

    def finish(self):
        self.current = self.total
        self._write(True)

    def _write(self, finish):
        fraction = min(self.current/self.total, 1.0)
        percent = int(round(100*fraction))
        if sys.stdout.isatty():
            filled = int(round(self.width*fraction))
            bar = "#"*filled + "-"*(self.width-filled)
            print(f"\r  {self.label}: [{bar}] {percent:3d}% "
                  f"({self.current}/{self.total})", end="\n" if finish else "",
                  flush=True)
        else:
            bucket = 10 if finish else percent//10
            if bucket > self.last_log_bucket:
                self.last_log_bucket = bucket
                print(f"  {self.label}: {10*bucket}%", flush=True)


def search_design(families=("quadratic", "asphere", "biconic", "freeform"),
                  proxy_maxiter=120, full_maxiter=80, popsize=8, restarts=2,
                  workers=-1):
    """Deterministic global integer search followed by 1-um refinement."""
    families = tuple(families)
    if (not families or min(proxy_maxiter, full_maxiter) < 0 or
            popsize < 5 or restarts < 1):
        raise ValueError("invalid search configuration")
    worker_pool = None
    worker_map = 1
    if workers != 1:
        from multiprocessing import Pool
        worker_pool = Pool(None if workers == -1 else workers)
        worker_map = worker_pool.map

    best = None
    method_results = {}
    try:
        for method_index, (central_family, side_family) in enumerate(
                (c, s) for c in families for s in families):
            _, bounds, x0, integrality = _search_spec(
                central_family, side_family)
            method_best = None
            for restart in range(restarts):
                seed = 1847 + 101*method_index + restart
                label = f"{central_family} + {side_family}"
                print(f"Optimizing {label}, restart {restart+1}/{restarts}...",
                      flush=True)
                proxy_progress = _ProgressBar(f"{label} proxy", proxy_maxiter)
                proxy = differential_evolution(
                    _search_objective, bounds,
                    args=(central_family, side_family, "proxy"),
                    strategy="best1bin", maxiter=proxy_maxiter,
                    popsize=popsize, tol=0.0, atol=0.0,
                    mutation=(0.5, 1.0), recombination=0.8,
                    rng=np.random.default_rng(seed), polish=False,
                    init="sobol", x0=x0,
                    updating="deferred", workers=worker_map,
                    integrality=integrality, callback=proxy_progress)
                proxy_progress.finish()
                full_progress = _ProgressBar(f"{label} full", full_maxiter)
                full = differential_evolution(
                    _search_objective, bounds,
                    args=(central_family, side_family, "full"),
                    strategy="best1bin", maxiter=full_maxiter,
                    popsize=popsize, tol=0.0, atol=0.0,
                    mutation=(0.5, 1.0), recombination=0.8,
                    rng=np.random.default_rng(seed+10_000), polish=False,
                    init="sobol", x0=proxy.x, updating="deferred",
                    workers=worker_map, integrality=integrality,
                    callback=full_progress)
                full_progress.finish()
                print(f"  proxy={-proxy.fun:.6g}, "
                      f"search sensitivity={full.fun:.6g} nT/sqrt(Hz)",
                      flush=True)
                if method_best is None or full.fun < method_best[0]:
                    method_best = (full.fun, full.x)

            merit, parameters = _coordinate_refine(
                method_best[1], central_family, side_family)
            surfaces = _search_surfaces(parameters, central_family, side_family)
            if surfaces is None or not np.isfinite(merit):
                continue
            central, side = surfaces
            result = evaluate_design(
                central, side, grid_n=81, depths=SEARCH_DEPTHS,
                red_lam=SEARCH_RED_LAM, red_w=SEARCH_RED_W, ray_grid=31)
            label = f"{central_family} + {side_family}"
            method_results[label] = {k: result[k] for k in
                                     ("raw_model_sensitivity_nt",
                                      "comparison_normalized_sensitivity_nt",
                                      "model_fiber_photons_s",
                                      "comparison_normalized_cps", "resolution_um")}
            print(f"  refined sensitivity={merit:.6g} nT/sqrt(Hz), "
                  f"parameters={design_parameters(central, side)}", flush=True)
            if best is None or merit < best[0]:
                best = (merit, central, side)
    finally:
        if worker_pool is not None:
            worker_pool.close()
            worker_pool.join()

    if best is None:
        raise RuntimeError("search found no manufacturable design")
    final_result = evaluate_design(best[1], best[2], grid_n=161)
    return dict(central=best[1], side=best[2], result=final_result,
                parameters=design_parameters(best[1], best[2]),
                method_comparison=method_results)


def _surface_json(surface):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in surface.items()}


def write_design_json(design, path):
    payload = dict(central=_surface_json(design["central"]),
                   side=_surface_json(design["side"]),
                   parameters=design.get(
                       "parameters", design_parameters(
                           design["central"], design["side"])),
                   result={k: v for k, v in design["result"].items()
                           if k not in ("central_stats", "side_stats")},
                   method_comparison=design.get("method_comparison", {}),
                   measured=MEASURED,
                   comparison_normalization=COMPARISON_NORMALIZATION,
                   assumptions=dict(mfd_um=MCF_MFD, ips_index=MCF_IPS_N,
                                    nv_ppm=PPM, nv_depth_um=list(NV_DEPTH_UM),
                                    spectrum_nm=list(NV_SPECTRUM_NM),
                                    print_limits_um=[PRINT_X_UM, PRINT_Y_UM,
                                                     PRINT_Z_UM],
                                    final_depth_points=len(DEPTHS),
                                    final_spectral_points=len(RED_LAM),
                                    spectral_quadrature="normalized trapezoidal",
                                    spatial_model="weighted traced-ray footprints with diffraction blur",
                                    green_fresnel="included once by trace_mode",
                                    method_search_quadrature_points=9))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def write_binary_stl(design, path, nr=55, nt=144, all_sides=False):
    """Write one watertight support with one or all six exposed side lenses."""
    c, s = design["central"], design["side"]
    surfaces = replicated_surfaces(c, s) if all_sides else [c, s]
    base_z = c["base_z"]
    radius = min(PRINT_X_UM/2.0, max(60.0, c["aperture"],
                                    s["center_r"]+s["aperture"]))
    support = base_z-8.0
    rings = np.linspace(0.0, radius, nr)
    ang = np.arange(nt)*2*np.pi/nt
    top = np.empty((nr, nt, 3))
    for i, r in enumerate(rings):
        x, y = r*np.cos(ang), r*np.sin(ang)
        z = np.full(nt, support)
        for surf in surfaces:
            z = np.minimum(z, surface_z(surf, x, y))
        top[i] = np.column_stack([x, y, z])
    top[0, :, :2] = 0.0
    top[0, :, 2] = np.min(top[0, :, 2])
    bottom = top.copy(); bottom[:, :, 2] = base_z
    tri = []
    for j in range(nt):
        k = (j+1) % nt
        tri += [[top[0, 0], top[1, j], top[1, k]],
                [bottom[0, 0], bottom[1, k], bottom[1, j]]]
    for i in range(1, nr-1):
        for j in range(nt):
            k = (j+1) % nt
            tri += [[top[i, j], top[i+1, j], top[i+1, k]],
                    [top[i, j], top[i+1, k], top[i, k]],
                    [bottom[i, j], bottom[i+1, k], bottom[i+1, j]],
                    [bottom[i, j], bottom[i, k], bottom[i+1, k]]]
    for j in range(nt):
        k = (j+1) % nt
        tri += [[top[-1, j], bottom[-1, j], bottom[-1, k]],
                [top[-1, j], bottom[-1, k], top[-1, k]]]
    tri = np.asarray(tri, dtype="<f4")
    normals = _unit(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])).astype("<f4")
    with open(path, "wb") as fh:
        label = b"MCF central plus six side physical lenses" if all_sides else \
                b"MCF central plus one side physical lens"
        fh.write(label.ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(tri)))
        for n, t in zip(normals, tri):
            fh.write(n.tobytes()); fh.write(t.tobytes()); fh.write(b"\0\0")
    return len(tri)


def validate_design(design):
    c, s = design["central"], design["side"]
    assert c["apex"] >= 5.0 and s["apex"] >= 5.0
    assert np.isclose(c["base_z"], s["base_z"])
    assert c["base_z"] <= PRINT_Z_UM+1e-9
    assert c["aperture"] <= PRINT_X_UM/2.0+1e-9
    assert s["center_r"]+s["aperture"] <= PRINT_X_UM/2.0+1e-9
    for surf in (c, s):
        lim = surface_limits(surf)
        assert lim["print_height"] <= MAX_PRINT+1e-9
        assert lim["min_z"] >= 0.0 and lim["max_z"] <= PRINT_Z_UM+1e-9
        assert lim["max_z"] < surf["base_z"]
        assert np.all(np.isfinite(surf["coef"]))
    r = design["result"]
    assert r["model_fiber_photons_s"] > 0
    assert r["comparison_normalized_cps"] > 0
    assert r["raw_model_sensitivity_nt"] > 0
    assert r["comparison_normalized_sensitivity_nt"] > 0
    return True


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(out, exist_ok=True)
    design = search_design()
    validate_design(design)
    write_design_json(design, os.path.join(out, "mcf_freeform_design.json"))
    ntri = write_binary_stl(design, os.path.join(out, "mcf_freeform_central_one_side.stl"))
    nfull = write_binary_stl(design, os.path.join(out, "mcf_freeform_full_seven_core.stl"),
                             all_sides=True)
    print(json.dumps({k: v for k, v in design["result"].items()
                      if k not in ("central_stats", "side_stats")}, indent=2))
    print(f"STL triangles: one-side={ntri}, full={nfull}")
