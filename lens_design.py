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
MCF_FULL_NA = float(os.environ.get("MCF_FULL_NA", "0.22"))
if not 0.0 < MCF_FULL_NA < MCF_IPS_N:
    raise ValueError("MCF_FULL_NA must be between 0 and the IP-S index")
PRINT_X_UM = PRINT_Y_UM = PRINT_Z_UM = 300.0
MAX_PRINT = PRINT_Z_UM  # compatibility name used by the fabrication scripts
NV_DEPTH_UM = (80.0, 90.0)
NV_SPECTRUM_NM = (650.0, 850.0)

# A printed cap is an optical surface only where its own core illuminates it.
# fit_surface constrains roughly one beam footprint, so an aperture far beyond
# that is unconstrained polynomial extrapolation: it can sculpt the polymer
# union, shadow the neighbouring caps, and move the reported apex into a region
# no light ever reaches.  The aperture is therefore derived from the beam
# footprint rather than searched freely, and every cap must additionally remain
# the exposed boundary over its own illuminated footprint.
APERTURE_MARGIN = 1.35   # cap radius / beam footprint, upper limit
EXPOSURE_MIN = 0.85      # min fraction of the lit footprint a cap must own
CLEARANCE_TOL = 0.5      # max (optical clearance - apex) / footprint

# Printability guard only: a height field cannot overhang, so this just keeps
# the rim off a near-vertical wall.  It is deliberately NOT set at the
# IP-S->air critical angle (41.1 deg, slope 0.87): a convex cap is steep at its
# rim by nature, that bound admits only ~6% of otherwise valid designs, and
# rays past the critical angle are already given zero transmission by
# _fresnel_refract.  The loss is therefore priced by the objective and reported
# as tir_fraction rather than forbidden here.
MAX_SURFACE_SLOPE = 3.0   # tan of the steepest allowed tilt (~72 deg)
MIN_POLYMER_UM = 5.0      # thinnest polymer left between a cap and its base
# The integration grid is snapped to this ladder so that two nearby candidates
# share an identical grid.  Without it the mesh pitch tracked each candidate's
# own beam width and the 1-um refinement partly chased discretization noise.
FIELD_LIMIT_QUANTUM_UM = 5.0
FIELD_LIMIT_MAX_UM = 140.0


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


def beam_footprint(height):
    """Radius a core's full-NA beam covers after ``height`` of polymer.

    Launch positions span 2*W_MODE across the mode field (``trace_full_na``),
    and every ray then diverges at the full-NA half angle inside the IP-S.
    """
    return 2.0*W_MODE + max(float(height), 0.0)*np.tan(
        np.arcsin(MCF_FULL_NA/MCF_IPS_N))


def lit_radius(surface):
    """Illuminated footprint radius of one printed cap."""
    return beam_footprint(surface["base_z"]-surface["apex"])


def optical_clearance(surface):
    """Lowest point of the cap inside the footprint its own core lights.

    ``apex`` is the minimum over the whole aperture, which is only the
    clearance the beam sees when the cap is sized to its own footprint.
    Reported alongside the apex so the two can never be silently conflated.
    """
    er = _frame(surface)[0]
    qx, qy, _ = _disk(41)
    points = surface["core_r"]*er + lit_radius(surface)*np.column_stack([qx, qy])
    z = surface_z(surface, points[:, 0], points[:, 1])
    z = z[np.isfinite(z)]
    return float(z.min()) if z.size else float("inf")


def exposure_fraction(index, all_surfaces, tol=1e-6):
    """Fraction of a cap's lit footprint on which it is the exposed boundary.

    Rays refract off whichever surface of the printed union is lowest.  A cap
    that loses this fraction is not the optic its own core is using.
    """
    surface = all_surfaces[index]
    er = _frame(surface)[0]
    qx, qy, _ = _disk(25)
    points = surface["core_r"]*er + lit_radius(surface)*np.column_stack([qx, qy])
    z_self = surface_z(surface, points[:, 0], points[:, 1])
    exposed = np.isfinite(z_self)
    for other_index, other in enumerate(all_surfaces):
        if other_index == index:
            continue
        exposed &= z_self <= surface_z(other, points[:, 0], points[:, 1])+tol
    return float(exposed.mean())


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
    # 34 halvings take a <0.5 rad bracket below 1e-10 rad; 55 was far past
    # double precision and this is the inner loop of every surface fit.
    for _ in range(34):
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
    """Fit a physical Snell surface from a full-NA core to the target."""
    surface = dict(family=family, role=role, center_r=float(center_r), angle=0.0,
                   core_r=0.0 if role == "central" else CORE_R,
                   apex=float(gap + (side_offset if role == "side" else 0.0)),
                   base_z=float(gap + base_height), aperture=float(aperture),
                   coef=np.zeros(8), shift=0.0)
    lam = 532.0 if role == "central" else 750.0
    n_dia = float(diamond_sellmeier(lam / 1000.0))
    er, et, center = _frame(surface)
    core_xy = surface["core_r"]*er
    origin = np.r_[core_xy, surface["base_z"]]
    theta_max = np.arcsin(MCF_FULL_NA/MCF_IPS_N)
    fit_radius = max(5.0, 1.15*(surface["base_z"]-surface["apex"])*np.tan(theta_max))
    qx, qy, _ = _disk(27)
    xy_all = core_xy + fit_radius*np.column_stack([qx, qy])
    local = xy_all-center
    u_all, v_all = local@er, local@et
    inside = u_all*u_all+v_all*v_all <= aperture*aperture
    u, v, xy = u_all[inside], v_all[inside], xy_all[inside]
    if len(u) < 20:
        raise ValueError("lens aperture does not intercept enough of the full-NA cone")
    xx, yy = u/aperture, v/aperture

    for _ in range(4):
        sag = _sag_grad(surface, u, v)[0]
        points = np.column_stack([xy, surface["apex"] + sag])
        offset = points-origin
        distance_sq = np.sum(offset*offset, axis=1)
        inc = _unit(offset)
        weight = np.maximum(-inc[:, 2], 0.0)/np.maximum(distance_sq, 1e-12)
        weight /= max(float(weight.max()), 1e-30)
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
    # Sample the illuminated Gaussian mode, not the potentially 300-um lens
    # diameter.  The old aperture-wide grid could have 10-um spacing for a
    # 10-um MFD and therefore mis-integrate the optical power.
    a = surface["aperture"]
    er, et, center = _frame(surface)
    lam = lam_nm*1e-3
    height = surface["base_z"]-surface["apex"]
    rayleigh = np.pi*MCF_IPS_N*W_MODE**2/lam
    mode_width = W_MODE*np.sqrt(1.0+(height/rayleigh)**2)
    sample_axis = np.linspace(-3.5*mode_width, 3.5*mode_width, int(n_grid))
    sx, sy = np.meshgrid(sample_axis, sample_axis, indexing="xy")
    core_xy = surface["core_r"]*er
    xy_all = core_xy + np.column_stack([sx.ravel(), sy.ravel()])
    local = xy_all-center
    u_all, v_all = local@er, local@et
    illuminated = u_all*u_all+v_all*v_all <= a*a
    u, v, xy = u_all[illuminated], v_all[illuminated], xy_all[illuminated]
    if not len(u):
        raise ValueError("lens aperture does not intercept the fiber mode")
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
    area = (sample_axis[1]-sample_axis[0])**2
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


def _union_boundary_arrays(all_surfaces, base_z, x, y):
    """Vectorized exposed polymer-union surface and outward normal."""
    x, y = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
    best_z = np.full(len(x), float(base_z)-8.0)
    best_normal = np.tile([0.0, 0.0, -1.0], (len(x), 1))
    for candidate in all_surfaces:
        u, v = _local(candidate, x, y)
        sag, gx, gy = _sag_grad(candidate, u, v)
        candidate_z = np.where(
            u*u+v*v <= candidate["aperture"]**2,
            candidate["apex"]+sag, np.inf)
        lower = candidate_z < best_z
        if not np.any(lower):
            continue
        er, et, _ = _frame(candidate)
        grad = gx[:, None]*er+gy[:, None]*et
        candidate_normal = _unit(
            np.column_stack([grad, -np.ones(len(x))]))
        best_z = np.where(lower, candidate_z, best_z)
        best_normal = np.where(lower[:, None], candidate_normal, best_normal)
    return best_z, best_normal


def _union_boundary(all_surfaces, base_z, x, y):
    z, normal = _union_boundary_arrays(all_surfaces, base_z, [x], [y])
    return float(z[0]), normal[0]


def trace_full_na(surface, all_surfaces, lam_nm, depths=SEARCH_DEPTHS,
                  n_grid=31, tilt_deg=0.0, fiber_na=MCF_FULL_NA):
    """Trace the Gaussian core area with a uniformly filled full-NA cone."""
    if not 0.0 < fiber_na < MCF_IPS_N or n_grid < 9:
        raise ValueError("invalid full-NA trace configuration")
    er, _, _ = _frame(surface)
    core_xy = surface["core_r"]*er

    # Thirteen Gaussian-weighted positions across the 10-um MFD, combined
    # with equal-solid-angle angular cells.  Total rays stay comparable to the
    # old aperture grid so the exhaustive search remains practical.
    px, py, _ = _disk(5)
    position_xy = core_xy+2.0*W_MODE*np.column_stack([px, py])
    position_weight = np.exp(-2.0*(px*px+py*py)*4.0)
    position_weight /= position_weight.sum()

    n_radial = max(3, int(round(n_grid/8)))
    n_phi = max(12, int(round(n_grid/2)))
    theta_max = float(np.arcsin(fiber_na/MCF_IPS_N))
    mu_edges = np.linspace(np.cos(theta_max), 1.0, n_radial+1)
    mu = 0.5*(mu_edges[:-1]+mu_edges[1:])
    directions, theta_rows = [], []
    for ring, cos_theta in enumerate(mu):
        theta = float(np.arccos(cos_theta))
        phi = 2*np.pi*(np.arange(n_phi)+0.5*(ring % 2))/n_phi
        directions.append(np.column_stack([
            np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),
            np.full(n_phi, -cos_theta)]))
        theta_rows.append(np.full(n_phi, theta))
    directions = np.vstack(directions)
    theta_rows = np.concatenate(theta_rows)
    n_angles = len(directions)
    origins = np.repeat(
        np.column_stack([position_xy,
                         np.full(len(position_xy), surface["base_z"])]),
        n_angles, axis=0)
    incident = np.tile(directions, (len(position_xy), 1))
    launch_theta = np.tile(theta_rows, len(position_xy))
    launch_weight = np.repeat(position_weight, n_angles)/n_angles

    lo = np.zeros(len(origins))
    hi = (origins[:, 2]+10.0)/np.maximum(-incident[:, 2], 1e-12)
    # 30 halvings resolve a 300-um bracket to <1e-6 um, well inside the
    # geometry tolerance; this bisection dominates the whole search cost.
    for _ in range(30):
        mid = 0.5*(lo+hi)
        trial = origins+mid[:, None]*incident
        boundary_z = _union_boundary_arrays(
            all_surfaces, surface["base_z"], trial[:, 0], trial[:, 1])[0]
        above = trial[:, 2] > boundary_z
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    distance = 0.5*(lo+hi)
    points = origins+distance[:, None]*incident
    boundary_z, normal = _union_boundary_arrays(
        all_surfaces, surface["base_z"], points[:, 0], points[:, 1])
    points[:, 2] = boundary_z
    air, t1, ok1 = _fresnel_refract(
        incident, normal, MCF_IPS_N, N_AIR)

    tilt = np.deg2rad(float(tilt_deg))
    diamond_normal = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    denominator = air@diamond_normal
    dt = np.divide(-(points@diamond_normal), denominator,
                   out=np.full(len(points), np.nan), where=denominator > 1e-12)
    at_surface = points+dt[:, None]*air
    n_dia = float(diamond_sellmeier(lam_nm/1000.0))
    dia, t2, ok2 = _fresnel_refract(
        air, np.tile(diamond_normal, (len(points), 1)), N_AIR, n_dia)
    valid = ok1 & ok2 & (denominator > 0.0) & np.isfinite(dt) & (dt >= 0.0)
    weight = launch_weight*t1*t2*valid
    hits = []
    for depth in np.atleast_1d(depths):
        td = np.divide(float(depth)-at_surface@diamond_normal,
                       dia@diamond_normal, out=np.full(len(points), np.nan),
                       where=dia@diamond_normal > 1e-12)
        hits.append(at_surface+td[:, None]*dia)
    # Share of the launched power turned back at the polymer->air surface,
    # i.e. rays meeting the cap past the critical angle.  Priced by the
    # objective through the zero transmission, surfaced here so a design that
    # buys its spot size by vignetting its own rim is visible rather than
    # merely implied.
    launched = max(float(launch_weight.sum()), 1e-30)
    tir_fraction = float((launch_weight*(~ok1)).sum())/launched
    return dict(origins=origins, points=points, air_surface=at_surface,
                hits=np.asarray(hits), weight=weight, tir_fraction=tir_fraction,
                throughput=float(weight.sum()), valid=valid,
                incident=incident, air=air, diamond=dia,
                mode_width=W_MODE, surface_normal=normal,
                diamond_normal=diamond_normal, tilt_deg=float(tilt_deg),
                launch_theta=launch_theta, theta_max=theta_max,
                fiber_na=float(fiber_na), ray_model="uniform_full_na")


def trace_na_cone(surface, all_surfaces, lam_nm, fiber_na=MCF_FULL_NA,
                  depth=85.0, n_theta=5, n_phi=18):
    """Trace an unweighted geometric full-NA envelope from one core.

    This draws the cone boundary/interior for visualization. Phase 3 uses the
    weighted equal-solid-angle quadrature in ``trace_full_na``.
    """
    if not 0.0 < fiber_na < MCF_IPS_N or n_theta < 2 or n_phi < 4:
        raise ValueError("invalid full-NA cone configuration")
    er, _, _ = _frame(surface)
    origin = np.r_[surface["core_r"]*er, surface["base_z"]]
    theta_max = float(np.arcsin(fiber_na/MCF_IPS_N))
    launch = [(0.0, 0.0, np.array([0.0, 0.0, -1.0]))]
    for theta in np.linspace(0.0, theta_max, int(n_theta))[1:]:
        for phi in np.linspace(0.0, 2*np.pi, int(n_phi), endpoint=False):
            launch.append((float(theta), float(phi), np.array([
                np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),
                -np.cos(theta)])))

    paths, theta_rows, phi_rows = [], [], []
    diamond_normal = np.array([0.0, 0.0, -1.0])
    n_dia = float(diamond_sellmeier(lam_nm/1000.0))
    for theta, phi, direction in launch:
        t_end = (origin[2]+10.0)/max(-direction[2], 1e-12)
        previous_t = 0.0
        previous_f = origin[2]-_union_boundary(
            all_surfaces, surface["base_z"], origin[0], origin[1])[0]
        bracket = None
        for trial_t in np.linspace(0.0, t_end, 161)[1:]:
            trial = origin+trial_t*direction
            boundary_z = _union_boundary(
                all_surfaces, surface["base_z"], trial[0], trial[1])[0]
            f = trial[2]-boundary_z
            if previous_f >= 0.0 and f <= 0.0:
                bracket = [previous_t, float(trial_t)]
                break
            previous_t, previous_f = float(trial_t), f
        if bracket is None:
            continue
        lo, hi = bracket
        for _ in range(40):
            mid = 0.5*(lo+hi)
            point = origin+mid*direction
            boundary_z = _union_boundary(
                all_surfaces, surface["base_z"], point[0], point[1])[0]
            if point[2]-boundary_z > 0.0:
                lo = mid
            else:
                hi = mid
        point = origin+0.5*(lo+hi)*direction
        point[2], normal = _union_boundary(
            all_surfaces, surface["base_z"], point[0], point[1])
        air, _, ok1 = _fresnel_refract(
            direction[None, :], normal[None, :], MCF_IPS_N, N_AIR)
        air = air[0]
        if not bool(ok1[0]) or air[2] >= 0.0:
            continue
        air_surface = point+(-point[2]/air[2])*air
        diamond, _, ok2 = _fresnel_refract(
            air[None, :], diamond_normal[None, :], N_AIR, n_dia)
        diamond = diamond[0]
        if not bool(ok2[0]) or diamond[2] >= 0.0:
            continue
        hit = air_surface+((-float(depth)-air_surface[2])/diamond[2])*diamond
        paths.append(np.vstack([origin, point, air_surface, hit]))
        theta_rows.append(theta)
        phi_rows.append(phi)
    return dict(paths=np.asarray(paths), theta=np.asarray(theta_rows),
                phi=np.asarray(phi_rows), theta_max=theta_max,
                fiber_na=float(fiber_na), depth=float(depth))


def _divergence_angle(trace, valid):
    """Power-weighted RMS cone angle inside the diamond.

    This sets the diffraction floor.  It replaces a 90th-percentile of the ray
    angles, which is a discrete order statistic: it stepped whenever a single
    ray entered or left the valid set, so the objective carried small jumps
    that the 1-um refinement could chase.  The second moment is smooth in the
    geometry and is the usual definition of beam divergence.
    """
    weight = trace["weight"][valid]
    angle = np.arccos(np.clip(
        trace["diamond"][valid] @ trace["diamond_normal"], -1.0, 1.0))
    total = max(float(weight.sum()), 1e-30)
    return max(float(np.sqrt((weight*angle*angle).sum()/total)), 1e-4)


def beam_stats(trace, lam_nm, depths=SEARCH_DEPTHS):
    """Weighted Gaussian-equivalent footprint, including the diffraction floor."""
    valid = (trace["valid"] & np.isfinite(trace["weight"]) &
             (trace["weight"] > 0.0))
    if not np.any(valid):
        raise ValueError("trace contains no valid transmitted rays")
    w = trace["weight"][valid]
    total = max(w.sum(), 1e-30)
    rows = []
    n_dia = float(diamond_sellmeier(lam_nm / 1000.0))
    tilt = np.deg2rad(float(trace.get("tilt_deg", 0.0)))
    tangent_x = np.array([np.cos(tilt), 0.0, np.sin(tilt)])
    tangent_y = np.array([0.0, 1.0, 0.0])
    theta = _divergence_angle(trace, valid)
    w_diff = (lam_nm*1e-3) / (np.pi*n_dia*np.sin(theta))
    for d, h in zip(np.atleast_1d(depths), trace["hits"]):
        h = h[valid]
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
    theta = _divergence_angle(trace, valid)
    diffraction_sigma = (lam_nm*1e-3)/(2*np.pi*n_dia*np.sin(theta))
    return gaussian_filter(
        density, diffraction_sigma/dx, mode="constant")/(dx*dx)


def _field_axis(all_stats, grid_n):
    """Shared transverse grid for the field integral.

    The extent still follows the beam so a tight spot stays resolved, but it is
    snapped to a fixed ladder: two candidates whose footprints differ slightly
    then integrate on exactly the same mesh, so their scores differ by optics
    rather than by discretization.  Used by every consumer of the field, which
    is what makes evaluate_design and combined_overlap_volume agree.
    """
    raw = max(5.0, max(np.max(np.abs(row["mean"]))+5.0*row["fwhm"]
                       for row in all_stats))
    limit = FIELD_LIMIT_QUANTUM_UM*np.ceil(
        min(raw, FIELD_LIMIT_MAX_UM)/FIELD_LIMIT_QUANTUM_UM)
    limit = float(min(limit, FIELD_LIMIT_MAX_UM))
    return np.linspace(-limit, limit, int(grid_n))


def _combined_overlap_plane(central_trace, side_traces, depth_index, axis):
    """Physical 532-nm excitation x summed six-core collection density."""
    intensity = P_GREEN_MW*_ray_density(
        central_trace, depth_index, axis, 532.0)
    sat = intensity/I_SAT
    excitation_rate = R_SAT*sat/(1.0+sat)

    max_saturation = float(sat.max()) if sat.size else 0.0

    collection = np.zeros((len(axis), len(axis)))
    for lam, spectral_weight, traces in side_traces:
        n_dia = float(diamond_sellmeier(lam/1000.0))
        theta_ips = np.arcsin(MCF_FULL_NA/MCF_IPS_N)
        solid_angle_ips = 2*np.pi*(1.0-np.cos(theta_ips))
        effective_core_area = 0.5*np.pi*W_MODE*W_MODE
        collection_area = (effective_core_area*MCF_IPS_N**2*solid_angle_ips /
                           (4*np.pi*n_dia*n_dia))
        if len(traces) == 1:
            density = _ray_density(
                traces[0], depth_index, axis, lam,
                np.arange(6)*np.pi/3.0)
        else:
            density = sum(_ray_density(trace, depth_index, axis, lam)
                          for trace in traces)
        collection += spectral_weight*collection_area*density
    # Clamping a collection probability at 1 is correct, but it also flattens
    # the objective wherever it bites, so report how much of the signal it
    # touched instead of applying it silently.
    clamped = collection > 1.0
    signal = RHO*excitation_rate*np.minimum(collection, 1.0)
    total = float(signal.sum())
    clamped_share = (float(signal[clamped].sum())/total
                     if total > 0.0 and clamped.any() else 0.0)
    return signal, dict(max_saturation=max_saturation,
                        clamped_signal_fraction=clamped_share)


def combined_overlap_volume(central, side, grid_n=101, tilt_deg=0.0,
                            depths=DEPTHS, red_lam=RED_LAM, red_w=RED_W,
                            ray_grid=35):
    """Return the exact 3-D field integrated by ``evaluate_design``.

    The field is the central 532-nm excitation rate multiplied by the summed,
    spectrum-weighted reciprocal collection of all six side cores.
    """
    depths = np.asarray(depths, dtype=float)
    red_lam = np.asarray(red_lam, dtype=float)
    red_w = np.asarray(red_w, dtype=float)
    if len(depths) < 2 or len(red_lam) != len(red_w):
        raise ValueError("depth and spectral quadratures are inconsistent")

    union = replicated_surfaces(central, side)
    central_trace = trace_full_na(central, union, 532.0, depths, ray_grid,
                                  tilt_deg=tilt_deg)
    central_stats = beam_stats(central_trace, 532.0, depths)
    side_stats, side_traces = [], []
    # At zero tilt the sixfold union is exactly rotationally symmetric, so one
    # side-core trace rotated six times is identical to tracing all six cores.
    # A tilted diamond breaks that symmetry and is therefore traced explicitly.
    side_surfaces = [side] if abs(float(tilt_deg)) < 1e-12 else union[1:]
    for lam, spectral_weight in zip(red_lam, red_w):
        traces = [trace_full_na(surface, union, lam, depths, ray_grid,
                                tilt_deg=tilt_deg) for surface in side_surfaces]
        rows = [beam_stats(trace, lam, depths) for trace in traces]
        side_stats.append((lam, spectral_weight, rows))
        side_traces.append((lam, spectral_weight, traces))

    all_stats = central_stats + [
        row for _, _, groups in side_stats for rows in groups for row in rows]
    axis = _field_axis(all_stats, grid_n)
    signal = np.stack([
        _combined_overlap_plane(central_trace, side_traces, iz, axis)[0]
        for iz in range(len(depths))])

    dx = axis[1]-axis[0]
    dz = np.empty_like(depths)
    dz[0] = 0.5*(depths[1]-depths[0])
    dz[-1] = 0.5*(depths[-1]-depths[-2])
    dz[1:-1] = 0.5*(depths[2:]-depths[:-2])
    weighted = signal*dz[:, None, None]*dx*dx
    total = float(weighted.sum())
    peak = float(signal.max())
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    mean_x = float((weighted*xx).sum()/total)
    mean_y = float((weighted*yy).sum()/total)
    mean_depth = float((weighted*depths[:, None, None]).sum()/total)
    rms = [
        float(np.sqrt((weighted*(xx-mean_x)**2).sum()/total)),
        float(np.sqrt((weighted*(yy-mean_y)**2).sum()/total)),
        float(np.sqrt((weighted*(depths[:, None, None]-mean_depth)**2).sum()/total)),
    ]
    half_max_volume = float(
        ((signal >= 0.5*peak)*dz[:, None, None]).sum()*dx*dx)
    summary = {
        "definition": ("central 532-nm excitation rate times summed six-side-core "
                       "650-850-nm reciprocal collection"),
        "depth_range_um": [float(depths[0]), float(depths[-1])],
        "integrated_signal_photons_s": total,
        "peak_signal_density_photons_s_um3": peak,
        "centroid_xyz_um": [mean_x, mean_y, -mean_depth],
        "rms_xyz_um": rms,
        "half_max_volume_um3": half_max_volume,
        "half_max_threshold_relative": 0.5,
    }
    return dict(axis_um=axis, depth_um=depths, signal_density=signal,
                relative_signal=signal/max(peak, 1e-30), summary=summary)


def surface_limits(surface):
    x, y, _ = _disk(81)
    sag, gx, gy = _sag_grad(surface, surface["aperture"]*x,
                            surface["aperture"]*y)
    return dict(min_z=float(surface["apex"] + sag.min()),
                max_z=float(surface["apex"] + sag.max()),
                max_slope=float(np.sqrt(gx*gx + gy*gy).max()),
                print_height=float(surface["base_z"] - surface["apex"] - sag.min()))


# Screening stage: the same objective at reduced fidelity, rather than a
# separate hand-built merit.  The previous throughput/FWHM^2 heuristic ranked
# designs with rank correlation +0.07 against the objective it fed -- its best
# fifth overlapped the true best fifth no better than chance -- so it steered
# the expensive stage with essentially no information.  Coarsening the real
# objective instead keeps 7.6x of the speed at rank correlation +0.68.
PROXY_DEPTHS = np.linspace(*NV_DEPTH_UM, 3)
PROXY_RED_LAM = np.array([750.0])      # band centre
PROXY_RED_W = np.array([1.0])
PROXY_GRID_N = 41
PROXY_RAY_GRID = 17


def evaluate_design(central, side, grid_n=121, tilt_deg=0.0, depths=DEPTHS,
                    red_lam=RED_LAM, red_w=RED_W, ray_grid=35):
    """Full 3-D layer integration, including a diamond-plane tilt about y.

    Fresnel transmission is already present in ``trace_full_na`` at both physical
    interfaces.  The excitation power is therefore multiplied only by the
    traced throughput here.
    """
    depths = np.asarray(depths, dtype=float)
    red_lam = np.asarray(red_lam, dtype=float)
    red_w = np.asarray(red_w, dtype=float)
    if len(depths) < 2 or len(red_lam) != len(red_w):
        raise ValueError("depth and spectral quadratures are inconsistent")
    union = replicated_surfaces(central, side)
    central_trace = trace_full_na(central, union, 532.0, depths, ray_grid,
                                  tilt_deg=tilt_deg)
    cstats = beam_stats(central_trace, 532.0, depths)
    sstats = []
    side_traces = []
    # The untilted optimization rotates this representative side-core density
    # through all six physical cores; nonzero tilt traces each core separately.
    side_surfaces = [side] if abs(float(tilt_deg)) < 1e-12 else union[1:]
    for lam, sw in zip(red_lam, red_w):
        traces = [trace_full_na(surface, union, lam, depths, ray_grid,
                                tilt_deg=tilt_deg) for surface in side_surfaces]
        rows = [beam_stats(trace, lam, depths) for trace in traces]
        sstats.append((lam, sw, rows))
        side_traces.append((lam, sw, traces))

    all_stats = cstats + [x for _, _, groups in sstats for rows in groups for x in rows]
    axis = _field_axis(all_stats, grid_n)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    dx = axis[1]-axis[0]
    dz = float(depths[-1]-depths[0])/(len(depths)-1)
    photons = 0.0
    moments = np.zeros(6)  # W, Wx, Wy, Wxx, Wyy, Wxy
    max_saturation = 0.0
    clamped_fraction = 0.0

    for iz, depth in enumerate(depths):
        sig, plane_diag = _combined_overlap_plane(
            central_trace, side_traces, iz, axis)
        max_saturation = max(max_saturation, plane_diag["max_saturation"])
        clamped_fraction = max(clamped_fraction,
                               plane_diag["clamped_signal_fraction"])
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
                max_saturation=float(max_saturation),
                clamped_signal_fraction=float(clamped_fraction),
                central_tir_fraction=float(central_trace["tir_fraction"]),
                side_tir_fraction=float(max(
                    trace["tir_fraction"]
                    for _, _, traces in side_traces for trace in traces)),
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
    "air_gap_um", "central_height_um", "side_height_um", "side_core_offset_um",
)
# Bounds are held to the region the 300-um print box and the cap-sizing rule
# can actually reach.  A post tall enough to throw a footprint wider than the
# 35-um core pitch makes its cap shadow its neighbours, so the old 295-um
# height ceilings only ever produced rejected candidates.
SEARCH_GEOMETRY_BOUNDS = (
    (5, 250), (5, 200), (5, 120), (-30, 110),
)
# Apertures are searched as a multiple of each cap's own beam footprint, so a
# candidate can never be fitted over one region and used over a much larger
# one.  The central/side overlap is then a reported consequence of the cap
# sizes and the core pitch, not a free parameter that can drive them apart.
SEARCH_APERTURE_NAMES = ("central_aperture_scale", "side_aperture_scale")
SEARCH_APERTURE_BOUNDS = ((1.0, APERTURE_MARGIN), (1.0, APERTURE_MARGIN))
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
    names = (SEARCH_GEOMETRY_NAMES + SEARCH_APERTURE_NAMES +
             tuple(f"central_{name}" for name in central_names) +
             tuple(f"side_{name}" for name in side_names))
    bounds = (SEARCH_GEOMETRY_BOUNDS + SEARCH_APERTURE_BOUNDS +
              tuple(SHAPE_BOUNDS[name] for name in central_names) +
              tuple(SHAPE_BOUNDS[name] for name in side_names))
    x0 = np.r_[25.0, 51.0, 51.0, 0.0, 1.03, 1.09,
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
    center = CORE_R + values["side_core_offset_um"]
    # Each cap is sized to the footprint its own core lights, so the fitted
    # region and the physical aperture coincide.  A decentered side cap must
    # additionally reach back over its core, hence the decenter term.
    decenter = abs(center - CORE_R)
    central_aperture = values["central_aperture_scale"]*beam_footprint(central_height)
    side_aperture = decenter + values["side_aperture_scale"]*beam_footprint(side_height)
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
        if (limits["min_z"] < 5.0 or
                # leave real polymer between the cap and its base rather than
                # pinching it to a sliver that cannot be printed
                limits["max_z"] > surface["base_z"]-MIN_POLYMER_UM or
                limits["max_z"] > PRINT_Z_UM or
                limits["print_height"] > PRINT_Z_UM or
                # a wall past the critical angle reflects the light that
                # reaches it, and will not print cleanly either
                limits["max_slope"] > MAX_SURFACE_SLOPE):
            return None
    # Rays refract off whichever surface of the printed union sits lowest, so a
    # cap that is shadowed over its own lit footprint is not the optic its core
    # actually uses.  Reject those before paying for a trace.
    union = replicated_surfaces(central, side)
    if (exposure_fraction(0, union) < EXPOSURE_MIN or
            exposure_fraction(1, union) < EXPOSURE_MIN):
        return None
    # The apex is what gets reported as the air gap, so it has to stay close to
    # the clearance the beam actually sees.  Checked here as well as in
    # validate_design so the search can never select a design the exporter
    # would reject.
    for surface in (central, side):
        if (optical_clearance(surface) >
                surface["apex"]+CLEARANCE_TOL*lit_radius(surface)):
            return None
    return central, side


def design_parameters(central, side):
    """Return the independent geometry and derived quantities in report units.

    ``air_gap_um`` is the apex, i.e. the closest approach of the whole cap to
    the diamond.  ``*_optical_clearance_um`` is the closest approach inside the
    footprint the core actually lights, which is the distance the beam sees.
    The two coincide only while each cap stays sized to its own footprint, so
    both are reported and ``validate_design`` checks they agree.
    """
    return {
        "central_lens_type": central["family"],
        "side_lens_type": side["family"],
        "air_gap_um": float(central["apex"]),
        "central_optical_clearance_um": float(optical_clearance(central)),
        "side_optical_clearance_um": float(optical_clearance(side)),
        "central_lit_radius_um": float(lit_radius(central)),
        "side_lit_radius_um": float(lit_radius(side)),
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
        else:
            result = evaluate_design(
                central, side, grid_n=PROXY_GRID_N, depths=PROXY_DEPTHS,
                red_lam=PROXY_RED_LAM, red_w=PROXY_RED_W,
                ray_grid=PROXY_RAY_GRID)
        # Both stages now minimize the same quantity, so the screening result
        # is a usable starting point for the full pass rather than the optimum
        # of a different function.
        merit = result["raw_model_sensitivity_nt"]
        return merit if np.isfinite(merit) else np.inf
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


def _surface_from_json(data):
    """Rebuild a surface dict written by _surface_json."""
    surface = dict(data)
    surface["coef"] = np.asarray(surface["coef"], dtype=float)
    return surface


def _search_signature(families, proxy_maxiter, full_maxiter, popsize, restarts):
    """Identity of a search configuration.

    A checkpoint is only reusable by an identical configuration.  Resuming a
    run whose bounds, families or physical constraints have changed would mix
    results from two different searches into one comparison table, so the
    signature is stored and checked rather than trusted.
    """
    return {"families": list(families), "proxy_maxiter": int(proxy_maxiter),
            "full_maxiter": int(full_maxiter), "popsize": int(popsize),
            "restarts": int(restarts),
            "geometry_bounds": [list(b) for b in SEARCH_GEOMETRY_BOUNDS],
            "aperture_bounds": [list(b) for b in SEARCH_APERTURE_BOUNDS],
            "constants": [APERTURE_MARGIN, EXPOSURE_MIN, CLEARANCE_TOL,
                          MAX_SURFACE_SLOPE, MIN_POLYMER_UM,
                          FIELD_LIMIT_QUANTUM_UM, MCF_FULL_NA]}


def _save_checkpoint(path, signature, done, finalists, method_results):
    """Persist completed family pairs.  Written atomically.

    A crash during the write would otherwise leave a truncated file that the
    next run would fail to parse, losing exactly the work being protected.
    """
    if not path:
        return
    payload = dict(version=1, signature=signature,
                   done=[list(pair) for pair in sorted(done)],
                   method_comparison=method_results,
                   finalists=[dict(merit=float(merit), label=label,
                                   central=_surface_json(central),
                                   side=_surface_json(side))
                              for merit, label, central, side in finalists])
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(temporary, path)


def _load_checkpoint(path, signature):
    """Return (done_pairs, finalists, method_results) from a usable checkpoint."""
    if not path or not os.path.exists(path):
        return set(), [], {}
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (ValueError, OSError):
        print(f"  checkpoint {path} is unreadable; starting fresh", flush=True)
        return set(), [], {}
    if payload.get("signature") != signature:
        print(f"  checkpoint {path} was written by a different search "
              "configuration; starting fresh", flush=True)
        return set(), [], {}
    finalists = [(row["merit"], row["label"],
                  _surface_from_json(row["central"]),
                  _surface_from_json(row["side"]))
                 for row in payload.get("finalists", [])]
    return ({tuple(pair) for pair in payload.get("done", [])},
            finalists, payload.get("method_comparison", {}))


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
                  workers=-1, rescore_count=3, checkpoint=None):
    """Deterministic global integer search followed by 1-um refinement.

    The family contest is decided on the same resolution the winner is
    published at: the best ``rescore_count`` candidates are re-evaluated on the
    final grid before one is selected.

    ``checkpoint`` is a path that, when given, is written after every family
    pair and read back on start, so an interrupted run resumes instead of
    repeating hours of completed work.  Delete the file to force a fresh
    search.  Per-pair seeds come from the position in the family product, not
    from a counter of executed pairs, so a resumed run reproduces exactly the
    sequence an uninterrupted one would have produced.
    """
    families = tuple(families)
    if (not families or min(proxy_maxiter, full_maxiter) < 0 or
            popsize < 5 or restarts < 1 or rescore_count < 1):
        raise ValueError("invalid search configuration")
    signature = _search_signature(families, proxy_maxiter, full_maxiter,
                                  popsize, restarts)
    completed, finalists, method_results = _load_checkpoint(checkpoint, signature)
    if completed:
        print(f"resuming from {checkpoint}: {len(completed)} of "
              f"{len(families)**2} family pairs already done", flush=True)
    worker_pool = None
    worker_map = 1
    if workers != 1:
        from multiprocessing import Pool
        worker_pool = Pool(None if workers == -1 else workers)
        worker_map = worker_pool.map

    def _evolve(**kwargs):
        """Run one DE pass, falling back to serial if the worker pool dies.

        A broken pool pipe previously destroyed a multi-hour run outright; the
        objective is cached and deterministic, so retrying serially costs time
        rather than correctness.
        """
        try:
            return differential_evolution(workers=worker_map, **kwargs)
        except Exception as exc:                      # pool died mid-generation
            print(f"  worker pool failed ({type(exc).__name__}); "
                  "retrying this pass serially", flush=True)
            return differential_evolution(workers=1, **kwargs)

    best = None
    try:
        for method_index, (central_family, side_family) in enumerate(
                (c, s) for c in families for s in families):
            if (central_family, side_family) in completed:
                print(f"Skipping {central_family} + {side_family} "
                      "(already in checkpoint)", flush=True)
                continue
            _, bounds, x0, integrality = _search_spec(
                central_family, side_family)
            method_best = None
            for restart in range(restarts):
                seed = 1847 + 101*method_index + restart
                label = f"{central_family} + {side_family}"
                print(f"Optimizing {label}, restart {restart+1}/{restarts}...",
                      flush=True)
                proxy_progress = _ProgressBar(f"{label} proxy", proxy_maxiter)
                proxy = _evolve(
                    func=_search_objective, bounds=bounds,
                    args=(central_family, side_family, "proxy"),
                    strategy="best1bin", maxiter=proxy_maxiter,
                    popsize=popsize, tol=0.0, atol=0.0,
                    mutation=(0.5, 1.0), recombination=0.8,
                    rng=np.random.default_rng(seed), polish=False,
                    init="sobol", x0=x0,
                    updating="deferred",
                    integrality=integrality, callback=proxy_progress)
                proxy_progress.finish()
                full_progress = _ProgressBar(f"{label} full", full_maxiter)
                full = _evolve(
                    func=_search_objective, bounds=bounds,
                    args=(central_family, side_family, "full"),
                    strategy="best1bin", maxiter=full_maxiter,
                    popsize=popsize, tol=0.0, atol=0.0,
                    mutation=(0.5, 1.0), recombination=0.8,
                    rng=np.random.default_rng(seed+10_000), polish=False,
                    init="sobol", x0=proxy.x, updating="deferred",
                    integrality=integrality,
                    callback=full_progress)
                full_progress.finish()
                print(f"  screen={proxy.fun:.6g}, "
                      f"search sensitivity={full.fun:.6g} nT/sqrt(Hz)",
                      flush=True)
                if method_best is None or full.fun < method_best[0]:
                    method_best = (full.fun, full.x)

            merit, parameters = _coordinate_refine(
                method_best[1], central_family, side_family)
            surfaces = _search_surfaces(parameters, central_family, side_family)
            if surfaces is None or not np.isfinite(merit):
                # still a completed pair: it yields no candidate, and repeating
                # it after a crash would only burn the same time again.
                completed.add((central_family, side_family))
                _save_checkpoint(checkpoint, signature, completed, finalists,
                                 method_results)
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
            method_results[label]["search_grid_sensitivity_nt"] = float(merit)
            print(f"  refined sensitivity={merit:.6g} nT/sqrt(Hz), "
                  f"parameters={design_parameters(central, side)}", flush=True)
            finalists.append((merit, label, central, side))
            completed.add((central_family, side_family))
            _save_checkpoint(checkpoint, signature, completed, finalists,
                             method_results)
    finally:
        if worker_pool is not None:
            worker_pool.close()
            worker_pool.join()

    # Every score above comes from the coarse search grid.  Ranking on it and
    # then publishing a fine-grid number let a ~5% quadrature shift decide a
    # contest whose top entries sat within a few percent of each other, so the
    # leaders are re-scored at the reporting resolution before one is chosen.
    # sort on the merit alone: a bare sorted() would fall through to comparing
    # the surface dicts on a tie, which raises.
    for merit, label, central, side in sorted(
            finalists, key=lambda row: row[0])[:rescore_count]:
        fine = evaluate_design(central, side, grid_n=161)
        method_results[label]["final_grid_sensitivity_nt"] = float(
            fine["raw_model_sensitivity_nt"])
        print(f"  re-scored {label}: {merit:.6g} -> "
              f"{fine['raw_model_sensitivity_nt']:.6g} nT/sqrt(Hz) at full "
              "resolution", flush=True)
        if best is None or fine["raw_model_sensitivity_nt"] < best[0]:
            best = (fine["raw_model_sensitivity_nt"], central, side, fine)

    if best is None:
        raise RuntimeError("search found no manufacturable design")
    # best[3] is the full-resolution evaluation the winner was chosen on, so
    # the exported result and the selection criterion are the same number.
    return dict(central=best[1], side=best[2], result=best[3],
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
                                    spatial_model=(
                                        "10-um Gaussian core area with uniformly filled full-NA "
                                        "equal-solid-angle rays, traced footprints, and diffraction blur"),
                                    fiber_na=MCF_FULL_NA,
                                    ray_model="uniform_full_na",
                                    optimization_score=(
                                        "80-90 um volume integral of central 532-nm excitation "
                                        "times summed six-side-core 650-850-nm reciprocal collection"),
                                    green_fresnel="included once by trace_full_na",
                                    aperture_rule=(
                                        "cap radius = decenter + scale x beam footprint, "
                                        f"scale in [1.0, {APERTURE_MARGIN}]; the fitted and "
                                        "physical apertures therefore coincide"),
                                    exposure_min=EXPOSURE_MIN,
                                    exposure_rule=(
                                        "each cap must be the lowest surface of the printed "
                                        "union over this fraction of its own lit footprint"),
                                    max_surface_slope=MAX_SURFACE_SLOPE,
                                    max_surface_slope_rule=(
                                        "tan of the steepest allowed tilt; bounds both "
                                        "TIR loss at the rim and printability"),
                                    min_polymer_um=MIN_POLYMER_UM,
                                    field_grid_quantum_um=FIELD_LIMIT_QUANTUM_UM,
                                    field_grid_rule=(
                                        "integration extent snapped to this ladder so nearby "
                                        "candidates share one mesh"),
                                    diffraction_angle="power-weighted RMS cone angle",
                                    family_choice=(
                                        "top candidates re-scored on the final grid before "
                                        "the winner is selected"),
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
    union = replicated_surfaces(c, s)
    for index, surf in ((0, c), (1, s)):
        lim = surface_limits(surf)
        assert lim["print_height"] <= MAX_PRINT+1e-9
        assert lim["min_z"] >= 0.0 and lim["max_z"] <= PRINT_Z_UM+1e-9
        assert lim["max_z"] <= surf["base_z"]-MIN_POLYMER_UM+1e-9
        assert lim["max_slope"] <= MAX_SURFACE_SLOPE+1e-9
        assert np.all(np.isfinite(surf["coef"]))
        # The cap covers the footprint its core lights, and does not run far
        # past it into unfitted extrapolation.
        footprint = lit_radius(surf)
        decenter = abs(surf["center_r"]-surf["core_r"])
        assert decenter+footprint <= surf["aperture"]+1e-6
        assert surf["aperture"] <= decenter+APERTURE_MARGIN*footprint+1e-6
        # It is also the surface its own core actually refracts off.
        assert exposure_fraction(index, union) >= EXPOSURE_MIN
        # With the cap sized to its footprint the apex is the clearance the
        # beam sees; a large gap between them means extrapolation crept back.
        assert optical_clearance(surf) <= surf["apex"]+CLEARANCE_TOL*footprint+1e-6
    r = design["result"]
    assert r["model_fiber_photons_s"] > 0
    assert r["comparison_normalized_cps"] > 0
    assert r["raw_model_sensitivity_nt"] > 0
    assert r["comparison_normalized_sensitivity_nt"] > 0
    return True


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(out, exist_ok=True)
    design = search_design(
        checkpoint=os.path.join(out, "phase3_checkpoint.json"))
    validate_design(design)
    write_design_json(design, os.path.join(out, "mcf_freeform_design.json"))
    ntri = write_binary_stl(design, os.path.join(out, "mcf_freeform_central_one_side.stl"))
    nfull = write_binary_stl(design, os.path.join(out, "mcf_freeform_full_seven_core.stl"),
                             all_sides=True)
    print(json.dumps({k: v for k, v in design["result"].items()
                      if k not in ("central_stats", "side_stats")}, indent=2))
    print(f"STL triangles: one-side={ntri}, full={nfull}")
