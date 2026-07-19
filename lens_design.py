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

import numpy as np

from physics import diamond_sellmeier, nv_emission_spectrum
from paper_figures import (
    E_PHOT_RED, I_SAT, MCF_IPS_N, MCF_MFD, P_GREEN_MW,
    RHO, R_SAT,
)
from sensitivity import MEASURED, linewidth_from_resolution, calibrated_sensitivity

N_AIR = 1.0
CORE_R = 35.0
W_MODE = MCF_MFD / 2.0
MAX_PRINT = 300.0


def _spectral_quadrature(points):
    """Normalized trapezoidal quadrature of the NV spectrum over 650--850 nm."""
    wavelength = np.linspace(650.0, 850.0, int(points))
    weight = nv_emission_spectrum(wavelength)
    weight[[0, -1]] *= 0.5
    return wavelength, weight/weight.sum()


# Cheap, deterministic screening is followed by a converged final evaluation.
SEARCH_DEPTHS = np.linspace(80.0, 90.0, 9)
SEARCH_RED_LAM, SEARCH_RED_W = _spectral_quadrature(9)
DEPTHS = np.linspace(80.0, 90.0, 33)
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


def _gaussian(x, y, mean, cov):
    inv = np.linalg.inv(cov)
    qx, qy = x-mean[0], y-mean[1]
    e = inv[0, 0]*qx*qx + 2*inv[0, 1]*qx*qy + inv[1, 1]*qy*qy
    return np.exp(-0.5*e)


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
    cstats = beam_stats(trace_mode(central, union, 532.0, depths, ray_grid,
                                   tilt_deg=tilt_deg), 532.0, depths)
    sstats = []
    side_surfaces = [side] if abs(float(tilt_deg)) < 1e-12 else union[1:]
    for lam, sw in zip(red_lam, red_w):
        rows = [beam_stats(trace_mode(surface, union, lam, depths, ray_grid,
                                      tilt_deg=tilt_deg), lam, depths)
                for surface in side_surfaces]
        sstats.append((lam, sw, rows))

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
        cs = cstats[iz]
        exc_shape = _gaussian(x, y, cs["mean"], cs["cov"])
        exc_norm = 2.0*np.pi*np.sqrt(np.linalg.det(cs["cov"]))
        intensity = P_GREEN_MW*cs["throughput"]*exc_shape/exc_norm
        sat = intensity/I_SAT
        rate = R_SAT*sat/(1.0+sat)

        eta = np.zeros_like(x)
        for lam, sw, groups in sstats:
            for k, rows in enumerate(groups if len(groups) > 1 else groups*6):
                ss = rows[iz]
                eig = np.sqrt(np.linalg.eigvalsh(ss["cov"]))
                wx, wy = 2.0*eig[0], 2.0*eig[1]
                n = float(diamond_sellmeier(lam/1000.0))
                eta_peak = ss["throughput"]*(lam*1e-3)**2/(4*np.pi**2*n*n*wx*wy)
                if len(groups) == 1:
                    a = k*np.pi/3.0
                    rot = np.array([[np.cos(a), -np.sin(a)],
                                    [np.sin(a),  np.cos(a)]])
                    mean, cov = rot@ss["mean"], rot@ss["cov"]@rot.T
                else:
                    mean, cov = ss["mean"], ss["cov"]
                eta += sw*eta_peak*_gaussian(x, y, mean, cov)
        eta = np.minimum(eta, 1.0)
        sig = RHO*rate*eta
        volume = dx*dx*dz*(0.5 if iz in (0, len(depths)-1) else 1.0)
        photons += sig.sum()*volume
        ww = sig*volume
        moments += [ww.sum(), (ww*x).sum(), (ww*y).sum(),
                    (ww*x*x).sum(), (ww*y*y).sum(), (ww*x*y).sum()]

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


def search_design():
    """Small deterministic search over method, position, height, and aperture."""
    families = ("quadratic", "asphere", "biconic", "freeform")
    shortlist = []
    for gap in (5.0, 25.0, 75.0, 150.0, 300.0, 500.0):
        for height in (120.0, 200.0, 280.0):
            central = []
            for family in families:
                for aperture in (20.0, 35.0, 50.0):
                    s = fit_surface(family, "central", gap, height, aperture)
                    lim = surface_limits(s)
                    if lim["print_height"] <= MAX_PRINT and lim["max_z"] < s["base_z"]:
                        central.append((candidate_merit(s, "central"), s))
            side = []
            for family in families:
                for aperture in (12.0, 17.5, 24.0):
                    for center in (17.5, 25.0, 35.0):
                        for offset in (10.0, 35.0, 70.0):
                            if offset >= height-10.0:
                                continue
                            s = fit_surface(family, "side", gap, height, aperture,
                                            center_r=center, side_offset=offset)
                            lim = surface_limits(s)
                            if lim["print_height"] <= MAX_PRINT and lim["max_z"] < s["base_z"]:
                                side.append((candidate_merit(s, "side"), s))
            # Keep two proxy-ranked shapes per family.  A single proxy winner
            # can discard a better excitation×collection result after overlap
            # and saturation are integrated.
            central_best = [q for family in families for q in
                            sorted((q for q in central if q[1]["family"] == family),
                                   key=lambda q: q[0], reverse=True)[:2]]
            side_best = [q for family in families for q in
                         sorted((q for q in side if q[1]["family"] == family),
                                key=lambda q: q[0], reverse=True)[:2]]
            for cm, c in central_best:
                for sm, s in side_best:
                    union = replicated_surfaces(c, s)
                    ct = trace_mode(c, union, 532.0, n_grid=25)
                    st = trace_mode(s, union, 750.0, n_grid=25)
                    overlap_merit = (cm*sm*ct["throughput"]*st["throughput"])
                    shortlist.append((overlap_merit, c, s))

    by_method = {}
    for row in shortlist:
        key = (row[1]["family"], row[2]["family"])
        by_method.setdefault(key, []).append(row)
    finalists = sorted(shortlist, key=lambda q: q[0], reverse=True)[:20]
    finalists += [row for rows in by_method.values()
                  for row in sorted(rows, key=lambda q: q[0], reverse=True)[:2]]
    unique = {(c["family"], s["family"], c["apex"], c["base_z"],
               c["aperture"], s["center_r"], s["aperture"], s["apex"]): (m, c, s)
              for m, c, s in finalists}
    best, method_results = None, {}
    for _, central, side in unique.values():
        result = evaluate_design(central, side, grid_n=81, depths=SEARCH_DEPTHS,
                                 red_lam=SEARCH_RED_LAM, red_w=SEARCH_RED_W,
                                 ray_grid=31)
        key = f"{central['family']} + {side['family']}"
        merit_key = "raw_model_sensitivity_nt"
        if key not in method_results or result[merit_key] < method_results[key][merit_key]:
            method_results[key] = {k: result[k] for k in
                                   ("raw_model_sensitivity_nt",
                                    "comparison_normalized_sensitivity_nt",
                                    "model_fiber_photons_s",
                                    "comparison_normalized_cps", "resolution_um")}
        if best is None or result[merit_key] < best[0]:
            best = (result[merit_key], central, side)
    final_result = evaluate_design(best[1], best[2], grid_n=161)
    return dict(central=best[1], side=best[2], result=final_result,
                method_comparison=method_results)


def _surface_json(surface):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in surface.items()}


def write_design_json(design, path):
    payload = dict(central=_surface_json(design["central"]),
                   side=_surface_json(design["side"]),
                   result={k: v for k, v in design["result"].items()
                           if k not in ("central_stats", "side_stats")},
                   method_comparison=design.get("method_comparison", {}),
                   measured=MEASURED,
                   comparison_normalization=COMPARISON_NORMALIZATION,
                   assumptions=dict(mfd_um=MCF_MFD, ips_index=MCF_IPS_N,
                                    nv_ppm=3.0, nv_depth_um=[80.0, 90.0],
                                    spectrum_nm=[650.0, 850.0], max_print_um=MAX_PRINT,
                                    final_depth_points=len(DEPTHS),
                                    final_spectral_points=len(RED_LAM),
                                    spectral_quadrature="normalized trapezoidal",
                                    green_fresnel="included once by trace_mode",
                                    method_search_quadrature_points=9))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def write_binary_stl(design, path, nr=55, nt=144, all_sides=False):
    """Write one watertight support with one or all six exposed side lenses."""
    c, s = design["central"], design["side"]
    surfaces = replicated_surfaces(c, s) if all_sides else [c, s]
    base_z = c["base_z"]
    radius = min(145.0, max(60.0, c["aperture"]+6.0,
                            s["center_r"]+s["aperture"]+6.0))
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
    assert c["base_z"] == s["base_z"]
    for surf in (c, s):
        lim = surface_limits(surf)
        assert lim["print_height"] <= MAX_PRINT+1e-9
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
