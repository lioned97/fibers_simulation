"""
paper_figures.py -- static publication figures for the SM / MM / MCF NV-diamond
fiber-probe comparison. Run `python paper_figures.py`; figures land in ./figures
(PNG 300 dpi + vector PDF). No UI, no CLI args: edit the constants block.

Model (shared Monte-Carlo ray tracer in physics.py, same as the app):
  * Diamond surface facing the fibers at Z = 0, diamond 100 um thick.
  * NV layer Z = -80..-90 um (centre 85 um); single-NV case: on axis at -85 um.
  * Gap g = air distance from the diamond surface to the fiber facet (SM/MM)
    or lens-tip plane (MCF), swept 0-200 um.
  * Collection: Snell + Fresnel + TIR at the diamond surface, then per-fiber
    coupling -- SM: Gaussian LP01 mode overlap; MM: geometric core+NA;
    MCF: 6 side cores, Gaussian aperture w0 = pitch/2, tilted boresights.
    MCF "as-built": the six physical side cores are 35 um off-axis and their
    red microlens pupils are decentered inward to 16.8434 um.  Their fixed
    acceptance angle is the 650 nm design from microlens.xlsx: 141.185 um of
    free air followed by a 30 um target depth (Shukhin et al., OMN 2024).
    MCF "re-aimed": boresights recomputed per gap to hit the NV-layer centre
    (exact Snell ray solve) -- an idealized redesign envelope.
  * Excitation: 10 mW of 532 nm out of the delivery aperture (SM/MM: same
    core that collects; MCF: central core, collimated per the paper design).
    Per-NV emission R = R_sat * s/(1+s), s = I/I_sat  (I_sat = 3 MW/cm^2,
    R_sat = 5e6 photons/s into 4pi -- literature-typical room-T values).
    Green Fresnel entry loss and divergence compression (Snell) included;
    green absorption across the 10 um layer neglected (<2% at 3 ppm).
  * Ensemble: 3 ppm -> 5.28e5 NV/um^3. Emitters are IMPORTANCE-SAMPLED from a
    mixture matched to the excitation footprint (two Gaussians + uniform box)
    and exactly reweighted by the true density / proposal pdf, so population
    totals are unbiased and every gap keeps hundreds of effective samples.
    Emission anisotropy: 4-axis NV dipole average.
  * Rays use a fixed, equal-solid-angle quadrature over the diamond escape
    cone, rather than wasting samples over the TIR hemisphere.  Tracer calls
    are chunked over emitters to keep the peak working set bounded.
  * Every reported efficiency is power coupled into the fiber/lens: diamond
    exit Fresnel, mode/geometric coupling, and the air-to-fiber/lens entrance
    Fresnel are included. No filter, detector QE, connector, or propagation
    loss is included -- these figures characterize the probe, not a detector.
  * Distance and resolution sweeps run at a representative 700 nm; the full
    spectral dependence is figure 2. Fixed seeds; fully reproducible.
"""
import csv
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from physics import (
    diamond_sellmeier, nv_emission_spectrum,
    GEOMETRIC_MODEL, MODE_OVERLAP_MODEL,
    get_collection_limit_radius, run_ray_tracing,
)


def bisect(f, lo, hi, iters=80):
    """Root of monotone f on [lo, hi] (f(lo) < 0 < f(hi)). No scipy needed."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# ============================== constants ==============================
# geometry (um).  The facing diamond surface is z=0, so depth is negative z.
NV_Z_MIN   = -90.0                   # deepest edge of the 3D NV layer
NV_Z_MAX   = -80.0                   # shallowest edge of the 3D NV layer
NV_DEPTH   = -0.5 * (NV_Z_MIN + NV_Z_MAX)
NV_WIDTH   = NV_Z_MAX - NV_Z_MIN
GAPS       = np.linspace(0.0, 200.0, 21)
N_MED      = 1.0                     # air gap

# wavelengths (nm)
LAM_RED    = 700.0                   # representative collection wavelength
LAM_GREEN  = 532.0
LAM_SPECTRUM = np.linspace(640.0, 800.0, 9)
N_DIA_RED   = float(diamond_sellmeier(LAM_RED / 1000.0))
N_DIA_GREEN = float(diamond_sellmeier(LAM_GREEN / 1000.0))

# excitation & photophysics
P_GREEN_MW = 10.0                    # green power out of the delivery aperture
I_SAT      = 30.0                    # mW/um^2  (= 3 MW/cm^2)
R_SAT      = 5.0e6                   # photons/s into 4pi at saturation
T_GREEN_IN = 1.0 - ((N_DIA_GREEN - 1.0) / (N_DIA_GREEN + 1.0)) ** 2
E_PHOT_RED = 6.62607015e-34 * 2.99792458e8 / (LAM_RED * 1e-9)   # J

# ensemble & Monte Carlo
PPM        = 3.0
DIAMOND_ATOM_DENSITY = 1.76e11       # carbon sites/um^3
RHO        = PPM * 1e-6 * DIAMOND_ATOM_DENSITY  # 5.28e5 NV/um^3
N_EMIT     = 600                     # importance-sampled emitters
# The tracer keeps several (n_emitters, n_rays) arrays plus a 3-vector array.
# 0.75 M ray pairs gives a reproducible workstation-safe peak memory use.
RAY_BUDGET = 750_000                 # max emitter-ray pairs per tracer call
SWEEP_WORKERS = min(4, os.cpu_count() or 1)  # gaps are independent; arrays are read-only

# MCF as-built: dimensions and surface curvature measured from the supplied
# Side lenses.STL + Central lens.STL, registered by Cylinder_lenses_job.gwl.
PITCH = 35.0                          # physical side-core radius (um)
LENS_R = 17.5                          # six side-lens vertices from STL (um)
W0_LENS = PITCH / 2.0                 # pupil-filling Gaussian 1/e^2 radius
MCF_RED_FREE_AIR = 141.185            # spreadsheet, 650 nm red design (um)
MCF_TARGET_DEPTH = 30.0               # spreadsheet, target below diamond face (um)
MCF_RED_LENS_HEIGHT = 158.815         # spreadsheet, side-lens IP-S thickness (um)
MCF_GREEN_LENS_HEIGHT = 194.041       # spreadsheet, central-lens IP-S thickness (um)
MCF_DESIGN_N_DIA = 2.4093             # spreadsheet, diamond index at 650 nm
MCF_IPS_N = 1.52
MCF_SIO2_N = 1.4565
MCF_MFD = 10.0                         # every MCF core, including the central core (um)
MCF_MODE_W = MCF_MFD / 2.0             # Gaussian 1/e^2 mode radius (um)
MCF_SIDE_TIP_OFFSET = MCF_GREEN_LENS_HEIGHT - MCF_RED_LENS_HEIGHT  # 35.225 um
# Quadratic fit to the STL outer surface in each lens's radial/tangential frame.
# The overlapping caps are one polymer union; this is its local external surface.
MCF_R_RADIAL, MCF_R_TANGENTIAL = 18.0, 95.0
# Central lens: axisymmetric quadratic fit to Central lens.STL.
MCF_CENTRAL_R = 74.91
MCF_CENTRAL_MODE_W = MCF_MODE_W
MCF_GREEN_LENS_T = 1.0 - ((MCF_IPS_N - N_MED) / (MCF_IPS_N + N_MED)) ** 2
EXPERIMENT_NOTE = "Independent experiment (photon counts; not fitted): SM = MCF; MM = 20 x SM"

def t_face(n_guide):                 # fiber entrance Fresnel (red, normal incidence)
    return 1.0 - ((n_guide - N_MED) / (n_guide + N_MED)) ** 2


def mcf_fixed_air_angle():
    """Air-side chief-ray angle of the spreadsheet's fixed red MCF design."""
    f = lambda t: (MCF_TARGET_DEPTH * np.tan(t)
                   + MCF_RED_FREE_AIR * np.tan(np.arcsin(MCF_DESIGN_N_DIA * np.sin(t)))
                   - LENS_R)
    theta_d = bisect(f, 1e-9, np.arcsin(1.0 / MCF_DESIGN_N_DIA) - 1e-6)
    return np.arcsin(MCF_DESIGN_N_DIA * np.sin(theta_d))


MCF_FIXED_AIR_ANGLE = mcf_fixed_air_angle()

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
# The supplied STL/GWL files are normally kept beside the user's Downloads.
# Override this when the paper script is moved to another machine.
MCF_ASSET_DIR = os.environ.get("MCF_ASSET_DIR",
                               os.path.join(os.path.expanduser("~"), "Downloads"))


def read_binary_stl(path, max_faces=10000):
    """Read a binary STL into triangle vertices, or return None if unavailable."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        n = struct.unpack_from("<I", data, 80)[0]
        if n <= 0 or 84 + 50 * n > len(data):
            return None
        dtype = np.dtype([("normal", "<f4", (3,)),
                          ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        tri = np.frombuffer(data, dtype=dtype, offset=84, count=n)["vertices"].copy()
    except (OSError, struct.error, ValueError):
        return None
    if not np.all(np.isfinite(tri)):
        return None
    if len(tri) > max_faces:
        tri = tri[np.linspace(0, len(tri) - 1, max_faces, dtype=int)]
    return tri


def mcf_printed_meshes(g):
    """Return the three actually-written STL parts in optical plot coordinates.

    STL/GWL uses the build axis as Y and the diamond-facing tip as the high-Y
    end.  The optical model measures distance outward from the diamond, so the
    plotted vertical coordinate is ``g + 194.041 - Y_STL``.  Each lens STL is
    centred from its own bounding box; this registers the central lens with
    the central core and the six side lenses with the 35-um-pitch fibre.
    """
    specs = (("Cylinder.STL", "cylinder", "#7b8790", "printed cylinder/pedestal"),
             ("Side lenses.STL", "side", "#d78500", "six side microlenses"),
             ("Central lens.STL", "central", "#1baf7a", "raised central microlens"))
    meshes = []
    for filename, kind, color, label in specs:
        tri = read_binary_stl(os.path.join(MCF_ASSET_DIR, filename))
        if tri is None:
            continue
        flat = tri.reshape(-1, 3)
        cx = 0.5 * (flat[:, 0].min() + flat[:, 0].max())
        cy = 0.5 * (flat[:, 2].min() + flat[:, 2].max())
        plot_tri = np.empty_like(tri, dtype=float)
        plot_tri[:, :, 0] = tri[:, :, 0] - cx
        plot_tri[:, :, 1] = tri[:, :, 2] - cy
        plot_tri[:, :, 2] = g + MCF_GREEN_LENS_HEIGHT - tri[:, :, 1]
        meshes.append((kind, plot_tri, color, label))
    return meshes

# =========================== fiber configs =============================
def mcf_fibers(g, reaim):
    fibs = []
    for i in range(6):
        a = i * np.pi / 3.0
        fx, fy = LENS_R * np.cos(a), LENS_R * np.sin(a)
        if reaim:
            # exact Snell aim at the NV-layer centre: solve exit angle in diamond
            f = lambda t: (NV_DEPTH * np.tan(t)
                           + (g + MCF_SIDE_TIP_OFFSET) * np.tan(np.arcsin(N_DIA_RED * np.sin(t))) - LENS_R)
            th_d = bisect(f, 1e-9, np.arcsin(1.0 / N_DIA_RED) - 1e-6)
            th_a = np.arcsin(N_DIA_RED * np.sin(th_d))
        else:
            th_a = MCF_FIXED_AIR_ANGLE
        b = np.array([np.cos(a) * np.sin(th_a), np.sin(a) * np.sin(th_a), np.cos(th_a)])
        fibs.append({'x': fx, 'y': fy, 'z': g + MCF_SIDE_TIP_OFFSET, 'd_core': PITCH,
                     'd_clad': PITCH * 1.3, 'na': 0.15, 'boresight': b, 'w0': W0_LENS,
                     'printed_lens': True, 'lens_axis': b,
                     'lens_height': MCF_RED_LENS_HEIGHT,
                     'lens_aperture': W0_LENS, 'mode_w': MCF_MODE_W})
    return fibs

# n1 / nen are equal-solid-angle escape-cone quadrature counts for the single
# NV and ensemble calculations.  They are deliberately shared by all probe
# types so differences are optical, not sampling artifacts.
CONFIGS = [
    dict(name="SM",            exc="SM",  model=MODE_OVERLAP_MODEL,
         fibers=lambda g: [{'x': 0.0, 'y': 0.0, 'z': g, 'd_core': 4.0,
                            'd_clad': 125.0, 'na': 0.12}],
         na=0.12, tf=t_face(1.46), n1=65_536, nen=16_384,
         color="#2a78d6", ls="-",  marker="o"),
    dict(name="MM",            exc="MM",  model=GEOMETRIC_MODEL,
         fibers=lambda g: [{'x': 0.0, 'y': 0.0, 'z': g, 'd_core': 50.0,
                            'd_clad': 125.0, 'na': 0.22}],
         na=0.22, tf=t_face(1.46), n1=65_536, nen=16_384,
         color="#1baf7a", ls="-",  marker="s"),
    dict(name="MCF fixed aim", exc="MCF", model=MODE_OVERLAP_MODEL,
           fibers=lambda g: mcf_fibers(g, reaim=False),
           # The pupil's diffraction-limited angular width needs a finer
           # escape-cone grid than the broad SM/MM acceptances.
           na=0.15, tf=1.0, n1=131_072, nen=65_536,
           color="#d78500", ls="-",  marker="^"),
    dict(name="MCF re-aimed",    exc="MCF", model=MODE_OVERLAP_MODEL,
         fibers=lambda g: mcf_fibers(g, reaim=True),
         na=0.15, tf=1.0, n1=131_072, nen=65_536,
         color="#d78500", ls="--", marker="v"),
]

# ============================ excitation ================================
def central_lens_width(depth, gap):
    """532-nm Gaussian propagated from the MCF core through Central lens.STL.

    The lens is an axisymmetric IP-S cap (R=74.91 um) on the 194.04 um
    central pillar.  A paraxial Gaussian-q transform captures its measured
    curvature without inventing an unmeasured asphere between STL vertices.
    """
    z_r = np.pi * MCF_IPS_N * MCF_CENTRAL_MODE_W ** 2 / (LAM_GREEN * 1e-3)
    q_ips = MCF_GREEN_LENS_HEIGHT + 1j * z_r
    # The convex IP-S-to-air exit surface is positive power when traversed
    # from the central fibre core toward the diamond.
    f_air = -MCF_CENTRAL_R / (MCF_IPS_N - N_MED)
    q_air = 1.0 / (1.0 / q_ips - 1.0 / f_air)
    q_dia = N_DIA_GREEN * (q_air + gap) + np.asarray(depth, dtype=float)
    inv_q = 1.0 / q_dia
    return np.sqrt(-LAM_GREEN * 1e-3 / (np.pi * N_DIA_GREEN * np.imag(inv_q)))

def exc_width(profile, d, g):
    """Excitation radius at depth d for gap g: Gaussian 1/e^2 w, or top-hat R (MM)."""
    if profile == "MM":
        th_a = np.arcsin(0.22)
        th_d = np.arcsin(np.sin(th_a) / N_DIA_GREEN)
        return 25.0 + g * np.tan(th_a) + d * np.tan(th_d)
    if profile == "SM":
        w0, th_a = 2.0, np.arcsin(0.12)
    else:                                    # MCF central fibre + printed central lens
        return central_lens_width(d, g)
    th_d = np.arcsin(np.sin(th_a) / N_DIA_GREEN)
    return np.sqrt(w0 ** 2 + (g * np.tan(th_a) + d * np.tan(th_d)) ** 2)

def exc_rate(profile, x, y, z, g):
    """Per-NV emission rate (photons/s into 4pi) at emitter (x, y, z<0), gap g."""
    d = -np.asarray(z, dtype=float)
    r2 = np.asarray(x) ** 2 + np.asarray(y) ** 2
    w = exc_width(profile, d, g)
    power = P_GREEN_MW * (MCF_GREEN_LENS_T if profile == "MCF" else 1.0)
    if profile == "MM":                      # top-hat
        I = np.where(r2 <= w * w, power * T_GREEN_IN / (np.pi * w * w), 0.0)
    else:                                    # Gaussian
        I = 2.0 * power * T_GREEN_IN / (np.pi * w * w) * np.exp(-2.0 * r2 / (w * w))
    s = I / I_SAT
    return R_SAT * s / (1.0 + s), float(np.max(s))


def escape_cone_quadrature(num_rays, n_dia, seed=42):
    """Equal-solid-angle, fixed-seed quadrature over the transmitting cone.

    ``run_ray_tracing`` normally expects directions drawn uniformly over the
    upper hemisphere and therefore applies its conventional 1/2 factor.  Here
    directions are instead restricted to the diamond-to-air escape cone; each
    dipole weight gets the importance factor ``1-cos(theta_c)`` so the same
    estimator remains exactly normalized to emission into 4pi.  A scrambled
    grid avoids the severe rare-event noise of uniform-hemisphere Monte Carlo
    for the MCF's diffraction-limited angular acceptance.
    """
    n_theta = max(1, int(np.floor(np.sqrt(num_rays))))
    n_phi = int(np.ceil(num_rays / n_theta))
    rng = np.random.default_rng(seed)
    u_shift, p_shift = rng.random(2)
    # Uniform in cos(theta) gives equal solid angle.  Keep every node safely
    # inside the critical angle so numerical roundoff cannot create TIR rays.
    cos_c = np.sqrt(1.0 - (N_MED / n_dia) ** 2)
    u = ((np.arange(n_theta) + 0.5 + u_shift) % n_theta) / n_theta
    p = ((np.arange(n_phi) + 0.5 + p_shift) % n_phi) / n_phi
    uu, pp = np.meshgrid(u, p, indexing="ij")
    cos_theta = 1.0 - uu.ravel() * (1.0 - cos_c)
    phi = 2.0 * np.pi * pp.ravel()
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    v0 = np.column_stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta))

    # Four-axis NV average: two orthogonal dipoles per axis, normalized to
    # unit 4pi-average emission.  This is also the orientation average used
    # for the single-NV panel, where the individual NV orientation is unknown.
    axes = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                     [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]) / np.sqrt(3.0)
    dipole_weight = 0.75 * (1.0 + np.mean((v0 @ axes.T) ** 2, axis=1))
    weights = dipole_weight * (1.0 - cos_c)
    return v0[:num_rays], weights[:num_rays]

# ===================== ensemble emitter sampling ========================
def sample_ensemble(cfg, n, rng):
    """
    Importance-sampled NV positions for the 3D layer, with density weights.
    Proposal in (x, y): mixture of two zero-centred Gaussians (excitation
    footprint at g = 0 and at g = max) + a uniform box covering the collection
    reach; z uniform across the layer. Returns (emitters (n,3), dens (n,))
    where dens_i = true NV density / proposal pdf, so any population total is
    estimated by mean(f_i * dens_i). Unbiased for every gap in the sweep.
    """
    fib0 = cfg['fibers'](0.0)
    r_core = max(f['d_core'] for f in fib0) / 2.0
    r_off = max(np.hypot(f['x'], f['y']) for f in fib0)
    L = get_collection_limit_radius(NV_DEPTH + NV_WIDTH / 2.0, GAPS[-1],
                                    cfg['na'], N_DIA_RED, N_MED, r_core) + r_off
    d_deep = NV_DEPTH + NV_WIDTH / 2.0
    sig_s = max(float(exc_width(cfg['exc'], d_deep, GAPS[0])) / 2.0, 1.0)
    sig_b = max(float(exc_width(cfg['exc'], d_deep, GAPS[-1])) / 2.0, sig_s)
    PW = np.array([0.4, 0.3, 0.3])           # small Gaussian, big Gaussian, uniform

    comp = rng.choice(3, size=n, p=PW)
    x, y = np.empty(n), np.empty(n)
    for c, sig in ((0, sig_s), (1, sig_b)):
        m = comp == c
        x[m] = rng.normal(0.0, sig, m.sum())
        y[m] = rng.normal(0.0, sig, m.sum())
    m = comp == 2
    x[m] = rng.uniform(-L, L, m.sum())
    y[m] = rng.uniform(-L, L, m.sum())
    z = rng.uniform(NV_Z_MIN, NV_Z_MAX, n)

    def gauss2(sig):
        return np.exp(-(x * x + y * y) / (2.0 * sig * sig)) / (2.0 * np.pi * sig * sig)
    in_box = (np.abs(x) <= L) & (np.abs(y) <= L)
    q_xy = PW[0] * gauss2(sig_s) + PW[1] * gauss2(sig_b) + PW[2] * in_box / (4.0 * L * L)
    dens = RHO / (q_xy * (1.0 / NV_WIDTH))    # q_z = 1/NV_WIDTH
    return np.column_stack([x, y, z]), dens

# ============================ collection ================================
def lensed_mcf_eta(emitters, V0, W0, lens, g, n_dia, lam):
    """Trace one continuous printed IP-S union, then apply sixfold symmetry.

    The cylinder, side lenses, and raised central lens are not treated as
    separate optical elements: the nearest exposed surface wins, one
    air-to-IP-S Fresnel transmission is applied, and the ray then remains in
    IP-S until the silica core face.
    """
    g_side = g + MCF_SIDE_TIP_OFFSET
    g_cylinder = g + MCF_GREEN_LENS_HEIGHT - 134.04
    raw = run_ray_tracing(emitters, V0, W0, n_dia, N_MED, g, [], MODE_OVERLAP_MODEL, lam)
    p = np.column_stack((raw['X_f'].ravel(), raw['Y_f'].ravel(), np.full(raw['X_f'].size, g)))
    v = raw['V1'].reshape(-1, 3)
    best_t = np.full(len(v), np.inf)
    normal = np.zeros_like(v)

    def add_cap(apex, er, et, rr, rt, footprint_u):
        q = p - apex
        u0, w0 = q @ er, q @ et
        du, dw, dz = v @ er, v @ et, v[:, 2]
        A = du * du / (2.0 * rr) + dw * dw / (2.0 * rt)
        B = u0 * du / rr + w0 * dw / rt - dz
        C = u0 * u0 / (2.0 * rr) + w0 * w0 / (2.0 * rt) - q[:, 2]
        disc = B * B - 4.0 * A * C
        t = (-B - np.sqrt(np.maximum(disc, 0.0))) / np.maximum(2.0 * A, 1e-15)
        surf = p + t[:, None] * v
        us, ws = (surf - apex) @ er, (surf - apex) @ et
        valid = ((disc >= 0.0) & (t >= 0.0)
                 & ((us - footprint_u) ** 2 + ws ** 2 <= W0_LENS ** 2))
        take = valid & (t < best_t)
        n = ((us / rr)[:, None] * er + (ws / rt)[:, None] * et
             - np.array([0.0, 0.0, 1.0]))
        n /= np.linalg.norm(n, axis=1)[:, None]
        best_t[take] = t[take]
        normal[take] = n[take]

    add_cap(np.array([0.0, 0.0, g]), np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]), MCF_CENTRAL_R, MCF_CENTRAL_R, 0.0)
    for i in range(6):
        a = i * np.pi / 3.0
        er = np.array([np.cos(a), np.sin(a), 0.0])
        et = np.array([-np.sin(a), np.cos(a), 0.0])
        add_cap(LENS_R * er + np.array([0.0, 0.0, g_side]), er, et,
                MCF_R_RADIAL, MCF_R_TANGENTIAL, PITCH - LENS_R)

    # Exposed top of the common cylinder/pedestal is part of the same IP-S union.
    t_flat = (g_cylinder - p[:, 2]) / np.maximum(v[:, 2], 1e-15)
    flat = p + t_flat[:, None] * v
    take = ((t_flat >= 0.0) & (flat[:, 0] ** 2 + flat[:, 1] ** 2 <= 75.0 ** 2)
            & (t_flat < best_t))
    best_t[take] = t_flat[take]
    normal[take] = np.array([0.0, 0.0, -1.0])

    hit = np.isfinite(best_t)
    surf = p + np.where(hit, best_t, 0.0)[:, None] * v
    cos_i = np.clip(-np.sum(v * normal, axis=1), 0.0, 1.0)
    eta = N_MED / MCF_IPS_N
    sin_t2 = eta * eta * (1.0 - cos_i * cos_i)
    transmit = hit & (sin_t2 < 1.0)
    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin_t2))
    vp = eta * v + (eta * cos_i - cos_t)[:, None] * normal
    # Average s/p Fresnel power transmission at the air-to-IP-S printed surface.
    rs = (N_MED * cos_i - MCF_IPS_N * cos_t) / (N_MED * cos_i + MCF_IPS_N * cos_t + 1e-15)
    rp = (MCF_IPS_N * cos_i - N_MED * cos_t) / (MCF_IPS_N * cos_i + N_MED * cos_t + 1e-15)
    ts = 1.0 - 0.5 * (rs * rs + rp * rp)
    target_er = np.array([lens['x'], lens['y'], 0.0]) / LENS_R
    core = PITCH * target_er + np.array([0.0, 0.0, g + MCF_GREEN_LENS_HEIGHT])
    dt = (core[2] - surf[:, 2]) / np.maximum(vp[:, 2], 1e-12)
    base = surf + dt[:, None] * vp
    miss2 = (base[:, 0] - core[0]) ** 2 + (base[:, 1] - core[1]) ** 2
    # The side-core mode is tilted outward.  Compare the internal ray to the
    # lens-to-core axis, not global +Z (which would reject the intended tilt).
    core_axis = core - np.array([lens['x'], lens['y'], g_side])
    core_axis /= np.linalg.norm(core_axis)
    sin2 = np.maximum(0.0, 1.0 - (vp @ core_axis) ** 2)
    na_mode = (lam * 1e-3) / (np.pi * MCF_IPS_N * lens['mode_w'])
    mode = np.exp(-2.0 * miss2 / lens['mode_w'] ** 2)
    mode *= np.exp(-2.0 * sin2 / na_mode ** 2)
    w = np.broadcast_to(raw['weights'], raw['X_f'].shape).ravel() * ts * mode * transmit
    # diamond exit, explicit air-to-IP-S lens, and IP-S-to-silica core face.
    core_t = 1.0 - ((MCF_IPS_N - MCF_SIO2_N) / (MCF_IPS_N + MCF_SIO2_N)) ** 2
    per = 0.5 * w.reshape(len(emitters), len(V0)).sum(axis=1) / len(V0)
    return 6.0 * core_t * per


def eta_per_emitter(emitters, V0, W0, fibers, g, model, lam=LAM_RED, n_dia=N_DIA_RED):
    """Collection efficiency per emitter (summed over cores), chunked for memory."""
    if fibers and fibers[0].get('printed_lens'):
        # The centred 3D ensemble is rotationally symmetric, so one fully
        # traced side lens times six is the exact symmetry reduction here.
        block = max(1, RAY_BUDGET // len(V0))
        return np.concatenate([lensed_mcf_eta(emitters[i:i + block], V0, W0, fibers[0], g, n_dia, lam)
                               for i in range(0, len(emitters), block)])
    block = max(1, RAY_BUDGET // len(V0))
    parts = []
    for i in range(0, len(emitters), block):
        res = run_ray_tracing(emitters[i:i + block], V0, W0, n_dia, N_MED, g,
                              fibers, model, lam)
        parts.append(sum(s['efficiencies'] for s in res['fiber_stats']))
    eta = np.concatenate(parts)
    assert np.all(np.isfinite(eta)) and np.all((eta >= 0.0) & (eta <= 1.0))
    return eta

# =============================== sweep ==================================
def run_sweeps():
    single = np.array([[0.0, 0.0, -NV_DEPTH]])
    out, s_max_seen = {}, 0.0

    for cfg in CONFIGS:
        print(f"  sweeping {cfg['name']} ...", flush=True)
        V0_1, W0_1 = escape_cone_quadrature(cfg['n1'], N_DIA_RED, seed=17)
        V0_e, W0_e = escape_cone_quadrature(cfg['nen'], N_DIA_RED, seed=23)
        em, dens = sample_ensemble(cfg, N_EMIT, np.random.default_rng(42))
        r_em = np.hypot(em[:, 0], em[:, 1])
        order = np.argsort(r_em)

        def one_gap(item):
            _, g = item
            fibs = cfg['fibers'](g)
            eta1 = cfg['tf'] * eta_per_emitter(single, V0_1, W0_1, fibs, g, cfg['model'])[0]
            R1, s1 = exc_rate(cfg['exc'], 0.0, 0.0, -NV_DEPTH, g)
            rate1 = R1 * eta1

            eta_i = eta_per_emitter(em, V0_e, W0_e, fibs, g, cfg['model'])
            Ri, sE = exc_rate(cfg['exc'], em[:, 0], em[:, 1], em[:, 2], g)
            u = Ri * eta_i * dens                     # signal carried per sample
            etaE = cfg['tf'] * u.sum() / (Ri * dens).sum()
            powE = u.mean() * cfg['tf'] * E_PHOT_RED * 1e9        # nW
            cu = np.cumsum(u[order])
            # A50 is the on-axis circular area on the NV layer containing
            # half the excitation x collection weighted ensemble signal.
            # This definition remains meaningful for the MCF's six lobes.
            r50 = r_em[order][np.searchsorted(cu, 0.5 * cu[-1])] if cu[-1] > 0 else np.nan
            return eta1, rate1, etaE, powE, np.pi * r50 * r50, max(s1, sE)

        with ThreadPoolExecutor(max_workers=SWEEP_WORKERS) as pool:
            rows = np.asarray(list(pool.map(one_gap, enumerate(GAPS))))
        eta1, rate1, etaE, powE, a50, s_seen = rows.T
        s_max_seen = max(s_max_seen, np.max(s_seen))

        out[cfg['name']] = dict(cfg=cfg, eta1=eta1, rate1=rate1,
                                etaE=etaE, powE=powE, a50=a50,
                                g_opt=float(GAPS[np.argmax(eta1)]))
    return out, s_max_seen

def run_spectra(results):
    """eta(lambda) for a single on-axis NV, each probe at its optimal gap."""
    single = np.array([[0.0, 0.0, -NV_DEPTH]])
    spectra = {}
    for name, r in results.items():
        cfg, g = r['cfg'], r['g_opt']
        # Cover the largest escape cone in the spectrum; the tracer then
        # applies wavelength-specific Snell, Fresnel, and TIR physics.
        n_min = min(float(diamond_sellmeier(lam / 1000.0)) for lam in LAM_SPECTRUM)
        V0, W0 = escape_cone_quadrature(cfg['n1'], n_min, seed=31)
        spectra[name] = np.array([
            cfg['tf'] * eta_per_emitter(single, V0, W0, cfg['fibers'](g), g, cfg['model'],
                                         lam=lam, n_dia=float(diamond_sellmeier(lam / 1000.0)))[0]
            for lam in LAM_SPECTRUM])
    return spectra

# =============================== figures ================================
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.5,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.0, "savefig.dpi": 300,
})

def style(cfg):
    return dict(color=cfg['color'], ls=cfg['ls'], marker=cfg['marker'],
                markersize=4.5, markevery=2, label=cfg['name'])

def save(fig, stem):
    fig.savefig(os.path.join(OUT, stem + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)

def fig1(results):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True)
    panels = [("eta1",  "Into-fiber efficiency $\\eta$ (%)",         100.0, "(a) single NV"),
              ("rate1", "Collected rate (kcps)",                       1e-3,  "(b) single NV, 10 mW excitation"),
              ("etaE",  "Excitation-weighted $\\bar\\eta$ (%)",      100.0, "(c) 3D ensemble, 3 ppm"),
              ("powE",  "Collected optical power (nW)",               1.0,   "(d) 3D ensemble, 10 mW excitation")]
    for ax, (key, ylab, scale, title) in zip(axs.ravel(), panels):
        for r in results.values():
            ax.plot(GAPS, np.asarray(r[key]) * scale, **style(r['cfg']))
        ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5, loc="left", color=INK2)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    for ax in axs[1]:
        ax.set_xlabel("Fiber-diamond gap $g$ ($\\mu$m)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig1_efficiency_vs_gap")

def fig2(results, spectra):
    fig, (ax, ax_s) = plt.subplots(
        2, 1, figsize=(5.2, 4.0), sharex=True,
        gridspec_kw=dict(height_ratios=[3.2, 1.0], hspace=0.12))
    for name, eta in spectra.items():
        cfg = results[name]['cfg']
        st = style(cfg)
        st['markevery'] = 1
        st['label'] = f"{name} (g = {results[name]['g_opt']:.0f} $\\mu$m)"
        ax.plot(LAM_SPECTRUM, eta * 100.0, **st)
    ax.set_yscale("log")
    ax.set_ylabel("Into-fiber efficiency $\\eta(\\lambda)$ (%)")
    ax.legend(fontsize=8, loc="center right")

    S = nv_emission_spectrum(LAM_SPECTRUM)
    ax_s.fill_between(LAM_SPECTRUM, S / S.max(), color="#e1e0d9")
    ax_s.set_ylabel("S($\\lambda$)\n(norm.)", fontsize=8.5, color=INK2)
    ax_s.set_yticks([0, 1])
    ax_s.set_xlabel("Wavelength $\\lambda$ (nm)")
    save(fig, "fig2_spectral_efficiency")

def fig3(results):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for r in results.values():
        ax.plot(GAPS, r['a50'], **style(r['cfg']))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fiber-diamond gap $g$ ($\\mu$m)")
    ax.set_ylabel("50%-signal collection area $A_{50}$ ($\\mu$m$^2$)")
    ax.legend(fontsize=8.5)
    save(fig, "fig3_resolution_vs_gap")


def fig6_efficiency_comparison(results):
    """Direct, same-gap comparison of the three physical fiber probes."""
    probes = (("SM", "SM"), ("MM", "MM"), ("MCF", "MCF fixed aim"))
    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(5.4, 5.0), sharex=True,
        gridspec_kw=dict(height_ratios=[2.2, 1.5], hspace=0.12))

    eta = {}
    for label, name in probes:
        r = results[name]
        eta[label] = np.asarray(r['etaE'])
        st = style(r['cfg'])
        st['label'] = label
        ax.plot(GAPS, 100.0 * eta[label], **st)
    ax.set_yscale("log")
    ax.set_ylabel("3D-ensemble $\\bar\\eta$ (%)")
    ax.set_title("3 ppm NV layer, $z=-90$ to $-80$ $\\mu$m",
                 fontsize=9.5, loc="left", color=INK2)
    ax.legend(fontsize=8.5, ncol=3)

    pairs = (("MM", "SM", "#1baf7a", "-"),
             ("MCF", "SM", "#d78500", "-"),
             ("MCF", "MM", "#d78500", "--"))
    for numerator, denominator, color, ls in pairs:
        ratio = np.divide(eta[numerator], eta[denominator],
                          out=np.full_like(eta[numerator], np.nan),
                          where=eta[denominator] > 0.0)
        ax_r.plot(GAPS, ratio, color=color, ls=ls, marker="o", markersize=3.8,
                  markevery=2, label=f"{numerator} / {denominator}")
    ax_r.axhline(1.0, color=INK2, lw=0.8)
    ax_r.set_yscale("log")
    ax_r.set_xlabel("Fiber-diamond gap $g$ ($\\mu$m)")
    ax_r.set_ylabel("Efficiency ratio")
    ax_r.legend(fontsize=8.0, ncol=3)
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.5, 0.025, EXPERIMENT_NOTE, ha="center", fontsize=7.8, color=INK2)
    save(fig, "fig6_collection_efficiency_comparison")


def write_efficiency_comparison(results):
    """Write every same-gap efficiency, pairwise difference, and ratio."""
    probes = (("SM", "SM"), ("MM", "MM"), ("MCF", "MCF fixed aim"))
    eta = {label: 100.0 * np.asarray(results[name]['etaE'])
           for label, name in probes}
    pairs = (("MM", "SM"), ("MCF", "SM"), ("MCF", "MM"))
    fields = (["gap_um"] + [f"{label}_eta_percent" for label, _ in probes]
              + [f"{a}_minus_{b}_percentage_points" for a, b in pairs]
              + [f"{a}_over_{b}" for a, b in pairs])
    path = os.path.join(OUT, "ensemble_collection_efficiency_comparison.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for k, gap in enumerate(GAPS):
            row = {"gap_um": f"{gap:.6g}"}
            row.update({f"{label}_eta_percent": f"{values[k]:.9g}"
                        for label, values in eta.items()})
            row.update({f"{a}_minus_{b}_percentage_points": f"{eta[a][k] - eta[b][k]:.9g}"
                        for a, b in pairs})
            row.update({f"{a}_over_{b}": f"{eta[a][k] / eta[b][k]:.9g}"
                        if eta[b][k] > 0.0 else "nan" for a, b in pairs})
            writer.writerow(row)
    return path


def mcf_display_paths(g, n_rays=4001):
    """Trace display rays through the first surface of the continuous print.

    This is the same union-envelope/refraction convention as ``lensed_mcf_eta``
    but returns the hit point and in-polymer direction so the interactive figure
    can draw a continuous path through the raised central lens and side-lens body.
    """
    all_fibers = mcf_fibers(g, reaim=False)
    theta_c = np.arcsin(N_MED / N_DIA_RED)
    theta = np.linspace(-0.999 * theta_c, 0.999 * theta_c, n_rays)
    v0 = np.column_stack((np.sin(theta), np.zeros_like(theta), np.cos(theta)))
    raw = run_ray_tracing(np.array([[0.0, 0.0, -NV_DEPTH]]), v0,
                          np.ones(len(v0)), N_DIA_RED, N_MED, g, [],
                          MODE_OVERLAP_MODEL, LAM_RED)
    x_int = raw['X_int'][0]
    p = np.column_stack((raw['X_f'][0], raw['Y_f'][0], np.full(n_rays, g)))
    v = raw['V1'][0]
    best_t = np.full(n_rays, np.inf)
    normal = np.zeros_like(v)

    def add_cap(apex, er, et, rr, rt, footprint_u):
        q = p - apex
        u0, w0 = q @ er, q @ et
        du, dw, dz = v @ er, v @ et, v[:, 2]
        A = du * du / (2.0 * rr) + dw * dw / (2.0 * rt)
        B = u0 * du / rr + w0 * dw / rt - dz
        C = u0 * u0 / (2.0 * rr) + w0 * w0 / (2.0 * rt) - q[:, 2]
        disc = B * B - 4.0 * A * C
        t = (-B - np.sqrt(np.maximum(disc, 0.0))) / np.maximum(2.0 * A, 1e-15)
        surf = p + t[:, None] * v
        us, ws = (surf - apex) @ er, (surf - apex) @ et
        valid = ((disc >= 0.0) & (t >= 0.0)
                 & ((us - footprint_u) ** 2 + ws ** 2 <= W0_LENS ** 2))
        take = valid & (t < best_t)
        n = ((us / rr)[:, None] * er + (ws / rt)[:, None] * et
             - np.array([0.0, 0.0, 1.0]))
        n /= np.linalg.norm(n, axis=1)[:, None]
        best_t[take] = t[take]
        normal[take] = n[take]

    g_side = g + MCF_SIDE_TIP_OFFSET
    add_cap(np.array([0.0, 0.0, g]), np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]), MCF_CENTRAL_R, MCF_CENTRAL_R, 0.0)
    for i in range(6):
        a = i * np.pi / 3.0
        er = np.array([np.cos(a), np.sin(a), 0.0])
        et = np.array([-np.sin(a), np.cos(a), 0.0])
        add_cap(LENS_R * er + np.array([0.0, 0.0, g_side]), er, et,
                MCF_R_RADIAL, MCF_R_TANGENTIAL, PITCH - LENS_R)

    g_cylinder = g + MCF_GREEN_LENS_HEIGHT - 134.04
    t_flat = (g_cylinder - p[:, 2]) / np.maximum(v[:, 2], 1e-15)
    flat = p + t_flat[:, None] * v
    take = ((t_flat >= 0.0) & (flat[:, 0] ** 2 + flat[:, 1] ** 2 <= 75.0 ** 2)
            & (t_flat < best_t))
    best_t[take] = t_flat[take]
    normal[take] = np.array([0.0, 0.0, -1.0])

    hit = np.isfinite(best_t)
    surf = p + np.where(hit, best_t, 0.0)[:, None] * v
    cos_i = np.clip(-np.sum(v * normal, axis=1), 0.0, 1.0)
    eta = N_MED / MCF_IPS_N
    sin_t2 = eta * eta * (1.0 - cos_i * cos_i)
    transmit = hit & (sin_t2 < 1.0)
    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin_t2))
    vp = eta * v + (eta * cos_i - cos_t)[:, None] * normal

    core_z = g + MCF_GREEN_LENS_HEIGHT
    dt = (core_z - surf[:, 2]) / np.maximum(vp[:, 2], 1e-12)
    base = surf + dt[:, None] * vp
    na_mode = (LAM_RED * 1e-3) / (np.pi * MCF_IPS_N * MCF_MODE_W)
    accepted_by = []
    for core_x in (PITCH, -PITCH):
        miss2 = (base[:, 0] - core_x) ** 2 + base[:, 1] ** 2
        lens_x = LENS_R if core_x > 0.0 else -LENS_R
        core_axis = np.array([core_x - lens_x, 0.0, core_z - g_side])
        core_axis /= np.linalg.norm(core_axis)
        sin2 = np.maximum(0.0, 1.0 - (vp @ core_axis) ** 2)
        mode = np.exp(-2.0 * miss2 / MCF_MODE_W ** 2)
        mode *= np.exp(-2.0 * sin2 / na_mode ** 2)
        accepted_by.append(transmit & (mode > np.exp(-2.0)))
    return dict(x_int=x_int, surface=surf, vp=vp, base=base, transmit=transmit,
                accepted_by=np.column_stack(accepted_by), all_fibers=all_fibers,
                g_side=g_side, core_z=core_z)


def fig4_mcf_ray_trace(results):
    """Full STL print, end-face layout, and deterministic ray trace."""
    cfg = results["MCF fixed aim"]['cfg']
    g = results["MCF fixed aim"]['g_opt']
    paths = mcf_display_paths(g)
    g_side, core_z = paths['g_side'], paths['core_z']
    all_fibers = paths['all_fibers']
    fibers = [all_fibers[0], all_fibers[3]]
    accepted_by = paths['accepted_by'].T
    accepted = np.any(accepted_by, axis=0)
    x_int, surface = paths['x_int'], paths['surface']

    fig = plt.figure(figsize=(11.2, 4.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.24, 0.82, 1.55], wspace=0.30)
    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax = fig.add_subplot(gs[0, 2])

    # Full printed assembly from the supplied binary STL files.  The common
    # cylinder/pedestal and both lens STLs are shown; Ring.STL is intentionally
    # absent because Cylinder_lenses_job.gwl comments out Ring_data.gwl.
    meshes = mcf_printed_meshes(g)
    for kind, tri, color, label in meshes:
        ax_3d.add_collection3d(Poly3DCollection(
            tri, facecolor=color, edgecolor="#4d4d4d", linewidth=0.08,
            alpha=0.26 if kind == "cylinder" else 0.58, label=label))
    if meshes:
        ax_3d.set_xlim(-82, 82)
        ax_3d.set_ylim(-82, 82)
        ax_3d.set_zlim(-NV_DEPTH - 8, g + MCF_GREEN_LENS_HEIGHT + 8)
    xx, yy = np.meshgrid(np.array([-75.0, 75.0]), np.array([-75.0, 75.0]))
    ax_3d.plot_surface(xx, yy, np.zeros_like(xx), color="#6aaec2", alpha=0.15,
                       shade=False, linewidth=0)
    ax_3d.scatter([0], [0], [-NV_DEPTH], s=24, color="#0b0b0b",
                  depthshade=False, label="single NV")
    # Draw the same accepted/rejected rays in 3D, with the side-core fold
    # continuing from the microlens plane to the fibre core plane.
    rejected = np.flatnonzero(~accepted)
    for i in rejected[np.linspace(0, len(rejected) - 1,
                                  min(45, len(rejected)), dtype=int)]:
        ax_3d.plot([0, x_int[i], surface[i, 0]], [0, 0, 0],
                   [-NV_DEPTH, 0, surface[i, 2]], color="#a7a59e", lw=0.35, alpha=0.18)
    collected = np.flatnonzero(accepted)
    for i in collected[np.linspace(0, len(collected) - 1,
                                   min(20, len(collected)), dtype=int)]:
        j = np.flatnonzero(accepted_by[:, i])[0]
        core_x = PITCH if all_fibers[j]['x'] > 0 else -PITCH
        ax_3d.plot([0, x_int[i], surface[i, 0]], [0, 0, 0],
                   [-NV_DEPTH, 0, surface[i, 2]], color="#c85b17", lw=0.72, alpha=0.88)
        ax_3d.plot([surface[i, 0], core_x], [0, 0],
                   [surface[i, 2], core_z], color="#c85b17",
                   lw=0.58, ls="--", alpha=0.68)
    ax_3d.set_box_aspect((1.0, 1.0, 2.35))
    ax_3d.view_init(elev=18, azim=-58)
    ax_3d.set_xlabel("x ($\u03bcm$)", labelpad=-2)
    ax_3d.set_ylabel("y ($\u03bcm$)", labelpad=-2)
    ax_3d.set_zlabel("distance from diamond ($\u03bcm$)", labelpad=2)
    ax_3d.set_title("(a) Full printed MCF tip", loc="left", fontsize=9.5, color=INK2)
    if meshes:
        ax_3d.legend(fontsize=6.1, loc="upper left", bbox_to_anchor=(-0.03, 1.03))
    else:
        ax_3d.text2D(0.04, 0.92, "STL files not found;\nshowing ray path only",
                     transform=ax_3d.transAxes, fontsize=7.2, color=INK2)

    angles = np.arange(6) * np.pi / 3.0
    core_xy = PITCH * np.column_stack((np.cos(angles), np.sin(angles)))
    pupil_xy = LENS_R * np.column_stack((np.cos(angles), np.sin(angles)))
    ax_top.scatter(core_xy[:, 0], core_xy[:, 1], s=52, facecolor="#2a78d6",
                   edgecolor="white", linewidth=0.7, label="physical side core")
    ax_top.scatter(pupil_xy[:, 0], pupil_xy[:, 1], s=30, facecolor="#d78500",
                   edgecolor="white", linewidth=0.7, label="red lens pupil")
    ax_top.scatter([0], [0], s=58, marker="s", color="#1baf7a", label="green core")
    for core, pupil in zip(core_xy, pupil_xy):
        ax_top.plot([core[0], pupil[0]], [core[1], pupil[1]], color="#a7a59e", lw=0.8)
    ax_top.set_aspect("equal")
    ax_top.set_xlim(-43, 43)
    ax_top.set_ylim(-43, 43)
    ax_top.set_xlabel("x ($\\mu$m)")
    ax_top.set_ylabel("y ($\\mu$m)")
    ax_top.set_title("(b) MCF end-face geometry", loc="left", fontsize=9.5, color=INK2)
    ax_top.legend(fontsize=6.4, loc="upper left", bbox_to_anchor=(0.0, 1.0),
                 borderaxespad=0.2, frameon=False)

    ax.axvspan(-100, 0, color="#c9e5f2", alpha=0.72, label="diamond")
    ax.axvspan(0, g, color="#f5f4ef", alpha=1.0, label="air to central tip")
    ax.axvspan(g, g + MCF_SIDE_TIP_OFFSET, color="#cfe8dd", alpha=0.72,
               label="central raised lens")
    ax.axvspan(g_side, g + MCF_GREEN_LENS_HEIGHT, color="#e8d8b7", alpha=0.8,
               label="common IP-S body")
    # STL-calibrated cylindrical cap; the six overlapping caps are drawn as
    # their outer envelope, never as stacked optical interfaces.
    cap_x = np.linspace(-W0_LENS, W0_LENS, 160)
    cap_z = g_side + cap_x * cap_x / (2.0 * MCF_R_RADIAL)
    for sign in (-1, 1):
        ax.plot(cap_z, sign * LENS_R + cap_x, color="#875a13", lw=1.15, zorder=4)
    central_x = np.linspace(-W0_LENS, W0_LENS, 160)
    central_z = g + central_x * central_x / (2.0 * MCF_CENTRAL_R)
    ax.plot(central_z, central_x, color="#178b63", lw=1.25, zorder=4)
    for i in rejected[np.linspace(0, len(rejected) - 1, min(140, len(rejected)), dtype=int)]:
        ax.plot([-NV_DEPTH, 0, surface[i, 2]], [0, x_int[i], surface[i, 0]], color="#a7a59e", lw=0.35, alpha=0.20)
    for i in collected[np.linspace(0, len(collected) - 1, min(45, len(collected)), dtype=int)]:
        j = np.flatnonzero(accepted_by[:, i])[0]
        core_x = PITCH if fibers[j]['x'] > 0 else -PITCH
        ax.plot([-NV_DEPTH, 0, surface[i, 2]], [0, x_int[i], surface[i, 0]], color="#c85b17", lw=0.9, alpha=0.92)
        ax.plot([surface[i, 2], core_z], [surface[i, 0], core_x], color="#c85b17",
                lw=0.65, ls="--", alpha=0.7)
    ax.scatter([0], [0], s=38, color="#0b0b0b", zorder=5, label="single NV")
    for sign in (-1, 1):
        ax.scatter([g_side], [sign * LENS_R], s=30, color="#d78500", zorder=5)
        ax.scatter([core_z], [sign * PITCH], s=34, color="#2a78d6", zorder=5)
    ax.scatter([core_z], [0], s=38, marker="s", color="#1baf7a", zorder=5)
    ax.axvline(0, color="#75808a", lw=0.8)
    ax.axvline(g_side, color="#75808a", lw=0.8)
    ax.set_xlim(-105, core_z + 12)
    ax.set_ylim(-52, 52)
    ax.set_xlabel("z from diamond surface ($\\mu$m)")
    ax.set_ylabel("x in meridional plane ($\\mu$m)")
    ax.set_title(f"(c) 700 nm rays at fixed-design optimum, g = {g:.0f} $\\mu$m", loc="left", fontsize=9.5, color=INK2)
    ax.text(0.98, 0.04, "orange = collected; dashed =\nin-polymer path to core",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color=INK2)
    fig.text(0.5, 0.005,
             "MCF geometry: Cylinder_job.gwl + Side lenses_data.gwl + Central lens_data.gwl; "
             "IP-S n=1.52, MFD=10 um. All three meshes are one printed IP-S piece; "
             "Ring_data.gwl is commented out and overlapping caps are one external polymer envelope.",
             ha="center", fontsize=7.2, color=INK2)
    # 3-D axes are not compatible with tight_layout on older Matplotlib; use
    # explicit margins so the vector/PDF layout is deterministic.
    fig.subplots_adjust(left=0.025, right=0.995, bottom=0.16, top=0.94, wspace=0.30)
    save(fig, "fig4_mcf_ray_trace")


def fig4_mcf_interactive(results):
    """Write a rotatable Plotly version of the complete printed-tip figure."""
    import plotly.graph_objects as go

    cfg = results["MCF fixed aim"]['cfg']
    g = results["MCF fixed aim"]['g_opt']
    paths = mcf_display_paths(g)
    g_side, core_z = paths['g_side'], paths['core_z']
    all_fibers = paths['all_fibers']
    accepted_by = paths['accepted_by'].T
    accepted = np.any(accepted_by, axis=0)
    x_int, surface = paths['x_int'], paths['surface']

    fig = go.Figure()

    def add_mesh(tri, name, color, opacity, legendgroup=None, showlegend=True):
        n = len(tri)
        xyz = tri.reshape(-1, 3)
        idx = np.arange(n) * 3
        fig.add_trace(go.Mesh3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            i=idx, j=idx + 1, k=idx + 2,
            name=name, color=color, opacity=opacity,
            legendgroup=legendgroup, showlegend=showlegend,
            flatshading=True, lighting=dict(ambient=0.55, diffuse=0.65,
                                            specular=0.18, roughness=0.8),
            hovertemplate=name + "<br>x=%{x:.1f} μm<br>y=%{y:.1f} μm<br>z=%{z:.1f} μm<extra></extra>"))

    for part_index, (kind, tri, color, label) in enumerate(mcf_printed_meshes(g)):
        # Colors still distinguish the STL regions visually, but the single
        # legend item and legendgroup make clear that they are one printed
        # polymer piece, not three serial optical interfaces.
        add_mesh(tri, "printed IP-S union" if part_index == 0 else label,
                 color, 0.18 if kind == "cylinder" else 0.38,
                 legendgroup="printed-union", showlegend=(part_index == 0))

    # Diamond surface and the on-axis emitter.
    d = np.array([-75.0, 75.0])
    fig.add_trace(go.Mesh3d(
        x=[d[0], d[1], d[1], d[0]], y=[d[0], d[0], d[1], d[1]], z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3], name="diamond surface",
        color="#6aaec2", opacity=0.16, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[-NV_DEPTH], mode="markers", name="single NV",
        marker=dict(size=5, color="#111111"),
        hovertemplate="single NV<br>z=%{z:.1f} μm<extra></extra>"))

    def add_path_trace(indices, name, color, dash="solid", width=2.0):
        xx, yy, zz = [], [], []
        for i in indices:
            xx += [0.0, float(x_int[i]), float(surface[i, 0]), None]
            yy += [0.0, 0.0, 0.0, None]
            zz += [-NV_DEPTH, 0.0, float(surface[i, 2]), None]
        fig.add_trace(go.Scatter3d(
            x=xx, y=yy, z=zz, mode="lines", name=name,
            line=dict(color=color, width=width, dash=dash),
            hoverinfo="skip"))

    rejected = np.flatnonzero(~accepted)
    collected = np.flatnonzero(accepted)
    add_path_trace(rejected[np.linspace(0, len(rejected) - 1,
                                      min(55, len(rejected)), dtype=int)],
                   "rejected rays", "#a7a59e", width=1.2)
    fold_x, fold_y, fold_z = [], [], []
    for i in collected[np.linspace(0, len(collected) - 1,
                                   min(28, len(collected)), dtype=int)]:
        j = np.flatnonzero(accepted_by[:, i])[0]
        core_x = PITCH if all_fibers[j]['x'] > 0 else -PITCH
        fold_x += [float(surface[i, 0]), core_x, None]
        fold_y += [0.0, 0.0, None]
        fold_z += [float(surface[i, 2]), core_z, None]
    add_path_trace(collected[np.linspace(0, len(collected) - 1,
                                         min(28, len(collected)), dtype=int)],
                   "collected rays", "#c85b17", width=2.5)
    fig.add_trace(go.Scatter3d(
        x=fold_x, y=fold_y, z=fold_z, mode="lines", name="lens-to-core fold",
        line=dict(color="#c85b17", width=2.2, dash="dash"), hoverinfo="skip"))

    angles = np.arange(6) * np.pi / 3.0
    core_xy = PITCH * np.column_stack((np.cos(angles), np.sin(angles)))
    pupil_xy = LENS_R * np.column_stack((np.cos(angles), np.sin(angles)))
    fig.add_trace(go.Scatter3d(
        x=core_xy[:, 0], y=core_xy[:, 1], z=np.full(6, core_z),
        mode="markers", name="physical side cores",
        marker=dict(size=4, color="#2a78d6"), hovertemplate="side core<extra></extra>"))
    fig.add_trace(go.Scatter3d(
        x=pupil_xy[:, 0], y=pupil_xy[:, 1], z=np.full(6, g_side),
        mode="markers", name="red lens pupils",
        marker=dict(size=3.5, color="#d78500"), hovertemplate="side lens pupil<extra></extra>"))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[core_z],
        mode="markers", name="green central core",
        marker=dict(size=5, color="#1baf7a", symbol="square"),
        hovertemplate="central core<extra></extra>"))

    legend_positions = [
        ("Legend: top-left", 0.01, 0.99, "left", "top"),
        ("Legend: top-right", 0.99, 0.99, "right", "top"),
        ("Legend: bottom-left", 0.01, 0.01, "left", "bottom"),
        ("Legend: bottom-right", 0.99, 0.01, "right", "bottom"),
    ]
    fig.update_layout(
        title=f"Interactive MCF printed structure and 700 nm ray paths (g = {g:.0f} μm)",
        template="plotly_white", height=760, margin=dict(l=0, r=0, t=92, b=0),
        scene=dict(xaxis_title="x (μm)", yaxis_title="y (μm)",
                   zaxis_title="distance from diamond (μm)",
                   aspectmode="manual", aspectratio=dict(x=1, y=1, z=2.35),
                   xaxis=dict(range=[-82, 82]), yaxis=dict(range=[-82, 82]),
                   zaxis=dict(range=[-NV_DEPTH - 8, core_z + 8]),
                   camera=dict(eye=dict(x=1.45, y=1.45, z=1.05))),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.78)", groupclick="togglegroup"),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.01, y=1.12,
            xanchor="left", yanchor="top", showactive=True,
            buttons=[dict(label=label, method="relayout",
                          args=[{"legend": {"x": x, "y": y,
                                             "xanchor": xa, "yanchor": ya}}])
                     for label, x, y, xa, ya in legend_positions])])
    path = os.path.join(OUT, "fig4_mcf_structure_interactive.html")
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                   config={"responsive": True, "displaylogo": False,
                           "scrollZoom": True})
    return path

# ================================ main ==================================
if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)

    results, s_max = run_sweeps()
    spectra = run_spectra(results)
    fig1(results)
    fig2(results, spectra)
    fig3(results)
    fig4_mcf_ray_trace(results)
    fig4_mcf_interactive(results)
    fig6_efficiency_comparison(results)
    comparison_csv = write_efficiency_comparison(results)

    print(f"\nfigures written to {OUT}  ({time.time()-t0:.0f} s)")
    print(f"max saturation parameter s = I/I_sat encountered: {s_max:.2e}"
          f"  ->  {'saturation matters' if s_max > 0.1 else 'deeply linear regime'}")
    hdr = (f"{'probe':>14} | {'g* (um)':>7} | {'eta1(g*) %':>10} | "
           f"{'rate1 (kcps)':>12} | {'P_ens (nW)':>10} | {'A50(g*) um2':>12}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for name, r in results.items():
        k = int(np.argmax(r['eta1']))
        print(f"{name:>14} | {r['g_opt']:7.0f} | {r['eta1'][k]*100:10.4f} | "
              f"{r['rate1'][k]*1e-3:12.5f} | {r['powE'][k]:10.3f} | {r['a50'][k]:12.1f}")
    print("\n3D 3 ppm ensemble collection-efficiency maxima (physical probes):")
    for label, name in (("SM", "SM"), ("MM", "MM"), ("MCF", "MCF fixed aim")):
        r = results[name]
        k = int(np.argmax(r['etaE']))
        sm = results['SM']['etaE'][k]
        print(f"  {label:>3}: {100.0*r['etaE'][k]:.6g}% at g={GAPS[k]:.0f} um; "
              f"{r['etaE'][k]/sm:.6g}x SM at the same gap")
    print(f"Same-gap pairwise values: {comparison_csv}")
    print(f"\n{EXPERIMENT_NOTE}")
    print("\nNotes: fixed-aim MCF uses the supplied STL/GWL geometry: 17.5 um side-lens radius,"
          "\n35.225 um central-to-side tip offset, IP-S n=1.52 and 10 um MFD; re-aimed MCF"
          "\nrecalculates the side-core boresights at every gap for the 85 um NV layer."
          "\nA50 is an on-axis circular area on the NV layer containing 50% of the"
          "\nexcitation x collection weighted ensemble signal. Single-NV rates are reported"
          "\nin kcps; ensemble operation is the practical mode for this deep NV layer.")
