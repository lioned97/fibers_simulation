"""
sensitivity.py -- shot-noise ODMR magnetic-sensitivity figure of merit Psi(z)
for each NV-diamond fiber probe, robust over the 80/85/90 um NV layer.

All physics is imported from paper_figures.py (which imports physics.py), so this
can never drift from the paper's ray-tracing model: the escape-cone quadrature,
the saturable green excitation, and each probe's collection efficiency.

Psi is a volume-normalized sensitivity: lower is better.  Per depth z,
  Psi(z) = dnu(z) / (C_eff(z) * sqrt(sbar(z)))
with C_eff the background-diluted ODMR contrast, sbar the signal-weighted mean
photon-density (voxel size cancels -- no area factor), and dnu the ODMR
linewidth (intrinsic + an optional gradient-broadening term).  psi_robust is the
worst (max) over the layer -- the number a probe is graded on.

Background is the confocal-rejection 2-knob model: autofluorescence generated
along the WHOLE excitation path through the 100 um diamond and collected wherever
the probe's collection sees it.  A single multiplicative knob b_auto is CALIBRATED
once on the MM baseline (S/(S+B)=0.5 at its optimum) and SHARED by every probe;
b0 is an additive floor.

No CLI, no disk cache, no matplotlib.  Fixed seeds -> fully reproducible.
"""
import numpy as np

from physics import trapz
from paper_figures import (
    CONFIGS, GAPS, NV_DEPTH, N_DIA_RED, RHO,
    exc_rate, exc_width, eta_per_emitter, escape_cone_quadrature,
)

# ---- NV / material defaults (overridable via function args) ----
C0        = 0.03                    # intrinsic ODMR contrast
T2_STAR   = 0.5e-6                  # s
DNU0      = 1.0 / (np.pi * T2_STAR)  # Hz  (intrinsic ODMR linewidth)
K_GRAD    = 0.0                     # Hz/um  (gradient broadening per FWHM um)
T_LAYER   = 10.0                    # um  (NV layer thickness)
DEPTHS    = (80.0, 85.0, 90.0)      # um below the fiber-facing surface

N_R, N_AZ = 120, 6                  # signal-grid radial pts x MCF azimuths/period
N_R_BG    = 60                      # coarser background-plane radial grid
BG_PLANES = np.linspace(5.0, 95.0, 7)  # autofluorescence planes through the diamond

MM_CFG = next(c for c in CONFIGS if c['name'] == "MM")
_B_AUTO_CACHE = {}                  # keyed by (cfg name, num_rays)


# ---- grid / profile helpers (the resolution_at_layer.py patterns, with a
#      num_rays override so tests can run at ~2000 rays in seconds) ----
def _radial_grid(cfg, gap, depth, n_r):
    """Power-law grid sized to the widest thing this probe does at this depth."""
    r_max = max(3.0 * float(np.atleast_1d(exc_width(cfg['exc'], depth, gap))[0]), 45.0)
    return r_max * np.linspace(0.0, 1.0, n_r) ** 1.5


def collection_profile(cfg, gap, depth, r, num_rays=None, seed=17):
    """Azimuthally averaged single-NV eta(r) on a ring grid at `depth`.

    SM/MM are axisymmetric (1 azimuth exact); the MCF needs the 6 midpoints of
    one 60-degree lens period.  Ray count defaults to cfg['nen']; pass num_rays
    to trade fidelity for speed (tests, background planes)."""
    n_az = N_AZ if cfg['exc'] == "MCF" else 1
    az = (np.arange(n_az) + 0.5) * (np.pi / 3.0) / n_az
    rr, aa = np.meshgrid(r, az, indexing="ij")
    em = np.column_stack([(rr * np.cos(aa)).ravel(), (rr * np.sin(aa)).ravel(),
                          np.full(rr.size, -depth)])
    V0, W0 = escape_cone_quadrature(num_rays or cfg['nen'], N_DIA_RED, seed=seed)
    eta = eta_per_emitter(em, V0, W0, cfg['fibers'](gap), gap, cfg['model'])
    return eta.reshape(len(r), n_az).mean(axis=1)


def width_half_max(r, p):
    """Outermost half-maximum DIAMETER of a radial profile (from resolution_at_layer)."""
    h = p.max() / 2.0
    above = np.nonzero(p >= h)[0]
    if len(above) == 0:                       # guard: empty/flat-zero profile
        return np.nan
    i = above[-1]
    if i == len(r) - 1:
        return 2.0 * r[-1]                     # not resolved inside the grid
    r_half = r[i] + (r[i + 1] - r[i]) * (p[i] - h) / (p[i] - p[i + 1])
    return 2.0 * r_half


# ---- signal and background integrals ----
def _signal(cfg, gap, depth, num_rays=None, seed=17):
    """Return (S, sbar, fwhm_sig) for one NV-layer depth. S in photons/s (per um
    of layer * T_LAYER); sbar in (photons/s/um^3)^2 / (photons/s/um^3); fwhm in um."""
    r = _radial_grid(cfg, gap, depth, N_R)
    R_exc = exc_rate(cfg['exc'], r, 0.0, -depth, gap)[0]
    eta = collection_profile(cfg, gap, depth, r, num_rays=num_rays, seed=seed)
    s = RHO * R_exc * eta                     # signal density [photons/s/um^3]
    w = 2.0 * np.pi * r
    flux = trapz(s * w, r)                     # = S / T_LAYER
    S = T_LAYER * flux
    sbar = trapz(s * s * w, r) / flux if flux > 0 else 0.0   # guard: zero signal
    return S, sbar, width_half_max(r, s)


def _background_integral(cfg, gap, num_rays=None, seed=29):
    """Path-integrated collected autofluorescence weight b_int (before b_auto).

    Reuses the SAME eta machinery per plane -- the expensive part -- at a reduced
    ray count (num_rays//4) and a coarser radial grid to keep runtime sane."""
    n_bg = max(1, (num_rays or cfg['nen']) // 4)
    dzp = BG_PLANES[1] - BG_PLANES[0]
    b_int = 0.0
    for zp in BG_PLANES:
        r = _radial_grid(cfg, gap, zp, N_R_BG)
        R_exc = exc_rate(cfg['exc'], r, 0.0, -zp, gap)[0]
        eta = collection_profile(cfg, gap, zp, r, num_rays=n_bg, seed=seed)
        b_int += trapz(R_exc * eta * 2.0 * np.pi * r, r) * dzp
    return b_int


def calibrate_b_auto(mm_cfg=MM_CFG, num_rays=None):
    """Calibrate the shared background knob so the MM baseline at z=85 um and its
    own optimal gap gives S/(S+B)=0.5 (i.e. B == S there).  Cached per ray count.

    All probes share this b_auto -- it is a property of the confocal-rejection
    assumption, not of any single probe."""
    key = (mm_cfg['name'], num_rays)
    if key not in _B_AUTO_CACHE:
        g = probe_optimum(mm_cfg, num_rays=num_rays)
        S, _, _ = _signal(mm_cfg, g, 85.0, num_rays=num_rays)
        b_int = _background_integral(mm_cfg, g, num_rays=num_rays)
        _B_AUTO_CACHE[key] = S / b_int if b_int > 0 else 0.0   # guard: no path signal
    return _B_AUTO_CACHE[key]


# ---- public API ----
def probe_optimum(cfg, num_rays=None):
    """Optimal gap = argmax single-NV (on-axis, NV_DEPTH) collection efficiency
    over the paper's 0-200 um gap sweep."""
    V0, W0 = escape_cone_quadrature(num_rays or cfg['n1'], N_DIA_RED, seed=17)
    single = np.array([[0.0, 0.0, -NV_DEPTH]])
    eta = [eta_per_emitter(single, V0, W0, cfg['fibers'](g), g, cfg['model'])[0]
           for g in GAPS]
    return float(GAPS[int(np.argmax(eta))])


def sensitivity_fom(cfg, gap, depths=DEPTHS, b_auto=None, b0=0.0, num_rays=None,
                    c0=C0, dnu0=DNU0, k_grad=K_GRAD):
    """Shot-noise ODMR magnetic-sensitivity FoM Psi(z) for one probe at one gap.

    Returns a dict: 'psi' (per depth, lower=better), 'psi_robust' (max over
    depths), and the per-depth diagnostics 'C_eff', 'S', 'B', 'sbar', 'fwhm_sig'.
    b_auto=None -> calibrate once on MM and share it (see calibrate_b_auto)."""
    if b_auto is None:
        b_auto = calibrate_b_auto(MM_CFG, num_rays=num_rays)
    # B is a probe+gap property, not depth-dependent; compute once.
    b_int = _background_integral(cfg, gap, num_rays=num_rays) if b_auto != 0.0 else 0.0
    B = b_auto * b_int + b0

    n = len(depths)
    S = np.empty(n); sbar = np.empty(n); fwhm = np.empty(n)
    C_eff = np.empty(n); psi = np.empty(n)
    for i, z in enumerate(depths):
        S[i], sbar[i], fwhm[i] = _signal(cfg, gap, z, num_rays=num_rays)
        C_eff[i] = c0 * S[i] / (S[i] + B) if (S[i] + B) > 0 else 0.0
        dnu = dnu0 + k_grad * fwhm[i]
        psi[i] = (dnu / (C_eff[i] * np.sqrt(sbar[i]))
                  if (C_eff[i] > 0 and sbar[i] > 0) else np.inf)  # guard: dead layer
    return dict(psi=psi, psi_robust=float(np.max(psi)), C_eff=C_eff, S=S,
                B=np.full(n, B), sbar=sbar, fwhm_sig=fwhm)
