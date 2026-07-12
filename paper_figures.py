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
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# geometry (um)
NV_DEPTH   = 85.0                    # layer centre below the facing surface
NV_WIDTH   = 10.0                    # layer spans 80-90 um
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
RHO        = PPM * 1.76e5            # NV/um^3
N_EMIT     = 600                     # importance-sampled emitters
# The tracer keeps several (n_emitters, n_rays) arrays plus a 3-vector array.
# 0.75 M ray pairs gives a reproducible workstation-safe peak memory use.
RAY_BUDGET = 750_000                 # max emitter-ray pairs per tracer call

# MCF as-built: Shukhin et al. (OMN 2024), refined with microlens.xlsx,
# orange/650-nm row.  The spreadsheet explicitly gives x=16.8434 um rather
# than the rounded 18 um decenter quoted in the paper.  The lens is represented
# as its measured/design effective pupil--the unpublished sag is not invented.
PITCH = 35.0                          # physical side-core radius (um)
LENS_R = 16.8434                      # side-lens pupil radius (um)
W0_LENS = PITCH / 2.0                 # pupil-filling Gaussian 1/e^2 radius
MCF_RED_FREE_AIR = 141.185            # spreadsheet, 650 nm red design (um)
MCF_TARGET_DEPTH = 30.0               # spreadsheet, target below diamond face (um)
MCF_RED_LENS_HEIGHT = 158.815         # spreadsheet, side-lens IP-S thickness (um)
MCF_GREEN_LENS_HEIGHT = 194.041       # spreadsheet, central-lens IP-S thickness (um)
MCF_DESIGN_N_DIA = 2.4093             # spreadsheet, diamond index at 650 nm

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

# =========================== fiber configs =============================
def mcf_fibers(g, reaim):
    fibs = []
    for i in range(6):
        a = i * np.pi / 3.0
        fx, fy = LENS_R * np.cos(a), LENS_R * np.sin(a)
        if reaim:
            # exact Snell aim at the NV-layer centre: solve exit angle in diamond
            f = lambda t: (NV_DEPTH * np.tan(t)
                           + g * np.tan(np.arcsin(N_DIA_RED * np.sin(t))) - LENS_R)
            th_d = bisect(f, 1e-9, np.arcsin(1.0 / N_DIA_RED) - 1e-6)
            th_a = np.arcsin(N_DIA_RED * np.sin(th_d))
        else:
            th_a = MCF_FIXED_AIR_ANGLE
        b = np.array([np.cos(a) * np.sin(th_a), np.sin(a) * np.sin(th_a), np.cos(th_a)])
        fibs.append({'x': fx, 'y': fy, 'z': g, 'd_core': PITCH,
                     'd_clad': PITCH * 1.3, 'na': 0.15, 'boresight': b, 'w0': W0_LENS})
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
           na=0.15, tf=t_face(1.51), n1=131_072, nen=65_536,
           color="#d78500", ls="-",  marker="^"),
    dict(name="MCF re-aimed",    exc="MCF", model=MODE_OVERLAP_MODEL,
         fibers=lambda g: mcf_fibers(g, reaim=True),
         na=0.15, tf=t_face(1.51), n1=131_072, nen=65_536,
         color="#d78500", ls="--", marker="v"),
]

# ============================ excitation ================================
def exc_width(profile, d, g):
    """Excitation radius at depth d for gap g: Gaussian 1/e^2 w, or top-hat R (MM)."""
    if profile == "MM":
        th_a = np.arcsin(0.22)
        th_d = np.arcsin(np.sin(th_a) / N_DIA_GREEN)
        return 25.0 + g * np.tan(th_a) + d * np.tan(th_d)
    if profile == "SM":
        w0, th_a = 2.0, np.arcsin(0.12)
    else:                                    # MCF central core, collimated
        # Gaussian propagation: z_R = pi*n*w0^2/lambda.  The central core is
        # collimated by design, so only this diffraction broadening is used.
        w0 = W0_LENS
        z_eff = g + d / N_DIA_GREEN
        return w0 * np.sqrt(1.0 + (LAM_GREEN * 1e-3 * z_eff
                                    / (np.pi * w0 * w0)) ** 2)
    th_d = np.arcsin(np.sin(th_a) / N_DIA_GREEN)
    return np.sqrt(w0 ** 2 + (g * np.tan(th_a) + d * np.tan(th_d)) ** 2)

def exc_rate(profile, x, y, z, g):
    """Per-NV emission rate (photons/s into 4pi) at emitter (x, y, z<0), gap g."""
    d = -np.asarray(z, dtype=float)
    r2 = np.asarray(x) ** 2 + np.asarray(y) ** 2
    w = exc_width(profile, d, g)
    if profile == "MM":                      # top-hat
        I = np.where(r2 <= w * w, P_GREEN_MW * T_GREEN_IN / (np.pi * w * w), 0.0)
    else:                                    # Gaussian
        I = 2.0 * P_GREEN_MW * T_GREEN_IN / (np.pi * w * w) * np.exp(-2.0 * r2 / (w * w))
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
    z = rng.uniform(-NV_DEPTH - NV_WIDTH / 2.0, -NV_DEPTH + NV_WIDTH / 2.0, n)

    def gauss2(sig):
        return np.exp(-(x * x + y * y) / (2.0 * sig * sig)) / (2.0 * np.pi * sig * sig)
    in_box = (np.abs(x) <= L) & (np.abs(y) <= L)
    q_xy = PW[0] * gauss2(sig_s) + PW[1] * gauss2(sig_b) + PW[2] * in_box / (4.0 * L * L)
    dens = RHO / (q_xy * (1.0 / NV_WIDTH))    # q_z = 1/NV_WIDTH
    return np.column_stack([x, y, z]), dens

# ============================ collection ================================
def eta_per_emitter(emitters, V0, W0, fibers, g, model, lam=LAM_RED, n_dia=N_DIA_RED):
    """Collection efficiency per emitter (summed over cores), chunked for memory."""
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

        eta1 = np.zeros_like(GAPS); rate1 = np.zeros_like(GAPS)
        etaE = np.zeros_like(GAPS); powE = np.zeros_like(GAPS); a50 = np.zeros_like(GAPS)
        for k, g in enumerate(GAPS):
            fibs = cfg['fibers'](g)
            eta1[k] = cfg['tf'] * eta_per_emitter(single, V0_1, W0_1, fibs, g, cfg['model'])[0]
            R1, s1 = exc_rate(cfg['exc'], 0.0, 0.0, -NV_DEPTH, g)
            rate1[k] = R1 * eta1[k]

            eta_i = eta_per_emitter(em, V0_e, W0_e, fibs, g, cfg['model'])
            Ri, sE = exc_rate(cfg['exc'], em[:, 0], em[:, 1], em[:, 2], g)
            u = Ri * eta_i * dens                     # signal carried per sample
            etaE[k] = cfg['tf'] * u.sum() / (Ri * dens).sum()
            powE[k] = u.mean() * cfg['tf'] * E_PHOT_RED * 1e9     # nW
            cu = np.cumsum(u[order])
            # A50 is the on-axis circular area on the NV layer containing
            # half the excitation x collection weighted ensemble signal.
            # This definition remains meaningful for the MCF's six lobes.
            r50 = r_em[order][np.searchsorted(cu, 0.5 * cu[-1])] if cu[-1] > 0 else np.nan
            a50[k] = np.pi * r50 * r50
            s_max_seen = max(s_max_seen, s1, sE)

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


def fig4_mcf_ray_trace(results):
    """MCF geometry plus a deterministic meridional ray trace at its best gap."""
    cfg = results["MCF fixed aim"]['cfg']
    g = results["MCF fixed aim"]['g_opt']
    all_fibers = cfg['fibers'](g)
    # The x-z slice intersects only the two opposite side lenses.  It is a
    # ray-trace view, not a fictitious lens-sag model.
    fibers = [all_fibers[0], all_fibers[3]]
    theta_c = np.arcsin(N_MED / N_DIA_RED)
    theta = np.linspace(-0.999 * theta_c, 0.999 * theta_c, 4001)
    v0 = np.column_stack((np.sin(theta), np.zeros_like(theta), np.cos(theta)))
    trace = run_ray_tracing(np.array([[0.0, 0.0, -NV_DEPTH]]), v0,
                            np.ones(len(v0)), N_DIA_RED, N_MED, g, fibers,
                            MODE_OVERLAP_MODEL, LAM_RED)
    accepted_by = np.vstack([s['collected_mask'][0] for s in trace['fiber_stats']])
    accepted = np.any(accepted_by, axis=0)
    x_int, x_f = trace['X_int'][0], trace['X_f'][0]

    fig, (ax_top, ax) = plt.subplots(1, 2, figsize=(8.0, 3.7),
                                     gridspec_kw=dict(width_ratios=[0.82, 1.48]))
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
    ax_top.set_title("(a) MCF end-face geometry", loc="left", fontsize=9.5, color=INK2)
    ax_top.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.37))

    ax.axvspan(-100, 0, color="#c9e5f2", alpha=0.72, label="diamond")
    ax.axvspan(0, g, color="#f5f4ef", alpha=1.0, label="air gap")
    ax.axvspan(g, g + MCF_RED_LENS_HEIGHT, color="#e8d8b7", alpha=0.8,
               label="IP-S red lens")
    rejected = np.flatnonzero(~accepted)
    for i in rejected[np.linspace(0, len(rejected) - 1, min(140, len(rejected)), dtype=int)]:
        ax.plot([-NV_DEPTH, 0, g], [0, x_int[i], x_f[i]], color="#a7a59e", lw=0.35, alpha=0.20)
    collected = np.flatnonzero(accepted)
    for i in collected[np.linspace(0, len(collected) - 1, min(45, len(collected)), dtype=int)]:
        j = np.flatnonzero(accepted_by[:, i])[0]
        core_x = PITCH if fibers[j]['x'] > 0 else -PITCH
        ax.plot([-NV_DEPTH, 0, g], [0, x_int[i], x_f[i]], color="#c85b17", lw=0.9, alpha=0.92)
        ax.plot([g, g + MCF_RED_LENS_HEIGHT], [x_f[i], core_x], color="#c85b17",
                lw=0.65, ls="--", alpha=0.7)
    ax.scatter([0], [0], s=38, color="#0b0b0b", zorder=5, label="single NV")
    for sign in (-1, 1):
        ax.scatter([g], [sign * LENS_R], s=30, color="#d78500", zorder=5)
        ax.scatter([g + MCF_RED_LENS_HEIGHT], [sign * PITCH], s=34, color="#2a78d6", zorder=5)
    ax.axvline(0, color="#75808a", lw=0.8)
    ax.axvline(g, color="#75808a", lw=0.8)
    ax.set_xlim(-105, g + MCF_RED_LENS_HEIGHT + 12)
    ax.set_ylim(-52, 52)
    ax.set_xlabel("z from diamond surface ($\\mu$m)")
    ax.set_ylabel("x in meridional plane ($\\mu$m)")
    ax.set_title(f"(b) 700 nm rays at fixed-design optimum, g = {g:.0f} $\\mu$m", loc="left", fontsize=9.5, color=INK2)
    ax.text(0.98, 0.04, "orange = collected; dashed =\neffective pupil-to-core map",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color=INK2)
    fig.text(0.5, 0.005,
             "MCF values: Shukhin et al., OMN 2024; microlens.xlsx orange (650 nm) design. "
             "The curved printed lens is modeled as an effective Gaussian pupil, not an assumed sag profile.",
             ha="center", fontsize=7.2, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig4_mcf_ray_trace")

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
    print("\nNotes: fixed-aim MCF uses the paper/spreadsheet 650 nm red design: 16.8434 um"
          "\nlens-pupil radius, 141.185 um air path and a 30 um target depth; re-aimed MCF"
          "\nrecalculates the side-core boresights at every gap for the 85 um NV layer."
          "\nA50 is an on-axis circular area on the NV layer containing 50% of the"
          "\nexcitation x collection weighted ensemble signal. Single-NV rates are reported"
          "\nin kcps; ensemble operation is the practical mode for this deep NV layer.")
