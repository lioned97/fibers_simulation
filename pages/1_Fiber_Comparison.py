"""
SM vs MM vs MCF collection-efficiency comparison for a single NV in bulk diamond.

MCF geometry (core pitch, lens decenter, working distance) is taken from the
fabricated probe in Shukhin, Halfon, Bar-Gill & Marom, "Multi-Core Fiber Tip
Optical Excitation/Collection of NV-diamond Quantum Magnetic Resonance
Sensor," OMN 2024, plus the authors' lens-design spreadsheet (core pitch
35 um, lens decenter 18 um, IP-S n=1.51, design convergence 300 um from the
fiber tip / 50 um deep in a 100 um diamond). The printed lens surface figure
itself was not published, so each side core's output aperture is modelled as
a diffraction-limited Gaussian relay (reusing the existing Mode-overlap
physics) rather than ray-traced through an invented lens curvature.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from physics import (
    diamond_sellmeier, nv_emission_spectrum, optics_survival,
    GEOMETRIC_MODEL, MODE_OVERLAP_MODEL, fiber_mode_params,
    sample_ray_directions, run_ray_tracing, trapz,
)

st.set_page_config(page_title="SM vs MM vs MCF", page_icon="🔬", layout="wide")
st.title("SM vs MM vs MCF — NV Collection Efficiency")
st.write("Single NV center in bulk diamond. Compares a bare single-mode fiber, a bare "
         "multi-mode fiber, and the 6-side-core multi-core-fiber (MCF) microlens probe.")

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("NV & Medium")
nv_depth = st.sidebar.slider("NV depth [µm]", 1.0, 200.0, 80.0, 1.0)
n_med = st.sidebar.number_input("Gap medium index", 1.0, 2.0, 1.00, 0.01, format="%.3f",
    help="1.0 = air, matching the MCF's free-space design ('FreeAir', N3=1 in the lens design).")
num_rays = st.sidebar.slider("Rays (single NV, so this is cheap)", 5000, 150000, 40000, 5000)

st.sidebar.markdown("---")
st.sidebar.subheader("SM / MM bare fibers")
air_gap = st.sidebar.slider("Air gap, SM & MM [µm]", 0.0, 5.0, 0.0, 0.5,
    help="0 = fiber pressed against the diamond; >0 = a standoff gap.")
sm_core = st.sidebar.number_input("SM core diameter [µm]", 1.0, 20.0, 4.0, 0.5)
sm_na = st.sidebar.number_input("SM NA", 0.05, 0.5, 0.12, 0.01)
mm_core = st.sidebar.number_input("MM core diameter [µm]", 10.0, 200.0, 50.0, 5.0)
mm_na = st.sidebar.number_input("MM NA", 0.05, 0.8, 0.22, 0.01)
st.sidebar.caption("SM/MM are generic — edit to your fiber's datasheet. The coupling model "
                   "(geometric vs. Gaussian mode-overlap) is chosen automatically from the "
                   "V-number, same as the main simulator.")

with st.sidebar.expander("MCF probe design (Shukhin et al., OMN 2024)"):
    pitch = st.number_input("Core pitch [µm]", 10.0, 100.0, 35.0, 1.0)
    decenter = st.number_input("Lens decenter, inward [µm]", 0.0, 30.0, 18.0, 1.0)
    standoff = st.number_input("Lens-to-diamond standoff [µm]", 50.0, 400.0, 250.0, 5.0,
        help="Fixed by the printed scaffold in the real device, not user-adjustable in practice.")
    target_depth = st.number_input("Design convergence depth [µm]", 5.0, 100.0, 50.0, 5.0,
        help="Where the 6 side-core cones are aimed (mid-plane of the paper's 100 µm diamond).")
    n_ips = st.number_input("Lens polymer index (IP-S)", 1.30, 1.70, 1.51, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Detector")
nv_rate = st.sidebar.number_input("Single NV emission rate [kcps into 4π]", 1.0, 10000.0, 150.0, 10.0)
alpha_db_km = st.sidebar.number_input("Fiber attenuation [dB/km]", 0.0, 2000.0, 10.0, 5.0)
length_m = st.sidebar.number_input("Fiber length [m]", 0.0, 10000.0, 2.0, 1.0)
filter_t = st.sidebar.number_input("Filter transmission", 0.0, 1.0, 0.90, 0.05)
det_qe = st.sidebar.number_input("Detector QE", 0.0, 1.0, 0.70, 0.05)
sweep_max = st.sidebar.slider("Depth-sweep range [µm]", 20.0, 300.0, 150.0, 10.0)

# ------------------------------------------------------------------
# Spectral setup (NV- band) + fixed ray directions (single on-axis NV)
# ------------------------------------------------------------------
LAM_MIN, LAM_MAX, N_BINS = 640.0, 800.0, 7
lambdas = np.linspace(LAM_MIN, LAM_MAX, N_BINS)
S_lambda = nv_emission_spectrum(lambdas)
S_lambda = S_lambda / trapz(S_lambda, lambdas)
n_dia_lambda = diamond_sellmeier(lambdas / 1000.0)

LAM_REP = 700.0  # representative NV emission wavelength, used only for the fast depth sweep
n_dia_rep = float(diamond_sellmeier(LAM_REP / 1000.0))

V0, W0 = sample_ray_directions(num_rays, "NV Symmetry Axis", "Ensemble (4-axis average)")

# ------------------------------------------------------------------
# Per-fiber-type efficiency at one wavelength (reuses the same ray tracer
# and Fresnel/TIR physics as the main simulator for all three fiber types)
# ------------------------------------------------------------------
def on_axis_eta(depth, d_core, na, gap, n_dia, lam_nm):
    """Bare fiber, on-axis. Coupling model follows the V-number automatically."""
    emitters = np.array([[0.0, 0.0, -depth]])
    V, _, _, _ = fiber_mode_params(d_core, na, lam_nm, n_med)
    model = MODE_OVERLAP_MODEL if V < 2.405 else GEOMETRIC_MODEL
    fibers = [{'x': 0.0, 'y': 0.0, 'd_core': d_core, 'na': na}]
    res = run_ray_tracing(emitters, V0, W0, n_dia, n_med, gap, fibers, model, lam_nm)
    return res['fiber_stats'][0]['avg_efficiency']

def mcf_eta(depth, n_dia, lam_nm):
    """
    Six side cores at radius `pitch`, each core's microlens decentered inward
    by `decenter` so the collection cone is steered to the shared on-axis
    point (0, 0, -target_depth). The lens exit aperture (not the bare core)
    is the effective collecting surface, sized to the pitch per the paper
    ("...total height is chosen to fill the apertures dictated by the core
    pitch"): waist w0 = pitch/2, with na_mode set by the diffraction limit
    (run_ray_tracing derives na_mode from w0 + the current wavelength).
    Returns the SUM over all 6 channels (independent, incoherent detectors).
    """
    emitters = np.array([[0.0, 0.0, -depth]])
    lens_r = pitch - decenter
    w0_lens = pitch / 2.0
    target = np.array([0.0, 0.0, -target_depth])
    fibers = []
    for i in range(6):
        ang = i * np.pi / 3.0
        fx, fy = lens_r * np.cos(ang), lens_r * np.sin(ang)
        boresight = np.array([fx, fy, standoff]) - target
        boresight = boresight / np.linalg.norm(boresight)
        fibers.append({'x': fx, 'y': fy, 'd_core': pitch, 'na': 0.0,
                       'boresight': boresight, 'w0': w0_lens})
    res = run_ray_tracing(emitters, V0, W0, n_dia, n_med, standoff, fibers,
                          coupling_model=MODE_OVERLAP_MODEL, lambda_nm=lam_nm)
    return sum(s['avg_efficiency'] for s in res['fiber_stats'])

SERIES = [
    ("SM (0 µm gap)", lambda d, nd, lam: on_axis_eta(d, sm_core, sm_na, 0.0, nd, lam), 1.46),
    ("SM (air gap)",  lambda d, nd, lam: on_axis_eta(d, sm_core, sm_na, air_gap, nd, lam), 1.46),
    ("MM (0 µm gap)", lambda d, nd, lam: on_axis_eta(d, mm_core, mm_na, 0.0, nd, lam), 1.46),
    ("MM (air gap)",  lambda d, nd, lam: on_axis_eta(d, mm_core, mm_na, air_gap, nd, lam), 1.46),
    ("MCF (6 cores)", lambda d, nd, lam: mcf_eta(d, nd, lam), n_ips),
]

def spectral_average(eta_fn, depth):
    vals = np.array([eta_fn(depth, n_dia_lambda[i], lambdas[i]) for i in range(N_BINS)])
    return float(trapz(S_lambda * vals, lambdas))

# ------------------------------------------------------------------
# Headline numbers at the current depth (full spectral averaging = accurate)
# ------------------------------------------------------------------
st.subheader(f"Collection efficiency at NV depth = {nv_depth:.0f} µm")

names, geo_pct, det_kcps = [], [], []
for name, eta_fn, n_core in SERIES:
    eta_geo = spectral_average(eta_fn, nv_depth)
    eta_optics, _ = optics_survival(True, n_core, n_med, alpha_db_km, length_m, filter_t, det_qe)
    names.append(name)
    geo_pct.append(eta_geo * 100.0)
    det_kcps.append(eta_geo * eta_optics * nv_rate)

fig_bar = go.Figure(go.Bar(x=names, y=geo_pct, marker_color='#00f2fe',
                           text=[f"{v:.3f}%" for v in geo_pct], textposition='outside'))
fig_bar.update_layout(template="plotly_dark", height=420,
                      yaxis=dict(title="Collection efficiency (%)"), margin=dict(t=30))
st.plotly_chart(fig_bar, width='stretch')

st.dataframe({
    "Fiber": names,
    "Collection η (%)": [f"{v:.4f}" for v in geo_pct],
    "Detected (kcps)": [f"{v:.3f}" for v in det_kcps],
}, width='stretch', hide_index=True)

best = int(np.argmax(det_kcps))
runner_up = sorted(det_kcps)[-2]
st.success(f"Best at {nv_depth:.0f} µm: **{names[best]}** — {det_kcps[best]:.3f} kcps detected "
          f"({geo_pct[best]:.4f}% η), vs {runner_up:.3f} kcps for the next best.")

# ------------------------------------------------------------------
# Depth sweep (single representative wavelength — fast trend view)
# ------------------------------------------------------------------
st.subheader("Collection efficiency vs. NV depth")
depths = np.linspace(2.0, sweep_max, 40)
colors = ['#00f2fe', '#4facfe', '#ff7f50', '#ffb347', '#00ffcc']

fig_line = go.Figure()
for (name, eta_fn, _), color in zip(SERIES, colors):
    curve = np.array([eta_fn(d, n_dia_rep, LAM_REP) for d in depths]) * 100.0
    fig_line.add_trace(go.Scatter(x=depths, y=curve, mode='lines', name=name,
                                  line=dict(width=2.5, color=color)))
fig_line.add_vline(x=nv_depth, line=dict(color='white', width=1, dash='dot'))
fig_line.update_layout(template="plotly_dark", height=480,
                       xaxis_title="NV depth [µm]", yaxis_title="Collection efficiency (%)",
                       margin=dict(t=30))
st.plotly_chart(fig_line, width='stretch')

st.caption(
    "Headline numbers spectrally average over the 640-800 nm NV⁻ band (7 bins) with diamond "
    "dispersion n(λ) from the Sellmeier fit; the depth-sweep curve above uses a single "
    "representative λ = 700 nm for speed. All three fiber types share the same ray tracer, "
    "Fresnel/TIR physics, and post-fiber optical-survival chain (attenuation, filter, QE) — "
    "only the entrance-face index differs (silica 1.46 for SM/MM vs. IP-S 1.51 for the MCF "
    "lenses). The MCF's central excitation core is shown in the design numbers only; this page "
    "compares fluorescence collection, not excitation."
)
