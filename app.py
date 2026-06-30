import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# numpy >= 2.0 renamed np.trapz -> np.trapezoid; stay compatible with both
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ==========================================
# Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="NV-Fiber Ray Tracing Simulator",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark theme styling
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Mono&display=swap');
    
    html, body, [data-testid="stSidebar"] {
        font-family: 'Outfit', sans-serif;
    }
    
    code, pre {
        font-family: 'Space Mono', monospace;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #a1a1aa;
        margin-bottom: 2rem;
    }
    
    /* Metrics panel */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: #00f2fe;
    }
    
    .metric-card {
        background-color: #181a27;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #2d3142;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin-bottom: 1rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8a8d9f;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .metric-unit {
        font-size: 0.9rem;
        color: #00f2fe;
        margin-left: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Crystal & Dipole Constants & Orientations
# ==========================================
# Standard diamond NV axes in (100) cut where normal is Z [001]
NV_AXES = {
    "[111]": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    "[1-1-1]": np.array([1.0, -1.0, -1.0]) / np.sqrt(3.0),
    "[-11-1]": np.array([-1.0, 1.0, -1.0]) / np.sqrt(3.0),
    "[-1-11]": np.array([-1.0, -1.0, 1.0]) / np.sqrt(3.0)
}

# ==========================================
# Spectral Model: Diamond Dispersion + NV Emission
# ==========================================

def diamond_sellmeier(lambda_um):
    """
    Refractive index of diamond vs wavelength via the Sellmeier equation
    (lambda in micrometres):
        n^2 = 1 + 0.3306 L/(L - 0.175^2) + 4.3356 L/(L - 0.106^2),  L = lambda^2
    Reproduces n = 2.4173 @ 589 nm and n = 2.4118 @ 637 nm (NV ZPL),
    valid across the visible / NV emission band.
    """
    l2 = np.asarray(lambda_um, dtype=float) ** 2
    n2 = 1.0 + 0.3306 * l2 / (l2 - 0.175 ** 2) + 4.3356 * l2 / (l2 - 0.106 ** 2)
    return np.sqrt(n2)

def nv_emission_spectrum(lambda_nm):
    """
    Approximate NV- room-temperature emission spectrum S(lambda) (un-normalised).
    A sharp zero-phonon line (ZPL) at 637 nm carrying roughly the Debye-Waller
    fraction (~4%) plus a broad phonon sideband peaking near 690 nm and trailing
    to ~800 nm. The caller normalises so that the integral of S over lambda = 1.
    """
    lam = np.asarray(lambda_nm, dtype=float)
    sideband = np.exp(-0.5 * ((lam - 690.0) / 50.0) ** 2)
    zpl = 0.52 * np.exp(-0.5 * ((lam - 637.0) / 4.0) ** 2)
    return sideband + zpl

def optics_survival(include, n_core, n_med, alpha_db_km, length_m, filter_t, det_qe):
    """
    Post-fibre optical survival factor:
        eta_optics = T_face * T_prop * T_filter * QE_detector
    where T_face is the normal-incidence Fresnel transmission at the fibre
    entrance face (gap medium -> core), T_prop is propagation survival from the
    fibre attenuation, T_filter the spectral-filter transmission and QE the
    detector quantum efficiency. Returns (eta, breakdown_dict).
    Returns 1.0 (and an empty breakdown) when disabled.
    """
    if not include:
        return 1.0, {}
    t_face = 1.0 - ((n_core - n_med) / (n_core + n_med)) ** 2
    t_prop = 10.0 ** (-(alpha_db_km * (length_m / 1000.0)) / 10.0)
    eta = t_face * t_prop * filter_t * det_qe
    return eta, {'face': t_face, 'prop': t_prop, 'filter': filter_t, 'qe': det_qe}

GEOMETRIC_MODEL = "Geometric (multimode)"
MODE_OVERLAP_MODEL = "Mode overlap (Gaussian)"

def fiber_mode_params(d_core, na, lambda_nm, n_med):
    """
    Step-index fiber modal parameters at a given wavelength:
      V        normalized frequency  V = 2*pi*a*NA / lambda
      n_modes  approximate guided mode count (large-V limit)  M ~ V^2 / 2
      w0       fundamental-mode (LP01) field radius via Marcuse's fit
                 w0/a = 0.65 + 1.619 V^-1.5 + 2.879 V^-6   (best for 1.2 < V < 2.4)
      na_mode  effective mode NA in the gap medium (Gaussian divergence)
                 sin(theta_mode) ~ lambda / (pi * n_med * w0)
    Lengths in micrometres.
    """
    a = d_core / 2.0
    lam = lambda_nm / 1000.0
    V = 2.0 * np.pi * a * na / lam
    n_modes = max(V * V / 2.0, 1.0)
    Vc = max(V, 1.2)                       # clamp: Marcuse fit diverges as V->0
    w0 = a * (0.65 + 1.619 * Vc ** -1.5 + 2.879 * Vc ** -6)
    na_mode = lam / (np.pi * n_med * w0)
    return V, n_modes, w0, na_mode

def compute_coupling(dist_sq, sin2_sq, tir_mask, r_core, na, n_med,
                     coupling_model, w0=0.0, na_mode=0.0):
    """
    Per-ray coupling weight in [0, 1] at the fiber facet.

    Geometric (multimode): hard acceptance, weight = 1 if the ray lands inside
    the core AND within the NA cone (and is not TIR), else 0. Correct in the
    many-mode limit where the étendue cells fill the geometric acceptance.

    Mode overlap (Gaussian): phase-space overlap with the fundamental LP01 mode,
    modelled as a Gaussian Wigner distribution that is separable in transverse
    position and angle:
        c = exp(-2 rho^2 / w0^2) * exp(-2 sin^2(theta2) / na_mode^2) * (not TIR)
    i.e. mode matching against the single guided mode of waist w0 / NA na_mode.
    """
    not_tir = ~tir_mask
    if coupling_model == MODE_OVERLAP_MODEL:
        spatial = np.exp(-2.0 * dist_sq / (w0 * w0))
        angular = np.exp(-2.0 * sin2_sq / (na_mode * na_mode))
        return spatial * angular * not_tir
    in_core = dist_sq <= r_core ** 2
    in_na = sin2_sq <= (na / n_med) ** 2
    return (in_core & in_na & not_tir).astype(float)

# ==========================================
# Physical Simulation Functions
# ==========================================

def get_collection_limit_radius(depth, z_fiber, na, n_dia, n_med, r_core):
    """
    Calculate the maximum physical radius from the fiber centers where NVs can be collected.
    Uses fiber NA and refraction equations to bound the simulation volume.
    """
    # Max angle inside diamond that can be accepted by the fiber NA
    # n_dia * sin(theta_dia) = n_med * sin(theta_med) <= NA
    sin_theta_dia_max = na / n_dia
    if sin_theta_dia_max >= 1.0:
        # No NA limit inside diamond (unlikely as n_dia is 2.417 and NA is ~0.2-0.5)
        theta_dia_max = np.arcsin(n_med / n_dia)  # Limited by TIR
    else:
        theta_dia_max = np.arcsin(sin_theta_dia_max)
        
    tan_theta_dia_max = np.tan(theta_dia_max)
    
    # Max angle in the air/coupling medium
    sin_theta_med_max = na / n_med
    if sin_theta_med_max >= 1.0:
        theta_med_max = np.pi / 2.0
    else:
        theta_med_max = np.arcsin(sin_theta_med_max)
        
    tan_theta_med_max = np.tan(theta_med_max)
    
    # Max lateral displacements
    l_dia = depth * tan_theta_dia_max
    l_med = z_fiber * tan_theta_med_max
    
    return l_dia + r_core + l_med

def generate_emitters(mode, depth, width, ppm, num_emitters, fibers, na, n_dia, n_med, seed=42):
    """
    Generate NV coordinates (in um) and density scaling factors.
    Returns:
        emitters: np.array of shape (num_emitters, 3)
        n_actual: estimated actual number of NVs in the active volume
        volume: volume or area of the active region
        box_dims: bounding box coordinates [xmin, xmax, ymin, ymax]
    """
    np.random.seed(seed)
    
    # Extract fiber coordinate range
    fiber_x = np.array([f['x'] for f in fibers])
    fiber_y = np.array([f['y'] for f in fibers])
    r_core = max([f['d_core'] for f in fibers]) / 2.0
    
    # Max collection radius from any fiber center
    r_collect = get_collection_limit_radius(depth + width/2.0, max(0.0, fibers[0]['z']), na, n_dia, n_med, r_core)
    
    # Bounding box of simulation region
    x_min = np.min(fiber_x) - r_collect
    x_max = np.max(fiber_x) + r_collect
    y_min = np.min(fiber_y) - r_collect
    y_max = np.max(fiber_y) + r_collect
    
    area_um2 = (x_max - x_min) * (y_max - y_min)
    
    # Volumetric density for 1 ppm of NVs: 1.76e5 NVs / um^3
    rho_vol = ppm * 1.76e5
    
    if mode == "Single NV":
        emitters = np.array([[0.0, 0.0, -depth]])
        n_actual = 1
        volume = 0.0
        box_dims = (0, 0, 0, 0)
    elif mode == "2D Layer (Plane)":
        # Generate NVs at a fixed depth Z = -depth
        x = np.random.uniform(x_min, x_max, num_emitters)
        y = np.random.uniform(y_min, y_max, num_emitters)
        z = np.full(num_emitters, -depth)
        emitters = np.column_stack((x, y, z))
        
        # For a 2D plane, sheet density is calculated assuming a nominal thickness (e.g. 1 nm monolayer = 0.001 um)
        thickness_2d = 0.001 # 1 nm
        rho_sheet = rho_vol * thickness_2d # NVs / um^2
        n_actual = rho_sheet * area_um2
        volume = area_um2
        box_dims = (x_min, x_max, y_min, y_max)
    else: # 3D Layer
        # Generate NVs distributed in Z over [-depth - width/2, -depth + width/2]
        x = np.random.uniform(x_min, x_max, num_emitters)
        y = np.random.uniform(y_min, y_max, num_emitters)
        z_min = -depth - width / 2.0
        z_max = -depth + width / 2.0
        z = np.random.uniform(z_min, z_max, num_emitters)
        emitters = np.column_stack((x, y, z))
        
        volume = area_um2 * width # um^3
        n_actual = rho_vol * volume
        box_dims = (x_min, x_max, y_min, y_max)
        
    return emitters, n_actual, volume, box_dims

def sample_ray_directions(num_rays, emitter_type, orientation, custom_dir=None, seed=42):
    """
    Generate unit direction vectors over the upper hemisphere and their respective emission weights.
    Returns:
        V0: np.array of shape (num_rays, 3)
        W0: np.array of shape (num_rays,) representing relative weights
    """
    np.random.seed(seed)
    
    # Uniform sampling of upper hemisphere
    phi = np.random.uniform(0.0, 2.0 * np.pi, num_rays)
    cos_theta = np.random.uniform(0.0, 1.0, num_rays) # cos(theta) from 0 to 1
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    
    v0x = sin_theta * np.cos(phi)
    v0y = sin_theta * np.sin(phi)
    v0z = cos_theta
    V0 = np.column_stack((v0x, v0y, v0z))
    
    # Compute relative emission weights
    if emitter_type == "Isotropic":
        W0 = np.ones(num_rays)
    elif emitter_type == "Single Dipole":
        if orientation == "Perpendicular to surface [001]":
            d = np.array([0.0, 0.0, 1.0])
        elif orientation == "Parallel to surface [100]":
            d = np.array([1.0, 0.0, 0.0])
        elif orientation == "Parallel to surface [010]":
            d = np.array([0.0, 1.0, 0.0])
        else: # Custom
            d = np.array(custom_dir) if custom_dir is not None else np.array([1.0, 0.0, 0.0])
            d_norm = np.linalg.norm(d)
            d = d / d_norm if d_norm > 0 else np.array([1.0, 0.0, 0.0])
            
        # Single dipole intensity: I(v) = 1.5 * (1 - (d.v)^2)
        dot_product = np.dot(V0, d)
        W0 = 1.5 * (1.0 - dot_product**2)
        
    else: # NV Symmetry Axis (2 orthogonal dipoles)
        if orientation == "Ensemble (4-axis average)":
            # Average over all 4 standard orientations
            W_sum = np.zeros(num_rays)
            for key, u in NV_AXES.items():
                # NV axis intensity: I(v) = 0.75 * (1 + (u.v)^2)
                dot_product = np.dot(V0, u)
                W_sum += 0.75 * (1.0 + dot_product**2)
            W0 = W_sum / 4.0
        else: # Specific NV axis
            u = NV_AXES[orientation]
            dot_product = np.dot(V0, u)
            W0 = 0.75 * (1.0 + dot_product**2)
            
    return V0, W0

def run_ray_tracing(emitters, V0, W0, n_dia, n_med, z_fiber, fibers,
                    coupling_model=GEOMETRIC_MODEL, lambda_nm=637.0):
    """
    Run vectorized ray tracing for all emitters and rays.
    coupling_model selects geometric (hard core+NA) or Gaussian mode-overlap
    acceptance; lambda_nm sets the fiber mode size in the mode-overlap case.
    Returns:
        results dict containing coordinates and collection stats.
    """
    n_nv = len(emitters)
    n_r = len(V0)
    
    # 1. Emitter positions (P0) broadcasted: shape (n_nv, 1, 3)
    P0 = emitters[:, np.newaxis, :]
    
    # 2. Ray directions (V0) broadcasted: shape (1, n_r, 3)
    V0_b = V0[np.newaxis, :, :]
    W0_b = W0[np.newaxis, :]
    
    # 3. Intersection with Z = 0 (diamond-medium interface)
    # Z_int = 0 => P0_z + t * V0_z = 0 => t = -P0_z / V0_z
    t = - P0[:, :, 2] / V0_b[:, :, 2] # shape (n_nv, n_r)
    
    X_int = P0[:, :, 0] + t * V0_b[:, :, 0] # shape (n_nv, n_r)
    Y_int = P0[:, :, 1] + t * V0_b[:, :, 1] # shape (n_nv, n_r)
    
    # 4. Refraction at interface
    cos1 = V0_b[:, :, 2] # cos(theta1), shape (1, n_r) -> broadcasted to (n_nv, n_r)
    sin1_sq = 1.0 - cos1**2
    
    # Critical angle condition: n_dia * sin(theta1) = n_med * sin(theta2)
    sin2_sq = (n_dia / n_med)**2 * sin1_sq
    tir_mask = sin2_sq > 1.0 # Total Internal Reflection
    
    cos2 = np.sqrt(np.maximum(0.0, 1.0 - sin2_sq))
    
    # Refracted direction vector V1 in the coupling medium
    # V1 = (n_dia/n_med)*V0_xy + cos(theta2)*Z_unit
    V1 = np.zeros((n_nv, n_r, 3))
    V1[:, :, 0] = (n_dia / n_med) * V0_b[:, :, 0]
    V1[:, :, 1] = (n_dia / n_med) * V0_b[:, :, 1]
    V1[:, :, 2] = cos2
    
    # 5. Fresnel Transmission coefficient (average of s & p polarization)
    # Avoid division by zero at normal incidence (though NumPy handles it gracefully)
    n1_cos1 = n_dia * cos1
    n2_cos2 = n_med * cos2
    n1_cos2 = n_dia * cos2
    n2_cos1 = n_med * cos1
    
    r_s = (n1_cos1 - n2_cos2) / (n1_cos1 + n2_cos2 + 1e-15)
    r_p = (n1_cos2 - n2_cos1) / (n1_cos2 + n2_cos1 + 1e-15)
    
    R_s = r_s**2
    R_p = r_p**2
    T = 0.5 * (2.0 - R_s - R_p)
    T[tir_mask] = 0.0 # No transmission for TIR
    
    # Transmitted Ray weights
    W = W0_b * T # shape (n_nv, n_r)
    
    # 6. Propagation to fiber facet plane Z = Z_fiber
    # Z_fiber is the air gap
    denom = np.where(V1[:, :, 2] > 0, V1[:, :, 2], 1.0)
    X_f = X_int + z_fiber * V1[:, :, 0] / denom
    Y_f = Y_int + z_fiber * V1[:, :, 1] / denom
    
    # 7. Evaluate collection by each fiber
    collection_stats = []
    
    for i, f in enumerate(fibers):
        r_core = f['d_core'] / 2.0
        na = f['na']

        # Transverse miss-distance at the facet
        dist_sq = (X_f - f['x'])**2 + (Y_f - f['y'])**2

        # Fundamental-mode size (only needed for the mode-overlap model)
        if coupling_model == MODE_OVERLAP_MODEL:
            _, _, w0, na_mode = fiber_mode_params(f['d_core'], na, lambda_nm, n_med)
        else:
            w0, na_mode = 0.0, 0.0

        # Per-ray coupling weight (binary geometric, or Gaussian mode overlap)
        coupling = compute_coupling(dist_sq, sin2_sq, tir_mask, r_core, na, n_med,
                                    coupling_model, w0, na_mode)

        # Boolean mask for visualization (geometric: accepted; mode: within 1/e^2)
        if coupling_model == MODE_OVERLAP_MODEL:
            collected = coupling > np.exp(-2.0)
        else:
            collected = coupling > 0.5

        # Efficiency of this fiber for each emitter
        # Divided by 2 because we only sampled the upper hemisphere (fraction 0.5 of 4pi)
        effs = 0.5 * np.sum(W * coupling, axis=1) / n_r

        collection_stats.append({
            'fiber_idx': i,
            'collected_mask': collected,
            'efficiencies': effs,
            'avg_efficiency': np.mean(effs)
        })
        
    return {
        'X_int': X_int, 'Y_int': Y_int,
        'X_f': X_f, 'Y_f': Y_f,
        'V1': V1,
        'tir_mask': tir_mask,
        'weights': W,
        'fiber_stats': collection_stats
    }

# ==========================================
# UI Layout & User Inputs
# ==========================================

st.write('<div class="main-title">NV-Fiber Optical Simulator</div>', unsafe_allow_html=True)
st.write('<div class="subtitle">Interactive 3D/2D optical ray tracing for NV center light collection into fiber bundles</div>', unsafe_allow_html=True)

# ----------------- Sidebar -----------------
st.sidebar.header("🔬 Simulation Parameters")

# 1. Numerical & General Config
num_rays = st.sidebar.slider("Number of Rays per Emitter", min_value=500, max_value=50000, value=10000, step=500,
                            help="More rays increase accuracy but slow down the 3D plots.")
num_vis_rays = st.sidebar.slider("Rays to Visualize (in 3D/2D)", min_value=10, max_value=500, value=100, step=10,
                                help="Number of representative rays plotted in the 3D visualizers.")

# 2. Material Refractive Indices
st.sidebar.markdown("---")
st.sidebar.subheader("💎 Materials & Coupling")
n_dia = st.sidebar.number_input("Refractive Index (Diamond)", min_value=1.0, max_value=4.0, value=2.417, step=0.01, format="%.3f",
                               help="Used only in monochromatic mode. With spectral averaging ON, n is computed from the Sellmeier dispersion n(λ).")
n_med = st.sidebar.number_input("Refractive Index (Gap Medium)", min_value=1.0, max_value=3.0, value=1.000, step=0.01, format="%.3f",
                               help="1.0 for Air. Use 1.4-1.5 for oil/gel interface coupling.")

# 2b. Spectral averaging configuration
st.sidebar.markdown("---")
st.sidebar.subheader("🌈 Spectral Model")
spectral_mode = st.sidebar.checkbox("Spectral averaging over NV emission", value=True,
    help="Average η over the NV- emission spectrum S(λ) with diamond dispersion n(λ): η = ∫S(λ)η(λ)dλ. If off, the run is monochromatic using the fixed diamond index above.")
if spectral_mode:
    cwl1, cwl2 = st.sidebar.columns(2)
    lam_min = cwl1.number_input("λ min [nm]", min_value=600.0, max_value=750.0, value=640.0, step=5.0)
    lam_max = cwl2.number_input("λ max [nm]", min_value=650.0, max_value=850.0, value=800.0, step=5.0)
    n_lambda = st.sidebar.slider("Spectral samples", min_value=3, max_value=25, value=9, step=2,
        help="Number of wavelength bins. Each bin re-runs the ray tracer, so higher = more accurate but slower.")
    if lam_max <= lam_min:
        st.sidebar.error("λ max must exceed λ min — falling back to 640–800 nm.")
        lam_min, lam_max = 640.0, 800.0
else:
    lam_min, lam_max, n_lambda = 637.0, 637.0, 1

# 3. NV Emitter Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("✨ NV Centers Emitters")
nv_mode = st.sidebar.selectbox("NV Source Type", ["Single NV", "2D Layer (Plane)", "3D Layer (Slab)"])

nv_depth = st.sidebar.number_input("NV Depth (d) [µm]", min_value=0.01, max_value=500.0, value=5.0, step=1.0, format="%.2f",
                                  help="Mean depth of NV centers below the surface (Z = 0).")

if nv_mode == "3D Layer (Slab)":
    nv_width = st.sidebar.number_input("NV Layer Width (in Z) [µm]", min_value=0.01, max_value=100.0, value=2.0, step=0.5, format="%.2f",
                                      help="Thickness of the NV-doped layer.")
else:
    nv_width = 0.0 # 2D or Single

if nv_mode in ["2D Layer (Plane)", "3D Layer (Slab)"]:
    nv_ppm = st.sidebar.number_input("NV Concentration [ppm]", min_value=0.0001, max_value=1000.0, value=0.1, step=0.05, format="%.4f",
                                    help="Parts per million. 1 ppm = 1.76e17 NVs / cm^3.")
    num_emitters = st.sidebar.number_input("Simulated Emitters", min_value=5, max_value=2000, value=100, step=20,
                                          help="Number of representative emitters generated in the simulation.")
else:
    nv_ppm = 0.0
    num_emitters = 1

# 4. Emitter Polarization / Dipole Type
st.sidebar.markdown("---")
st.sidebar.subheader("🧲 Dipole / Symmetry Axis")
emitter_type = st.sidebar.selectbox("Emission Pattern", ["Isotropic", "NV Symmetry Axis", "Single Dipole"])

if emitter_type == "NV Symmetry Axis":
    orientation = st.sidebar.selectbox("Symmetry Axis Orientation", 
                                       ["Ensemble (4-axis average)", "[111]", "[1-1-1]", "[-11-1]", "[-1-11]"],
                                       help="NV center axis in (100) diamond. Ensemble averages over all 4 directions.")
    custom_dir = None
elif emitter_type == "Single Dipole":
    orientation = st.sidebar.selectbox("Dipole Direction", 
                                       ["Parallel to surface [100]", "Parallel to surface [010]", "Perpendicular to surface [001]", "Custom Vector"])
    if orientation == "Custom Vector":
        col1, col2, col3 = st.sidebar.columns(3)
        dx = col1.number_input("dx", value=1.0, step=0.1)
        dy = col2.number_input("dy", value=0.0, step=0.1)
        dz = col3.number_input("dz", value=0.0, step=0.1)
        custom_dir = [dx, dy, dz]
    else:
        custom_dir = None
else:
    orientation = "Isotropic"
    custom_dir = None

# 5. Fibers Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("🔌 Fibers Configuration")

coupling_model = st.sidebar.radio(
    "Coupling Model",
    [GEOMETRIC_MODEL, MODE_OVERLAP_MODEL],
    help="Geometric: hard core + NA acceptance — valid for highly multimode fibers "
         "(many guided modes). Mode overlap: Gaussian phase-space overlap with the "
         "fundamental LP01 mode (waist w0, mode NA) — the single-/few-mode coupling "
         "or 'mode matching' efficiency. The V-number and mode count are reported above the plots."
)

num_fibers = st.sidebar.number_input("Number of Fibers", min_value=1, max_value=50, value=1, step=1)

fiber_layout = st.sidebar.selectbox("Fiber Arrangement", ["Single Fiber", "Linear Array (X-axis)", "Custom Coordinates"])

if fiber_layout == "Single Fiber":
    d_core = st.sidebar.number_input("Core Diameter [µm]", min_value=1.0, max_value=1000.0, value=50.0, step=5.0)
    d_clad = st.sidebar.number_input("Cladding Diameter [µm]", min_value=1.0, max_value=2000.0, value=125.0, step=5.0)
    fiber_na = st.sidebar.number_input("Numerical Aperture (NA)", min_value=0.01, max_value=1.0, value=0.22, step=0.01)
    
    st.sidebar.markdown("**Offsets**")
    z_fiber = st.sidebar.number_input("Z Offset (Air Gap) [µm]", min_value=0.0, max_value=5000.0, value=10.0, step=1.0,
                                     help="Distance between the diamond surface Z=0 and fiber facets. Can be 0.")
    x_offset = st.sidebar.number_input("X Offset [µm]", min_value=-500.0, max_value=5000.0, value=0.0, step=1.0)
    y_offset = st.sidebar.number_input("Y Offset [µm]", min_value=-500.0, max_value=5000.0, value=0.0, step=1.0)
    
    fibers = [{'x': x_offset, 'y': y_offset, 'z': z_fiber, 'd_core': d_core, 'd_clad': d_clad, 'na': fiber_na}]
    
elif fiber_layout == "Linear Array (X-axis)":
    d_core = st.sidebar.number_input("Core Diameter (for each) [µm]", min_value=1.0, max_value=1000.0, value=50.0, step=5.0)
    d_clad = st.sidebar.number_input("Cladding Diameter (for each) [µm]", min_value=1.0, max_value=2000.0, value=125.0, step=5.0)
    fiber_na = st.sidebar.number_input("Numerical Aperture (NA)", min_value=0.01, max_value=1.0, value=0.22, step=0.01)
    
    fiber_pitch = st.sidebar.number_input("Fiber Pitch (spacing center-to-center) [µm]", min_value=1.0, max_value=2000.0, value=125.0, step=5.0,
                                         help="Spacing between adjacent fiber centers. Defaults to cladding diameter.")
    
    st.sidebar.markdown("**Offsets**")
    z_fiber = st.sidebar.number_input("Z Offset (Air Gap) [µm]", min_value=0.0, max_value=5000.0, value=10.0, step=1.0)
    x_offset = st.sidebar.number_input("Array Center X Offset [µm]", min_value=-1000.0, max_value=1000.0, value=0.0, step=1.0)
    y_offset = st.sidebar.number_input("Array Center Y Offset [µm]", min_value=-1000.0, max_value=1000.0, value=0.0, step=1.0)
    
    fibers = []
    for i in range(num_fibers):
        # Spaced symmetrically along X axis
        x_c = x_offset + (i - (num_fibers - 1) / 2.0) * fiber_pitch
        fibers.append({
            'x': x_c, 'y': y_offset, 'z': z_fiber,
            'd_core': d_core, 'd_clad': d_clad, 'na': fiber_na
        })

else: # Custom coordinates
    d_core = st.sidebar.number_input("Core Diameter (for all) [µm]", min_value=1.0, max_value=1000.0, value=50.0, step=5.0)
    d_clad = st.sidebar.number_input("Cladding Diameter (for all) [µm]", min_value=1.0, max_value=2000.0, value=125.0, step=5.0)
    fiber_na = st.sidebar.number_input("Numerical Aperture (NA)", min_value=0.01, max_value=1.0, value=0.22, step=0.01)
    z_fiber = st.sidebar.number_input("Z Offset (Air Gap) [µm]", min_value=0.0, max_value=5000.0, value=10.0, step=1.0)
    
    x_coords_str = st.sidebar.text_input("X coordinates (comma separated) [µm]", "0, 125, -125")
    y_coords_str = st.sidebar.text_input("Y coordinates (comma separated) [µm]", "0, 0, 0")
    
    try:
        xs = [float(x.strip()) for x in x_coords_str.split(",")]
        ys = [float(y.strip()) for y in y_coords_str.split(",")]
        
        # Ensure sizes match
        n_c = min(len(xs), len(ys), num_fibers)
        fibers = []
        for i in range(n_c):
            fibers.append({
                'x': xs[i], 'y': ys[i], 'z': z_fiber,
                'd_core': d_core, 'd_clad': d_clad, 'na': fiber_na
            })
    except Exception as e:
        st.sidebar.error("Error parsing custom coordinates. Using default (0,0).")
        fibers = [{'x': 0.0, 'y': 0.0, 'z': z_fiber, 'd_core': d_core, 'd_clad': d_clad, 'na': fiber_na}]

# Est. total power parameter
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Detector Config")
nv_photon_rate = st.sidebar.number_input("Single NV Saturation Emission Rate [kps]", min_value=1.0, max_value=10000.0, value=150.0, step=50.0,
                                       help="Photon emission rate of a single NV center into 4pi (in kilo-photons per second). Used to estimate absolute collected counts.")

# Optical survival (post-fiber): fiber-face Fresnel, attenuation, filter, detector QE
st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Optical Survival (η_optics)")
include_optics = st.sidebar.checkbox("Apply η_optics to detected counts", value=True,
    help="Multiply the geometric collection efficiency by the post-fiber optical path: entrance-face Fresnel, propagation loss, filter, and detector QE.")
if include_optics:
    n_core = st.sidebar.number_input("Fiber Core Index n_core", min_value=1.0, max_value=2.0, value=1.460, step=0.01, format="%.3f",
        help="Fused silica ≈ 1.46. Sets the entrance-face Fresnel loss against the gap medium.")
    alpha_db_km = st.sidebar.number_input("Fiber Attenuation [dB/km]", min_value=0.0, max_value=2000.0, value=10.0, step=5.0,
        help="Multimode silica ≈ 8-15 dB/km in the 650-800 nm band.")
    length_m = st.sidebar.number_input("Fiber Length [m]", min_value=0.0, max_value=10000.0, value=2.0, step=1.0)
    filter_t = st.sidebar.number_input("Filter Transmission", min_value=0.0, max_value=1.0, value=0.90, step=0.05,
        help="Long-pass / band-pass transmission over the NV emission band.")
    det_qe = st.sidebar.number_input("Detector Quantum Efficiency", min_value=0.0, max_value=1.0, value=0.70, step=0.05,
        help="e.g. Si single-photon APD ≈ 0.65-0.75 at 700 nm.")
else:
    n_core, alpha_db_km, length_m, filter_t, det_qe = 1.46, 0.0, 0.0, 1.0, 1.0


# ==========================================
# Execution of Main Simulation
# ==========================================

# Extract representative core and cladding radii for visualizers
r_core = fibers[0]['d_core'] / 2.0 if len(fibers) > 0 else 25.0
r_clad = fibers[0]['d_clad'] / 2.0 if len(fibers) > 0 else 62.5

# 1. Build the wavelength sampling and per-λ diamond index
if spectral_mode:
    lambdas = np.linspace(lam_min, lam_max, int(n_lambda))
    S_lambda = nv_emission_spectrum(lambdas)
    S_lambda = S_lambda / trapz(S_lambda, lambdas)         # ∫ S(λ) dλ = 1
    n_dia_lambda = diamond_sellmeier(lambdas / 1000.0)     # λ nm -> µm
else:
    lambdas = np.array([637.0])
    S_lambda = np.array([1.0])
    n_dia_lambda = np.array([n_dia])

# Representative wavelength = emission centroid; drives all 3D/2D visualizers and sweeps
if len(lambdas) > 1:
    lam_centroid = float(trapz(S_lambda * lambdas, lambdas))   # ∫λS(λ)dλ (∫S=1)
    rep_idx = int(np.argmin(np.abs(lambdas - lam_centroid)))
else:
    rep_idx = 0
n_dia_rep = float(n_dia_lambda[rep_idx])
lam_rep = float(lambdas[rep_idx])

# 2. Generate NV center emitters (bounding box uses representative index)
emitters, n_actual, act_volume, box_dims = generate_emitters(
    nv_mode, nv_depth, nv_width, nv_ppm, num_emitters, fibers, fiber_na, n_dia_rep, n_med
)

# 3. Sample global unit ray directions (geometry is wavelength-independent)
V0, W0 = sample_ray_directions(num_rays, emitter_type, orientation, custom_dir)

# 4. Solve ray tracing at each wavelength; keep the full result for the representative λ
eff_lambda = np.zeros(len(lambdas))
tir_lambda = np.zeros(len(lambdas))
per_fiber_eff_lambda = np.zeros((max(len(fibers), 1), len(lambdas)))
sim_results = None
for li, nd in enumerate(n_dia_lambda):
    res = run_ray_tracing(emitters, V0, W0, nd, n_med, z_fiber, fibers,
                          coupling_model=coupling_model, lambda_nm=lambdas[li])
    fiber_effs = np.array([s['avg_efficiency'] for s in res['fiber_stats']])
    per_fiber_eff_lambda[:len(fiber_effs), li] = fiber_effs
    eff_lambda[li] = fiber_effs.sum()
    tir_lambda[li] = float(np.mean(res['tir_mask']))
    if li == rep_idx:
        sim_results = res

# 5. Spectral averages:  η = ∫ S(λ) η(λ) dλ   (single sample falls back to the value)
if len(lambdas) > 1:
    total_eff = float(trapz(S_lambda * eff_lambda, lambdas))
    tir_loss = float(trapz(S_lambda * tir_lambda, lambdas))
else:
    total_eff = float(eff_lambda[0])
    tir_loss = float(tir_lambda[0])

# 6. Post-fiber optical survival and detected efficiency
eta_optics, optics_breakdown = optics_survival(
    include_optics, n_core, n_med, alpha_db_km, length_m, filter_t, det_qe
)
total_eff_detected = total_eff * eta_optics

# Modal parameters of the representative fiber at the representative wavelength
V_disp, M_disp, w0_disp, na_mode_disp = fiber_mode_params(
    fibers[0]['d_core'], fibers[0]['na'], lam_rep, n_med
)
single_mode = V_disp < 2.405
mode_regime = "single-mode" if single_mode else f"≈{M_disp:.0f} modes"
is_mode_overlap = (coupling_model == MODE_OVERLAP_MODEL)

# ----------------- Dashboard Metrics -----------------
max_eff = float(np.max(per_fiber_eff_lambda)) if len(fibers) > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)

eff_label = "Mode-Matched Coupling" if is_mode_overlap else "Geometric Collection Efficiency"

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{eff_label}</div>
        <div class="metric-val">{total_eff*100:.3f}<span class="metric-unit">%</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if nv_mode == "Single NV":
        est_counts_val = total_eff_detected * nv_photon_rate
        counts_lbl = "Detected Photon Rate" if include_optics else "Collected Photon Rate"
    else:
        est_counts_val = n_actual * total_eff_detected * nv_photon_rate
        counts_lbl = "Est. Total Detected Count Rate" if include_optics else "Est. Total Active Count Rate"
        
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{counts_lbl}</div>
        <div class="metric-val">{est_counts_val:.1f}<span class="metric-unit">kcps</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Diamond Surface TIR Loss</div>
        <div class="metric-val">{tir_loss*100:.1f}<span class="metric-unit">%</span></div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active Emitters (In Box)</div>
        <div class="metric-val">{n_actual:.0f}<span class="metric-unit">NVs</span></div>
    </div>
    """, unsafe_allow_html=True)

# Model-state caption: spectral mode, representative index, modal regime, optical budget
cap_parts = [f"Representative λ = {lambdas[rep_idx]:.0f} nm · n_dia(λ) = {n_dia_rep:.4f}"]
if spectral_mode and len(lambdas) > 1:
    cap_parts.append(f"spectral avg over {len(lambdas)} bins ({lam_min:.0f}–{lam_max:.0f} nm)")
else:
    cap_parts.append("monochromatic")
mode_part = f"fiber V = {V_disp:.2f} ({mode_regime})"
if is_mode_overlap:
    mode_part += f", mode w₀ = {w0_disp:.2f} µm, mode NA = {na_mode_disp:.3f}"
cap_parts.append(mode_part)
if include_optics:
    cap_parts.append(
        f"η_optics = {eta_optics*100:.1f}% "
        f"(face {optics_breakdown['face']*100:.1f}% · prop {optics_breakdown['prop']*100:.1f}% · "
        f"filter {filter_t*100:.0f}% · QE {det_qe*100:.0f}%) → detected η = {total_eff_detected*100:.3f}%"
    )
st.caption("  ·  ".join(cap_parts))

# Unpack common simulation results for visualizers
X_int = sim_results['X_int']
Y_int = sim_results['Y_int']
X_f = sim_results['X_f']
Y_f = sim_results['Y_f']
tir_mask = np.broadcast_to(sim_results['tir_mask'], X_f.shape)

# ----------------- Tabs Layout -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 3D Interactive Ray Tracing",
    "🎯 Spot Diagram (Fiber Plane)",
    "📐 2D Ray Trajectories (Cross-section)",
    "📈 Alignment Tolerance Sweeps",
    "🌈 Spectral Response"
])

# -----------------------------------------------
# TAB 1: 3D INTERACTIVE PLOT
# -----------------------------------------------
with tab1:
    st.subheader("3D Interactive Visualizer")
    st.write("Rotate, zoom, and pan to inspect individual ray paths from the NV centers to the fibers.")
    
    # 3D Plotly creation
    fig3d = go.Figure()
    
    # Draw interface plane (Z = 0)
    grid_lim = max(r_core * 3.0, 50.0) if len(fibers) == 1 else max(np.max(np.abs([f['x'] for f in fibers])) * 1.5, 100.0)
    
    # Draw interface plane
    fig3d.add_trace(go.Surface(
        x=np.array([[-grid_lim, grid_lim], [-grid_lim, grid_lim]]),
        y=np.array([[-grid_lim, -grid_lim], [grid_lim, grid_lim]]),
        z=np.array([[0.0, 0.0], [0.0, 0.0]]),
        colorscale=[[0, 'rgba(128, 128, 128, 0.15)'], [1, 'rgba(128, 128, 128, 0.15)']],
        showscale=False,
        name="Diamond Interface Z=0",
        hoverinfo='skip'
    ))
    
    # Draw Emitters
    fig3d.add_trace(go.Scatter3d(
        x=emitters[:, 0],
        y=emitters[:, 1],
        z=emitters[:, 2],
        mode='markers',
        marker=dict(size=4, color='#00ffcc', symbol='circle', opacity=0.9),
        name="NV Centers"
    ))
    
    # Draw Fibers (Facet + short cylinder)
    for i, f in enumerate(fibers):
        r_c = f['d_core'] / 2.0
        r_cl = f['d_clad'] / 2.0
        
        # Facet Cladding circle
        theta = np.linspace(0, 2 * np.pi, 60)
        cx = f['x'] + r_cl * np.cos(theta)
        cy = f['y'] + r_cl * np.sin(theta)
        cz = np.full_like(cx, z_fiber)
        fig3d.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz,
            mode='lines',
            line=dict(color='#888888', width=2),
            name=f"Fiber {i+1} Cladding ({f['d_clad']}µm)",
            legendgroup=f"fib{i}"
        ))
        
        # Facet Core filled surface
        r_grid, theta_grid = np.meshgrid(np.linspace(0, r_c, 8), theta)
        x_mesh = f['x'] + r_grid * np.cos(theta_grid)
        y_mesh = f['y'] + r_grid * np.sin(theta_grid)
        z_mesh = np.full_like(x_mesh, z_fiber)
        
        fig3d.add_trace(go.Surface(
            x=x_mesh, y=y_mesh, z=z_mesh,
            colorscale=[[0, 'rgba(0, 242, 254, 0.3)'], [1, 'rgba(0, 242, 254, 0.3)']],
            showscale=False,
            name=f"Fiber {i+1} Core ({f['d_core']}µm)",
            legendgroup=f"fib{i}"
        ))
        
        # 3D core Cylinder
        cylinder_len = 15.0 # length of drawn fiber cylinder
        z_cyl = np.linspace(z_fiber, z_fiber + cylinder_len, 2)
        theta_grid, z_grid = np.meshgrid(theta, z_cyl)
        x_cyl = f['x'] + r_cl * np.cos(theta_grid)
        y_cyl = f['y'] + r_cl * np.sin(theta_grid)
        
        fig3d.add_trace(go.Surface(
            x=x_cyl, y=y_cyl, z=z_grid,
            colorscale=[[0, 'rgba(100, 100, 100, 0.1)'], [1, 'rgba(100, 100, 100, 0.1)']],
            showscale=False,
            hoverinfo='skip',
            legendgroup=f"fib{i}",
            showlegend=False
        ))
        
    # Draw subset of rays
    # Sub-sample emitters to draw
    n_draw_emitters = min(10, len(emitters))
    draw_emitter_indices = np.random.choice(len(emitters), n_draw_emitters, replace=False)
    
    # Sub-sample direction vectors for visualization
    n_draw_rays = min(num_vis_rays, num_rays)
    draw_ray_indices = np.random.choice(num_rays, n_draw_rays, replace=False)
    
    # Combine collected masks for any fiber
    any_collected = np.zeros_like(X_f, dtype=bool)
    for f_stat in sim_results['fiber_stats']:
        any_collected |= f_stat['collected_mask']
        
    for em_idx in draw_emitter_indices:
        em_pos = emitters[em_idx]
        
        for r_idx in draw_ray_indices:
            x_i = X_int[em_idx, r_idx]
            y_i = Y_int[em_idx, r_idx]
            
            # Inside diamond segment (Source -> Interface)
            fig3d.add_trace(go.Scatter3d(
                x=[em_pos[0], x_i],
                y=[em_pos[1], y_i],
                z=[em_pos[2], 0.0],
                mode='lines',
                line=dict(color='rgba(255, 127, 80, 0.25)', width=1.5),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Outside diamond segment
            if tir_mask[em_idx, r_idx]:
                # Draw small reflection back downwards
                v_in = V0[r_idx]
                t_refl = -em_pos[2] / 2.0 # short length
                fig3d.add_trace(go.Scatter3d(
                    x=[x_i, x_i + t_refl * v_in[0]],
                    y=[y_i, y_i + t_refl * v_in[1]],
                    z=[0.0, -t_refl * v_in[2]],
                    mode='lines',
                    line=dict(color='rgba(128, 128, 128, 0.15)', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            else:
                # Transmitted ray (Interface -> Fiber plane)
                x_f = X_f[em_idx, r_idx]
                y_f = Y_f[em_idx, r_idx]
                
                is_coll = any_collected[em_idx, r_idx]
                ray_color = 'rgba(0, 255, 100, 0.65)' if is_coll else 'rgba(255, 50, 50, 0.25)'
                ray_width = 2.0 if is_coll else 1.2
                
                fig3d.add_trace(go.Scatter3d(
                    x=[x_i, x_f],
                    y=[y_i, y_f],
                    z=[0.0, z_fiber],
                    mode='lines',
                    line=dict(color=ray_color, width=ray_width),
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # Layout configuration
    fig3d.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis=dict(title='X (µm)', gridcolor='#222222', showbackground=False),
            yaxis=dict(title='Y (µm)', gridcolor='#222222', showbackground=False),
            zaxis=dict(title='Z (µm)', gridcolor='#222222', showbackground=False),
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.2, y=1.2, z=1.0)
            )
        ),
        height=700
    )
    
    st.plotly_chart(fig3d, use_container_width=True)

# -----------------------------------------------
# TAB 2: SPOT DIAGRAM (FIBER FACET PLANE)
# -----------------------------------------------
with tab2:
    st.subheader("Facet Intersection Spot Diagram")
    st.write("Spatial distribution of rays intersecting the fiber facet plane ($Z = Z_{fiber}$).")
    
    # We sample a subset of rays for plotting to avoid lagging the frontend
    max_scatter_pts = 10000
    total_pts = len(emitters) * num_rays
    
    if total_pts > max_scatter_pts:
        # Downsample rays
        step = int(np.ceil(total_pts / max_scatter_pts))
        X_plot = sim_results['X_f'].flatten()[::step]
        Y_plot = sim_results['Y_f'].flatten()[::step]
        tir_plot = sim_results['tir_mask'].flatten()[::step]
    else:
        X_plot = sim_results['X_f'].flatten()
        Y_plot = sim_results['Y_f'].flatten()
        tir_plot = sim_results['tir_mask'].flatten()
        
    # Exclude TIR rays (they never make it to the facet plane)
    non_tir_indices = np.where(~tir_plot)[0]
    X_non_tir = X_plot[non_tir_indices]
    Y_non_tir = Y_plot[non_tir_indices]
    
    fig_spot = go.Figure()
    
    # Plot ray intersections
    fig_spot.add_trace(go.Scattergl(
        x=X_non_tir,
        y=Y_non_tir,
        mode='markers',
        marker=dict(
            color='#ff4500',
            size=2.0,
            opacity=0.3
        ),
        name="Exiting Rays"
    ))
    
    # Draw fiber cores and cladding boundary circles in 2D
    for i, f in enumerate(fibers):
        r_core = f['d_core'] / 2.0
        r_clad = f['d_clad'] / 2.0
        
        # Add core circle shape
        fig_spot.add_shape(type="circle",
            xref="x", yref="y",
            x0=f['x'] - r_core, y0=f['y'] - r_core,
            x1=f['x'] + r_core, y1=f['y'] + r_core,
            line=dict(color="#00f2fe", width=2.5, dash="solid"),
            fillcolor="rgba(0, 242, 254, 0.08)",
            name=f"Fiber {i+1} Core"
        )
        
        # Add cladding circle shape
        fig_spot.add_shape(type="circle",
            xref="x", yref="y",
            x0=f['x'] - r_clad, y0=f['y'] - r_clad,
            x1=f['x'] + r_clad, y1=f['y'] + r_clad,
            line=dict(color="#888888", width=1.5, dash="dash"),
            name=f"Fiber {i+1} Clad"
        )
        
    # Layout configuration
    lim = max(r_core * 4.0, 60.0) if len(fibers) == 1 else max(np.max(np.abs([f['x'] for f in fibers])) * 1.5, 120.0)
    fig_spot.update_layout(
        template="plotly_dark",
        xaxis=dict(title="X Position (µm)", range=[-lim, lim], gridcolor='#222222'),
        yaxis=dict(title="Y Position (µm)", range=[-lim, lim], scaleanchor="x", scaleratio=1, gridcolor='#222222'),
        height=600,
        margin=dict(l=40, r=40, b=40, t=40)
    )
    
    st.plotly_chart(fig_spot, use_container_width=True)

# -----------------------------------------------
# TAB 3: 2D RAY TRAJECTORIES (XZ PLANE)
# -----------------------------------------------
with tab3:
    st.subheader("2D Trajectory Profile (XZ Cross-Section)")
    st.write("2D side-view of ray refraction and collection paths across the diamond-medium-fiber interfaces.")
    
    # 2D cross section using Matplotlib (cleaner rendering for overlapping curves)
    fig_2d, ax = plt.subplots(figsize=(12, 6.5), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    
    # Draw Diamond surface boundary
    ax.axhline(0, color='w', linestyle='-', linewidth=1.5, label='Diamond Surface (Z=0)')
    ax.fill_between([-grid_lim, grid_lim], -grid_lim, 0, color='#1c1d24', alpha=0.5, label='Diamond')
    
    # Draw Emitters
    ax.scatter(emitters[:, 0], emitters[:, 2], color='#00ffcc', s=15, zorder=5, label='NV Centers')
    
    # Draw Fibers (Cores at Z = Z_fiber)
    for i, f in enumerate(fibers):
        r_c = f['d_core'] / 2.0
        r_cl = f['d_clad'] / 2.0
        
        # Core line
        ax.plot([f['x'] - r_c, f['x'] + r_c], [z_fiber, z_fiber], color='#00f2fe', linewidth=4, zorder=4, label=f'Fiber {i+1} Core' if i==0 else "")
        # Cladding line
        ax.plot([f['x'] - r_cl, f['x'] - r_c], [z_fiber, z_fiber], color='#888888', linewidth=2, zorder=3, label=f'Cladding' if i==0 else "")
        ax.plot([f['x'] + r_c, f['x'] + r_cl], [z_fiber, z_fiber], color='#888888', linewidth=2, zorder=3)
        
        # Render a shaded rectangle for fiber body
        rect_clad = plt.Rectangle((f['x'] - r_cl, z_fiber), f['d_clad'], 15.0, facecolor='grey', alpha=0.1, edgecolor='none')
        rect_core = plt.Rectangle((f['x'] - r_c, z_fiber), f['d_core'], 15.0, facecolor='#00f2fe', alpha=0.08, edgecolor='none')
        ax.add_patch(rect_clad)
        ax.add_patch(rect_core)

    # Plot rays
    n_draw_em = min(5, len(emitters))
    draw_em_ids = np.random.choice(len(emitters), n_draw_em, replace=False)
    
    n_draw_r = min(num_vis_rays, num_rays)
    draw_r_ids = np.random.choice(num_rays, n_draw_r, replace=False)
    
    any_collected = np.zeros_like(X_f, dtype=bool)
    for f_stat in sim_results['fiber_stats']:
        any_collected |= f_stat['collected_mask']
        
    for em_idx in draw_em_ids:
        em_pos = emitters[em_idx]
        
        for r_idx in draw_r_ids:
            x_i = X_int[em_idx, r_idx]
            
            # Inside diamond segment (Source -> Interface)
            ax.plot([em_pos[0], x_i], [em_pos[2], 0.0], color='#ff7f50', alpha=0.15, linewidth=0.8)
            
            # Outside diamond segment
            if tir_mask[em_idx, r_idx]:
                # TIR reflected ray segment (stops at diamond surface for clean visual, or short bounce)
                v_in = V0[r_idx]
                ax.plot([x_i, x_i + 2.0 * v_in[0]], [0.0, -2.0 * v_in[2]], color='#555555', alpha=0.08, linewidth=0.6)
            else:
                x_f = X_f[em_idx, r_idx]
                is_coll = any_collected[em_idx, r_idx]
                color = '#00ff64' if is_coll else '#ff3232'
                alpha = 0.35 if is_coll else 0.08
                width = 1.2 if is_coll else 0.7
                ax.plot([x_i, x_f], [0.0, z_fiber], color=color, alpha=alpha, linewidth=width)
                
    # Labels & Limits
    ax.set_xlabel('X Offset (µm)', color='white')
    ax.set_ylabel('Z Depth / Height (µm)', color='white')
    ax.tick_params(colors='white')
    ax.grid(color='#222222', linestyle='--', alpha=0.5)
    
    x_view_lim = max(r_core * 3.5, 60.0) if len(fibers) == 1 else max(np.max(np.abs([f['x'] for f in fibers])) * 1.6, 120.0)
    ax.set_xlim(-x_view_lim, x_view_lim)
    ax.set_ylim(-nv_depth * 1.5 - nv_width, z_fiber + 10.0)
    
    # Clean legends
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', facecolor='#0e1117', edgecolor='#2d3142', labelcolor='white')
    
    st.pyplot(fig_2d)

# -----------------------------------------------
# TAB 4: ALIGNMENT TOLERANCE SWEEPS
# -----------------------------------------------
with tab4:
    st.subheader("Alignment Tolerance Curves")
    st.write("Analyze how misalignment in X or Z direction impacts the light collection efficiency.")
    
    sweep_type = st.radio("Alignment Axis to Sweep", ["X Axis (Lateral Offset)", "Z Axis (Air Gap)"])
    
    if sweep_type == "X Axis (Lateral Offset)":
        st.write(r"Sweeping the lateral displacement ($\Delta X$) of the fiber/array center relative to the emitter.")
        
        sweep_range = st.slider("Sweep Range (X) [µm]", min_value=1.0, max_value=200.0, value=80.0, step=5.0)
        num_sweep_pts = st.slider("Number of Sweep Points (X)", min_value=10, max_value=150, value=60, step=5)
        
        x_offsets_sweep = np.linspace(-sweep_range, sweep_range, num_sweep_pts)
        
        # Optimize sweep: propagate rays once (already done in sim_results)
        # We only need to check which rays fall within the fibers for each offset.
        X_f = sim_results['X_f']
        Y_f = sim_results['Y_f']
        sin2_sq = (n_dia_rep / n_med)**2 * (1.0 - V0[np.newaxis, :, 2]**2)
        tir_mask = sim_results['tir_mask']
        W = sim_results['weights']
        
        # Storage for curves
        sweep_effs = np.zeros((num_fibers, num_sweep_pts))
        
        for p_idx, dx in enumerate(x_offsets_sweep):
            for f_idx, f in enumerate(fibers):
                r_core = f['d_core'] / 2.0
                na = f['na']
                
                # Fiber center is shifted by dx
                shifted_x = f['x'] + dx

                # Check intersections
                dist_sq = (X_f - shifted_x)**2 + (Y_f - f['y'])**2
                if coupling_model == MODE_OVERLAP_MODEL:
                    _, _, w0_s, na_mode_s = fiber_mode_params(f['d_core'], na, lam_rep, n_med)
                else:
                    w0_s, na_mode_s = 0.0, 0.0
                coupling = compute_coupling(dist_sq, sin2_sq, tir_mask, r_core, na, n_med,
                                            coupling_model, w0_s, na_mode_s)

                # Compute average efficiency for this fiber
                effs = 0.5 * np.sum(W * coupling, axis=1) / num_rays
                sweep_effs[f_idx, p_idx] = np.mean(effs)
                
        # Combined trace
        combined_eff = np.sum(sweep_effs, axis=0)
        
        # Plot curves using Plotly
        fig_sweep = go.Figure()
        
        for f_idx in range(num_fibers):
            fig_sweep.add_trace(go.Scatter(
                x=x_offsets_sweep,
                y=sweep_effs[f_idx] * 100,
                mode='lines',
                name=f"Fiber {f_idx+1}",
                line=dict(width=2)
            ))
            
        if num_fibers > 1:
            fig_sweep.add_trace(go.Scatter(
                x=x_offsets_sweep,
                y=combined_eff * 100,
                mode='lines+markers',
                name="Combined Bundle",
                line=dict(color='#00ffcc', width=3, dash='dash')
            ))
            
        fig_sweep.update_layout(
            template="plotly_dark",
            title="Collection Efficiency vs. Lateral Offset",
            xaxis=dict(title="Lateral Misalignment ΔX (µm)", gridcolor='#222222'),
            yaxis=dict(title="Collection Efficiency (%)", gridcolor='#222222'),
            height=500
        )
        st.plotly_chart(fig_sweep, use_container_width=True)
        
    else: # Z Axis Sweep
        st.write("Sweeping the air gap distance ($Z_{fiber}$) between the diamond surface and the fiber plane.")
        
        z_max_sweep = st.slider("Max Z Sweep Distance [µm]", min_value=1.0, max_value=250.0, value=60.0, step=5.0)
        num_sweep_pts = st.slider("Number of Sweep Points (Z)", min_value=10, max_value=100, value=40, step=5)
        
        z_offsets_sweep = np.linspace(0.0, z_max_sweep, num_sweep_pts) # Supports 0 air gap
        
        # For Z sweep, we must re-calculate the facet intersections for each Z step
        X_int = sim_results['X_int']
        Y_int = sim_results['Y_int']
        V1 = sim_results['V1']
        sin2_sq = (n_dia_rep / n_med)**2 * (1.0 - V0[np.newaxis, :, 2]**2)
        tir_mask = sim_results['tir_mask']
        W = sim_results['weights']
        
        # Precompute denominator to avoid divide-by-zeros
        denom = np.where(V1[:, :, 2] > 0, V1[:, :, 2], 1.0)
        
        # Storage
        sweep_effs = np.zeros((num_fibers, num_sweep_pts))
        
        for p_idx, z_val in enumerate(z_offsets_sweep):
            # Calculate coordinates at new Z plane
            X_f_new = X_int + z_val * V1[:, :, 0] / denom
            Y_f_new = Y_int + z_val * V1[:, :, 1] / denom
            
            for f_idx, f in enumerate(fibers):
                r_core = f['d_core'] / 2.0
                na = f['na']
                
                # Check intersections
                dist_sq = (X_f_new - f['x'])**2 + (Y_f_new - f['y'])**2
                if coupling_model == MODE_OVERLAP_MODEL:
                    _, _, w0_s, na_mode_s = fiber_mode_params(f['d_core'], na, lam_rep, n_med)
                else:
                    w0_s, na_mode_s = 0.0, 0.0
                coupling = compute_coupling(dist_sq, sin2_sq, tir_mask, r_core, na, n_med,
                                            coupling_model, w0_s, na_mode_s)

                # Compute average efficiency
                effs = 0.5 * np.sum(W * coupling, axis=1) / num_rays
                sweep_effs[f_idx, p_idx] = np.mean(effs)
                
        # Combined trace
        combined_eff = np.sum(sweep_effs, axis=0)
        
        # Plot
        fig_sweep = go.Figure()
        
        for f_idx in range(num_fibers):
            fig_sweep.add_trace(go.Scatter(
                x=z_offsets_sweep,
                y=sweep_effs[f_idx] * 100,
                mode='lines',
                name=f"Fiber {f_idx+1}",
                line=dict(width=2)
            ))
            
        if num_fibers > 1:
            fig_sweep.add_trace(go.Scatter(
                x=z_offsets_sweep,
                y=combined_eff * 100,
                mode='lines+markers',
                name="Combined Bundle",
                line=dict(color='#00ffcc', width=3, dash='dash')
            ))
            
        fig_sweep.update_layout(
            template="plotly_dark",
            title="Collection Efficiency vs. Air Gap (Z)",
            xaxis=dict(title="Air Gap Distance Z (µm)", gridcolor='#222222'),
            yaxis=dict(title="Collection Efficiency (%)", gridcolor='#222222'),
            height=500
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

# -----------------------------------------------
# TAB 5: SPECTRAL RESPONSE
# -----------------------------------------------
with tab5:
    st.subheader("Spectral Response")
    if not spectral_mode or len(lambdas) < 2:
        st.info("Enable **Spectral averaging** in the sidebar to see wavelength-resolved "
                "curves. The simulation is currently monochromatic "
                f"(n_dia = {n_dia_rep:.4f}).")
    else:
        st.write(r"""
        Wavelength-resolved collection efficiency $\eta(\lambda)$, the normalized NV$^-$
        emission spectrum $S(\lambda)$ (with $\int S(\lambda)\,d\lambda = 1$), and the
        diamond dispersion $n_{dia}(\lambda)$. The reported efficiency is the spectral
        average $\eta = \int S(\lambda)\,\eta(\lambda)\,d\lambda$.
        """)

        # η(λ) overlaid on the emission spectrum (dual y-axis)
        fig_sp = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sp.add_trace(go.Scatter(
            x=lambdas, y=S_lambda / np.max(S_lambda),
            name="S(λ) (normalized)", fill='tozeroy',
            line=dict(color='#ff7f50', width=2)
        ), secondary_y=False)
        fig_sp.add_trace(go.Scatter(
            x=lambdas, y=eff_lambda * 100,
            name="η(λ) [%]", mode='lines+markers',
            line=dict(color='#00f2fe', width=3)
        ), secondary_y=True)
        fig_sp.add_vline(x=lambdas[rep_idx], line=dict(color='#00ffcc', width=1, dash='dash'),
                         annotation_text="rep. λ", annotation_position="top")
        fig_sp.update_layout(template="plotly_dark", height=460,
                             margin=dict(l=40, r=40, b=40, t=40),
                             xaxis_title="Wavelength λ (nm)")
        fig_sp.update_yaxes(title_text="S(λ) (normalized)", secondary_y=False, gridcolor='#222222')
        fig_sp.update_yaxes(title_text="Collection Efficiency η(λ) [%]", secondary_y=True)
        st.plotly_chart(fig_sp, use_container_width=True)

        # Diamond dispersion n(λ)
        fig_n = go.Figure()
        fig_n.add_trace(go.Scatter(
            x=lambdas, y=n_dia_lambda, mode='lines+markers',
            name="n_dia(λ)", line=dict(color='#00ffcc', width=2)
        ))
        fig_n.update_layout(template="plotly_dark", height=320,
                            margin=dict(l=40, r=40, b=40, t=30),
                            xaxis=dict(title="Wavelength λ (nm)", gridcolor='#222222'),
                            yaxis=dict(title="Diamond Refractive Index n(λ)", gridcolor='#222222'))
        st.plotly_chart(fig_n, use_container_width=True)

        st.caption(
            f"Spectrally-averaged geometric η = {total_eff*100:.3f}%   ·   "
            f"band edges: {eff_lambda[0]*100:.3f}% @ {lambdas[0]:.0f} nm → "
            f"{eff_lambda[-1]*100:.3f}% @ {lambdas[-1]:.0f} nm   ·   "
            f"n_dia: {n_dia_lambda[0]:.4f} → {n_dia_lambda[-1]:.4f}"
        )

# ----------------- Physics Information Card -----------------
st.markdown("---")
with st.expander("📝 Physical Principles of the Simulation"):
    st.write(r"""
    This simulator models the ray-optical path of light emitted by single or layered NV centers (Nitrogen-Vacancy color centers) inside bulk diamond, escaping through a flat surface into a coupling medium (e.g. air or oil), and then coupling into a fiber bundle.
    
    ### 1. Emission Angular Distribution
    * **Isotropic**: $I(\mathbf{v}) = \text{const}$. Rays are sampled uniformly in the upper hemisphere.
    * **Single Dipole**: Light emitted from a single linear dipole vector $\mathbf{d}$ has angular power density $I(\mathbf{v}) \propto 1 - (\mathbf{d} \cdot \mathbf{v})^2 = \sin^2(\theta_{\text{dipole}})$.
    * **NV Symmetry Axis**: An NV center has two orthogonal dipole transitions in the plane perpendicular to its symmetry axis $\mathbf{u}_{\text{NV}}$. The combined emission profile is $I(\mathbf{v}) \propto 1 + (\mathbf{u}_{\text{NV}} \cdot \mathbf{v})^2 = 1 + \cos^2(\theta_{\text{NV}})$.
    
    ### 2. Refraction & Fresnel Reflection
    When a ray hits the boundary between diamond ($n_1 = 2.417$) and the gap medium ($n_2$, e.g. air $1.0$), it refracts according to Snell's Law:
    $$n_1 \sin\theta_1 = n_2 \sin\theta_2$$
    Since $n_1 > n_2$, rays incident at angles exceeding the critical angle $\theta_c = \arcsin(n_2 / n_1) \approx 24.4^\circ$ experience **Total Internal Reflection (TIR)** and are lost.
    
    For escaping rays, we calculate the power transmission $T$ using the Fresnel equations for unpolarized light:
    $$R_s = \left( \frac{n_1 \cos\theta_1 - n_2 \cos\theta_2}{n_1 \cos\theta_1 + n_2 \cos\theta_2} \right)^2, \quad R_p = \left( \frac{n_2 \cos\theta_1 - n_1 \cos\theta_2}{n_2 \cos\theta_1 + n_1 \cos\theta_2} \right)^2$$
    $$T = 1 - \frac{R_s + R_p}{2}$$
    At normal incidence, this gives $T \approx 82.8\%$ for diamond-to-air (17.2% reflection loss).
    
    ### 3. Fiber Core & Acceptance Cone (Numerical Aperture)
    Rays propagating in the medium will hit the fiber facet at height $Z = Z_{\text{fiber}}$. A ray is accepted if:
    1. **Spatial Overlap**: Its intersection coordinate $(X_f, Y_f)$ lies inside the fiber core of radius $R_{\text{core}} = D_{\text{core}} / 2$.
    2. **Angular Overlap**: Its angle with the fiber axis (Z) in the gap medium is less than the fiber acceptance angle, which is defined by the numerical aperture $NA$:
       $$\sin\theta_2 \le \frac{NA}{n_{\text{med}}}$$
    
    ### 4. Layer Density Scaling
    The actual number of NVs in the active simulation volume is computed from the volumetric density corresponding to the ppm concentration:
    $$\rho = C_{\text{ppm}} \times 1.76 \times 10^5 \text{ NVs/µm}^3$$
    The sheet density of a 2D layer is evaluated by modeling the layer as a nominal $1\,\text{nm}$ thick slab, representing a delta-doped monolayer.

    ### 5. Spectral Averaging
    The NV$^-$ emission is broadband (ZPL at 637 nm + phonon sideband to ~800 nm), and diamond is dispersive, so the reported efficiency is the spectrum-weighted average of the monochromatic efficiency:
    $$\eta = \int S(\lambda)\,\eta(\lambda)\,d\lambda, \qquad \int S(\lambda)\,d\lambda = 1$$
    The diamond index follows the Sellmeier dispersion (λ in µm):
    $$n^2(\lambda) = 1 + \frac{0.3306\,\lambda^2}{\lambda^2 - 0.175^2} + \frac{4.3356\,\lambda^2}{\lambda^2 - 0.106^2}$$
    which gives $n = 2.4173$ at 589 nm and $n = 2.4118$ at the 637 nm ZPL. Each wavelength bin re-runs the ray tracer with its own $n_{dia}(\lambda)$; the integral is evaluated by the trapezoid rule. All 3D/2D visualizers and the alignment sweeps use the *representative* wavelength (the spectrum peak).

    ### 6. Post-Fiber Optical Survival
    Detected counts also depend on the optical path after the fiber face. The geometric efficiency is multiplied by
    $$\eta_{\text{optics}} = T_{\text{face}}\cdot T_{\text{prop}}\cdot T_{\text{filter}}\cdot \mathrm{QE}$$
    where the entrance-face Fresnel transmission is $T_{\text{face}} = 1 - \left(\frac{n_{\text{core}} - n_{\text{med}}}{n_{\text{core}} + n_{\text{med}}}\right)^2$ (≈96.5% for silica/air), the propagation survival is $T_{\text{prop}} = 10^{-\alpha L / 10}$ for attenuation $\alpha$ [dB/km] over length $L$ [km], and $T_{\text{filter}}$, $\mathrm{QE}$ are the filter transmission and detector quantum efficiency. The detected efficiency is $\eta_{\text{det}} = \eta\cdot\eta_{\text{optics}}$.

    ### 7. Coupling Model & Mode Matching
    The fiber acceptance can be evaluated two ways:

    **Geometric (multimode)** — a ray couples (weight 1) if it lands inside the core *and* within the NA cone. This is exact in the highly-multimode limit, where the guided modes tile the geometric étendue. The number of guided modes follows from the normalized frequency (V-number):
    $$V = \frac{2\pi a}{\lambda}\,\mathrm{NA}, \qquad M \approx \frac{V^2}{2}\;\;(V \gg 1)$$
    with core radius $a = D_{\text{core}}/2$. A fiber is single-mode when $V < 2.405$.

    **Mode overlap (Gaussian)** — for single-/few-mode coupling, the binary aperture is replaced by the phase-space overlap with the fundamental $\text{LP}_{01}$ mode, approximated as a Gaussian of field radius $w_0$ (Marcuse fit) and effective mode NA:
    $$\frac{w_0}{a} = 0.65 + \frac{1.619}{V^{3/2}} + \frac{2.879}{V^{6}}, \qquad \mathrm{NA}_{\text{mode}} = \frac{\lambda}{\pi\,n_{\text{med}}\,w_0}$$
    The per-ray coupling weight is the (separable) Gaussian Wigner overlap in transverse position $\rho$ and angle $\theta_2$:
    $$c(\rho,\theta_2) = \exp\!\left(-\frac{2\rho^2}{w_0^2}\right)\exp\!\left(-\frac{2\sin^2\theta_2}{\mathrm{NA}_{\text{mode}}^2}\right)$$
    so $\eta = \tfrac12\langle w\,T\,c\rangle$ is the **mode-matching efficiency** into the single guided mode. Coupling NV emission into the $\text{LP}_{01}$ mode of a large multimode core is intrinsically small (the mode fills only a tiny phase-space cell) — this regime is where the geometric model is the relevant one.
    """)
