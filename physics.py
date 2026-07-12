"""
Pure physics/math for the NV-fiber ray tracer: no Streamlit calls, so it can
be imported by every page (app.py and pages/*.py) without re-running UI code.
Extracted verbatim from app.py — behavior is unchanged, only the location moved.
"""
import numpy as np

# numpy >= 2.0 renamed np.trapz -> np.trapezoid; stay compatible with both
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

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

def angle_to_boresight(V1, boresight):
    """
    sin^2 of each ray's angle relative to an arbitrary unit boresight vector,
    instead of the global Z axis — for a collection channel whose acceptance
    cone is tilted (e.g. by a decentered microlens) rather than pointing
    straight along the fiber. V1 has shape (..., 3); boresight is length-3.
    """
    cos_b = V1[..., 0] * boresight[0] + V1[..., 1] * boresight[1] + V1[..., 2] * boresight[2]
    # TIR rays carry a non-unit V1 (norm = n*sin(theta1) > 1), so cos_b can
    # exceed 1 and 1-cos^2 would go negative -> exp overflow -> NaN downstream.
    # sin^2 is >= 0 by definition; TIR rays are zeroed by the not_tir mask anyway.
    return np.maximum(1.0 - cos_b ** 2, 0.0)

def lens_dome_mesh(center, boresight, radius, height, n_phi=16, n_rho=8):
    """
    Small paraboloid dome (base radius `radius`, sag `height`) representing a
    printed microlens, tilted so its own +Z axis points along `boresight`
    instead of straight up — a visual stand-in for the real lens surface
    (whose curvature wasn't published), sized/positioned to match the
    core's actual acceptance geometry so the tilt itself reads correctly.
    Returns (X, Y, Z) grids ready for a 3D surface plot.
    """
    rho, phi = np.linspace(0.0, radius, n_rho), np.linspace(0.0, 2 * np.pi, n_phi)
    rho_g, phi_g = np.meshgrid(rho, phi)
    x = rho_g * np.cos(phi_g)
    y = rho_g * np.sin(phi_g)
    z = height * (1.0 - (rho_g / radius) ** 2)

    z_axis = np.array([0.0, 0.0, 1.0])
    boresight = np.asarray(boresight, dtype=float)
    v = np.cross(z_axis, boresight)
    c = np.dot(z_axis, boresight)
    if np.linalg.norm(v) < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1.0 + c)

    pts = R @ np.stack([x.ravel(), y.ravel(), z.ravel()])
    X = pts[0].reshape(x.shape) + center[0]
    Y = pts[1].reshape(y.shape) + center[1]
    Z = pts[2].reshape(z.shape) + center[2]
    return X, Y, Z

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

    sin2_sq is the sin^2 of the ray's angle relative to whatever axis the
    caller measured it against — normally the global Z (on-axis fiber), but a
    caller may pass an angle measured against a tilted boresight instead (e.g.
    an off-axis collection channel steered by a decentered microlens) without
    any change needed here.
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
    `fibers` may be an empty list — the top-level arrays (X_f, Y_f, V1,
    tir_mask, weights) are still returned so a caller can evaluate its own
    (e.g. tilted-boresight) acceptance test against them.
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

        # A fiber may carry its own boresight (e.g. a microlens-steered MCF
        # channel whose acceptance cone doesn't point straight along Z) —
        # measure the angle against that instead of the global interface angle.
        boresight = f.get('boresight')
        sin2_local = angle_to_boresight(V1, boresight) if boresight is not None else sin2_sq

        # Fundamental-mode size (only needed for the mode-overlap model). A
        # fiber may override w0 directly (e.g. a lensed aperture whose waist
        # isn't the Marcuse fit for a bare step-index core, such as an MCF
        # channel's beam expanded to fill the lens aperture) — na_mode still
        # follows the diffraction limit at the CURRENT wavelength, so this
        # stays correct under spectral averaging (na_mode scales with lambda).
        if coupling_model == MODE_OVERLAP_MODEL:
            if 'w0' in f:
                w0 = f['w0']
                na_mode = (lambda_nm / 1000.0) / (np.pi * n_med * w0)
            else:
                _, _, w0, na_mode = fiber_mode_params(f['d_core'], na, lambda_nm, n_med)
        else:
            w0, na_mode = 0.0, 0.0

        # Per-ray coupling weight (binary geometric, or Gaussian mode overlap)
        coupling = compute_coupling(dist_sq, sin2_local, tir_mask, r_core, na, n_med,
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
