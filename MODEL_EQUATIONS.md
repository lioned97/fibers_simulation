# NV-to-Fiber Collection Model — Equations & Parameter Justification

Documentation of the optical model implemented in [`app.py`](app.py).
The simulator is a vectorised Monte-Carlo ray tracer: light emitted by NV
centres inside bulk diamond escapes the flat top surface (Snell + Fresnel +
TIR) into a gap medium, propagates across an air gap, and is collected by one
or more step-index fibres (core overlap + numerical-aperture acceptance).

All symbols use SI-ish lab units: lengths in µm, angles in rad.

---

## 0. Master efficiency equation

The collection efficiency is built per ray, averaged over the Monte-Carlo ray
set (index $j=1\dots N_r$), then over the emitter ensemble (index
$i=1\dots N_{\mathrm{NV}}$). For a **single emitter** the geometric collection
efficiency is

$$
\eta \;=\; \underbrace{\tfrac{1}{2}}_{\text{hemisphere}}\;
\frac{1}{N_r}\sum_{j=1}^{N_r}
w_j\;\;
\underbrace{T(\theta_j)}_{\eta_{\text{surface}}}\;\;
\underbrace{\mathbb{1}\!\left[\,\rho_j \le R_{\text{core}}\,\right]}_{C_{\text{core}}}\;\;
\underbrace{\mathbb{1}\!\left[\,\sin\theta_{2,j} \le \tfrac{\mathrm{NA}}{n_{\text{med}}}\,\right]}_{A_{\text{NA}}}\;\;
\underbrace{(1-\text{TIR}_j)}_{\text{escape}}
$$

where $w_j$ is the normalised dipole emission weight, $T$ is the Fresnel power
transmission, $\rho_j=\sqrt{(X_{f,j}-x_f)^2+(Y_{f,j}-y_f)^2}$ is the radial
miss-distance at the fibre facet, and $\mathbb{1}[\cdot]$ is the indicator
(1 if true, 0 otherwise). This is exactly the kernel
`0.5 * np.sum(W * collected, axis=1) / n_r` in `run_ray_tracing`.

**Ensemble average** over emitters and **bundle sum** over fibres $k$:

$$
\eta_{\text{tot}} \;=\; \sum_{k}\;\frac{1}{N_{\mathrm{NV}}}\sum_{i=1}^{N_{\mathrm{NV}}}\eta_{i}^{(k)}
$$

**Spectral average** (the physically complete quantity — see note below): the
NV emission is broadband, so the reported efficiency should be the
emission-spectrum-weighted average of the monochromatic efficiency,

$$
\boxed{\;\eta \;=\; \int S(\lambda)\,\eta(\lambda)\,\mathrm{d}\lambda\,,
\qquad \int S(\lambda)\,\mathrm{d}\lambda = 1\;}
$$

where $S(\lambda)$ is the normalised NV$^-$ emission spectrum (ZPL at 637 nm
plus phonon sideband, 637–800 nm). The wavelength enters $\eta(\lambda)$
through the diamond dispersion $n_{\text{dia}}(\lambda)$. **This is implemented**
(`spectral_mode`): the ray tracer is re-run at each wavelength bin with its own
$n_{\text{dia}}(\lambda)$ from the Sellmeier equation (§7), and the integral is
evaluated by the trapezoid rule with $S(\lambda)$ normalised to unit area.
Disabling spectral mode falls back to the monochromatic value at the fixed
$n_{\text{dia}}=2.417$.

A more complete budget multiplies in the post-fibre optical survival:

$$
\eta_{\text{detected}} \;=\; \eta_{\text{tot}}\;\cdot\;\eta_{\text{optics}}
$$

with $\eta_{\text{optics}}=T_{\text{face}}\cdot T_{\text{prop}}\cdot T_{\text{filter}}\cdot\mathrm{QE}$
(fibre-face Fresnel · propagation loss · filter · detector QE). **This is
implemented** (`include_optics`); see §7. Disabling it sets $\eta_{\text{optics}}\equiv1$.

---

## 1. Emission angular distribution & hemisphere sampling

Directions are sampled uniformly **in solid angle** over the upper hemisphere
($\cos\theta\sim\mathcal U[0,1]$, $\phi\sim\mathcal U[0,2\pi]$), which makes
every ray carry equal solid-angle weight $\mathrm{d}\Omega$:

$$
\mathbf v_j = (\sin\theta\cos\phi,\;\sin\theta\sin\phi,\;\cos\theta),\qquad
\cos\theta\sim\mathcal U[0,1]
$$

The dipole weight $w_j$ for each emission model:

| Model | Weight $w(\mathbf v)$ | Normalisation |
|---|---|---|
| Isotropic | $w = 1$ | trivially $\langle w\rangle_{4\pi}=1$ |
| Single dipole $\mathbf d$ | $w = \tfrac{3}{2}\left[\,1-(\mathbf d\cdot\mathbf v)^2\,\right]=\tfrac32\sin^2\theta_d$ | $\langle\sin^2\rangle_{4\pi}=\tfrac23\Rightarrow\langle w\rangle=1$ |
| NV (2 ⟂ dipoles) about axis $\mathbf u$ | $w = \tfrac{3}{4}\left[\,1+(\mathbf u\cdot\mathbf v)^2\,\right]$ | $\langle 1+\cos^2\rangle_{4\pi}=\tfrac43\Rightarrow\langle w\rangle=1$ |
| NV ensemble (4 axes) | $w = \tfrac14\sum_{m=1}^{4}\tfrac34\left[1+(\mathbf u_m\cdot\mathbf v)^2\right]$ | average of the above |

The NV axes in a (100)-cut crystal (surface normal $\parallel[001]$):

$$
\mathbf u \in \tfrac{1}{\sqrt3}\{[111],[1\bar1\bar1],[\bar11\bar1],[\bar1\bar11]\}
$$

**The $\tfrac12$ prefactor** in the master equation: only the upper hemisphere
($2\pi$ of $4\pi$) is sampled because downward rays cannot reach the top
surface in this single-interface model. The collected fraction is therefore
weighted by $2\pi/4\pi=\tfrac12$.

---

## 2. Surface escape — Snell, TIR, Fresnel  →  $\eta_{\text{surface}}$

**Snell's law** at the diamond ($n_1=n_{\text{dia}}$) / gap ($n_2=n_{\text{med}}$) interface:

$$
n_1\sin\theta_1 = n_2\sin\theta_2,\qquad
\sin^2\theta_2=\left(\frac{n_1}{n_2}\right)^2\sin^2\theta_1
$$

**Total internal reflection** when $\sin^2\theta_2>1$, i.e. for incidence
beyond the critical angle

$$
\theta_c=\arcsin\!\left(\frac{n_2}{n_1}\right)
=\arcsin\!\left(\frac{1.000}{2.417}\right)\approx 24.4^\circ
$$

TIR rays are removed ($T=0$).

**Refracted unit direction** (tangential direction component scaled by $n_1/n_2$):

$$
\mathbf v' = \left(\tfrac{n_1}{n_2}v_x,\;\tfrac{n_1}{n_2}v_y,\;\cos\theta_2\right),
\qquad \cos\theta_2=\sqrt{1-\sin^2\theta_2}
$$

**Fresnel power transmission** for unpolarised light (mean of s- and p-pol):

$$
R_s=\left(\frac{n_1\cos\theta_1-n_2\cos\theta_2}{n_1\cos\theta_1+n_2\cos\theta_2}\right)^2,
\qquad
R_p=\left(\frac{n_2\cos\theta_1-n_1\cos\theta_2}{n_2\cos\theta_1+n_1\cos\theta_2}\right)^2
$$

$$
\eta_{\text{surface}} \equiv T(\theta_1)=1-\frac{R_s+R_p}{2}
$$

At normal incidence ($\theta_1=0$): $R=\big(\tfrac{n_1-n_2}{n_1+n_2}\big)^2=\big(\tfrac{1.417}{3.417}\big)^2=0.172$,
so $T\approx 0.828$ (17.2 % reflection loss).

---

## 3. Fibre acceptance  →  $C_{\text{core}}$ (core_fraction) and $A_{\text{NA}}$ (NA_fraction)

**Propagation to the facet plane** $Z=Z_{\text{fiber}}$. First intersect the
surface $Z=0$:

$$
t=-\frac{P_{0,z}}{v_z},\quad
X_{\text{int}}=P_{0,x}+t\,v_x,\quad
Y_{\text{int}}=P_{0,y}+t\,v_y
$$

then carry the refracted ray across the air gap:

$$
X_f=X_{\text{int}}+Z_{\text{fiber}}\frac{v'_x}{v'_z},\qquad
Y_f=Y_{\text{int}}+Z_{\text{fiber}}\frac{v'_y}{v'_z}
$$

**Core spatial overlap** (`core_fraction`, indicator per ray):

$$
C_{\text{core}}=\mathbb{1}\!\left[(X_f-x_f)^2+(Y_f-y_f)^2\le R_{\text{core}}^2\right],
\qquad R_{\text{core}}=\frac{D_{\text{core}}}{2}
$$

**Numerical-aperture acceptance** (`NA_fraction`, indicator per ray):

$$
A_{\text{NA}}=\mathbb{1}\!\left[\sin\theta_2\le\frac{\mathrm{NA}}{n_{\text{med}}}\right]
\quad\Longleftrightarrow\quad
\sin^2\theta_2\le\left(\frac{\mathrm{NA}}{n_{\text{med}}}\right)^2
$$

After Monte-Carlo averaging, $\langle C_{\text{core}}\rangle$ and
$\langle A_{\text{NA}}\rangle$ become the *fractions* of rays meeting each
criterion — hence the names.

---

## 4. Simulation-volume bounding (collection-limit radius)

To avoid generating NVs that can never be collected, the emitter box radius
around each fibre is bounded by back-propagating the NA cone:

$$
\theta_{\text{dia}}^{\max}=\arcsin\!\frac{\mathrm{NA}}{n_{\text{dia}}},\qquad
\theta_{\text{med}}^{\max}=\arcsin\!\frac{\mathrm{NA}}{n_{\text{med}}}
$$

$$
R_{\text{collect}} = d\,\tan\theta_{\text{dia}}^{\max} + R_{\text{core}} + Z_{\text{fiber}}\,\tan\theta_{\text{med}}^{\max}
$$

(`get_collection_limit_radius`; $d$ = NV depth incl. half slab width.)

---

## 5. Density scaling & absolute count rate

Volumetric NV density from concentration $C_{\text{ppm}}$:

$$
\rho = C_{\text{ppm}}\times 1.76\times10^5\ \text{NVs/µm}^3
$$

2-D / 3-D active emitter counts:

$$
N_{\mathrm{NV}}^{\text{(3D)}}=\rho\,(A_{\text{box}}\cdot W),\qquad
N_{\mathrm{NV}}^{\text{(2D)}}=\rho\,t_{2D}\cdot A_{\text{box}}\;\;(t_{2D}=1\,\text{nm})
$$

Estimated detected count rate:

$$
\dot N_{\text{phot}} = N_{\mathrm{NV}}\cdot\eta_{\text{tot}}\cdot R_{\text{sat}}
\quad\text{(single NV: }N_{\mathrm{NV}}=1\text{)}
$$

---

## 6. Parameter table — values and how they were obtained

| Symbol (code) | Value | How the value was obtained |
|---|---|---|
| $n_{\text{dia}}$ (`n_dia`) | **2.417** | Refractive index of diamond in the visible (Sellmeier / sodium-D value, ~589 nm). Diamond is nearly non-dispersive over the NV band: $n\approx2.41$ at 637 nm. Using 2.417 is the standard literature value; for a spectral model use $n_{\text{dia}}(\lambda)$. |
| $n_{\text{med}}$ (`n_med`) | **1.000** (air); 1.4–1.5 oil/gel | $n=1$ for an air gap. Index-matching oil/gel (1.46) raises $\theta_c$ and cuts Fresnel/TIR loss — that is the physical motivation for the option. |
| $\theta_c$ | **24.4°** | Derived: $\arcsin(n_{\text{med}}/n_{\text{dia}})=\arcsin(1/2.417)$. Not a free input. |
| $T(0)$ | **0.828** | Derived from Fresnel at normal incidence: $1-\big(\tfrac{n_1-n_2}{n_1+n_2}\big)^2$. Diamond–air gives 17.2 % reflection. |
| $\rho_{1\text{ppm}}$ (`1.76e5`) | **$1.76\times10^5$ NVs/µm³** | From the carbon number density of diamond. Lattice $a=3.567$ Å, 8 atoms/cell ⇒ $n_C=8/a^3=1.76\times10^{23}\,\text{cm}^{-3}$. 1 ppm $=1.76\times10^{17}\,\text{cm}^{-3}=1.76\times10^{5}\,\text{µm}^{-3}$ (since $1\,\text{cm}^3=10^{12}\,\text{µm}^3$). |
| $t_{2D}$ (`thickness_2d`) | **1 nm = 0.001 µm** | Modelling choice: a δ-doped NV monolayer is treated as a 1 nm slab so the volumetric density converts to a sheet density $\rho_{\text{sheet}}=\rho\,t_{2D}$. |
| NA (`fiber_na`) | **0.22** | Standard step-index multimode silica fibre NA (e.g. Thorlabs FG050LGA). Sets the acceptance half-angle $\theta_a=\arcsin(\mathrm{NA}/n_{\text{med}})\approx12.7°$ in air. |
| $D_{\text{core}}$ (`d_core`) | **50 µm** | Standard 50/125 multimode fibre core. Drives $C_{\text{core}}$. |
| $D_{\text{clad}}$ (`d_clad`) | **125 µm** | Standard silica cladding O.D.; sets array pitch / packing, not the collection physics. |
| $D_{\text{core}}/D_{\text{clad}}$ pitch | **125 µm** | Default array pitch = cladding diameter (fibres touching). |
| $Z_{\text{fiber}}$ (`z_fiber`) | **10 µm** (default) | User alignment parameter (air gap). Swept in Tab 4. Larger gap → larger spot → lower $C_{\text{core}}$. |
| $d$ (`nv_depth`) | **5 µm** (default) | NV depth below surface; experimental input. Deeper NV ⇒ larger surface spot ⇒ lower $C_{\text{core}}$. |
| $W$ (`nv_width`) | **2 µm** (3D slab) | Doped-layer thickness; experimental input. |
| $R_{\text{sat}}$ (`nv_photon_rate`) | **150 kcps** (default) | Representative single-NV saturated emission rate into $4\pi$. Order-of-magnitude lab value; user-tunable. Only scales absolute counts, not $\eta$. |
| $\tfrac32,\ \tfrac34$ (dipole prefactors) | **1.5, 0.75** | Normalisation constants so each pattern's $4\pi$ average weight = 1: $\langle\sin^2\rangle=2/3\Rightarrow\tfrac32$; $\langle1+\cos^2\rangle=4/3\Rightarrow\tfrac34$. Derived, not fitted. |
| $\tfrac12$ (hemisphere) | **0.5** | Solid-angle fraction of the upper hemisphere, $2\pi/4\pi$. Derived. |
| $\mathbf u_m$ (`NV_AXES`) | $\tfrac1{\sqrt3}[\pm1\pm1\pm1]$ | The four ⟨111⟩ NV symmetry axes in a (100)-oriented crystal; crystallography, not fitted. |
| seed | **42** | Fixed RNG seed for reproducibility. |

### Factors requested in the example, mapped to the code

| Requested factor | In this model | Source |
|---|---|---|
| `η_surface` | $T(\theta_1)=1-\tfrac{R_s+R_p}{2}$ | Fresnel, §2 — physically derived from $n_1,n_2$ |
| `core_fraction` ($C_{\text{core}}$) | $\langle\mathbb{1}[\rho\le R_{\text{core}}]\rangle$ | Monte-Carlo geometry, §3 |
| `NA_fraction` ($A_{\text{NA}}$) | $\langle\mathbb{1}[\sin\theta_2\le \mathrm{NA}/n_{\text{med}}]\rangle$ | Monte-Carlo geometry, §3 |
| `optic_survival` ($\eta_{\text{optics}}$) | $T_{\text{face}}T_{\text{prop}}T_{\text{filter}}\mathrm{QE}$ | implemented, §7 — multiplied onto $\eta_{\text{tot}}$ for detected counts |

---

## 7. Spectral dispersion & optical survival (implemented)

**Diamond dispersion** — Sellmeier equation ($\lambda$ in µm):

$$
n_{\text{dia}}^2(\lambda) = 1 + \frac{0.3306\,\lambda^2}{\lambda^2-0.175^2}
+ \frac{4.3356\,\lambda^2}{\lambda^2-0.106^2}
$$

Verified: $n=2.4173$ @ 589 nm, $2.4118$ @ 637 nm, $2.4062$ @ 700 nm, $2.4001$ @ 800 nm.

**NV emission spectrum** $S(\lambda)$ — a phonon sideband Gaussian (centre
690 nm, $\sigma=50$ nm) plus a ZPL Gaussian (centre 637 nm, $\sigma=4$ nm,
amplitude 0.52 giving the ~4 % Debye–Waller area fraction), then normalised so
$\int S\,\mathrm d\lambda=1$. The **representative wavelength** used for the
3-D/2-D visualisers and the alignment sweeps is the emission centroid
$\bar\lambda=\int\lambda S(\lambda)\,\mathrm d\lambda\approx700$ nm.

**Optical survival**:

$$
T_{\text{face}} = 1-\left(\frac{n_{\text{core}}-n_{\text{med}}}{n_{\text{core}}+n_{\text{med}}}\right)^2,
\qquad
T_{\text{prop}} = 10^{-\alpha L/10}
$$

with $\alpha$ in dB/km and $L$ in km. Defaults
($n_{\text{core}}=1.46$, $\alpha=10$ dB/km, $L=2$ m, $T_{\text{filter}}=0.90$,
$\mathrm{QE}=0.70$) give $\eta_{\text{optics}}=0.965\cdot0.995\cdot0.90\cdot0.70\approx0.605$.

| New parameter (code) | Value | How obtained |
|---|---|---|
| Sellmeier coeffs | $B_1{=}0.3306,\,C_1{=}0.175^2$; $B_2{=}4.3356,\,C_2{=}0.106^2$ | Standard diamond Sellmeier fit (visible/IR); reproduces the 2.417 sodium-D value. |
| $S(\lambda)$ shape | sideband 690 nm/$\sigma$50, ZPL 637 nm/$\sigma$4, DW≈4% | Analytic approximation of the measured room-temperature NV$^-$ emission band. |
| $n_{\text{core}}$ | 1.46 | Fused-silica fibre core index. |
| $\alpha$ | 10 dB/km | Typical multimode silica attenuation at 650–800 nm. |
| $T_{\text{filter}}$ | 0.90 | Representative long-/band-pass transmission over the NV band. |
| QE | 0.70 | Si single-photon APD QE near 700 nm. |

---

## 8. Coupling model & mode matching (implemented)

The fiber acceptance is selectable (`coupling_model`):

**Geometric (multimode)** — the binary acceptance of §3, exact in the
many-mode limit where guided modes tile the geometric étendue. The mode
content of the fiber is set by the **normalised frequency** (V-number):

$$
V = \frac{2\pi a}{\lambda}\,\mathrm{NA},\qquad
M \approx \frac{V^2}{2}\;\;(V\gg1),\qquad a = \frac{D_{\text{core}}}{2}
$$

with single-mode operation for $V < 2.405$. (Reported in the dashboard caption.)

**Mode overlap (Gaussian)** — for single-/few-mode coupling the hard aperture
is replaced by the **phase-space overlap with the fundamental $\text{LP}_{01}$
mode**, modelled as a Gaussian of field radius $w_0$ (Marcuse fit) with an
effective mode NA:

$$
\frac{w_0}{a} = 0.65 + \frac{1.619}{V^{3/2}} + \frac{2.879}{V^{6}},
\qquad
\mathrm{NA}_{\text{mode}} = \frac{\lambda}{\pi\,n_{\text{med}}\,w_0}
$$

The per-ray coupling weight is the separable Gaussian Wigner overlap in
transverse position $\rho$ and angle $\theta_2$ at the facet:

$$
c(\rho,\theta_2) = \exp\!\left(-\frac{2\rho^2}{w_0^2}\right)
\exp\!\left(-\frac{2\sin^2\theta_2}{\mathrm{NA}_{\text{mode}}^2}\right)
$$

so the master kernel of §0 uses $C_{\text{core}}\,A_{\text{NA}} \to c$ and the
result is the **mode-matching efficiency** into the single guided mode,
$\eta = \tfrac12\langle w\,T\,c\rangle$. Because $w_0$ and $\mathrm{NA}_{\text{mode}}$
both scale with $\lambda$, the mode-matched efficiency is wavelength-dependent
even where the geometric one is flat — so §0's spectral average is non-trivial here.

| New parameter (code) | Value / formula | How obtained |
|---|---|---|
| $V$ | $2\pi a\,\mathrm{NA}/\lambda$ | Definition of the step-index normalised frequency. |
| $M$ | $V^2/2$ | Large-$V$ guided-mode count (both polarisations). |
| $w_0$ | Marcuse fit (above) | Standard $\text{LP}_{01}$ Gaussian-equivalent field radius, best for $1.2<V<2.4$ (clamped at $V{=}1.2$ for display). |
| $\mathrm{NA}_{\text{mode}}$ | $\lambda/(\pi n_{\text{med}} w_0)$ | Gaussian-beam far-field divergence of the mode. |

> Coupling NV emission into the $\text{LP}_{01}$ mode of a *large* multimode core
> is intrinsically tiny (the mode fills a single small phase-space cell, e.g.
> $\mathrm{NA}_{\text{mode}}\approx0.014$ for a 50 µm/0.22 fiber) — that regime is
> exactly where the **geometric** model is the relevant one. Use mode overlap for
> single-/few-mode fibers ($V \lesssim$ a few).

---

### Note on what is and isn't modelled

- **Spectral averaging — implemented**: $n_{\text{dia}}(\lambda)$ varies over the
  NV band, weighted by $S(\lambda)$. In practice $\eta(\lambda)$ is nearly flat
  because $n_{\text{dia}}$ moves only ~0.011 across 640–800 nm, so the spectral
  correction to the *geometric* efficiency is small (the dominant
  wavelength dependence would come from $T_{\text{filter}}(\lambda)$ /
  $\mathrm{QE}(\lambda)$, currently taken constant).
- **Optical survival — implemented**: face Fresnel, attenuation, filter, QE.
  Connector losses and $\lambda$-dependent filter/QE curves are not yet broken out.
- **Single interface**: only the top surface; no substrate reflections, no
  multiple bounces, no waveguiding back into the bulk.
- **Coupling model — both implemented**: geometric (multimode étendue) and
  Gaussian fundamental-mode overlap (mode matching), selectable in the sidebar
  (§8). Full vector-mode (LP$_{\ell m}$) overlap and few-mode summation are not
  modelled — the Gaussian approximates only the $\text{LP}_{01}$ mode.
