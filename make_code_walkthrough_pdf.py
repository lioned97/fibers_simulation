"""Build a standalone walkthrough of the NV-fiber calculation and code."""
import json
import os

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from make_comparison_pdf import (GRID, INK, MUTED, NAVY, PALE,
                                 key_value_table, styles)


ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "output", "pdf")
PDF = os.path.join(OUT, "nv_fiber_simulation_code_walkthrough.pdf")

pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Consolas", r"C:\Windows\Fonts\consola.ttf"))


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.line(18*mm, height-13*mm, width-18*mm, height-13*mm)
    canvas.setFont("Arial", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18*mm, height-10*mm, "NV-fiber simulation: code and calculation walkthrough")
    canvas.drawRightString(width-18*mm, 10*mm, f"Page {doc.page}")
    canvas.restoreState()


def code(text, style):
    return Preformatted(text.strip("\n"), style["mono"])


def table(headers, rows, widths, font=7.5):
    body = [[Paragraph(str(value), STYLES["small"]) for value in headers]]
    body += [[Paragraph(str(value), STYLES["body"]) for value in row] for row in rows]
    item = Table(body, colWidths=[w*mm for w in widths], repeatRows=1)
    item.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("LEADING", (0, 0), (-1, -1), font+2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.3, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return item


def page(title, subtitle=None):
    items = [PageBreak(), Paragraph(title, STYLES["title"])]
    if subtitle:
        items.append(Paragraph(subtitle, STYLES["subtitle"]))
    return items


def build():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(FIG, "mcf_freeform_design.json"), encoding="utf-8") as fh:
        design = json.load(fh)
    with open(os.path.join(FIG, "mcf_numerical_validation.json"), encoding="utf-8") as fh:
        validation = json.load(fh)
    result = design["result"]
    measured = design["measured"]
    norm = design["comparison_normalization"]

    resolution_cal = np.array([3.5, 49.0, 1.5])
    linewidth_cal = np.array([3.69, 4.52, 3.44])
    lw_a, lw_b = np.linalg.lstsq(
        np.column_stack([np.ones(3), resolution_cal**2]),
        linewidth_cal**2, rcond=None)[0]

    story = [
        Paragraph("NV-fiber simulation", STYLES["title"]),
        Paragraph("Complete code and calculation walkthrough with pseudocode", STYLES["h1"]),
        Paragraph("Purpose: make every major numerical step inspectable, show which outputs are direct optical calculations and which are empirical mappings, and identify what still requires experimental validation.", STYLES["subtitle"]),
        Paragraph("Bottom line", STYLES["h1"]),
        Paragraph(f"The corrected redesigned-MCF model returns {result['model_fiber_photons_s']:.4g} fiber photons/s and a {result['resolution_um']:.4f} um Gaussian-equivalent optical FWHM. Snell-law residuals are at machine precision and the final quadratures are numerically converged. The predicted {result['comparison_normalized_sensitivity_nt']:.2f} nT/sqrt(Hz) value is not a direct detector prediction: it uses an empirical normalization and retains the measured old-MCF contrast.", STYLES["callout"]),
        table(["Layer", "What the code does", "Confidence status"], [
            ("Geometry and refraction", "Vector rays, local normals, Snell law, Fresnel power transmission and monolithic surface visibility.", "Strong internal consistency; not yet compared with an independent optical package."),
            ("NV ensemble", "3 ppm density integrated through z=-80...-90 um with excitation saturation and collection weighting.", "Numerically converged for the implemented model."),
            ("New-lens collection", "Reciprocal Gaussian-mode tracing plus a Gaussian-equivalent diffraction/etendue approximation.", "Useful design model; not full-wave electromagnetic optics."),
            ("ODMR linewidth/contrast", "Linewidth is fitted from three supplied probe points; contrast is fixed to measured MCF contrast.", "Empirical, not independently predicted."),
            ("Sensitivity", "Measured MCF sensitivity is rescaled by linewidth, contrast and square-root count rate.", "Comparison model only; experimental confirmation required."),
        ], [35, 88, 40]),
        Spacer(1, 4*mm),
        Paragraph("How to read this document", STYLES["h2"]),
        Paragraph("Sections 1-6 explain the original SM/MM/as-built-MCF calculation. Sections 7-11 explain the redesigned lens, sensitivity mapping and fabrication output. Sections 12-14 give validation evidence and complete end-to-end pseudocode.", STYLES["body"]),
    ]

    story += page("1. Code map and execution order")
    story += [
        table(["File", "Responsibility", "Main entry points"], [
            ("physics.py", "Shared diamond dispersion, NV spectrum, emission directions, bare-fiber ray tracing and coupling.", "diamond_sellmeier; nv_emission_spectrum; run_ray_tracing"),
            ("paper_figures.py", "Static SM, MM, as-built MCF and ideal re-aimed MCF sweeps and figures.", "run_sweeps; run_spectra; eta_per_emitter"),
            ("lens_design.py", "New central/side surface fitting, physical ray tracing, 3-D ensemble integral, optimization and STL export.", "search_design; evaluate_design; write_binary_stl"),
            ("sensitivity.py", "ND correction, measured references, linewidth fit and reference-scaled sensitivity.", "linewidth_from_resolution; calibrated_sensitivity"),
            ("phase3_optimize.py", "Runs the deterministic new-lens search and writes design JSON plus two STLs.", "main"),
            ("redesign_fig.py", "Static/interactive rays, method comparison, gap/tilt plots and fabrication drawing.", "full_3d_interactive; alignment_figure"),
            ("validate_model.py", "Grid/quadrature convergence and Snell residual report.", "main"),
            ("test_*.py; fab_check.py", "Small executable checks for physics, sensitivity, symmetry and watertight geometry.", "run each file with Python"),
        ], [34, 86, 43]),
        Paragraph("Actual run order", STYLES["h1"]),
        code("""
paper_figures.py
    -> physics.py
    -> figures 1-7 and baseline CSV

phase3_optimize.py
    -> lens_design.search_design()
    -> mcf_freeform_design.json
    -> one-side STL + full seven-core STL

redesign_fig.py
    -> reload final design
    -> static rays + interactive HTML + gap/angle sweep

validate_model.py
    -> recompute selected numerical variants
    -> mcf_numerical_validation.json
""", STYLES),
        Paragraph("There are two related optical models", STYLES["h1"]),
        Paragraph("The baseline comparison and the new-lens optimizer share physical constants, but they are not the same numerical method. The baseline traces emitter-to-fiber rays with deterministic angular quadrature and importance-sampled emitter Monte Carlo. The redesign uses reciprocal fiber-mode propagation and deterministic volume integration. Their resolution definitions also differ; this is documented explicitly in sections 7 and 10.", STYLES["callout"]),
    ]

    story += page("2. Coordinates, geometry and fixed inputs")
    story += [
        key_value_table([
            ("Optical coordinates", "Diamond surface z=0. Diamond occupies negative z. The 3-D NV layer is z=-80 to -90 um. Fibers and printed polymer are at positive z."),
            ("Single NV", "One emitter on the optical axis at z=-85 um."),
            ("Baseline distance", "g is diamond-surface to bare-fiber facet for SM/MM and diamond-surface to central lens tip for MCF; baseline sweep is 0-200 um."),
            ("New-design distance", "Allowed 5-500 um. The selected central tip is 5 um from the diamond."),
            ("Diamond", "100 um thick; air gap; green absorption over the 10 um NV layer is neglected."),
            ("Population", "3 ppm x 1.76e11 carbon sites/um^3 = 5.28e5 NV/um^3."),
            ("Excitation", "10 mW at 532 nm from each delivery aperture; R_sat=5e6 photons/s; I_sat=30 mW/um^2."),
            ("MCF", "Central core plus six side cores at radius 35 um; every core MFD=10 um; printed polymer n=1.52."),
        ], STYLES),
        Spacer(1, 3*mm),
        table(["Probe", "Excitation", "Collection"], [
            ("SM", "Gaussian w0 about 2 um, NA 0.12", "4 um core, Gaussian fundamental-mode overlap"),
            ("MM", "25 um-radius top-hat cone, NA 0.22", "50 um core, geometric core+NA acceptance"),
            ("MCF old", "Central printed lens, Gaussian q propagation", "Six physical side lenses/cores, reciprocal mode overlap"),
            ("MCF new", "Optimized central polynomial surface", "Six copies of optimized side polynomial surface"),
        ], [27, 68, 68]),
        Paragraph("Spectral windows are not identical", STYLES["h1"]),
        Paragraph("paper_figures.py evaluates its explicit spectrum at 9 wavelengths from 640 to 800 nm. lens_design.py evaluates the redesigned lens at 33 trapezoid-weighted wavelengths from 650 to 850 nm. Distance sweeps use a representative 700 nm in the baseline. This difference is deliberate in the current code but should be standardized before a final apples-to-apples publication table.", STYLES["callout"]),
        code("""
rho_NV = 3 ppm * 1e-6 * 1.76e11 carbon_sites_per_um3
       = 5.28e5 NV_per_um3

new-design number in a differential cell:
dN = rho_NV * dx * dy * dz
""", STYLES),
    ]

    story += page("3. Shared physics: spectrum, modes and emitted rays")
    story += [
        Paragraph("Diamond dispersion", STYLES["h1"]),
        code("""
L = wavelength_um^2
n_diamond^2 = 1 + 0.3306*L/(L-0.175^2)
                  + 4.3356*L/(L-0.106^2)
""", STYLES),
        Paragraph("NV spectrum", STYLES["h1"]),
        Paragraph("nv_emission_spectrum() is an analytic room-temperature approximation: a broad Gaussian phonon sideband centered near 690 nm plus a narrow Gaussian zero-phonon line at 637 nm. Callers normalize the sampled values before integration.", STYLES["body"]),
        Paragraph("Four-axis dipole average", STYLES["h1"]),
        code("""
for each <111> NV axis u:
    angular_weight_u(v) = 0.75 * (1 + dot(u,v)^2)
ensemble_weight(v) = mean(angular_weight_u(v) over four axes)
""", STYLES),
        Paragraph("For equal populations of the four tetrahedral axes, mean(dot(u,v)^2)=1/3 for every direction v, so the four-axis ensemble average is exactly one. Therefore the redesigned ensemble can use an isotropic angular average without losing the equal-axis dipole physics.", STYLES["callout"]),
        Paragraph("Bare-fiber mode models", STYLES["h1"]),
        table(["Model", "Per-ray coupling"], [
            ("Geometric", "1 when the facet hit is inside the core and sin(theta)<=NA/n; otherwise 0."),
            ("Gaussian mode", "exp(-2*rho^2/w0^2) x exp(-2*sin(theta)^2/NA_mode^2)."),
        ], [38, 125]),
        code("""
PSEUDOCODE: sample_ray_directions
    sample phi uniformly in [0, 2*pi)
    sample cos(theta) uniformly in [0,1]
    build unit direction v on upper hemisphere
    evaluate isotropic, single-dipole or NV-axis weight
    return directions and normalized relative weights
""", STYLES),
    ]

    story += page("4. Baseline SM/MM/old-MCF calculation", "This is the path that creates the original distance, spectrum and collection-area figures.")
    story += [
        code("""
PSEUDOCODE: paper_figures.run_sweeps
for probe in [SM, MM, MCF fixed aim, MCF re-aimed]:
    build deterministic escape-cone ray quadrature
    importance-sample 600 emitters in the 3-D NV layer

    for gap g from 0 to 200 um:
        construct fiber/lens configuration at g

        # single NV
        eta_single = collection_efficiency(on-axis NV)
        R_single = saturated_emission_rate(on-axis NV, g)
        collected_single = eta_single * R_single

        # 3-D ensemble
        eta_i = collection_efficiency(each sampled emitter)
        R_i = saturated_emission_rate(each sampled emitter, g)
        signal_weight_i = R_i * eta_i * density_importance_weight_i

        ensemble_efficiency = sum(signal_weight_i) / sum(R_i*density_weight_i)
        ensemble_power = mean(signal_weight_i) * photon_energy
        A50 = pi * radius_containing_half_of_signal^2

    store all gap curves and the gap with maximum single-NV efficiency
""", STYLES),
        Paragraph("Why importance sampling is used", STYLES["h1"]),
        Paragraph("A uniform emitter box would spend most samples where the pump is negligible. sample_ensemble() draws from a mixture of a narrow Gaussian, a broad Gaussian and a uniform box. Each emitter receives density/proposal_pdf, so the estimator remains unbiased for the modeled finite collection region.", STYLES["body"]),
        code("""
proposal q(x,y,z) = q_xy(x,y) * (1/NV_width)
density_weight_i = rho_NV / q(x_i,y_i,z_i)

integral of f over NV volume ~= mean(f_i * density_weight_i)
""", STYLES),
        Paragraph("No detector chain", STYLES["h1"]),
        Paragraph("The baseline publication figures stop at power coupled into the fiber. Filter transmission, connector loss and detector QE are not included in paper_figures.py outputs.", STYLES["callout"]),
    ]

    story += page("5. Excitation and saturation")
    story += [
        code("""
s = intensity / I_sat
R_NV = R_sat * s/(1+s)

Gaussian: I(r) = 2*P_transmitted/(pi*w^2) * exp(-2*r^2/w^2)
Top hat:  I(r) = P_transmitted/(pi*w^2) inside radius w
""", STYLES),
        table(["Probe", "How beam radius at the NV is obtained"], [
            ("SM", "w = sqrt(w0^2 + (gap*tan(theta_air)+depth*tan(theta_dia))^2), w0=2 um."),
            ("MM", "Top-hat radius = 25 um + gap*tan(theta_air)+depth*tan(theta_dia)."),
            ("MCF old", "Gaussian q propagation through the fitted central STL curvature, polymer-air propagation and diamond."),
            ("MCF new", "Physical mode rays through the optimized central surface; weighted hits are reduced to a Gaussian covariance at every depth."),
        ], [31, 132]),
        Paragraph("Fresnel accounting", STYLES["h1"]),
        Paragraph("For the baseline bare fibers, T_GREEN_IN is the 532 nm air-to-diamond entry loss. For the old printed MCF, MCF_GREEN_LENS_T is the polymer-to-air lens loss and T_GREEN_IN is the separate air-to-diamond loss. For the new design, trace_mode() already contains polymer-air and air-diamond transmission, so evaluate_design() does not multiply another T_GREEN_IN. This was the duplicate-loss bug corrected in the current code.", STYLES["callout"]),
        code("""
PSEUDOCODE: excitation rate
    depth = -z_NV
    beam_radius = profile_specific_radius(depth, gap)
    transmitted_power = 10 mW * all physical interface transmissions
    intensity = normalized spatial profile * transmitted_power
    saturation = intensity / 30 mW_per_um2
    return 5e6 * saturation/(1+saturation)
""", STYLES),
        Paragraph("Model boundary", STYLES["h1"]),
        Paragraph("Green absorption through the 10 um NV layer, background fluorescence, charge-state conversion, NV-NV interactions, local pump depletion and heating are not modeled.", STYLES["body"]),
    ]

    story += page("6. Baseline ray collection and fiber coupling")
    story += [
        code("""
PSEUDOCODE: physics.run_ray_tracing
for every emitter and upper-hemisphere ray:
    intersect ray with diamond surface z=0
    apply Snell law diamond -> air
    mark total-internal-reflection rays
    apply unpolarized Fresnel transmission
    propagate transmitted ray across the air gap to fiber plane

    for every fiber:
        calculate transverse miss distance
        calculate angle relative to fiber or lens boresight
        apply geometric or Gaussian-mode coupling
        efficiency_per_emitter = 0.5 * mean(dipole_weight*Fresnel*coupling)
""", STYLES),
        Paragraph("Why the factor 0.5 appears", STYLES["h1"]),
        Paragraph("Directions represent only the upper 2*pi hemisphere, while NV emission rate is defined into 4*pi. Multiplication by 0.5 restores the full-sphere fraction.", STYLES["body"]),
        Paragraph("Old printed MCF", STYLES["h1"]),
        Paragraph("lensed_mcf_eta() first traces diamond-to-air escape. It then intersects the nearest exposed surface of the continuous printed union, applies air-to-IP-S Snell/Fresnel, propagates inside IP-S to the silica core plane, and applies spatial plus angular overlap with the 10 um MFD side-core mode. One side is traced and multiplied by six only for the centered, rotationally symmetric ensemble.", STYLES["body"]),
        code("""
PSEUDOCODE: old MCF collection
    trace emitter ray to air
    find earliest hit among central cap, six side caps and support plane
    refract air -> IP-S at exposed union surface
    propagate inside one continuous IP-S body to side-core plane
    mode_weight = Gaussian(position miss) * Gaussian(angle miss)
    apply IP-S -> silica normal-incidence transmission
    return 6 * hemisphere_average(weighted mode coupling)
""", STYLES),
        Paragraph("Geometric limitation", STYLES["callout"]),
        Paragraph("This is geometrical optics plus Gaussian mode matching. It does not solve coherent diffraction, phase errors, interference between overlapping apertures, polarization-dependent vector modes, surface roughness scattering or fabrication shrinkage.", STYLES["body"]),
    ]

    story += page("7. Baseline spectrum and resolution outputs")
    story += [
        Paragraph("Spectrum", STYLES["h1"]),
        Paragraph("run_spectra() repeats single-NV collection at every wavelength, recomputing diamond index, Fresnel transmission and wavelength-dependent mode divergence. It reports eta(lambda), not the NV spectrum multiplied by eta(lambda).", STYLES["body"]),
        Paragraph("Ensemble collection efficiency", STYLES["h1"]),
        code("""
eta_ensemble = sum(R_i * eta_i * density_weight_i)
               / sum(R_i * density_weight_i)

power_nW = mean(R_i * eta_i * density_weight_i)
           * photon_energy * 1e9
""", STYLES),
        Paragraph("50%-signal area", STYLES["h1"]),
        code("""
sort emitters by radius from optical axis
cumulative_signal = cumulative_sum(R_i*eta_i*density_weight_i)
r50 = first radius where cumulative_signal >= 0.5*total_signal
A50 = pi*r50^2
""", STYLES),
        Paragraph("A50 is an on-axis circular area, not the mathematically smallest arbitrary area containing half the signal. It remains interpretable for six MCF lobes but can hide azimuthal structure.", STYLES["callout"]),
        Paragraph("Important metric difference", STYLES["h1"]),
        Paragraph("The redesigned-lens code does not return A50. It returns a Gaussian-equivalent FWHM derived from second moments of excitation x collection. A50 and FWHM are related only for an ideal Gaussian and should not be presented as identical resolution metrics.", STYLES["callout"]),
        code("""
PSEUDOCODE: baseline spectral figure
for probe:
    choose probe's optimum baseline gap
    for wavelength in 640...800 nm:
        n_diamond = Sellmeier(wavelength)
        eta[wavelength] = collection_efficiency(single_NV, wavelength)
plot eta versus wavelength for each probe
""", STYLES),
    ]

    story += page("8. New lens surface design")
    story += [
        Paragraph("Polynomial height surface", STYLES["h1"]),
        code("""
xi=u/aperture; nu=v/aperture
sag = aperture * [p0*xi + 0.5*p1*xi^2 + 0.5*p2*nu^2
       + (p3/3)*xi^3 + p4*xi*nu^2 + (p5/4)*xi^4
       + 0.5*p6*xi^2*nu^2 + (p7/4)*nu^4] + shift
z_surface = apex + sag
""", STYLES),
        Paragraph("How a surface is fitted", STYLES["h1"]),
        code("""
PSEUDOCODE: fit_surface
choose role, family, gap, height, aperture and lateral center
choose design wavelength: 532 nm central or 750 nm side
sample pupil points inside circular aperture

repeat four times:
    calculate current surface points and Gaussian input directions
    solve desired air direction that reaches target through diamond
    required_normal = normalize(n_IPS*incident - n_air*desired)
    convert normal to required x/y surface slopes
    least-squares fit allowed polynomial slope basis
    shift sag so no point extends behind the requested apex

return fitted physical surface
""", STYLES),
        Paragraph("Why the normal formula works", STYLES["h1"]),
        Paragraph("Vector Snell refraction conserves the tangential wave-vector component. Therefore n1*v_incident - n2*v_transmitted is parallel to the interface normal. Fitting that normal produces a surface that redirects the modeled input wavefront toward the target.", STYLES["body"]),
        table(["Family label", "Allowed slope basis"], [
            ("quadratic", "Rotational quadratic power plus optional linear steering"),
            ("asphere", "Rotational quadratic plus radial fourth-order term"),
            ("biconic", "Independent radial/tangential quadratic power plus fourth order"),
            ("freeform", "Eight-term asymmetric polynomial basis"),
        ], [40, 123]),
        Paragraph("This is a finite polynomial search, not an unconstrained freeform optimizer or proof of global optimality.", STYLES["callout"]),
    ]

    story += page("9. New-lens physical ray tracing")
    story += [
        code("""
PSEUDOCODE: trace_mode(surface, wavelength, depths)
sample Gaussian-mode pupil points inside the lens aperture
calculate each point's local sag and normal

for every other overlapping lens domain:
    hide the current point if another surface is farther outward

calculate Gaussian input direction from the fiber core
refract IP-S -> air using vector Snell law
apply unpolarized Fresnel transmission
intersect the possibly tilted diamond plane
refract air -> diamond and apply Fresnel transmission
propagate each valid ray to every requested NV depth
return hits, directions and power weights
""", STYLES),
        Paragraph("Vector refraction", STYLES["h1"]),
        code("""
eta = n1/n2
sin(theta_t)^2 = eta^2 * (1-cos(theta_i)^2)
v_t = normalize(eta*v_i + (cos_t-eta*cos_i)*normal)
T = 1 - 0.5*(R_s + R_p)
""", STYLES),
        Paragraph("Beam statistics", STYLES["h1"]),
        Paragraph("beam_stats() computes weighted centroid and 2x2 covariance of ray hits. It adds an isotropic diffraction floor w_diff=lambda/[pi*n*sin(theta_90)] before converting covariance to a Gaussian-equivalent FWHM.", STYLES["body"]),
        code("""
mean = sum(weight_i * hit_i) / sum(weight_i)
cov  = sum(weight_i * (hit_i-mean)(hit_i-mean)^T) / sum(weight_i)
cov += identity * (w_diff/2)^2
FWHM = 2*sqrt(2*ln(2)) * sqrt(mean(eigenvalues(cov)))
""", STYLES),
        Paragraph("The diffraction floor is a Gaussian approximation derived from angular spread. It is not a coherent propagation of the polynomial aperture phase.", STYLES["callout"]),
    ]

    story += page("10. New 3-D ensemble and reciprocal collection")
    story += [
        code("""
PSEUDOCODE: evaluate_design
trace central 532 nm mode to 33 NV depths
trace one side mode at 33 wavelengths to the same depths
construct x-y integration grid covering all Gaussian footprints

for each depth plane:
    excitation = normalized Gaussian from central covariance
    intensity = 10 mW * traced central throughput * excitation
    R(x,y) = R_sat * (intensity/I_sat)/(1+intensity/I_sat)

    eta(x,y) = 0
    for wavelength and six rotated side cores:
        calculate Gaussian collection footprint
        calculate eta_peak from throughput, wavelength and mode area
        eta += spectral_weight * eta_peak * Gaussian_footprint

    signal_density = rho_NV * R(x,y) * min(eta(x,y),1)
    integrate signal_density * dx*dy*dz by trapezoid rule in z
    accumulate signal-weighted first and second moments

convert moments to Gaussian-equivalent optical FWHM
map optical FWHM to ODMR linewidth and sensitivity
""", STYLES),
        Paragraph("Collection by reciprocity", STYLES["h1"]),
        code("""
wx, wy = Gaussian 1/e^2 radii from covariance eigenvalues
eta_peak = throughput * wavelength^2 / (4*pi^2*n_diamond^2*wx*wy)
eta_local = eta_peak * Gaussian(x,y; mean,covariance)
""", STYLES),
        Paragraph("The model launches the fiber mode outward and uses optical reciprocity to infer coupling from an emitter back into that mode. This is valid for a linear reciprocal optical system, but the conversion from traced ray covariance to eta_peak is a Gaussian/etendue approximation. It is the largest optical-model approximation in the redesigned-lens result.", STYLES["callout"]),
        Paragraph("Final quadrature", STYLES["h1"]),
        Paragraph("The selected design is recomputed with 33 depth planes, 33 normalized trapezoid-weighted wavelengths from 650 to 850 nm, a 35-point pupil grid and a 161x161 lateral grid. Search screening uses 9 depth planes and 9 wavelengths to control runtime.", STYLES["body"]),
    ]

    story += page("11. Resolution, linewidth, contrast and sensitivity")
    story += [
        table(["Probe", "Corrected cps", "Contrast", "FWHM", "Sensitivity"], [
            ("SM", f"{measured['SM']['cps']:.3g}", f"{100*measured['SM']['contrast']:.2f}%", f"{measured['SM']['fwhm_mhz']:.2f} MHz", f"{measured['SM']['sensitivity_nt']:.0f} nT/sqrt(Hz)"),
            ("MM", f"{measured['MM']['cps']:.3g}", f"{100*measured['MM']['contrast']:.2f}%", f"{measured['MM']['fwhm_mhz']:.2f} MHz", f"{measured['MM']['sensitivity_nt']:.0f} nT/sqrt(Hz)"),
            ("MCF old", f"{measured['MCF']['cps']:.3g}", f"{100*measured['MCF']['contrast']:.2f}%", f"{measured['MCF']['fwhm_mhz']:.2f} MHz", f"{measured['MCF']['sensitivity_nt']:.0f} nT/sqrt(Hz)"),
        ], [29, 34, 31, 34, 35]),
        Paragraph("ND correction", STYLES["h1"]),
        code("corrected_cps = observed_kcps * 1000 * 10^OD", STYLES),
        Paragraph("Linewidth mapping", STYLES["h1"]),
        Paragraph(f"The code fits FWHM^2 = a + b*w^2 to assumed optical resolutions [3.5, 49, 1.5] um paired with the supplied SM/MM/MCF linewidths. The fitted values are a={lw_a:.7g} MHz^2 and b={lw_b:.7g} MHz^2/um^2, giving a floor sqrt(a)={np.sqrt(lw_a):.4f} MHz.", STYLES["body"]),
        Paragraph("Those three optical-resolution calibration values are hard-coded model inputs. They are not re-derived from the current baseline A50 curves inside sensitivity.py. This mapping should be replaced by measured or consistently computed widths before claiming a first-principles linewidth prediction.", STYLES["callout"]),
        Paragraph("Contrast", STYLES["h1"]),
        Paragraph("The new design does not calculate contrast from optical background or microwave physics. It sets contrast equal to the measured old-MCF value, 8.10%.", STYLES["callout"]),
        Paragraph("Sensitivity mapping", STYLES["h1"]),
        code("""
eta_B = 103 nT/sqrt(Hz)
        * (FWHM_new/3.44 MHz)
        * (0.081/contrast_new)
        * sqrt(2e9 / rate_new)
""", STYLES),
        Paragraph(f"The raw convention inserts {result['model_fiber_photons_s']:.4g} fiber photons/s as if every photon produced one count, giving {result['raw_model_sensitivity_nt']:.2f} nT/sqrt(Hz). The comparison convention multiplies by {norm['factor']:.5g}, giving {result['comparison_normalized_sensitivity_nt']:.2f} nT/sqrt(Hz). Because that factor exceeds one, it is not a detector efficiency.", STYLES["body"]),
    ]

    story += page("12. Search, alignment and STL construction")
    story += [
        code("""
PSEUDOCODE: search_design
for gap in [5,25,75,150,300,500] um:
  for base height in [120,200,280] um:
    fit quadratic/asphere/biconic/freeform central candidates
    fit side candidates over aperture, radial center and vertical offset
    reject candidates exceeding 300 um print height
    keep two proxy-ranked shapes per family
    form central+side pairs and rank by cheap overlap merit

keep global and per-method finalists
evaluate finalists with 9-depth x 9-wavelength screening integral
select lowest raw-model sensitivity
recompute winner with 33 depths, 33 wavelengths and grid 161
""", STYLES),
        Paragraph("Search status", STYLES["h1"]),
        Paragraph("The search is deterministic and broader than a single proxy winner, but it is still a discrete grid with polynomial families. It does not prove that the selected quadratic-central plus biconic-side pair is the global optimum.", STYLES["callout"]),
        Paragraph("Gap and tilt", STYLES["h1"]),
        Paragraph("design_at_gap() translates the unchanged printed geometry. alignment_sweep() screens gap, chooses maximum modeled fiber photons, and then rotates the diamond plane about the fiber y-axis. At nonzero tilt all six side lenses are traced separately because rotational symmetry is broken.", STYLES["body"]),
        Paragraph("STL", STYLES["h1"]),
        code("""
PSEUDOCODE: write_binary_stl
replicate side surface at six azimuths
sample radial rings and 144 angular positions
at each x,y choose the most outward of support/central/side surfaces
create top height field and flat fiber-attachment bottom
triangulate top, bottom and outer wall
write binary STL and verify every mesh edge belongs to exactly two faces
""", STYLES),
        Paragraph("Overlaps are max-height/min-z unions, not added sags. The STL contains no internal refractive boundary between printed parts.", STYLES["callout"]),
    ]

    depth_diff = validation["relative_photon_difference_from_final"]["depth17"]
    spectrum_diff = validation["relative_photon_difference_from_final"]["spectrum17"]
    grid_diff = validation["relative_photon_difference_from_final"]["grid121"]
    snell_max = max(value for row in validation["snell_law"].values()
                    for value in row.values())
    story += page("13. What the validation proves - and does not prove")
    story += [
        table(["Check", "Result", "Meaning"], [
            ("121 vs 161 lateral grid", f"{100*grid_diff:.3g}% photon-rate change", "Lateral numerical integral is converged for this design."),
            ("17 vs 33 depth planes", f"{100*depth_diff:.4f}%", "Depth trapezoid is converged below 0.1%."),
            ("17 vs 33 wavelengths", f"{100*spectrum_diff:.4f}%", "Spectral trapezoid is converged well below 0.1%."),
            ("Vector Snell residual", f"maximum {snell_max:.3g}", "Refraction satisfies n*sin(theta) to floating-point precision."),
            ("STL topology", "31,104 triangles; every edge used twice", "Export is closed and watertight."),
            ("Tilt symmetry", "+/- tilt agrees within numerical tolerance", "Sixfold tracing behaves consistently."),
        ], [42, 42, 79]),
        Paragraph("Executable checks", STYLES["h1"]),
        code("""
python test_physics.py
python test_sensitivity.py
python test_lens_design.py
python fab_check.py
python validate_model.py
""", STYLES),
        Paragraph("These checks prove implementation consistency, not physical truth", STYLES["h1"]),
        table(["Not established", "Required next evidence"], [
            ("Absolute photon prediction", "Measure input power, unattenuated linear counts, transmission/QE and background."),
            ("Freeform wave optics", "Independent Zemax/Code V physical-optics propagation or FDTD/FEM benchmark."),
            ("Fabricated surface fidelity", "Profilometer/confocal map registered to STL, including shrinkage and roughness."),
            ("Contrast model", "Measured background and ODMR contrast versus optical footprint and MW settings."),
            ("Linewidth model", "Consistent measured/computed resolution metric for all four probes."),
            ("Global optimum", "Continuous optimization or denser parameter study with manufacturing constraints."),
        ], [63, 100]),
    ]

    story += page("14. Complete end-to-end pseudocode and rerun checklist")
    story += [
        code("""
INPUTS
    diamond: 100 um; NV layer 80-90 um; density 3 ppm
    fibers: SM 4/0.12, MM 50/0.22, MCF 35 um core radius and 10 um MFD
    pump: 10 mW at 532 nm; R_sat and I_sat
    old lens geometry: paper + spreadsheet + supplied STL/GWL

BASELINE COMPARISON
    construct each fiber configuration
    sample single NV and importance-weighted 3-D NV ensemble
    for every gap:
        calculate saturated excitation at each emitter
        ray-trace collection with Snell, Fresnel, TIR and fiber coupling
        integrate single-NV rate, ensemble power, efficiency and A50
    repeat single-NV collection versus wavelength
    write baseline figures

NEW LENS DESIGN
    fit candidate central and one-side polynomial surfaces to Snell normals
    replicate side surface sixfold and treat overlaps as one exterior union
    screen discrete geometry candidates with 9x9 quadrature
    select best candidate by raw-model sensitivity
    recompute winner with 33 depths, 33 wavelengths and fine lateral grid
    report raw fiber photons and separately normalized comparison rate
    export one-side and complete watertight STLs

ALIGNMENT AND REPORTING
    translate fixed tip through gap sweep
    rotate diamond plane through angle sweep
    generate static and interactive ray figures
    run convergence, Snell and STL checks
""", STYLES),
        Paragraph("Recommended rerun order", STYLES["h1"]),
        code("""
python paper_figures.py
python phase3_optimize.py
python redesign_fig.py --angle-deg 0
python validate_model.py
python test_sensitivity.py
python test_lens_design.py
python fab_check.py
""", STYLES),
        Paragraph("Review checklist before using a number in a paper", STYLES["h1"]),
        table(["Question", "Required answer"], [
            ("Is this baseline A50 or redesigned Gaussian FWHM?", "Name the metric explicitly; do not silently compare them."),
            ("Is the rate fiber photons or detector counts?", "Use model_fiber_photons_s for optics; label normalized cps as empirical."),
            ("Was contrast predicted?", "No. It is the measured old-MCF contrast unless a new model/data set is supplied."),
            ("Was the final design experimentally validated?", "No. Treat it as a fabrication target."),
            ("Were all physical interfaces counted once?", "Yes in the corrected new model; preserve this when editing."),
            ("Did geometry or spectral inputs change?", "Rerun optimization, figures and validation together."),
        ], [77, 86]),
        Spacer(1, 3*mm),
        Paragraph("The most important next correction for scientific comparability is to use one common resolution definition and one common spectral window for SM, MM, old MCF and new MCF. The most important experimental validation is an absolute optical-throughput measurement of the fabricated full STL at controlled gap and tilt.", STYLES["callout"]),
    ]

    doc = SimpleDocTemplate(
        PDF, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
        topMargin=19*mm, bottomMargin=16*mm,
        title="NV-fiber simulation code and calculation walkthrough",
        author="NV-diamond fiber simulation")
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    if os.path.getsize(PDF) < 20_000:
        raise RuntimeError("walkthrough PDF is unexpectedly small")
    return PDF


STYLES = styles()
for name in ("subtitle", "body", "small", "caption"):
    STYLES[name].fontName = "Arial"
for name in ("title", "h1", "h2", "callout"):
    STYLES[name].fontName = "Arial-Bold"
STYLES["mono"].fontName = "Consolas"


if __name__ == "__main__":
    print(build())
