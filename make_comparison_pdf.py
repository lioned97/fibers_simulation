"""Build the measured-vs-model fiber comparison PDF."""
import json
import os
import argparse
import hashlib

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle, Preformatted)


ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
TMP = os.path.join(ROOT, "tmp", "pdfs")
OUT = os.path.join(ROOT, "output", "pdf")
PDF = os.path.join(OUT, "sm_mm_mcf_comparison_and_replication_guide.pdf")

NAVY = colors.HexColor("#17324D")
PALE = colors.HexColor("#EEF3F7")
GRID = colors.HexColor("#C9D2D9")
INK = colors.HexColor("#263238")
MUTED = colors.HexColor("#5F6B73")


def load_data():
    with open(os.path.join(FIG, "mcf_freeform_design.json"), encoding="utf-8") as fh:
        design = json.load(fh)
    with open(os.path.join(FIG, "mcf_gap_angle_sweep.json"), encoding="utf-8") as fh:
        alignment = json.load(fh)
    measured = design["measured"]
    new = design["result"]
    rows = [
        dict(name="SM", status="Measured", **measured["SM"]),
        dict(name="MM", status="Measured", **measured["MM"]),
        dict(name="MCF old", status="Measured", **measured["MCF"]),
        dict(name="MCF new", status="Normalized model", cps=new["comparison_normalized_cps"],
             contrast=new["contrast"], fwhm_mhz=new["fwhm_mhz"],
             sensitivity_nt=new["comparison_normalized_sensitivity_nt"],
             raw_model_photons_s=new["model_fiber_photons_s"],
             raw_model_sensitivity_nt=new["raw_model_sensitivity_nt"],
             snr=None, od=None),
    ]
    return design, alignment, rows


def comparison_chart(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = os.path.join(TMP, "comparison_metrics.png")
    labels = [r["name"] for r in rows]
    palette = ["#2A78D6", "#1BAF7A", "#D78500", "#D7263D"]
    specs = [
        ("Corrected / projected photon rate", "cps", [r["cps"] for r in rows], True),
        ("Shot-noise sensitivity (lower is better)", "nT/sqrt(Hz)",
         [r["sensitivity_nt"] for r in rows], True),
        ("ODMR contrast", "%", [100*r["contrast"] for r in rows], False),
        ("ODMR FWHM", "MHz", [r["fwhm_mhz"] for r in rows], False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.45))
    for ax, (title, unit, values, log) in zip(axes.flat, specs):
        bars = ax.bar(labels, values, color=palette, width=0.68)
        bars[-1].set_hatch("///"); bars[-1].set_edgecolor("#7A1421")
        if log:
            ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=9, fontweight="semibold")
        ax.set_ylabel(unit, fontsize=8)
        ax.grid(axis="y", color="#D6DDE2", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelrotation=17, labelsize=7.5)
        ax.tick_params(axis="y", labelsize=7.5)
        for bar, value in zip(bars, values):
            label = f"{value:.2g}" if log else f"{value:.2f}"
            ax.annotate(label, (bar.get_x()+bar.get_width()/2, value),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.008,
             "Solid bars: measured experiment. Hatched bar: empirically normalized model target.",
             ha="center", fontsize=7.5, color="#4F5B62")
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.25, w_pad=1.1)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=23, leading=27, textColor=NAVY,
                                alignment=TA_LEFT, spaceAfter=5*mm),
        "subtitle": ParagraphStyle("Subtitle", parent=base["BodyText"],
                                   fontName="Helvetica", fontSize=10.5, leading=15,
                                   textColor=MUTED, spaceAfter=5*mm),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold",
                             fontSize=15, leading=18, textColor=NAVY,
                             spaceBefore=2*mm, spaceAfter=2.5*mm),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=11, leading=14, textColor=NAVY,
                             spaceBefore=1.5*mm, spaceAfter=1.5*mm),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica",
                               fontSize=8.7, leading=12.2, textColor=INK,
                               spaceAfter=2.2*mm),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica",
                                fontSize=7.3, leading=9.5, textColor=MUTED),
        "callout": ParagraphStyle("Callout", parent=base["BodyText"], fontName="Helvetica-Bold",
                                  fontSize=9.2, leading=13, textColor=NAVY,
                                  backColor=PALE, borderPadding=8, spaceAfter=4*mm),
        "caption": ParagraphStyle("Caption", parent=base["BodyText"], fontName="Helvetica",
                                  fontSize=7.2, leading=9.2, textColor=MUTED,
                                  alignment=TA_CENTER, spaceBefore=1.2*mm),
        "mono": ParagraphStyle("Mono", parent=base["Code"], fontName="Courier",
                               fontSize=7.2, leading=9.6, textColor=INK,
                               backColor=PALE, borderPadding=7, spaceAfter=3*mm),
    }


def data_table(rows, s):
    head = ["Probe", "Evidence", "Photon rate*", "Contrast", "FWHM", "Sensitivity"]
    body = [head]
    for r in rows:
        body.append([
            r["name"], r["status"], f"{r['cps']:.3g} cps",
            f"{100*r['contrast']:.2f}%", f"{r['fwhm_mhz']:.2f} MHz",
            f"{r['sensitivity_nt']:.2f} nT/sqrt(Hz)",
        ])
    table = Table(body, colWidths=[18*mm, 29*mm, 32*mm, 22*mm, 22*mm, 40*mm],
                  repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#FDECEF")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("LEADING", (0, 0), (-1, -1), 9),
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return [table, Spacer(1, 1.2*mm),
            Paragraph("* Measured rates are corrected for ND attenuation. The new-MCF entry is a comparison-normalized count-rate proxy, not a physical detector conversion. Its separate raw output is reported below.", s["small"])]


def key_value_table(items, s):
    table = Table([[Paragraph(k, s["body"]), Paragraph(v, s["body"])]
                   for k, v in items], colWidths=[45*mm, 118*mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def page_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID); canvas.setLineWidth(0.5)
    canvas.line(18*mm, height-13*mm, width-18*mm, height-13*mm)
    canvas.setFont("Helvetica", 7); canvas.setFillColor(MUTED)
    canvas.drawString(18*mm, height-10*mm, "NV-diamond fiber probe comparison")
    canvas.drawRightString(width-18*mm, 10*mm, f"Page {doc.page}")
    canvas.restoreState()


def image_fit(path, width_mm, max_height_mm):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    width = width_mm*mm
    height = min(max_height_mm*mm, width*h/w)
    width = height*w/h
    return Image(path, width=width, height=height)


def build_pdf(generate_chart=True):
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    design, alignment, rows = load_data()
    chart = (comparison_chart(rows) if generate_chart else
             os.path.join(TMP, "comparison_metrics.png"))
    if not os.path.exists(chart):
        raise FileNotFoundError("Generate the chart first with --chart-only")
    s = styles()
    doc = SimpleDocTemplate(PDF, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=19*mm, bottomMargin=16*mm,
                            title="SM vs MM vs old and new MCF comparison",
                            author="NV-diamond fiber simulation")
    story = []

    story += [Paragraph("SM vs MM vs MCF optical probe comparison", s["title"]),
              Paragraph("Measured SM, MM and original MCF performance compared with the new monolithic seven-lens MCF model", s["subtitle"]),
              Paragraph("Evidence boundary: SM, MM and MCF old are measured experimental values. MCF new is a shot-noise-limited optical-model prediction and has not yet been experimentally validated. Raw fiber photons and the empirical comparison normalization are reported separately.", s["callout"]),
              image_fit(chart, 174, 105), Spacer(1, 2*mm),
              Paragraph("Primary comparison", s["h1"])]
    story += data_table(rows, s)

    old, new = rows[2], rows[3]
    story += [Paragraph("Main result", s["h2"]),
              Paragraph(f"Under the explicitly labeled empirical normalization, the new design gives a {new['cps']/old['cps']:.1f}x count-rate proxy and {old['sensitivity_nt']/new['sensitivity_nt']:.1f}x lower shot-noise sensitivity than the measured original MCF, while retaining the measured 8.10% MCF contrast. The physical optical-model output is {new['raw_model_photons_s']:.3g} fiber photons/s, corresponding to {new['raw_model_sensitivity_nt']:.2f} nT/sqrt(Hz) if one fiber photon is treated as one detected count. The measured MM remains the best demonstrated sensitivity at 29 nT/sqrt(Hz).", s["body"]),
              PageBreak()]

    central, side = design["central"], design["side"]
    story += [Paragraph("New MCF design and optical model", s["title"]),
              key_value_table([
                  ("Printed structure", "One monolithic IP-S body (n = 1.52): central lens plus six side lenses on a common support"),
                  ("Central channel", f"532 nm excitation; {central['family']} polynomial surface; {central['aperture']:.0f} um aperture; {central['base_z']-central['apex']:.0f} um protrusion"),
                  ("Side channels", f"650-850 nm collection; six {side['family']} polynomial surfaces; {side['aperture']:.0f} um aperture; 35 um core pitch; 10 um MFD"),
                  ("Best working gap", f"{alignment['best_gap_um']:.1f} um central-lens-to-diamond clearance for the fixed optimized tip"),
                  ("NV ensemble", "3 ppm NV centers throughout the 80-90 um layer in a 100 um diamond"),
                  ("Optical physics", "3-D rays, local surface normals, vector Snell law, each Fresnel interface applied once, wavelength-dependent diamond index, saturation and excitation-collection overlap"),
                  ("Final quadrature", f"{design['assumptions']['final_depth_points']} NV-depth planes and {design['assumptions']['final_spectral_points']} trapezoid-weighted wavelengths"),
              ], s), Spacer(1, 3*mm),
              image_fit(os.path.join(FIG, "fig10_full_mcf_3d_raytrace.png"), 120, 108),
              Paragraph("Figure 2. Complete seven-core printed-lens union. Green paths are 532 nm excitation rays. Red paths are reciprocal 750 nm collection rays. Refraction occurs only at the polymer-air and air-diamond boundaries.", s["caption"]),
              Spacer(1, 2*mm),
              Paragraph("Sensitivity model", s["h2"]),
              Paragraph(f"Both outputs use eta_new = 103 x (FWHM_new / 3.44) x (0.081 / C_new) x sqrt(2e9 / R_new), with eta in nT/sqrt(Hz). For the raw optical result, R is {design['result']['model_fiber_photons_s']:.3g} fiber photons/s. For comparison with the measurements, R is the explicitly normalized proxy {design['result']['comparison_normalized_cps']:.3g} cps. The normalization factor is {design['comparison_normalization']['factor']:.4g}; because it exceeds one, it is not interpreted as detector efficiency.", s["body"]),
              PageBreak()]

    gap_peak = max(row["model_fiber_photons_s"] for row in alignment["gap"])
    near_10 = min(alignment["gap"], key=lambda row: abs(row["gap_um"]-10.0))
    zero_angle = min(alignment["angle"], key=lambda row: abs(row["angle_deg"]))
    def angle_retention(target):
        row = min(alignment["angle"], key=lambda q: abs(q["angle_deg"]-target))
        return row["angle_deg"], (row["model_fiber_photons_s"] /
                                  zero_angle["model_fiber_photons_s"])
    angle_5, retain_5 = angle_retention(5.0)
    angle_10, retain_10 = angle_retention(10.0)

    story += [Paragraph("Mounting tolerance and interpretation", s["title"]),
              image_fit(os.path.join(FIG, "fig11_gap_and_angle_tolerance.png"), 174, 82),
              Paragraph("Figure 3. Fixed-lens Z-distance response and diamond-plane angular response. At nonzero tilt, all six side lenses are traced independently; the diamond and the 80-90 um NV layer rotate together about the fiber y-axis.", s["caption"]),
              Spacer(1, 3*mm),
              key_value_table([
                  ("Z sensitivity", f"The sampled fixed-tip optimum is {alignment['best_gap_um']:.2f} um. At the sampled point nearest 10 um ({near_10['gap_um']:.2f} um), modeled fiber fluorescence is {100*near_10['model_fiber_photons_s']/gap_peak:.1f}% of the peak."),
                  ("Angular sensitivity", f"At the sampled angles {angle_5:.2f} and {angle_10:.2f} degrees, predicted fiber-photon retention is {100*retain_5:.2f}% and {100*retain_10:.2f}%, respectively."),
                  ("Measured ordering", "SM and old MCF both give about 2e9 corrected cps. MM gives about 2e10 corrected cps and is therefore 10x higher in the supplied experiment."),
                  ("Raw optical output", f"{new['raw_model_photons_s']:.3g} fiber photons/s and {new['raw_model_sensitivity_nt']:.2f} nT/sqrt(Hz) under the one-photon/one-count convention."),
                  ("Normalized comparison", f"{new['cps']:.3g} cps proxy, {100*new['contrast']:.2f}% contrast, {new['fwhm_mhz']:.3f} MHz FWHM and {new['sensitivity_nt']:.2f} nT/sqrt(Hz)."),
              ], s), Spacer(1, 3*mm),
              Paragraph("Limits of the comparison", s["h1"]),
              Paragraph("The new-design fiber-photon rate is an ideal smooth-surface geometrical-optics projection. The separate empirical normalization references the legacy old-MCF model checkpoint and must not be read as detector efficiency. The model does not include wave-optical diffraction at detailed freeform features, detector saturation or dead time, fabrication roughness and shrinkage, alignment drift, real wavelength-dependent core modes, fluorescence background, filter transmission or detector quantum efficiency. No background-derived contrast penalty is applied because no measured background-versus-footprint calibration was supplied.", s["body"]),
              Paragraph("Accordingly, compare the new MCF with the measured probes as a design target and expected trend, not as a demonstrated experimental result. The next decisive test is to fabricate the full STL, measure the unattenuated linear photon range, and repeat contrast/FWHM/sensitivity measurements at controlled gap and tilt.", s["callout"]),
              Paragraph("Reproducibility", s["h2"]),
              Paragraph("Inputs: figures/mcf_freeform_design.json and figures/mcf_gap_angle_sweep.json. Geometry: figures/mcf_freeform_full_seven_core.stl. Regenerate design figures with 'python redesign_fig.py --angle-deg ANGLE'. Generate this report with 'python make_comparison_pdf.py'.", s["small"])]

    support_radius = max(60.0, central["aperture"]+6.0,
                         side["center_r"]+side["aperture"]+6.0)
    story += [PageBreak(),
              Paragraph("Lens replication drawing", s["title"]),
              Paragraph("Use the full seven-core STL as the authoritative fabrication geometry. The drawing below is for alignment and metrology; do not reconstruct the surface from the drawing alone.", s["callout"]),
              image_fit(os.path.join(FIG, "fig12_mcf_fabrication_blueprint.png"), 174, 92),
              Paragraph("Figure 4. Fabrication coordinates use the fiber facet as Zf = 0 and positive Zf outward toward the diamond. Transparent aperture circles show where lens domains overlap; the printed result is one continuous height union.", s["caption"]),
              Spacer(1, 2*mm),
              key_value_table([
                  ("Authoritative file", "figures/mcf_freeform_full_seven_core.stl; binary STL; 31,104 triangles; one watertight body"),
                  ("STL units", "The STL is unitless but every coordinate is in micrometers. Select um on import. If the CAD system assumes millimeters, apply scale 0.001."),
                  ("Fiber coordinates", "Central core at (0,0). Six side cores at radius 35 um and azimuths 0, 60, 120, 180, 240 and 300 degrees. All cores have 10 um MFD."),
                  ("Support", f"Circular footprint diameter {2*support_radius:.0f} um; 8 um support thickness at the fiber facet."),
                  ("Central lens", f"Centered on the central core; aperture diameter {2*central['aperture']:.0f} um; maximum outward height 120 um."),
                  ("Side lenses", f"Six rotated copies; aperture diameter {2*side['aperture']:.0f} um; aperture centers coincide with the six side cores."),
                  ("Assembly spacing", "Place the diamond surface 125 um from the fiber facet, giving a 5 um gap from the 120 um central tip. The NV layer is then 80-90 um inside the diamond."),
                  ("Overlap rule", "All lens and support domains are fused as one polymer body. Where surfaces overlap, keep the most outward surface; never add their heights and never preserve an internal optical boundary."),
              ], s)]

    central_coef = ", ".join(f"{value:.10g}" for value in central["coef"])
    side_coef = ", ".join(f"{value:.10g}" for value in side["coef"])
    equation = """For each lens, xi = u/a and nu = v/a, with u^2 + v^2 <= a^2.
S(u,v) = a*[p0*xi + 0.5*p1*xi^2 + 0.5*p2*nu^2
          + (p3/3)*xi^3 + p4*xi*nu^2 + (p5/4)*xi^4
          + 0.5*p6*xi^2*nu^2 + (p7/4)*nu^4] + shift
Hj(u,v) = base_z - apex_j - S(u,v)
H(x,y) = max(8, Hcentral, Hside_0, ... Hside_5)
Printed volume: x^2+y^2 <= 65^2 and 0 <= Zf <= H(x,y)."""
    stl_path = os.path.join(FIG, "mcf_freeform_full_seven_core.stl")
    with open(stl_path, "rb") as fh:
        stl_sha256 = hashlib.sha256(fh.read()).hexdigest().upper()
    story += [PageBreak(),
              Paragraph("Exact CAD surface definition", s["title"]),
              Paragraph("This equation reproduces the same monolithic height field used by the STL generator. Distances are in micrometers. The STL remains preferable because it also fixes tessellation, footprint and watertight closure.", s["body"]),
              Preformatted(equation, s["mono"]),
              key_value_table([
                  ("Coordinate transform", "The simulation uses z_model = 125 - Zf. The fiber-facet plane is z_model = 125; the diamond-facing direction is decreasing z_model."),
                  ("Central local frame", "u=x, v=y; a=20; apex=5; base_z=125; shift=0."),
                  ("Central coefficients", f"p0...p7 = [{central_coef}]"),
                  ("Side local frames", "For k=0...5, phi=k*60 deg, center=35*(cos(phi),sin(phi)); u=(r-center).(cos(phi),sin(phi)); v=(r-center).(-sin(phi),cos(phi))."),
                  ("Side parameters", f"a=24; apex=75; base_z=125; shift={side['shift']:.12g}."),
                  ("Side coefficients", f"p0...p7 = [{side_coef}]"),
              ], s), Spacer(1, 3*mm),
              Paragraph("Artifact identity", s["h1"]),
              Paragraph(f"Full STL SHA-256: {stl_sha256}", s["small"]),
              Paragraph("Prototype file figures/mcf_freeform_central_one_side.stl contains only the central lens and one side domain. Use the full seven-core file for fabrication.", s["body"]),
              Paragraph("Important orientation check", s["h1"]),
              Paragraph("Before slicing, display the fiber attachment plane and the lens tips in the machine preview. The flat disk at z_model=125 attaches to the fiber; the central tip is at z_model=5 and must project away from the fiber. The original job used InvertZAxis 1, but that command is machine-coordinate dependent and must not replace a visual orientation check.", s["callout"])]

    process_rows = [
        ("Printer/objective", "Nanoscribe Photonics Professional GT; 25X objective"),
        ("Photoresist", "IP-S photopolymer; paper reports n approximately 1.5"),
        ("Slicing/hatching", "100 nm slicing distance; 200 nm hatching distance"),
        ("Exposure/delay", "Laser power 33%; 3 ms delay after every written layer"),
        ("Writing mode", "GalvoScanMode; ContinuousMode"),
        ("Speeds", "Support ScanSpeed 50000; lens ScanSpeed 10000"),
        ("Stage/control", "PiezoSettlingTime 10; GalvoAcceleration 10; StageVelocity 200; PowerScaling 1.0"),
        ("Original orientation", "InvertZAxis 1"),
    ]
    story += [PageBreak(),
              Paragraph("Fabrication and verification procedure", s["title"]),
              Paragraph("The geometry is complete; the exposure recipe is not universally transferable. Follow this sequence and calibrate the printer/polymer combination with a coupon before writing on the MCF.", s["callout"]),
              Paragraph("1. Verify the fiber", s["h2"]),
              Paragraph("Image the facet and fit the seven core centers. Confirm one central core and six cores on a 35 um radius. Use the measured central core as the x=y=0 registration point; use two or more side cores to determine rotation and scale.", s["body"]),
              Paragraph("2. Import and register the full STL", s["h2"]),
              Paragraph("Import in micrometers, preserve aspect ratio, and align the STL central axis to the central core. Rotate about that axis until every side-lens center coincides with its side core. The support disk may be cropped only after confirming it remains fully attached to the actual fiber facet.", s["body"]),
              Paragraph("3. Slice as one monolithic solid", s["h2"]),
              Paragraph("Union the geometry before slicing. Do not slice seven independent lenses and do not reuse the old AddZOffset 134.04 or 24.77 commands: the new STL already contains the complete relative heights. Start from the paper's 100 nm slice and 200 nm hatch settings, confirm them with an IP-S shrinkage coupon, then inspect the toolpath for uninterrupted support-to-lens attachment and no duplicate exposure in overlap regions.", s["body"]),
              Paragraph("4. Reported paper and DeScribe starting settings", s["h2"]),
              key_value_table(process_rows, s),
              Paragraph("The printer, objective, slice, hatch, power and inter-layer delay are reported in the supplied paper. The remaining controls and scan speeds come from Cylinder_lenses_job.gwl. LaserPower 33% is still instrument-specific and must be verified for the actual objective, resin batch, interface and laser calibration.", s["small"]),
              Paragraph("5. Develop, inspect and mount", s["h2"]),
              Paragraph("Use a validated IP-S development, rinse and drying procedure and record the exact solvents and times; the supplied paper states only that unpolymerized material was removed in solvent and the structure was dried. Inspect for delamination, voids and stair-stepping. Measure the 130 um footprint, 8 um support, 120 um central height, 40 um central aperture, 48 um side apertures and 35 um side-core radius. Compare a profilometer or confocal surface scan directly with the STL, especially across overlap seams.", s["body"]),
              Paragraph("6. Optical acceptance test", s["h2"]),
              Paragraph("With the diamond parallel to the fiber, start at a 5 um central-tip gap. Sweep Z before comparing count rates: the model falls to about 90% at 5.97 um and about 4.3% near 10.15 um. Verify central-core 532 nm delivery first, then inject or collect 650-850 nm light through each side core separately before summing all six.", s["body"]),
              Paragraph("Missing process data", s["h2"]),
              Paragraph("The supplied paper does not identify the commercial MCF part number or cladding diameter, nor the development solvent/time, interface-finding method, shrinkage compensation, acceptable surface roughness or calibrated absolute laser dose for this new STL. Record those values from the actual fiber and a successful calibration coupon before another laboratory attempts process-level replication. The CAD geometry itself is fully specified here.", s["callout"])]

    doc.build(story, onFirstPage=page_header_footer, onLaterPages=page_header_footer)
    return PDF


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart-only", action="store_true")
    parser.add_argument("--pdf-only", action="store_true")
    args = parser.parse_args()
    if args.chart_only:
        os.makedirs(TMP, exist_ok=True)
        print(comparison_chart(load_data()[2]))
        raise SystemExit
    path = build_pdf(generate_chart=not args.pdf_only)
    assert os.path.getsize(path) > 100_000
    print(path)
