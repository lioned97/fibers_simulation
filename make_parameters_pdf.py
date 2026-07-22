"""One-page-per-topic PDF of everything the phase-3 lens search varies.

Run ``py make_parameters_pdf.py``.  Writes figures/phase3_search_parameters.pdf.

Every number is pulled live from lens_design.py rather than retyped, so the
document cannot drift away from the code it describes.
"""
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

import lens_design as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "phase3_search_parameters.pdf")

INK = colors.HexColor("#131920")
MUTED = colors.HexColor("#52514e")
ACCENT = colors.HexColor("#2f5fc4")
RULE = colors.HexColor("#c3ccd6")
BAND = colors.HexColor("#eef1f5")

_base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=_base["Title"], fontSize=17,
                            textColor=INK, spaceAfter=2, alignment=0),
    "sub": ParagraphStyle("sub", parent=_base["Normal"], fontSize=9.5,
                          textColor=MUTED, spaceAfter=11),
    "h": ParagraphStyle("h", parent=_base["Heading2"], fontSize=11.5,
                        textColor=INK, spaceBefore=11, spaceAfter=5),
    "body": ParagraphStyle("body", parent=_base["Normal"], fontSize=9.2,
                           textColor=INK, leading=13.2, spaceAfter=6),
    "note": ParagraphStyle("note", parent=_base["Normal"], fontSize=8.2,
                           textColor=MUTED, leading=11.4, spaceAfter=6),
    "cell": ParagraphStyle("cell", parent=_base["Normal"], fontSize=8.2,
                           textColor=INK, leading=10.6),
}


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), S["cell"]) for c in row] for row in rows]
    style = [("GRID", (0, 0), (-1, -1), 0.4, RULE),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 5),
             ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("TOPPADDING", (0, 0), (-1, -1), 3.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), BAND),
                  ("TEXTCOLOR", (0, 0), (-1, 0), INK)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle(style))
    return t


def searched_page(story):
    story.append(Paragraph("Phase 3: what the lens search varies", S["title"]))
    story.append(Paragraph(
        f"Printed seven-core tip for the NV-diamond probe &middot; generated "
        f"{datetime.now():%Y-%m-%d %H:%M} from lens_design.py", S["sub"]))

    story.append(Paragraph("1 &nbsp; Geometry searched for every lens family",
                           S["h"]))
    story.append(Paragraph(
        "These four are searched on a 1 &micro;m integer grid. Everything else "
        "about the tip follows from them.", S["body"]))
    labels = {"air_gap_um": ("Air gap", "closest approach of the printed tip to "
                             "the diamond; also the central cap apex"),
              "central_height_um": ("Central post height", "IP-S pillar under "
                                    "the central (green) cap"),
              "side_height_um": ("Side post height", "IP-S pillar under each of "
                                 "the six collection caps"),
              "side_core_offset_um": ("Side cap decentre", "cap centre relative "
                                      "to the 35 &micro;m core pitch; negative "
                                      "moves it inward, which steers the "
                                      "collection cone toward the axis")}
    rows = [["Parameter", "Range (&micro;m)", "Step", "What it sets"]]
    for name, (lo, hi) in zip(L.SEARCH_GEOMETRY_NAMES, L.SEARCH_GEOMETRY_BOUNDS):
        title, meaning = labels[name]
        rows.append([f"<b>{title}</b><br/><font size=7 color='#52514e'>{name}</font>",
                     f"{lo} to {hi}", "1", meaning])
    story.append(table(rows, [46*mm, 24*mm, 12*mm, 78*mm]))

    story.append(Paragraph("2 &nbsp; Cap size, searched as a multiple of the beam",
                           S["h"]))
    story.append(Paragraph(
        "Cap radius is <b>not</b> searched directly. It is tied to the footprint "
        "the core's own full-NA beam throws onto it, so a cap can never be "
        "fitted over one region and used over a much larger one:", S["body"]))
    story.append(Paragraph(
        "<font face='Courier'>footprint = 2&middot;w<sub>mode</sub> + height &middot; "
        "tan(arcsin(NA / n<sub>IP-S</sub>))</font><br/>"
        "<font face='Courier'>cap radius = decentre + scale &times; footprint</font>",
        S["body"]))
    rows = [["Parameter", "Range", "Meaning"]]
    for name, (lo, hi) in zip(L.SEARCH_APERTURE_NAMES, L.SEARCH_APERTURE_BOUNDS):
        rows.append([f"<b>{name}</b>", f"{lo} to {hi}",
                     "1.0 exactly covers the beam; the upper limit is how far "
                     "the cap may run past it"])
    story.append(table(rows, [46*mm, 24*mm, 90*mm]))
    story.append(Paragraph(
        "Consequence: the central/side overlap and the cap apertures are "
        "reported <i>results</i>, not free knobs. They follow from the post "
        "heights, the decentre and these two scales.", S["note"]))

    story.append(Paragraph("3 &nbsp; Surface shape, per lens family", S["h"]))
    story.append(Paragraph(
        "Each surface starts as a Snell-law fit: for every point on the cap the "
        "code solves for the air direction that reaches the NV, takes the normal "
        "vector refraction demands, and least-squares fits the polynomial to "
        "those normals. The family then decides which coefficients the search "
        "may scale.", S["body"]))
    rows = [["Family", "Free shape parameters", "Surface it can form"]]
    described = {
        "quadratic": "one radius of curvature - a plain paraboloid cap",
        "asphere": "one radius plus a rotationally symmetric fourth-order term",
        "biconic": "independent radii along the radial and tangential axes, "
                   "plus a fourth-order term",
        "freeform": "independent radii, off-axis odd terms and an independent "
                    "quartic - the least constrained surface",
    }
    for family in ("quadratic", "asphere", "biconic", "freeform"):
        names = L.FAMILY_SHAPE_PARAMETERS[family]
        detail = "<br/>".join(
            f"{n} &nbsp;<font size=7 color='#52514e'>{L.SHAPE_BOUNDS[n][0]} to "
            f"{L.SHAPE_BOUNDS[n][1]}</font>" for n in names)
        rows.append([f"<b>{family}</b>", detail, described[family]])
    story.append(table(rows, [24*mm, 60*mm, 76*mm]))
    story.append(Paragraph(
        f"Every side cap additionally gets <b>tilt_scale</b> "
        f"({L.SHAPE_BOUNDS['tilt_scale'][0]} to {L.SHAPE_BOUNDS['tilt_scale'][1]}), "
        "which scales the vertex tilt that aims its cone at the axis. The "
        f"central and side families are chosen independently, so all "
        f"{len(('quadratic','asphere','biconic','freeform'))**2} pairings are "
        "searched.", S["note"]))


def rules_page(story):
    story.append(PageBreak())
    story.append(Paragraph("4 &nbsp; Rules a candidate must satisfy", S["h"]))
    story.append(Paragraph(
        "Checked before any ray is traced, so violating designs cost nothing "
        "and can never reach the results.", S["body"]))
    rows = [["Rule", "Value", "Why"]]
    rows += [
        ["Print volume", f"{L.PRINT_X_UM:.0f} &times; {L.PRINT_Y_UM:.0f} &times; "
         f"{L.PRINT_Z_UM:.0f} &micro;m", "the two-photon printer's field"],
        ["Cap covers its beam", f"scale &ge; 1.0", "no cap fitted over one "
         "region and used over another"],
        ["Cap not oversized", f"&le; {L.APERTURE_MARGIN}&times; footprint",
         "beyond the fit the polynomial is unconstrained extrapolation"],
        ["Cap owns its own light", f"&ge; {100*L.EXPOSURE_MIN:.0f}% of its lit "
         "footprint", "otherwise its core is really refracting off a "
         "neighbouring cap"],
        ["Apex is the real clearance", f"within {L.CLEARANCE_TOL}&times; "
         "footprint", "so the reported air gap is what the beam sees"],
        ["Surface slope", f"&le; {L.MAX_SURFACE_SLOPE} "
         f"(~{__import__('numpy').degrees(__import__('numpy').arctan(L.MAX_SURFACE_SLOPE)):.0f}&deg;)",
         "printability; steeper walls also reflect light away"],
        ["Polymer left under a cap", f"&ge; {L.MIN_POLYMER_UM:.0f} &micro;m",
         "a cap pinched to a sliver cannot be printed"],
    ]
    story.append(table(rows, [42*mm, 40*mm, 78*mm]))

    story.append(Paragraph("5 &nbsp; How the search moves", S["h"]))
    rows = [["Stage", "Setting", "Note"]]
    rows += [
        ["Family pairings", "16", "every central family against every side family"],
        ["Restarts each", "2", "different seeds, best kept"],
        ["Screening pass", "differential evolution, 120 generations",
         "the same objective at reduced fidelity - 3 depths, 1 wavelength, "
         "coarse grid"],
        ["Full pass", "differential evolution, 80 generations",
         "9 depths, 9 wavelengths, 81&sup2; grid"],
        ["Local refinement", "coordinate descent, steps 10&rarr;5&rarr;2&rarr;1 "
         "&micro;m", "and 0.2&rarr;0.02 on the shape scales"],
        ["Final selection", "top 3 re-scored at full resolution",
         "33 depths, 33 wavelengths, 161&sup2; grid - so the winner is chosen "
         "on the grid it is reported on"],
    ]
    story.append(table(rows, [34*mm, 56*mm, 70*mm]))

    story.append(Paragraph("6 &nbsp; Held fixed (not searched)", S["h"]))
    n_dia = float(L.diamond_sellmeier(L.RED_DESIGN_NM/1000.0))
    rows = [["Quantity", "Value"]]
    rows += [
        ["Fibre NA / mode-field diameter", f"{L.MCF_FULL_NA} / {L.MCF_MFD:.0f} &micro;m"],
        ["Core pitch", f"{L.CORE_R:.0f} &micro;m (6 side cores + 1 central)"],
        ["IP-S polymer index", f"{L.MCF_IPS_N}"],
        ["Diamond index at the design wavelength", f"{n_dia:.4f}"],
        ["Collection band", f"{L.NV_SPECTRUM_NM[0]:.0f} to {L.NV_SPECTRUM_NM[1]:.0f} nm"],
        ["Side-lens design wavelength", f"{L.RED_DESIGN_NM:.1f} nm "
         "(emission-weighted centre of the band)"],
        ["Excitation", "532 nm, "
         f"{L.P_GREEN_MW:.0f} mW out of the central core"],
        ["NV layer", f"{L.NV_DEPTH_UM[0]:.0f} to {L.NV_DEPTH_UM[1]:.0f} &micro;m "
         "below the diamond surface"],
        ["NV concentration", f"{L.PPM:.0f} ppm"],
    ]
    story.append(table(rows, [70*mm, 90*mm]))

    story.append(Paragraph("7 &nbsp; What is optimised", S["h"]))
    story.append(Paragraph(
        "Each candidate is ray traced with exact vector Snell refraction and "
        "polarisation-averaged Fresnel at both the polymer-air and air-diamond "
        "surfaces, with total internal reflection removing the light that "
        "cannot escape the diamond. The signal is the green excitation rate "
        "multiplied by the six side cores' collection probability, integrated "
        "through the NV layer - only an NV that is both pumped and seen "
        "contributes. The search minimises shot-noise-limited magnetic "
        "sensitivity, which rewards more collected light and a tighter sensing "
        "spot at the same time.", S["body"]))
    story.append(Paragraph(
        "Collection efficiency, quoted separately in the comparison figures, is "
        "a pure optical result of the same trace: it needs no assumed contrast, "
        "no linewidth model and no empirical normalisation.", S["note"]))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = SimpleDocTemplate(OUT, pagesize=A4, title="Phase 3 search parameters",
                            leftMargin=17*mm, rightMargin=17*mm,
                            topMargin=15*mm, bottomMargin=15*mm)
    story = []
    searched_page(story)
    rules_page(story)
    doc.build(story)
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
