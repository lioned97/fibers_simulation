"""What an NV actually emits, and how little of it ever leaves the diamond.

Run ``py figure_nv_emission.py``.  Writes figures/comparison_collection/.

Every ray drawn carries the same power: directions are sampled uniformly in
cos(theta), which is uniform in solid angle.  Ray density on the page is
therefore light density, and the picture cannot flatter the near-axis
direction the way an evenly-spaced fan of angles would.
"""
import json
import os

import numpy as np

from lens_design import RED_DESIGN_NM, diamond_sellmeier
from method_export import list_methods

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = os.path.join(HERE, "figures", "methods")
OUT = os.path.join(HERE, "figures", "comparison_collection")
NV_DEPTH_UM = 85.0

DIAMOND, NVLAYER = "#cfe6f2", "#9dc6dc"
ESCAPE, TRAPPED, CAP, INK, MUTED, RED = "#2f5fc4", "#b9c0c9", "#d99414", "#131920", "#52514e", "#c0392b"


def geometry():
    designs = list_methods(METHODS)
    if not designs:
        raise SystemExit(f"no exported designs in {METHODS}")
    best = designs[0]
    with open(os.path.join(best["directory"], "design.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    parameters = payload["parameters"]
    return dict(label=best["label"],
                n_dia=float(diamond_sellmeier(RED_DESIGN_NM/1000.0)),
                gap=parameters["air_gap_um"],
                central_radius=parameters["central_aperture_um"],
                side_inner=max(0.0, payload["side"]["center_r"] -
                               parameters["side_aperture_um"]),
                side_outer=payload["side"]["center_r"] +
                parameters["side_aperture_um"])


def equal_power_rays(count, theta_max=np.pi/2):
    """Directions carrying equal power: uniform in cos(theta), not in theta."""
    cosines = np.linspace(np.cos(theta_max), 1.0, count, endpoint=False)
    cosines += 0.5*(cosines[1]-cosines[0]) if count > 1 else 0.0
    return np.arccos(np.clip(cosines, -1.0, 1.0))


def main():
    os.makedirs(OUT, exist_ok=True)
    geo = geometry()
    n_dia = geo["n_dia"]
    theta_c = np.arcsin(1.0/n_dia)
    escape_share = 0.5*(1.0-np.cos(theta_c))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2),
                                  gridspec_kw=dict(width_ratios=[1.0, 1.25]))

    # ---- (a) the whole upper hemisphere, so the escape cone is in proportion
    span = 150.0
    ax.axhspan(-NV_DEPTH_UM-40, 0, color=DIAMOND)
    ax.axhspan(-90, -80, color=NVLAYER)
    ax.axhline(0, color=INK, lw=1.4)

    for theta in equal_power_rays(150):
        for sign in (-1, 1):
            reach = (NV_DEPTH_UM/np.cos(theta)) if np.cos(theta) > 1e-3 else 1e4
            end = np.array([sign*reach*np.sin(theta), -NV_DEPTH_UM + reach*np.cos(theta)])
            if theta < theta_c:
                ax.plot([0, end[0]], [-NV_DEPTH_UM, 0.0], color=ESCAPE,
                        lw=0.8, alpha=0.75, zorder=3)
            else:
                stop = min(reach, 1.35*span/max(np.sin(theta), 1e-6))
                ax.plot([0, sign*stop*np.sin(theta)],
                        [-NV_DEPTH_UM, -NV_DEPTH_UM + stop*np.cos(theta)],
                        color=TRAPPED, lw=0.7, alpha=0.55, zorder=2)
    ax.add_patch(Wedge((0, -NV_DEPTH_UM), NV_DEPTH_UM/np.cos(theta_c),
                       90-np.degrees(theta_c), 90+np.degrees(theta_c),
                       facecolor=ESCAPE, alpha=0.13, zorder=1))
    ax.plot(0, -NV_DEPTH_UM, marker="*", ms=17, color="#111111", zorder=6)
    ax.annotate("NV", (0, -NV_DEPTH_UM), xytext=(6, -NV_DEPTH_UM-11),
                fontsize=10, color=INK, fontweight="bold")
    # boxed and moved clear of the ray fan, which the labels sat on top of
    box = dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3.0)
    ax.annotate(f"escape cone  $\\pm${np.degrees(theta_c):.1f}$\\degree$\n"
                f"only {100*escape_share:.1f}% of the emission gets out",
                xy=(16, -20), xytext=(46, 30), fontsize=9.5, color=ESCAPE,
                fontweight="bold", bbox=box,
                arrowprops=dict(arrowstyle="->", color=ESCAPE, lw=1.1))
    ax.annotate(f"trapped by total internal reflection\n"
                f"{100*(1-escape_share):.1f}% never leaves the diamond",
                xy=(0, -NV_DEPTH_UM-30), ha="center", fontsize=9, color=MUTED,
                bbox=box)
    ax.set_xlim(-span, span)
    ax.set_ylim(-NV_DEPTH_UM-40, 62)
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z from the diamond surface (um)")
    ax.set_title("(a) an NV emits in every direction - almost none gets out",
                 loc="left", fontsize=10.5)

    # ---- (b) inside the escape cone: where the surviving light lands
    cap_z = geo["gap"]
    thetas = equal_power_rays(220, theta_c)
    landing = []
    for theta in thetas:
        theta_air = np.arcsin(np.clip(n_dia*np.sin(theta), -1.0, 1.0))
        x_surface = NV_DEPTH_UM*np.tan(theta)
        x_cap = x_surface + cap_z*np.tan(theta_air)
        landing.append(x_cap)
        for sign in (-1, 1):
            on_cap = geo["side_inner"] <= x_cap <= geo["side_outer"]
            ax2.plot([0, sign*x_surface, sign*x_cap],
                     [-NV_DEPTH_UM, 0.0, cap_z],
                     color=ESCAPE if on_cap else TRAPPED,
                     lw=0.85 if on_cap else 0.6,
                     alpha=0.8 if on_cap else 0.4, zorder=3 if on_cap else 2)
    landing = np.asarray(landing)

    ax2.axhspan(-NV_DEPTH_UM-18, 0, color=DIAMOND, zorder=0)
    ax2.axhspan(-90, -80, color=NVLAYER, zorder=0)
    ax2.axhline(0, color=INK, lw=1.4, zorder=4)
    for sign in (-1, 1):
        ax2.plot([sign*geo["side_inner"], sign*geo["side_outer"]],
                 [cap_z, cap_z], color=CAP, lw=6, solid_capstyle="butt", zorder=5)
    ax2.plot([-geo["central_radius"], geo["central_radius"]], [cap_z, cap_z],
             color=MUTED, lw=6, solid_capstyle="butt", zorder=5)
    ax2.plot(0, -NV_DEPTH_UM, marker="*", ms=15, color="#111111", zorder=6)

    reach = geo["side_outer"]*1.9
    inside = np.abs(landing) <= reach
    caught = ((landing >= geo["side_inner"]) & (landing <= geo["side_outer"])).mean()
    ax2.annotate("six side caps (collect)", (geo["side_outer"], cap_z),
                 xytext=(geo["side_outer"]*0.60, cap_z+16), fontsize=9,
                 color=CAP, fontweight="bold")
    ax2.annotate("central cap\n(delivers green)", (0, cap_z),
                 xytext=(-reach*0.97, cap_z+14), fontsize=8.4, color=MUTED)
    ax2.annotate(f"{100*caught:.0f}% of the escaping light\nlands on the side caps",
                 xy=(-reach*0.96, -NV_DEPTH_UM*0.78), ha="left", fontsize=10.5,
                 color=ESCAPE, fontweight="bold",
                 bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3.0))
    ax2.set_xlim(-reach, reach)
    ax2.set_ylim(-NV_DEPTH_UM-18, cap_z+30)
    ax2.set_xlabel("x (um)")
    ax2.set_title("(b) inside the escape cone: equal-power rays, so density is light",
                  loc="left", fontsize=10.5)

    for axis in (ax, ax2):
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(False)

    fig.text(0.01, 0.015,
             f"{geo['label']}, NV at {NV_DEPTH_UM:.0f} um, cap plane {geo['gap']:.0f} um "
             "above the surface. Every ray carries equal power (uniform in cos theta), "
             "so where the rays crowd is where the light is.",
             fontsize=7.6, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(OUT, "nv_emission.png")
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)

    print(f"escape cone     : +/-{np.degrees(theta_c):.1f} deg = "
          f"{100*escape_share:.2f}% of 4pi")
    print(f"trapped by TIR  : {100*(1-escape_share):.2f}%")
    print(f"of what escapes, {100*caught:.0f}% lands on the side caps")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
