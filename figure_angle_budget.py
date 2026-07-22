"""Where an NV's escapable light actually goes, and which cap catches it.

Run ``py figure_angle_budget.py``.  Writes figures/comparison_collection/.

Answers the obvious objection to a high collection efficiency: the central core
delivers green and collects nothing, so how can six side cores catch a large
share of the light?  Because solid angle goes as sin(theta) d(theta) -- a cone
around the axis holds almost none of the emission, while the outer part of the
escape cone holds most of it, and that is exactly where the side caps sit.

Geometry is read from the best exported design, so the figure always describes
the design it is shown next to.
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

GREY, BLUE, RED, INK, MUTED = "#b9c0c9", "#2f5fc4", "#c0392b", "#131920", "#52514e"


def geometry():
    """Best design's cap layout, plus the diamond index at the design band."""
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


def landing_radius(theta_dia, geo):
    """Where a ray leaving the NV at theta_dia meets the cap plane."""
    sin_air = geo["n_dia"]*np.sin(theta_dia)
    inside = sin_air < 1.0
    theta_air = np.arcsin(np.clip(sin_air, -1.0, 1.0))
    radius = np.where(inside,
                      NV_DEPTH_UM*np.tan(theta_dia) +
                      geo["gap"]*np.tan(np.where(inside, theta_air, 0.0)),
                      np.inf)
    return radius


def angle_at(radius, geo, theta_max):
    """Invert landing_radius: the emission angle that lands at ``radius``."""
    low, high = 0.0, theta_max-1e-9
    for _ in range(80):
        mid = 0.5*(low + high)
        if landing_radius(np.array(mid), geo) < radius:
            low = mid
        else:
            high = mid
    return 0.5*(low + high)


def main():
    os.makedirs(OUT, exist_ok=True)
    geo = geometry()
    theta_max = np.arcsin(1.0/geo["n_dia"])

    # Share of escapable light per unit angle: dOmega = 2 pi sin(theta) dtheta,
    # normalised so the whole escape cone integrates to 1.
    theta = np.linspace(0.0, theta_max*(1-1e-9), 2000)
    density = np.sin(theta)/(1.0-np.cos(theta_max))
    degrees = np.degrees(theta)

    inner = angle_at(geo["side_inner"], geo, theta_max)
    outer = angle_at(geo["side_outer"], geo, theta_max)
    share = lambda a, b: (np.cos(a)-np.cos(b))/(1.0-np.cos(theta_max))
    to_side = share(inner, outer)
    before = share(0.0, inner)
    beyond = share(outer, theta_max)
    first_two = share(0.0, np.radians(2.0))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    ax.fill_between(degrees, density, where=theta <= inner, color=GREY)
    ax.fill_between(degrees, density,
                    where=(theta >= inner) & (theta <= outer), color=BLUE)
    ax.fill_between(degrees, density, where=theta >= outer, color=GREY, alpha=0.45)
    ax.plot(degrees, density, color=INK, lw=1.2)
    ax.axvline(np.degrees(inner), color=INK, lw=0.8, ls=":")
    ax.axvline(np.degrees(outer), color=INK, lw=0.8, ls=":")

    # Percentages sit inside their own band; the wording goes in the legend, so
    # nothing has to compete for the same patch of canvas.
    peak = density.max()
    from matplotlib.patches import Patch
    ax.annotate(f"{100*to_side:.0f}%",
                xy=(0.5*(np.degrees(inner)+np.degrees(outer)), peak*0.30),
                ha="center", va="center", fontsize=19, color="white",
                fontweight="bold")
    ax.annotate(f"{100*beyond:.0f}%",
                xy=(np.degrees(outer)+0.5*(np.degrees(theta_max)-np.degrees(outer)),
                    peak*0.42), ha="center", va="center", fontsize=13, color=MUTED)
    ax.annotate(f"{100*before:.0f}%  into the central cap",
                xy=(np.degrees(inner)*0.75, peak*0.05),
                xytext=(np.degrees(inner)+1.1, peak*0.62), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.annotate(f"the first 2 deg carries only {100*first_two:.1f}%",
                xy=(2.0, density[np.searchsorted(degrees, 2.0)]),
                xytext=(6.0, peak*0.90), fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.legend(handles=[Patch(facecolor=BLUE, label="reaches the six side caps"),
                       Patch(facecolor=GREY, label="into the central cap"),
                       Patch(facecolor=GREY, alpha=0.45,
                             label="lands beyond the caps")],
              fontsize=8.4, loc="upper left", frameon=False)
    ax.set_xlim(0, np.degrees(theta_max))
    ax.set_ylim(0, peak*1.20)
    ax.set_xlabel("emission angle inside the diamond (deg)")
    ax.set_ylabel("share of escapable light per degree")
    ax.set_title("(a) the light is in the outer angles, not near the axis",
                 loc="left", fontsize=10.5)

    finite = theta < theta_max*(1-1e-6)
    ax2.plot(degrees[finite], landing_radius(theta[finite], geo),
             color=INK, lw=1.8)
    ax2.axhspan(geo["side_inner"], geo["side_outer"], color=BLUE, alpha=0.22)
    ax2.axhspan(0.0, geo["central_radius"], color=GREY, alpha=0.45)
    ax2.annotate("six side caps", (np.degrees(theta_max)*0.97,
                                   0.5*(geo["side_inner"]+geo["side_outer"])),
                 fontsize=9.5, color=BLUE, fontweight="bold", va="center",
                 ha="right")
    ax2.annotate("central cap\ndelivers green, collects nothing",
                 (np.degrees(theta_max)*0.97, geo["central_radius"]*0.30),
                 fontsize=8.2, color=MUTED, va="center", ha="right")
    ax2.set_yscale("log")
    ax2.set_ylim(1.0, 400.0)
    ax2.set_xlim(0, np.degrees(theta_max))
    ax2.set_xlabel("emission angle inside the diamond (deg)")
    ax2.set_ylabel("where the ray meets the cap plane (um)")
    ax2.set_title("(b) so the caps are placed where that light lands",
                  loc="left", fontsize=10.5)

    for axis in (ax, ax2):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#e1e0d9", lw=0.6)
        axis.set_axisbelow(True)

    fig.text(0.01, 0.015,
             f"{geo['label']}, NV at {NV_DEPTH_UM:.0f} um, gap {geo['gap']:.0f} um. "
             f"Beyond {np.degrees(theta_max):.1f} deg the light is trapped by total "
             "internal reflection and never leaves the diamond.",
             fontsize=7.6, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = os.path.join(OUT, "angle_budget.png")
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)

    print(f"escape cone half-angle      : {np.degrees(theta_max):.1f} deg")
    print(f"side caps span r            : {geo['side_inner']:.1f}-{geo['side_outer']:.1f} um"
          f"  (theta {np.degrees(inner):.1f}-{np.degrees(outer):.1f} deg)")
    print(f"  reaching the side caps    : {100*to_side:.1f}%")
    print(f"  inside the central cap    : {100*before:.1f}%")
    print(f"  landing beyond the caps   : {100*beyond:.1f}%")
    print(f"  in the first 2 deg        : {100*first_two:.1f}%")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
