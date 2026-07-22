# Handoff — NV/fiber collection simulation

Paste this into a new chat to resume.

---

## Context

Working dir: `F:\fibers_simulation` (separate from the NV_LAB repo).
Goal: a defensible optical model of an NV-diamond fiber probe for a physics paper,
plus a side-by-side comparison to show a PI that a new Nanoscribe-printed MCF lens
design beats the multimode fiber and the as-built MCF tip.

Physics: single NV / 3 ppm ensemble in a ~100 um diamond, NV layer 80-90 um deep.
MCF = 7 cores, 1 central (532 nm delivery, ~10 mW) + 6 side (collection), IP-S
printed caps. Comparison baselines: SM, MM (0 and 1-5 um air gap), as-built MCF.

## State of the exported results — resolved, no action needed

There was a hazard here and it did **not** fire. Recording it so nobody re-opens it:

A phase-3 search was running from before the physics correction, holding the old
pre-fix `lens_design.py` in memory. The worry was that its final write would
clobber the corrected files. It exited by 09:03:54 without writing anything after
08:03:13, so the corrected state is intact:

- `recompute_corrected.py` ran 2026-07-22 08:02:52 and rewrote all 16
  `figures/methods/*/{design,summary}.json` plus the top-level winner and STLs
  (all stamped 08:03:13) onto the corrected model.
- `figures/mcf_freeform_design.json` carries `clamped_signal_fraction` and reads
  13.28 nT/sqrt(Hz) — the corrected-model markers. Verified.
- No python processes remain.

**Do not re-run `recompute_corrected.py`** — it is idempotent on the numbers but
would overwrite the `unclamped` keys with already-corrected values, destroying the
pre-correction record.

## The critical bug (fixed) — the thing that matters most

Collection probability was built as `A_coll x rho` (etendue over 4pi times a ray
density). That conserves etendue **in the integral** but bounds nothing **at a
point**. The only clamp was `np.minimum(collection, 1.0)` — about 28x looser than
physics allows.

Symptom the user caught ("could that be correct? it seems too good to be true"):
peak collection 10.43% against an honest escape-cone ceiling of 3.5635% (2.93x
over), with 40.8% of excited NVs sitting above the ceiling.

Fix, in `lens_design.py`: `escape_ceiling(lam_nm)` — Fresnel-weighted integral over
the diamond escape cone, `lru_cache`d — with a per-wavelength clamp applied inside
`_combined_overlap_plane`, and mirrored in `compare_collection.py` and
`figure_photon_budget.py`.

**Corrected headline numbers (use these, not the old ones):**
- the chosen design is **`quadratic + biconic`** — the geometry in the top-level
  `mcf_freeform_design.json` and in the printed STLs. Every figure is pinned to
  it via `method_export.headline_label()`; do not let a figure re-sort and pick
  its own "best" (see the tie below)
- chosen design collection **1.874%**, as-built **0.00963%**, **x195**
  (NOT 3.6% / x840, not x490, not x456, not x212, not x204 — the escape ceiling,
  the as-built geometry, the field grid and the design pinning have each moved it)
- photon budget: 0.96 vs 187 collected per 10,000 emitted
- corrected winner: `quadratic + biconic`, 13.28 nT/sqrt(Hz) (was 10.42 unclamped)
- top designs collapse to 13.28-13.48 nT — **statistically tied, family ranking is
  not trustworthy**; do not claim one lens family beats another

Pre-correction values are preserved under an `unclamped` key in each summary.json
and as `*_unclamped.json` at top level. Labelled "do not quote".

`paper_figures.py` was **verified unaffected** — it never used the `A_coll`
formulation (per-ray bounded sum instead).

## The second real error: as-built geometry (fixed 2026-07-22)

The user asked whether the as-built tip is really as narrow as the ray figure
draws it. It is not. `AS_BUILT` in `compare_probes.py` sized every printed cap at
a 17.5 um pupil, taken from `paper_figures`' constants rather than from the
parts. Measured off the supplied STLs (`MCF_ASSET_DIR`, default
`C:\Users\owner\Downloads`):

| | STL | old model |
|---|---|---|
| side lens span | r = 17.5 -> 63 um, vertex at the inner rim | r = 0 -> 35, vertex at cap centre |
| side R_radial | 39.82 um | 18.0 um |
| side R_tangential | 1139.27 um | 95.0 um |
| pedestal | 150 um diameter, 134.04 um tall | flat base plane |
| central lens | 35 um diameter, 36 um tall | 17.5 um aperture — correct |

Each printed side lens is one wedge running from the central pillar out to the
fibre edge, so the core at r=35 looks through the **middle** of its own lens. The
old 17.5 um pupil stopped exactly at the core, so the outer half of every
collection cone met flat polymer, and the half it did catch was bent by an 18 um
radius instead of 40 um.

Fix: `_cap(..., vertex_at_inner_rim=True)` writes the sag about x=-1
(`coef[0] = coef[1]`, `shift = a*coef[1]/2` so `apex` still means the lowest
point of the cap, which the clearance checks assume). The rebuilt surface tracks
the STL to under 0.6 um across the whole 45 um run.

**Effect: the as-built baseline collects 2.15x more than before** — eta
0.00430% -> 0.00924% at its best 10 um standoff, gain x456 -> **x212**. The new
design is untouched; only the baseline moved. Model apertures are disks while the
real lenses are 60-degree wedges, so adjacent caps overlap slightly in the union.

## The third error: the field grid was too coarse (fixed 2026-07-22)

Found while building the alignment-tolerance sweep, where the tilt curve came
out non-monotone and jumped 15% between neighbouring angles.

The signal is the **product** of two sharply peaked densities, and the new
design's sensing spot is about 1.2-1.9 um across against a diffraction blur of
sigma ~0.5 um. At `grid_n=81` the transverse mesh is ~1 um, so the product
integral is not resolved. Convergence at the design's own standoff:

| grid_n | as-built eta | new eta | ratio |
|---|---|---|---|
| 81 | 0.00924% | 1.9622% | 212 |
| 121 | 0.00964% | 1.8237% | 189 |
| 241 | 0.00963% | 1.8723% | 194 |
| 321 | 0.00973% | 1.8857% | 194 |

The as-built tip was already resolved at 81 — its spot is far broader — which
is why the error landed on the **ratio** and not on either level. `grid_n` is
now **241** in `compare_collection._quadrature` and `compare_probes.evaluate`.

Two things this leaves behind. First, the top-eta design at 241 is
`freeform + biconic` and at 81 it was `asphere + biconic`, i.e. the family
ranking flips with the grid — more evidence for the existing "statistically
tied, do not rank families" rule. Second, `lens_design`'s own search still runs
at its search grid, so the geometries were selected on a mesh that cannot
resolve the spots they produce.

## Which design the figures show (fixed 2026-07-22)

The user caught the figures showing a different tip than the chosen one. Three
orderings existed and disagreed:

| ordering | top |
|---|---|
| top-level `mcf_freeform_design.json` + printed STLs | **quadratic + biconic**, 13.279 nT |
| `list_methods`, by summary sensitivity | **quadratic + biconic** |
| `compare_collection`, by eta | freeform + biconic at grid 241, asphere + biconic at grid 81 |

Spread across the top four: **4.5% on eta, 1.8% on sensitivity** — inside the
tie, so "best" is whichever sort ran last. `method_export.headline_label()` now
reads the chosen pairing from the top-level design JSON, and
`compare_collection.headline_row()`, `figure_ray_budget` and
`figure_alignment_tolerance` all select through it. `figure_photon_budget`
already used `list_methods[0]`, which agrees. The eta table stays sorted by eta
— it is a table, not a choice — but its footer now names the chosen design.

## Alignment tolerance (2026-07-22)

`figure_alignment_tolerance.py`, at grid_n=241, each tip normalised to its own
peak:

| | as-built | `quadratic + biconic` |
|---|---|---|
| best standoff | 12 um | 18 um |
| half-signal standoff window | 5 - 84 um | **15.3 - 20.5 um** |
| signal at +/-2 deg tilt | 97% | 98% |
| signal at +/-5 deg tilt | 98% | 90% |
| green-red shared volume at its best standoff | 6,221 um3 | 24.4 um3 |

The new tip buys its 195x with a standoff window of about +/-2.5 um. Tilt is a
non-issue for both out to +/-10 deg. The few-percent ripple on the as-built
standoff curve reproduces at grid_n 241/321/401 (spread 0.5-1.5%), so it is the
model's ray optics — hard apertures, no wave averaging — not numerical noise.

The green excitation was already in the swept quantity: `_combined_overlap_plane`
returns `RHO * excitation_rate * collection`, the traced 532 nm pump times the
six-core collection, and `evaluate_design` re-traces **both** at every tilt. What
was missing was a way to see it, so `evaluate_design` now also returns
`model_excited_rate_s`, `collection_efficiency` (identical to
`compare_collection`'s to all printed digits — cross-checked both ways) and
`effective_volume_um3`. That last one is the participation ratio
`(int s)^2 / int s^2` of the green-times-red field: the half-maximum core volume
is a threshold statistic and jumped two orders of magnitude between neighbouring
standoffs, while the participation ratio is smooth and needs no threshold.

## File inventory

Created:
- `physics.py` — extracted verbatim from app.py (AST-verified identical) so pages
  import without launching Streamlit. Has `angle_to_boresight` (TIR guard:
  non-unit V1 made sin^2 negative -> exp overflow -> NaN) and `lens_dome_mesh`
  (paraboloid cap; the spherical version had a sag/radius convention bug).
- `lens_design.py` — phase-3 optimizer. Derived apertures, exposure gating,
  checkpointing with atomic writes + config signature, escape ceiling.
- `method_export.py` — per-method folders: design.json, summary.json,
  raytrace.png, 2 STLs. Ray alpha is power-weighted.
- `phase3_gui.py` — Tkinter: Run tab (checkpoint status, Continue / Start over /
  Stop, live log) + Designs tab (browse and view exported designs).
- `compare_collection.py`, `compare_probes.py` — the PI deliverables.
- `figure_photon_budget.py` — 3 panels: as-built map, new-design map (shared log
  scale, green excitation contour), photon budget bars.
- `figure_alignment_tolerance.py` — standoff and diamond-tilt tolerance, both
  MCF tips, each against its own peak. Reuses `lens_design.alignment_sweep`.
- `figure_ray_budget.py` — 5 panels: SM, MM, as-built and new-design meridional
  ray traces on one vertical scale for the same NV at 85 um, plus the photon
  budget bars. Asserts every efficiency stays under the escape ceiling.
- `figure_angle_budget.py`, `figure_nv_emission.py`, `make_parameters_pdf.py`,
  `recompute_corrected.py`, `resolution_at_layer.py`, `paper_figures.py`,
  `pages/1_Fiber_Comparison.py`, `test_physics.py`.

Modified: `app.py` (batched 3D traces 5000->4 via None separators, LineCollection
in 2D, fiber-type selector, Z-sweep 0-500).

## Pending

1. ~~Confirm the stale search is dead~~ — done, and do **not** re-run
   `recompute_corrected.py` (see above).
2. ~~`figure_photon_budget.py` without `--coarse`~~ — done 2026-07-22, on the
   corrected as-built geometry.
3. Re-run `paper_figures.py` at full fidelity. `compare_collection.py`,
   `compare_probes.py`, `figure_photon_budget.py` and `figure_ray_budget.py` have
   all been re-run on the corrected as-built geometry.
4. The search's own transverse grid is still the coarse one — a fresh search
   should run the objective at a mesh that resolves the spots it is selecting
   for, or it will keep preferring designs the scoring cannot integrate.
5. Decide whether to launch a **fresh** search under the corrected model — the
   current geometries were optimized against the wrong objective, so they are
   valid designs scored honestly, not designs optimized honestly.
5. Open question for the user: `paper_figures.py` integrates 640-800 nm while
   `lens_design.py` now uses 650-800 nm. Pick one for the paper.

## Working agreements

- Do **not** assume magnetic sensitivity in the PI comparison — compare
  **collection efficiency**. (Explicit user instruction.)
- Printed lens caps must bulge **toward the diamond**, not the fiber. A regression
  test locks the sign; do not "fix" it back.
- MM is flagged separately in comparisons: different tracer, a bare fiber cannot
  be expressed as a printed cap, so new-vs-old is same-model but MM is not.
- Verify every edit actually landed (a heredoc `str.replace` once silently matched
  0 times and reported success).
- The user catches real physics errors. When a result looks too good, check it
  against a hard bound before defending it.
- Any claim about the fabricated tip's geometry is measured off the delivered
  STLs, never copied from a constant in another module. The 17.5 um as-built
  pupil survived for weeks because it was inherited rather than measured.
