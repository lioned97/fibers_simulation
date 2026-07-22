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

## READ THIS FIRST — live hazard

A phase-3 lens search (`lens_design.search_design`) is **still running** from before
the physics correction. Python loaded `lens_design.py` into memory *before* the
escape-ceiling fix landed, so that process is computing with the **old, wrong
model**. Its final step writes `figures/mcf_freeform_design.json` and the two STLs.

`recompute_corrected.py` has **already been run** (2026-07-22 08:02:52) and rewrote
all 16 `figures/methods/*/{design,summary}.json` plus the top-level winner onto the
corrected model.

**So: when that stale search process exits it will clobber the corrected top-level
design JSON and STLs with old-model numbers.** First action in the new chat: check
whether it is still alive, and if it has since exited, re-run
`py recompute_corrected.py` before trusting anything.

```powershell
Get-Process python* -EA SilentlyContinue    # any hit = stale search still running
```

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
- new design collection ~1.9%, **x490 vs as-built** (NOT 3.6% / x840)
- photon budget: 0.46 vs 182 collected per 10,000 emitted = **x392**
- corrected winner: `quadratic + biconic`, 13.28 nT/sqrt(Hz) (was 10.42 unclamped)
- top designs collapse to 13.28-13.48 nT — **statistically tied, family ranking is
  not trustworthy**; do not claim one lens family beats another

Pre-correction values are preserved under an `unclamped` key in each summary.json
and as `*_unclamped.json` at top level. Labelled "do not quote".

`paper_figures.py` was **verified unaffected** — it never used the `A_coll`
formulation (per-ray bounded sum instead).

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
- `figure_angle_budget.py`, `figure_nv_emission.py`, `make_parameters_pdf.py`,
  `recompute_corrected.py`, `resolution_at_layer.py`, `paper_figures.py`,
  `pages/1_Fiber_Comparison.py`, `test_physics.py`.

Modified: `app.py` (batched 3D traces 5000->4 via None separators, LineCollection
in 2D, fiber-type selector, Z-sweep 0-500).

## Pending

1. Confirm the stale search is dead, then re-run `py recompute_corrected.py`.
2. Re-run `py figure_photon_budget.py` **without** `--coarse` for clean maps.
3. Re-run `compare_collection.py`, `compare_probes.py`, `paper_figures.py` at full
   fidelity.
4. Decide whether to launch a **fresh** search under the corrected model — the
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
