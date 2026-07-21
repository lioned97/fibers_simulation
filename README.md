# fibers_simulation

## Lens optimization

Run the physical seven-core lens search from PowerShell:

```powershell
.\.venv\Scripts\python.exe phase3_optimize.py
```

The progress bars cover every central/side lens-family pair. The optimizer
traces a 10-um Gaussian core area combined with a uniformly filled full-NA
angular cone for the central core and all six side cores. It scores the
deposited footprints, including Fresnel loss and diffraction blur, rather
than replacing a split or folded bundle with one Gaussian spot.

The score is the 80--90 um volume integral of the central-core 532 nm
excitation rate multiplied by the summed reciprocal 650--850 nm collection
of all six side cores. `redesign_fig.py` draws the rays from the fiber cores
into the diamond and writes the combined 3-D field to
`figures/mcf_combined_overlap_volume.npz` with a readable summary in
`figures/mcf_combined_overlap_volume.json`.

The default Phase 3 NA is 0.22. Set the measured value before starting both
optimization and figure generation:

```powershell
$env:MCF_FULL_NA = "0.22"
.\.venv\Scripts\python.exe phase3_optimize.py
```

Figure 13 shows the weighted equal-solid-angle Phase 3 rays together with the
full-NA boundary for the central core and all six side cores.

Optimized parameters are the central and side lens families, central-to-side
overlap, side-lens offset from its fibre core, independent central and side
heights, central aperture (the side aperture follows from overlap), air gap,
and the applicable radius/asphere/biconic/freeform factors. NV concentration
(3 ppm), NV depth (80-90 um), emission band (650-850 nm), and the
300 x 300 x 300 um print volume are fixed.
