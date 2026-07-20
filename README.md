# fibers_simulation

## Lens optimization

Run the physical seven-core lens search from PowerShell:

```powershell
.\.venv\Scripts\python.exe phase3_optimize.py
```

The progress bars cover every central/side lens-family pair. The optimizer
scores the deposited traced-ray footprints, including Fresnel loss and
diffraction blur, rather than replacing a split or folded bundle with one
Gaussian spot.

Optimized parameters are the central and side lens families, central-to-side
overlap, side-lens offset from its fibre core, independent central and side
heights, central aperture (the side aperture follows from overlap), air gap,
and the applicable radius/asphere/biconic/freeform factors. NV concentration
(3 ppm), NV depth (80-90 um), emission band (650-850 nm), and the
300 x 300 x 300 um print volume are fixed.
