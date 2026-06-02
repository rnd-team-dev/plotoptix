Release history
===============

`v0.19.2` - unreleased
----------------------

Changed
~~~~~~~
- Intel denoiser libs updated to 2.4.1.

Fixed
~~~~~
- FFmpeg libs in Windows were not accessible for python installed via Microsoft Store, even if the correct folder was included in PATH.


`v0.19.1` - 2026-05-28
----------------------

Added
~~~~~
- Function to get current noise level (value updated after each accumulation frame): ``get_current_noise_level()``.

Changed
~~~~~~~
- Moved to OptiX 9.1, recompliled with CUDA 13.2 (note: minimal requirements are now driver r590 and compute capability 7.5).
- ``rt_completed`` callback is now executed only after the last accumulation frame instead of every accumulation frame. Also,
  the video encoder adds frame to the output clip on the "accumulation done" signal instead of "launch finished". This allows
  for using all the work balancing modes in animation workflows, while it is still possible to configure
  `min_accumulation_step == max_accumulation_frames` to get the same behavior as before.


`v0.19.0` - 2026-03-25
----------------------

Added
~~~~~

- Bokeh swirl and asymmetrical vignette in cameras with depth of field simulation, allowing for rendering vintage lens effects.

- Roving capsules intersection mode added to curve geoms: BSplineQuadRocaps, BSplineCubicRocaps, CatmullRomRocaps, BezierRocaps,
  significantly improving performance on a longer strand segments, see 2_beziers.py sample code. Minor improvemts on other curve
  geom due to simplified vertex access code.

- HEVC codec support.

- View orientation changes in GUI, per user request. F5-F8 will flip/rotate the view.

Changed
~~~~~~~

- Moved to OptiX 9.0.
- Moved to NVENC SDK 13.0. Note, preset names are changed accordingly to P1 - P7. *Note:* FFmpeg 8.0 shared libs are recommended.
- Updated `GpuArchitecture` enum: added Ada Lovelace and Blackwell architectures.

Removed
~~~~~~~

- LDR denoiser based on OptiX driver (discontinued). Denoiser enum now points to DenoiserHDR. OIDN denoiser still supports LDR mode.

`v0.18.4` - 2025-02-12
----------------------

Changed
~~~~~~~

- Reworked heuristics for the noise-balanced ray sampling.
- NoiseBalanced work distribution mode is now alias to AbsNoiseBalanced since it turns more often practical in terms of visual result
  and stable compute time.

`v0.18.3` - 2025-01-03
----------------------

Changed
~~~~~~~

- Intel denoiser libs updated to 2.3.1. Accumulation frames, before the final one, are denoised using the fast model, the final frame is
  denoised using the high quality / slower model.


`v0.18.2` - 2024-12-13
----------------------

Changed
~~~~~~~

- Moved to OptiX 8.1.
- Encoder code upgraded to use FFmpeg 7.0; FFmpeg libs excluded from the Windows release (too large to keep them within the package).


`v0.18.1` - 2024-03-09
----------------------

Added
~~~~~

- Intel Open Image Denoiser libraries: HDR/LDR, supporting RGB/Albedo/Normal inputs.
- Optional counting of transmission path segments (previously they were counted always and included in
  the segment range limits).
- Const-size geometry primitives (same radii, or u/v/w vectors for all data points) with much lower memory
  consumption, beneficial e.g. for volume plots.
- Volume coloring material for semi-transparent volume visualization (experimental).

Changed
~~~~~~~

- Reviewed and improved camera kernels.

Fixed
~~~~~

- Require min SM 5.0 instead of 6.0.
- Minor lighting issues with masked transparency materials.


`v0.17.1` - 2023-10-06
----------------------

Added
~~~~~

- Noise-balanced work distribution: absolute and relative noise modes.


`v0.17.0` - 2023-08-18
----------------------

Added
~~~~~

- Noise-balanced work distribution.


`v0.16.1` - 2023-06-04
----------------------

**NOTE:** Binaries recompiled with OptiX 7.7, NVIDIA driver r530 or above is required.

Added
~~~~~

- CuPy support for texture and geometry data updates
- New curve geometries: Ribbon, Beziers

Changed
~~~~~~~

- no dedicated PyTorch/CuPy/... methods, all types handled with single corresponding method


`v0.16.0` - 2023-05-17
----------------------

Added
~~~~~

- set texture data from pytorch tensors
- texture filter modes enabled, default is *trilinear*, optionally can be changed to *nearest*

- set geometry data directly from numpy array or pytorch tensor (no host copy is mada)
- synchronization method to update host copy with data from gpu: NpOptiX.sync_raw_data()

- convenience method to get a copy of geometry data (more simple than PinnedBuffer, though returns a copy): NpOptiX.get_data()

`v0.15.1` - 2023-03-16
----------------------

Added
~~~~~

- OpenCV-like intrinsic matrix in camera setup
- configurable denoiser start frame

Changed
~~~~~~~

- ovrlaping refractive volumes handling


`v0.15.0` - 2022-12-30
----------------------

**NOTE:** NVIDIA driver r520 or above is required.

Added
~~~~~

- AI upsampler and HDR denoiser
- Catmull-Rom curve primitive

Changed
~~~~~~~

- binaries updated to OptiX 7.6
- raytracing and callbacks workflow explained in the docs 

.. _`v0.19.1`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.19.1
.. _`v0.19.0`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.19.0
.. _`v0.18.4`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.18.4
.. _`v0.18.3`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.18.3
.. _`v0.18.2`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.18.2
.. _`v0.18.1`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.18.1
.. _`v0.17.1`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.17.1
.. _`v0.17.0`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.17.0
.. _`v0.16.1`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.16.1
.. _`v0.16.0`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.16.0
.. _`v0.15.1`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.15.1
.. _`v0.15.0`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.15.0
