Release history
===============

v0.13.0 - unreleased
--------------------

Added
~~~~~

- improved memory model: some buffers are not allocated until needed, and will use host
  memory if no space on device
- custom projection camera with ray target positions provided in a texture
- minimum ray tracing size can be even 1 pixel now

`v0.12.0` - 2020-11-17
----------------------

Added
~~~~~

- make_material method for easy configuration of material shaders
- metalness and metalness texture support
- enable changes of material in geometry update methods (why it was not possible before...?)

`v0.11.1` - 2020-10-21
----------------------

Code updated to OptiX 7.2. This is a minor step, preparing for the new features of the low
level library.

Fixed
~~~~~

- bug when new geometry families were added dynamically, e.g. b-splines to the scene with meshes only
- bug in deserialization of bezier and bspline geometries

`v0.11.0` - 2020-09-27
----------------------

Added
~~~~~

- direct access to internal geometry buffers (memory shared with ndarrays on the python side)
- graph / mesh wireframe geometry, available also for all surface plots
- m_shadow_catcher, material useful for preparation of packshot style images

Fixed
~~~~~

- clear the shader compilation cache on installing (incompatible code was surviving updates)
- several fixes in b-splines geometry

`v0.10.1`_ - 2020-08-30
-----------------------

Added
~~~~~

- enabled orthogonal projection camera

Fixed
~~~~~

- crash on empty geometries that appeared with the driver 452

`v0.10.0`_ - 2020-08-17
-----------------------

Added
~~~~~

- fisheye camera, custom projection camera
- thin lens and fisheye camera variants supporting chromatic aberration (transverse and longitudinal)
- zero-copy access to device buffers wrapped in ndarrays: 8/32bps image, hit and object info, albedo, normals
- configurable denoiser inputs: rgb-only, rgb+albedo, rgb+albedo+normals

Fixed
~~~~~

- more accurate light dispersion

`v0.9.0`_ - 2020-07-20
----------------------

NVIDIA driver >= 450 is required to run this release.

Added
~~~~~

- enabled normal buffer in AI denoiser
- new geometries for curves: BSplineQuad and BSplineCubic approximating data points, SegmentChain for a piecewise-linear plot

Changes
~~~~~~~

- update to OptiX 7.1 SDK and CUDA 11 (note: CUDA toolkit is not required in your system to run PlotOptiX)

`v0.8.2`_ - 2020-07-12
----------------------

Added
~~~~~

- method to update parallelogram light using center/target 3D points and scalar lengths of u/v sides (missing in v0.8.0)

Changes
~~~~~~~

- lower memory usage on both host and gpu
- tested with pythonnet 2.5.1 and Mono 6.x - linux installation made easier

`v0.8.1`_ - 2020-06-14
----------------------

Added
~~~~~

- camera mode for baking 360 degree panoramic views
- support 16 bit per channel and hdr (32 bit fp per channel) output to image files and ndarray
- support reading hdr images

Fixed
~~~~~

- correct light emission in volumes
- fix restoring scene global variables from json
- more verbose messaging on initialization problems
- fix camera switching when ray generation program changes

`v0.8.0`_ - 2020-06-04
----------------------

Added
~~~~~

- diffuse/reflective/plastic material transparency handled with alpha channel of textures
- load multiple meshes from .obj with materials specified in a dictionary, and an option to select parent mesh (then transormations of parent are applied to children meshes as well)
- setup parallelogram light using center/target 3D points and scalar lengths of u/v sides
- method to select objects for manual manipulation in gui (if e.g. cannot click object invisible in the view)

Fixed
~~~~~

- scatterng in volumes: support enabled in background modes AmbientAndVolume, TextureFixed, and TextureEnvironment;
  subsurface color added to material parameters
- keep_on_host argument of load_displacement() and load_normal_tilt() removed (value always set to false now; it was a bug in linux);

`v0.7.2`_ - 2020-05-13
----------------------

Added
~~~~~

- raw mesh geometry (defined explicitly with vertices, faces, normals, and uv mapping)
- selection of devices

Fixed
~~~~~

- color range scaling for arrays of const coloe (utility function)

`v0.7.1`_ - 2020-04-11
----------------------

Added
~~~~~

- set/release gimbal lock in camera rotations
- geometry scaling by vector and w.r.t. provided point
- sub-launch loop breaking on UI events (e.g. camera rotation)

Fixed
~~~~~

- nan's in mesh normal calculatons
- improved bvh memory allocations can handle more primitives in data sets
- texture values prescale when gamma is 1.0

`v0.7.0`_ - 2020-03-27
----------------------

*PlotOptiX has moved to OptiX 7 framework in this version.* This is a major change, basically a rewrite of entire
ray-tracting engine, followed by several breaking changes in the Python API. Denoiser binaries included in GPU
driver and improved compilation of shaders code are among advantages of the new framework. The long lasting issues
with using PlotOptiX on some hardware configurations, related to the shader compilation should be resolved now.

OptiX 7 shifts significant parts of functionality to the application side. Multi-GPU support and most of the
ray-tracting host state is now maintained by PlotOptiX code. Be warned that this code is fresh! If you spot
problems, go ahead and submit issue to the tracker on GitHub.

Changes
~~~~~~~

- no need to install denoiser binaries separately, no OptiX binaries shipped with PlotOptiX package (these libraries
  are now included in the GPU driver)
- setup_denoiser() removed, denoising is now configured with add_postproc() method
- uniform configuration of textures used by materials, geometries, background, etc., see load_texture() and
  set_texture_2d() methods
- material textures are now referenced by texture name instead of full texture description included in the
  material definition
- some of material properties names changed, see updated pre-defined materials
- NormalTilt removed from GeomAttributeProgram, surface normals are modulated with material textures
- tonal correction parameter tonemap_igamma (inverse value of gamma) changed to tonemap_gamma (gamma value)
- JSON structure changed and not backward-compatible for several scene components, which means scenes saved
  with earlier releases wont load with v0.7.0

Added
~~~~~

- surface roughness textures
- load_texture() method to facilitate reading textures from file

`v0.5.2`_ - 2019-10-15
----------------------

Fixed
~~~~~

- dependency on vcruntime140_1.dll in Windows binaries, introduced in v0.5.1 with the VS tools upgrade 

`v0.5.1`_ - 2019-09-27
----------------------

Added
~~~~~

- ray tracing timeout parameter, use set_param(rt_timeout=n_ms) and get_param("rt_timeout")

Fixed
~~~~~

- timeout instead of freeze if stucked in the internal OptiX launch() function
- default lighting was not initialized properly after refactoring made in v0.5.0

`v0.5.0`_ - 2019-09-20
----------------------

Added
~~~~~

- scene saving/loading in JSON file format or python's dictionary (note, format is not finally freezed and may
  change on migration to OptiX 7)
- callbacks re-configurable after initialization
- load selected/all/merged objects from Wavefront .obj files
- thin-walled material

Changes
~~~~~~~

- load_mesh_obj() method renamed to load_merged_mesh_obj(); the new load_mesh_obj() loads meshes selected by
  the name or loads all meshes from the file with no merging
- light shading mode configured with set_param() and get_param() methods

`v0.4.8`_ - 2019-09-07
----------------------

Added
~~~~~

- Oren-Nayar diffuse reflectance model (in addition to the default Lambertian), adjustable surface roughness
- adjustable surface rougness also for metalic and dielectric (glass) materials, improved predefined materials

Changes
~~~~~~~

- metalic and mirror materials use primitive colors to colorize the reflection (primitive color overrides
  surface albedo) so color data can be effectively used also with these materials

`v0.4.7`_ - 2019-08-28
----------------------

Added
~~~~~

- select and rotate/move/scale objects and lights in GUI with mouse (same as for the camera)
- status bar in GUI, shows selected item, 2D/3D coordinates of the surface under the pointer, and FPS
- method to set fixed size of the ray-tracing output in GUI (or go back to auto-fit to window size)

`v0.4.6`_ - 2019-08-19
----------------------

Added
~~~~~

- methods to rotate camera about given point, eye about target, target about eye, in local and global coordinates

Changes
~~~~~~~

- calculate normal tilt on the fly in the surface displacement mode, speed not affected, much lower gpu memory footprint

Fixed
~~~~~

- normal tilt mode in textured parallelepipeds bug resulting with transparent walls in some configs

`v0.4.5`_ - 2019-08-11
----------------------

Added
~~~~~

- particles geometry with 3D orientation (so textures can be applied), textured glass color
- shading normal tilt (particles, parallelograms, parellelepipeds, tetrahedrons) and surface displacement (particles) using texture data
- overlay a texture in 2D postprocessing

`v0.4.2`_ - 2019-07-23
----------------------

Added
~~~~~

- method to get light source parameters in a dictionary
- examples installer - so examples compatible with the recent PyPI release can be downloaded locally without cloning the repository

Fixed
~~~~~

- OptiX-CUDA interop: readback buffer pointer is now obtained for a single GPU in multi-GPU systems, this solves issue on multi-GPU systems

`v0.4.1`_ - 2019-07-14
----------------------

Added
~~~~~

- 2D color preprocessing utility
- reading normalized images

Fixed
~~~~~

- read_image method name in linux library loader

`v0.4.0`_ - 2019-07-06
----------------------

Added
~~~~~

- AI denoiser
- light dispersion in refractions
- method to update material properties after construction
- enable textured materials
- utilities for reading image files to numpy array, support for huge tiff images (>>GB)

Fixed
~~~~~

- update of parallelogram light properties
- selection of SM architecture

`v0.3.1`_ - 2019-06-26
----------------------

Added
~~~~~

- textured background (fixed texture or environment map, from numpy array or image file)
- json converters for vector types (more compact scene description)

Fixed
~~~~~

- removed dependency on CUDA release, CUDA required for video encoding features only

`v0.3.0`_ - 2019-06-09
----------------------

Added
~~~~~

- **linux support**
- parametric surface

Changes
~~~~~~~

- update to NVIDIA Video Codec SDK 9.0 and FFmpeg 4.1
- no need for CUDA_PATH environment variable

`v0.2.2`_ - 2019-05-26
----------------------

Added
~~~~~

- color calculation convenience method: scaling, exposure and inverted gamma correction
- h.264 encoder profile and preset selection

Changes
~~~~~~~

- major speed improvement in general, plus faser convergence in out of focus regions
- refactoring for linux support

Fixed
~~~~~

- missing parallelogram support

`v0.2.1`_ - 2019-05-19
----------------------

Added
~~~~~

- OpenSimplex noise generator
- basic interface to the video encoder (save video output to mp4 files)
- save current image to file

`v0.2.0`_ - 2019-05-12
----------------------

Added
~~~~~

- RTX-accelerated mesh geometry for surface plots, reading 3D meshes from Wavefront .obj fromat
- several configurable 2D postprocessing stages

Fixed
~~~~~

- bug on geometry update when data size was changed with u/v/w vectors not provided

`v0.1.4`_ - 2019-04-25
----------------------

Added
~~~~~

- methods to rotate geometry/primitive about provided 3D point
- autogenerated documentation, improved and completed docstring in the code

Changed
~~~~~~~

- use tuples instead of x, y, z arguments in rotation/move methods

`v0.1.3`_ - 2019-04-19
----------------------

Two weeks and some steps from the initial release. Starting changelog.

Added
~~~~~

- RTX-accelerated tetrahedrons geometry
- generate aligned or randomly rotated data markers if some vectors are missing
- methods to read back camera eye/target, light position, color and r/u/v
- get_param() to read back the rt parameters
- this changelog, markdown description content type tag for PyPI
- use [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

.. _`v0.12.0`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.12.0
.. _`v0.11.0`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.11.0
.. _`v0.10.1`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.10.1
.. _`v0.10.0`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.10.0
.. _`v0.9.0`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.9.0
.. _`v0.8.2`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.8.2
.. _`v0.8.1`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.8.1
.. _`v0.8.0`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.8.0
.. _`v0.7.2`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.7.2
.. _`v0.7.1`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.7.1
.. _`v0.7.0`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.7.0
.. _`v0.5.2`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.5.2
.. _`v0.5.1`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.5.1
.. _`v0.5.0`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.5.0
.. _`v0.4.8`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.4.8
.. _`v0.4.7`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.4.7
.. _`v0.4.6`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.4.6
.. _`v0.4.5`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.4.5
.. _`v0.4.2`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.4.2
.. _`v0.4.1`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.4.1
.. _`v0.4.0`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.4.0
.. _`v0.3.1`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.3.1
.. _`v0.3.0`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.3.0
.. _`v0.2.2`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.2.2
.. _`v0.2.1`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.2.1
.. _`v0.2.0`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.2.0
.. _`v0.1.4`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.1.4
.. _`v0.1.3`: https://github.com/rnd-team-dev/plotoptix/releases/tag/v0.1.3

