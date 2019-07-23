PlotOptiX
=========

.. image:: https://img.shields.io/pypi/v/plotoptix.svg
   :target: https://pypi.org/project/plotoptix
   :alt: Latest PlotOptiX version
.. image:: https://img.shields.io/pypi/dm/plotoptix.svg
   :target: https://pypi.org/project/plotoptix
   :alt: PlotOptiX downloads by pip install
.. image:: https://img.shields.io/badge/support%20project-paypal-brightgreen.svg
   :target: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=RG47ZEL5GKLNA&source=url
   :alt: Support project

**Data visualisation in Python based on NVIDIA OptiX ray tracing framework.**

**Note:** active development is continuing, expect changes.

**GPU drivers note:** *NVIDIA drivers 419 (Windows) and 418 (Ubuntu, CentOS) are recommended if you experience problems with the most recent drivers release 430/431*.

3D `ray tracing <https://en.wikipedia.org/wiki/Ray_tracing_(graphics)>`__ package for Python, aimed at easy and aesthetic visualization
of large datasets (and small as well). Data features can be represented on plots as a position, size/thickness and color of markers
of several basic shapes, finished with a photorealistic lighting and depth of field.

No need to write shaders, intersection algorithms, handle 3D scene technicalities. Basic usage is even more simple than with
`matplotlib <https://matplotlib.org/gallery/mplot3d/scatter3d.html>`__:

.. code-block:: python

   import numpy as np
   from plotoptix import TkOptiX

   n = 1000000                                  # 1M points, better not try this with matplotlib
   xyz = 3 * (np.random.random((n, 3)) - 0.5)   # random 3D positions
   r = 0.02 * np.random.random(n) + 0.002       # random radii

   plot = TkOptiX()
   plot.set_data("my plot", xyz, r=r)
   plot.show()

... but PlotOptiX is much faster on large data and, with all the raytraced shades and DoF, more readable and eye catching.

`Documentation pages <https://plotoptix.rnd.team>`__ are currently generated from the source code docstrings. Please,
see `examples on GitHub <https://github.com/rnd-team-dev/plotoptix/tree/master/examples>`__
for practical code samples.

PlotOptiX is based on `NVIDIA OptiX <https://developer.nvidia.com/optix>`_ framework wrapped in RnD.SharpOptiX C#/C++ libraries
and completed with custom CUDA shaders by R&D Team. PlotOptiX makes use of RTX-capable GPU's.

.. image:: https://plotoptix.rnd.team/images/screenshots.jpg
   :alt: PlotOptiX screenshots, scatter and line plots ray tracing

Features
--------

- progressive path tracing with explicit light sampling
- pinhole cameras and cameras with depth of field simulation
- geometries: particle (sphere), parallelepiped, parallelogram, tetrahedron, bezier line, surface mesh
- parameterized materials shading: flat, diffuse, reflective, refractive (including light dispersion and nested volumes)
- spherical and parallelogram light sources
- environmental light and ambient occlusion
- post-processing: tonal correction curves, levels adjustment, mask overlay, AI denoiser
- GPU acceleration using RT Cores, multi-GPU support, and everything else what comes with `OptiX 6.0 <https://developer.nvidia.com/optix>`__
- callbacks at the scene initialization, start and end of each frame raytracing, end of progressive accumulation
- image output to `numpy <http://www.numpy.org>`__ array, or save to popular image file formats
- hardware accelerated video output to MP4 file format using `NVENC 9.0 <https://developer.nvidia.com/nvidia-video-codec-sdk>`__
- Tkinter based UI or headless raytracer

System Requirements
-------------------

- a `CUDA-enabled GPU <https://developer.nvidia.com/cuda-gpus>`__ with compute capability 5.0 (Maxwell) to latest (Turing)
- **Python 3 64-bit**
- Windows:
   - `.NET Framework <https://dotnet.microsoft.com/download/dotnet-framework>`__ >= 4.6.1 (present in normally updated Windows)
- Linux:
   - `Mono <https://www.mono-project.com/download/stable/#download-lin>`__ Common Language Runtime >= 5.2
   - `pythonnet <http://pythonnet.github.io>`__ >= 2.4
   - `FFmpeg <https://ffmpeg.org/download.html>`__ >= 4.1
- for video encoding: `CUDA Toolkit v10.x <https://developer.nvidia.com/cuda-downloads>`__ (tested with v10.0 and v10.1)

What's Included
---------------

- OptiX 6.0.0 libraries
- RnD.SharpOptiX and RnD.SharpEncoder libraries
- all other supporting 3'rd party libraries: FFmpeg (Windows only), LibTiff, Newtonsoft.Json
- Python examples
