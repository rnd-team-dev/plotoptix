PlotOptiX
=========

.. image:: https://img.shields.io/pypi/v/plotoptix.svg
   :target: https://pypi.org/project/plotoptix
   :alt: Latest PlotOptiX version
.. image:: https://img.shields.io/pypi/dm/plotoptix.svg
   :target: https://pypi.org/project/plotoptix
   :alt: PlotOptiX downloads by pip install
.. image:: https://img.shields.io/badge/PATREON-Become%20a%20Patron!-008a04.svg
   :target: https://www.patreon.com/bePatron?u=33442314
   :alt: Become a Patron!

**Data visualisation and ray tracing in Python based on NVIDIA OptiX framework.**

- Check what we are doing with PlotOptiX on `Behance <https://www.behance.net/RnDTeam>`__, `Facebook <https://www.facebook.com/rndteam>`__, and `Instagram <https://www.instagram.com/rnd.team.studio/>`__.
- Join us on `Patreon <https://www.patreon.com/rndteam?fan_landing=true>`__ for news, release plans and hi-res content.

PlotOptiX is a 3D `ray tracing <https://en.wikipedia.org/wiki/Ray_tracing_(graphics)>`__ package for Python, aimed at easy and aesthetic visualization
of large datasets (and small as well). Data features can be represented in images as a position, size/thickness and color of primitives
of several basic shapes, or projected onto surfaces of objects in form of a color textures and displacement maps. Triangular meshes,
generated in the code or loaded from a file, are supported as well. All is finished with a photorealistic lighting, depth of field, and many other physically based effects simulated with a high quality.

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

... but PlotOptiX is much faster on large data, and, with all the raytraced shades and DoF, more readable and eye catching.

Check `examples on GitHub <https://github.com/rnd-team-dev/plotoptix/tree/master/examples>`__ for practical code samples and `documentation pages <https://plotoptix.rnd.team>`__ for a complete API reference.

PlotOptiX is a set of CUDA shaders by `R&D Team <https://rnd.team>`_ wrapped in C#/C++ libraries with a Python API. PlotOptiX is based on `NVIDIA OptiX 7.5 <https://developer.nvidia.com/optix>`_ framework and makes use of RTX-capable GPU's.

You can quickly display data in a simple plot:

.. image:: https://plotoptix.rnd.team/images/screenshots.jpg
   :alt: PlotOptiX screenshots, scatter and line plots ray tracing

or prepare complex scenes, combining your generated data with meshes modeled elsewhere:

.. image:: https://plotoptix.rnd.team/images/screenshot2.jpg
   :alt: PlotOptiX screenshots, ray tracing for generative art

Features
--------

- progressive path tracing with explicit light sampling
- *cameras*: orthographic, pinhole, thin-lens and fisheye with depth of field and chromatic aberration simulation, panoramic camera for making 360 deg environment maps, user-defined projection for shooting rays at any angle, and from any origin
- *geometries*: particles (spheres), parallelepipeds, parallelograms, tetrahedrons, linear segments, bezier curves, b-splines; *meshes*: shaded surface or wireframe, automatically generated from a parametric surface or f(x,y) data, or defined with vertices and faces, e.g. created with `pygmsh <https://github.com/nschloe/pygmsh>`__, or loaded from a file, e.g. supported by `trimesh <https://github.com/mikedh/trimesh>`__, or loaded from a Wavefront .obj file with a native loader
- *materials*: flat, diffuse, reflective, refractive; including: light dispersion, surface roughness and metalness, volume scattering, and nested volumes
- *light sources*: spherical and parallelogram, light emission in volumes, uniform environmental light or environment map
- *post-processing*: tonal correction curves, levels adjustment, apply mask/overlay, AI denoiser
- *callbacks*: at the scene initialization, start and end of each frame raytracing, end of progressive accumulation
- 8/16/32bps(hdr) image output to `numpy <http://www.numpy.org>`__ array, or save to popular image file formats
- zero-copy access to GPU buffers wrapped in ndarrays: 8/32bps image, hit and object info, albedo, normals
- GPU acceleration using RT Cores and everything else what comes with `OptiX <https://developer.nvidia.com/optix>`__
- hardware accelerated video output to MP4 file format using `NVENC 9.0 <https://developer.nvidia.com/nvidia-video-codec-sdk>`__
- Tkinter based simple GUI window or a headless raytracer
- configurable multi-GPU support

System Requirements
-------------------

- a `CUDA-enabled GPU <https://developer.nvidia.com/cuda-gpus>`__ with compute capability 5.0 (Maxwell) to latest (Ampere);
   - NVIDIA driver >= r515;
- **Python 3 64-bit**
- Windows:
   - `.NET Framework <https://dotnet.microsoft.com/download/dotnet-framework>`__ >= 4.6.1 (present in normally updated Windows)
- Linux:
   - `Mono <https://www.mono-project.com/download/stable/#download-lin>`__ Common Language Runtime >= 5.2
   - `pythonnet <http://pythonnet.github.io>`__ 2.5.1 or 2.5.2 (before 3.0 is released, these are the only supported pythonnet versions, thus require **Python <= 3.8**)
   - `FFmpeg <https://ffmpeg.org/download.html>`__ >= 4.1

What's Included
---------------

- RnD.SharpOptiX and RnD.SharpEncoder libraries
- all other supporting 3'rd party libraries: FFmpeg (Windows only), LibTiff, Newtonsoft.Json
- Python examples

Installation
============

**Note**, at this point, PlotOptiX binaries are tested in: Windows 10/11, Ubuntu 18.04, CentOS 7.

PlotOptiX was also successfully tested on the `Google Cloud Platform <https://cloud.google.com/>`__, using Compute Engine instance with 2x V100 GPU's and Ubuntu 18.04 image.
Here are the `installation steps <https://github.com/rnd-team-dev/plotoptix/blob/master/gcp_install_gpu.txt>`__ so you can save some precious seconds (FFmpeg not included).

Windows should be ready to go in most cases. You need to do some more typing in Linux.

Windows prerequisites
---------------------

*.NET Framework:*

Most likely you already got the right version with your Windows installation. Just in case, here is the command verifying this::

   C:\>reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\full" /v version
   
   HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\full
       version    REG_SZ    4.7.03056

If the number in your output is < 4.6.1, visit `download page <https://dotnet.microsoft.com/download/dotnet-framework>`__ and
install the most recent release.

Linux prerequisites
-------------------

*Mono runtime:*

Check if / which Mono release is present in your system::

   mono -V
   
   Mono JIT compiler version 5.18.1.3 (tarball Tue Apr  9 16:16:30 UTC 2019)
      Copyright (C) 2002-2014 Novell, Inc, Xamarin Inc and Contributors. www.mono-project.com
	   TLS:           __thread
      ... (output cropped for clarity) ...

If ``mono`` command is not available, or the reported version is < 5.2, visit `Mono download page <https://www.mono-project.com/download/stable/#download-lin>`__ and follow instructions related to your Linux distribution. You want to install **mono-complete** package.

*pythonnet:*

Note, current pythonnet release supports Python up to 3.8.

The `pythonnet <http://pythonnet.github.io>`__ package is available from `PyPI <https://pypi.org/project/pythonnet>`__, however, some prerequisities are needed. Instuctions below are based on APT, replace ``apt`` with ``yum`` depending on your OS::

   apt update
   apt install clang libglib2.0-dev python-dev
   
You may also need to install development tools, if not already present in your system, e.g. in Ubuntu::

   apt install build-essential
   
or in CentOS::

   yum group install "Development Tools" 
   
Then, update required packages and install ``pythonnet``::

   pip install -U setuptools wheel pycparser
   pip install -U pythonnet
   
After successful installation you should be able to do python's import:

.. code-block:: python

   import clr
   print(clr.__version__)

*FFmpeg:*

FFmpeg shared libraries >= 4.1 are required to enable video encoding features in PlotOptiX. Uninstall older version first. Visit `FFmpeg site <https://ffmpeg.org/download.html>`__ and download the most recent release sources. Unpack it to a new folder, cd to it. Configure, compile and install as below::

   ./configure --enable-shared
   make
   sudo make install

Add FFmpeg's shared library path to your config::

   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
   sudo ldconfig

PlotOptiX
---------

Using pip::

   pip install -U plotoptix

From GitHub sources::

   git clone https://github.com/rnd-team-dev/plotoptix.git
   cd plotoptix
   python setup.py install

Then, try running code from the top of this readme, or one of the examples. You may also need to install ``tkinter`` and/or ``PyQt`` packages, if not shipped with your Python environment.

Development path
================

This is still an early version. There are some important features not available yet, eg. ticks and labels on plot axes.

PlotOptiX is basically an interface to RnD.SharpOptiX library which we are developing and using in our Studio. RnD.SharpOptiX offers
much more functionality than it is now available through PlotOptiX. We'll progressively add more to PlotOptiX if there is interest in
this project (download, star, and `become our Patron <https://www.patreon.com/rndteam>`__
if you like it!).

The idea for development is:

1. Binaries for Linux (done in v0.3.0).
2. Migrate to OptiX 7.0 (done in v0.7.0).
3. Complete the plot layout / cover more raytracing features.
4. Convenience functions for various plot styles. Other GUI's.

   *Here, the community input is possible and warmly welcome!*

Examples
========

Looking at examples is the best way to get started and explore PlotOptiX features. Have a look at the
`readme and sample codes here <https://github.com/rnd-team-dev/plotoptix/tree/master/examples>`__.

Examples in the repository head may use features not yet available in the PyPI release. In order to download examples
compatible with PyPI release install the package::

	python -m plotoptix.install examples

This will create a folder with examples in the current directory.

.. image:: https://plotoptix.rnd.team/images/surface_plot.jpg
   :alt: Surface plot ray tracing with PlotOptiX
