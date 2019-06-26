PlotOptiX
=========

.. image:: https://img.shields.io/pypi/v/plotoptix.svg
   :target: https://pypi.org/project/plotoptix
   :alt: Latest PlotOptiX version
.. image:: https://img.shields.io/pypi/dm/plotoptix.svg
   :target: https://pypi.org/project/plotoptix
   :alt: Number of PlotOptiX downloads
.. image:: https://img.shields.io/badge/support%20project-paypal-brightgreen.svg
   :target: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=RG47ZEL5GKLNA&source=url
   :alt: Support project

**Data visualisation in Python based on NVIDIA OptiX ray tracing framework.**

**Note:** active development is continuing, expect changes.

**GPU drivers note:** *NVIDIA drivers 419 (Windows) and 418 (Ubuntu) are recommended currently. You may experience problems with the most recent drivers release 430* (see this `issue <https://devtalk.nvidia.com/default/topic/1056174/optix/windows-10-update-broke-optix-6-0-on-my-machine/>`__).

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
- parameterized materials: flat, diffuse, reflective, refractive shading
- spherical and parallelogram light sources
- environmental light and ambient occlusion
- post-processing: tonal correction curves, levels adjustment, mask overlay
- GPU acceleration using RT Cores, and everything else what comes with `OptiX 6.0 <https://developer.nvidia.com/optix>`__
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

Installation
============

**Note**, at this point, PlotOptiX binaries are tested in: Windows 10, Ubuntu 18.04.

Windows should be ready to go in most cases. You need to do some more typing in Linux. For video encoding you need to install CUDA toolkit in both Linux and Windows.

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
   
   Mono JIT compiler **version 5.18.1.3** (tarball Tue Apr  9 16:16:30 UTC 2019)
      Copyright (C) 2002-2014 Novell, Inc, Xamarin Inc and Contributors. www.mono-project.com
	   TLS:           __thread
      ... (output cropped for clarity) ...

If ``mono`` command is not available, or the reported version is < 5.2, visit `Mono download page <https://www.mono-project.com/download/stable/#download-lin>`__ and follow instructions related to your Linux distribution. You want to install **mono-complete** package.

*pythonnet:*

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

CUDA Toolkit
------------

CUDA libraries are not included in the package and required only for the video encoding features of PlotOptiX. Visit
`CUDA download page <https://developer.nvidia.com/cuda-downloads>`__, select your operating system and CUDA version **10.x**.
Download and run the installer.

*Linux note:* Install the GPU driver before installing CUDA toolkit, it makes things easier.

PlotOptiX
---------

Using pip::

   pip install -U plotoptix

From GitHub sources::

   git clone https://github.com/rnd-team-dev/plotoptix.git
   cd plotoptix
   python setup.py install

Then, try running code from the top of this readme, or one of the examples.

You may also need to install ``tkinter`` and/or ``PyQt`` packages, if not shipped with your python environment.

Development path
================

This is an early version. There are some important features not available yet, eg. AI denoiser or even ticks and labels on plot axes.

PlotOptiX is basically an interface to RnD.SharpOptiX library which we are developing and using in our Studio. RnD.SharpOptiX offers
much more functionality than it is now available through PlotOptiX. We'll progressively add more to PlotOptiX if there is interest in
this project (download, star, and `support <https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=RG47ZEL5GKLNA&source=url>`__
if you like it!).

The idea for development is:

1. Binaries for Linux (done in v0.3.0).
2. Complete the plot layout / cover more raytracing features.
3. Convenience functions for various plot styles. Other GUI's.

   *Here, the community input is possible and warmly welcome!*

Examples
========

Looking at examples is the best way to get started and explore PlotOptiX features. Have a look at the
`readme and sample codes here <https://github.com/rnd-team-dev/plotoptix/tree/master/examples>`__.

.. image:: https://plotoptix.rnd.team/images/surface_plot.jpg
   :alt: Surface plot ray tracing with PlotOptiX
