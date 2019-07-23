.. include:: docs/about.rst

Installation
============

**Note**, at this point, PlotOptiX binaries are tested in: Windows 10, Ubuntu 18.04, CentOS 7.

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

Then, try running code from the top of this readme, or one of the examples. You may also need to install ``tkinter`` and/or ``PyQt`` packages, if not shipped with your Python environment.

Denoiser binaries are optional and can be downloaded after PlotOptiX installation (the package size is ~370 MB, administrator rights are required for the installation)::

   python -m plotoptix.install denoiser

Development path
================

This is an early version. There are some important features not available yet, eg. ticks and labels on plot axes.

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

Examples in the repository head may use features not yet available in the PyPI release. In order to download examples
compatible with PyPI release install the package::

	python -m plotoptix.install examples

This will create a folder with examples in the current directory.

.. image:: https://plotoptix.rnd.team/images/surface_plot.jpg
   :alt: Surface plot ray tracing with PlotOptiX
