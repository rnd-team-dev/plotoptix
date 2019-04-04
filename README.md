# PlotOptiX

<a href="https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=RG47ZEL5GKLNA&source=url"><img src="https://img.shields.io/badge/support%20project-paypal-brightgreen.svg"></a>
<a href="https://pypi.org/project/plotoptix/"><img alt="Latest PlotOptiX version" src="https://img.shields.io/pypi/v/plotoptix.svg" /></a>
<a href="https://pypi.org/project/plotoptix/"><img alt="Number of PlotOptiX downloads" src="https://img.shields.io/pypi/dm/plotoptix.svg" /></a>

3D raytracing package for Python, aimed at easy and aesthetic visualization of large datasets (and small as well). Data features can be represented on plots as a position, size and color of markers of several basic shapes, finished with a photorealistic lighting and depth of field.

No need to write shaders, intersection algorithms, handle 3D scene technicalities. Basic usage is even more simple than with [matplotlib](https://matplotlib.org/):

```python
import numpy as np
from plotoptix import TkOptiX

n = 1000000                                  # 1M points, better not try this with matplotlib
xyz = 3 * (np.random.random((n, 3)) - 0.5)   # random 3D positions
r = 0.02 * np.random.random(n) + 0.002       # random radii

plot = TkOptiX()
plot.set_data("my plot", xyz, r=r)
plot.show()
```

...but PlotOptiX is much faster on large data and, with all the raytraced shades and DoF, more readable and eye catching.

Just as a decoration of this readme, here is a couple of sample images made with PlotOptiX:

![screenshots](https://github.com/robertsulej/plotoptix/blob/master/screenshots.jpg "PlotOptiX screenshots")

See examples for code details and more usage options.

PlotOptiX is based on [NVIDIA OptiX framework](https://developer.nvidia.com/optix) wrapped in RnD.SharpOptiX C#/C++/CUDA libraries
by R&D Team. In this early version we start with:

### Features

- progressive path tracing with explicit light sampling
- pinhole cameras and cameras with depth of field simulation
- particle (sphere), parallelepiped, parallelogram and bezier geometries
- flat, diffuse, specular and glass shading
- spherical and parallelogram light sources
- environmental light and ambient occlusion
- GPU acceleration using RT Cores, and everything else what comes with [OptiX](https://developer.nvidia.com/optix)
- callbacks at the scene initialization, start and end of each frame raytracing, end of progressive accumulation
- image output in [numpy](http://www.numpy.org/) array
- Tkinter based UI or headless raytracer

### System Requirements

- operating system, *currently*: Windows
- [.NET Framework](https://dotnet.microsoft.com/download/dotnet-framework) >= 4.6.1 (most likely you already have it)
- a [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus) with compute capability 5.0 (Maxwell) to latest
- [CUDA Toolkit **10.1**](https://developer.nvidia.com/cuda-downloads)
- Python 3

### What's Included

- OptiX 6.0.0 libraries
- RnD.SharpOptiX and RnD.SharpEncoder libraries
- all other required 3'rd party libraries: FFmpeg, LibTiff, Newtonsoft.Json, .NET bits
- python examples


## Installation

**Note**, at this point, PlotOptiX binaries are prepared for Windows only. We should be able to extend to Linux / MacOS X, but today you are dealing with the very first version of the project.

#### .NET Framework

Most likely you already got the right version with your Windows installation. Just in case, here is the command verifying this:

```shell session
C:\reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\full" /v version

HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\full
    version    REG_SZ    4.7.03056
```
If the number in your output is < 4.6.1, visit [download page](https://dotnet.microsoft.com/download/dotnet-framework) and install the most recent release.

#### CUDA Toolkit

CUDA libraries are not included in the package. They are rather huge, and the installation is quite straight-forward. Simply visit [CUDA download page](https://developer.nvidia.com/cuda-downloads), select your operating system and CUDA version **10.1** (we keep binaries compatible with the latest CUDA release). Download and run the installer.

Make sure the CUDA_PATH environment variable is configured:

```shell session
C:\>echo %CUDA_PATH%
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
```
It is also a good idea to keep your GPU driver up to date.

#### PlotOptiX

Using pip:

```shell session
C:\>pip install plotoptix
```

From GitHub sources:

```shell session
C:\plotoptix>python setup.py install
```

Then, try running code from the top of this readme, or one of the examples.

## Development path

This is the first, beta version, with binaries released for Windows only. Everything what is implemented should work, but there are some important features not available yet, eg. AI denoiser or even ticks and labels on plot axes.

PlotOptiX is basically an interface to RnD.SharpOptiX library which we are developing and using in our Studio. RnD.SharpOptiX offers much more functionality than it is now available through PlotOptiX. Eg. live streaming of the raytraced video output. We'll progressively add more to PlotOptiX if there is interest in this project (download, star, and <a href="https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=RG47ZEL5GKLNA&source=url">donate</a> if you like it!).

The idea for development is:

1. Binaries for Linux.
2. Complete the plot layout / cover more raytracing features.
3. Convenience functions for various plot styles. Other UI's if useful, eg. Qt.

   *Here, the community input is possible and warmly welcome!*

## Examples

Looking at examples is the best way to get started and explore PlotOptiX features, before we write a complete reference manual. Have look at the <a href="https://github.com/rnd-team-dev/plotoptix/tree/master/examples">readme and sample codes here</a>.
