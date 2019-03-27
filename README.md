# PlotOptiX

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

PlotOptiX is based on NVIDIA OptiX framework and RnD.SharpOptiX C#/C++/CUDA libraries
by R&D Team. In **version 0.1.1.1** we start with:

### Features

- progressive path tracing with explicit light sampling
- pinhole cameras and cameras with depth of field simulation
- particle (sphere), parallelepiped, parallelogram and bezier geometries
- flat, diffuse, specular and glass shading
- spherical and parallelogram light sources
- environmental light and ambient occlusion
- GPU acceleration using RT Cores, and everything else what comes with [OptiX](https://developer.nvidia.com/optix)
- image output in [numpy](http://www.numpy.org/) array
- callbacks at the scene initialization, start and end of each frame raytracing, end of progressive accumulation
- Tkinter based UI

### System Requirements

- operating system, *currently*: Windows
- a [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus) with compute capability 2.0 to latest
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Python 3

### What's Included

- OptiX 6.0.0 libraries
- RnD.SharpOptiX libraries
- all other required 3'rd party libraries: FFmpeg, LibTiff, Newtonsoft.Json, .NET Core
- python examples


## Installation

**Note**, at this point PlotOptiX binaries are prepared for Windows only. We should extend to Linux / MacOS X soon, but today you are dealing with the very first version of the project.

#### CUDA Toolkit

CUDA libraries are not included in the package. They are rather huge, PlotOptiX is not coupled to a particular CUDA release, and the installation is quite straight-forward. Simply visit [CUDA download page](https://developer.nvidia.com/cuda-downloads), select your operating system and CUDA version (latest should be fine, PlotOptiX was tested with 9.1, 10.0, 10.1). Download and run the installer.

Make sure the CUDA_PATH environment variable is configured:

```shell session
C:\>echo %CUDA_PATH%
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
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

Then, try running the code from the top of this readme, or something from the examples.

## Development path

This is the first, beta version. Everything what is implemented should work, but there are some important features not available yet, eg. ticks and labels on plot axes.

PlotOptiX is basically an interface to RnD.SharpOptiX library which we are developing and using in our Studio. RnD.SharpOptiX offers much more functionality than it is now available through PlotOptiX. Eg. live streaming of the video raytracing output. We'll progressively add more to PlotOptiX if there is interest in this project (star or donate if you like!).

The idea for development is:

1. Binaries for Linux.
2. Complete the plot layout / continue adding raytracing features.
3. Convenience functions for various plot styles.

   Here the community input is possible and welcome.

## Examples

Progressing...
