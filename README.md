# PlotOptiX

3D raytracing package for Python, aimed at easy and aesthetic visualization of large datasets (and small as well). Data features can be represented on plots as a position, size and color of markers of several basic shapes, finished with a photorealistic lighting and depth of field.

No need to write shaders, intersection algorithms, handle 3D scene technicalities. Basic usage is even more simple than [matplotlib](https://matplotlib.org/):

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

...but it is much faster and, with all the nice raytraced shades and DoF, more readable and eye catching.

Just as a decoration of this readme, here are a few sample images made with PlotOptiX:

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

## Development

## Examples
