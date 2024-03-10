"""
Transparent volume plot.
(NOTE, faster material kernels to be released)

This example shows how to:
   - use const-sized cubes to plot semi-transparent volume plot
   - present data feature as a color
   - use matplotlib color map (see https://matplotlib.org/users/colormaps.html)
"""

import numpy as np
from plotoptix import TkOptiX
from plotoptix.materials import m_volume_color # implements transparent, colored volume
from plotoptix.utils import map_to_colors      # feature to color conversion
from plotoptix.enums import DenoiserKind       # deniser options

from plotoptix.utils import simplex            # noise generator
_ = simplex(np.zeros((3,3)))


def main():

    # Make some data first:

    # Full volume is visualized in this example. For efficiency one can
    # use cell selection like in the opaque volume plot example to skip
    # completely transparent cells.

    # Note, these dimensions result with >60M cells which will require
    # about 5-6GB of GPU memory to store all the structures. Reduce
    # dimensions in case of GPU with smaller memory.
    kx, ky, kz = (500, 500, 250)
    scale = 0.02

    x = np.linspace(0, kx, kx)
    y = np.linspace(0, ky, ky)
    z = np.linspace(0, kz, kz)
    xv, yv, zv = np.meshgrid(x, y, z)
    xyz = scale * np.stack((xv, yv, zv), axis=-1)

    vol = simplex(xyz)
    print(f"Volume shape: {vol.shape}, {kx * ky * kz / 1.0e6 :.1f}M cells.")

    colors = map_to_colors(vol, "Reds")

    # Setup ray-tracing:

    rt = TkOptiX() # create and configure, show the window later

    max_frames = 16 # accumulation frames (rays/pixel in other words)
    rt.set_param(
        min_accumulation_step=4,
        max_accumulation_frames=max_frames
    )

    rt.set_float("scene_epsilon", 1.0e-05) # expect small cubes, lower epsilon to ensure details are correctly presented
    rt.set_uint("path_seg_range", 1, 1)    # single segment is enough (transmission ray segments are not counted)

    rt.set_background(0.99) # plot background color
    rt.set_ambient(0.8)     # uniform environment light

    # setup tone mapping and denoiser for best image quality
    exposure = 1.1
    gamma = 2.2
    rt.set_float("tonemap_exposure", exposure)
    rt.set_float("tonemap_gamma", gamma)
    rt.set_uint("denoiser_start", max_frames) # denoise only the final image
    rt.set_int("denoiser_kind", DenoiserKind.Rgb.value)
    rt.add_postproc("OIDenoiser")
    
    # a simple and fast material kernel, dedicated to such visualisation;
    # more generic features of this material are under developent
    rt.setup_material("air", m_volume_color)

    rt.setup_camera("cam1",
        eye=scale * np.array([kx/2, ky/2, 5*kz]),
        target=0.5 * scale * np.array([kx, ky, kz]),
        up=[0, 1, 0],
        fov=35
    )
    
    # Create plot:
    
    rt.set_data(
        "cubes", pos=xyz,
        r=scale, # equivalent to: u=[scale, 0, 0], v=[0, scale, 0], w=[0, 0, scale],
    
        # color values in transmissive materials are interpreted as
        # attenuation length, higher values are more transparent and
        # sepend on actual sizes of objects (not limited to 0-1 range)
        c=8 * colors,
    
        # all cells will have the same shape and size:
        geom="ParallelepipedsConstSize",
        
        # use dedicated material
        mat="air"
    )
    
    rt.show()

    print("done")

if __name__ == '__main__':
    main()
