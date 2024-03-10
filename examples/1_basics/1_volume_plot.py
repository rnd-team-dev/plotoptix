"""
Opaque volume plot.

This example shows how to:
   - use const-sized cubes to plot volume shape (opaque)
   - present data feature as a color
   - use matplotlib color map (see https://matplotlib.org/users/colormaps.html)
"""

import numpy as np
from plotoptix import TkOptiX
from plotoptix.utils import map_to_colors  # feature to color conversion
from plotoptix.enums import DenoiserKind   # deniser options

from plotoptix.utils import simplex        # noise generator
_ = simplex(np.zeros((3,3)))


def main():

    # Make some data first:

    kx, ky, kz = (500, 500, 250)
    scale = 0.02

    x = np.linspace(0, kx, kx)
    y = np.linspace(0, ky, ky)
    z = np.linspace(0, kz, kz)
    xv, yv, zv = np.meshgrid(x, y, z)
    xyz = scale * np.stack((xv, yv, zv), axis=-1)

    vol = simplex(xyz)
    print("Volume shape:", vol.shape)
    
    vmin, vmax = (-0.1, 0.1)

    selection = (vol > vmin) & (vol < vmax)
    sel_cubes = xyz[selection]
    colors = map_to_colors(vol[selection], "Reds")

    print(f"{np.sum(selection) / 1.0e6 :.1f}M seleced cells.")

    # Setup ray-tracing:

    rt = TkOptiX() # create and configure, show the window later

    max_frames = 32 # accumulation frames (rays/pixel in other words)
    rt.set_param(
        min_accumulation_step=4,
        max_accumulation_frames=max_frames
    )

    rt.set_float("scene_epsilon", 1.0e-05) # expect small cubes, lower epsilon to ensure details are correctly presented
    rt.set_uint("path_seg_range", 4, 16)   # shape can have copmplex cavities that look better with multiple ray scattering

    rt.set_background(0.99) # plot background color
    rt.set_ambient(0.8)     # uniform environment light

    # setup tone mapping and denoiser for best image quality
    exposure = 1.1
    gamma = 2.2
    rt.set_float("tonemap_exposure", exposure)
    rt.set_float("tonemap_gamma", gamma)
    rt.set_uint("denoiser_start", max_frames) # denoise only the final image
    rt.set_int("denoiser_kind", DenoiserKind.RgbAlbedoNormal.value)
    rt.add_postproc("OIDenoiser")

    rt.setup_camera("cam1",
        eye=scale * np.array([kx/2, ky/2, 5*kz]),
        target=0.5 * scale * np.array([kx, ky, kz]),
        up=[0, 1, 0],
        fov=35
    )
    
    # Create plot:
    
    rt.set_data(
        "cubes", pos=sel_cubes,
        r=scale, # equivalent to: u=[scale, 0, 0], v=[0, scale, 0], w=[0, 0, scale],
    
        # constant color:
        #c=0.9,
    
        # or use color data for each cell:
        c=0.95 * colors,
    
        # all cells will have the same shape and size:
        geom="ParallelepipedsConstSize"
    )
    
    rt.show()

    print("done")

if __name__ == '__main__':
    main()
