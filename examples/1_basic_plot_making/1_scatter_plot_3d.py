"""
Scatter plot 3D.

This example shows how to:
   - create a basic scatter plot
   - present data feature as a color or size
   - use matplotlib color map (see https://matplotlib.org/users/colormaps.html)
   - turn on coordinate system (under development!).
"""

import numpy as np
from plotoptix import TkOptiX
from plotoptix.utils import map_to_colors  # feature to color conversion


def main():

    # Make some data first:

    n = 50000

    xyz = 3 * (np.random.random((n, 3)) - 0.5)
    r = 0.1 * (1 - (xyz[:,0]/3 + 0.5)) + 0.02
    s = np.sum(xyz, axis=1)

    particles = xyz[s > 0.2]
    rp = r[s > 0.2]
    # Use xyz positions to calculate RGB color components:
    cp = particles/3 + 0.5

    cubes = xyz[s < -0.2]
    rc = r[s < -0.2]
    # Map Y-coordinate to matplotlib's color map RdYlBu: map_to_colors()
    # function is automatically scaling the data to fit <0; 1> range.
    # Any other mapping is OK, just keep the result in shape
    # (n-data-points, 3), where 3 stands for RGB color
    # components.
    cc = map_to_colors(cubes[:,1], "RdYlBu")

    # Create the plots:

    optix = TkOptiX() # create and configure, show the window later

    # white background
    optix.set_background(0.99)

    # add plots, ParticleSet geometry is default
    optix.set_data("particles", pos=particles, r=rp, c=cp)
    # and use geom parameter to specify cubes Parallelepipeds geometry;
    # Parallelepipeds can be described precisely with U, V, W vectors,
    # but here we only provide the r parameter - this results with
    # randomly rotated cubes of U, V, W lenghts equal to r 
    optix.set_data("cubes", pos=cubes, r=rc, c=cc, geom="Parallelepipeds")

    # if you prefer cubes aligned with xyz:
    #nc = rc.shape[0]
    #u = np.zeros((nc,3)); u[:,0] = rc[:]
    #v = np.zeros((nc,3)); v[:,1] = rc[:]
    #w = np.zeros((nc,3)); w[:,2] = rc[:]
    #optix.set_data("cubes", pos=cubes, u=u, v=v, w=w, c=cc, geom="Parallelepipeds")

    # show coordinates box
    optix.set_coordinates()

    # show the UI window here - this method is calling some default
    # initialization for us, e.g. creates camera, so any modification
    # of these defaults should come below (or we provide on_initialization
    # callback)
    optix.show()

    # camera and lighting configured by hand
    optix.update_camera(eye=[5, 0, -8])
    optix.setup_light("light1", color=15*np.array([0.99, 0.9, 0.7]), radius=2)

    # accumulate up to 30 frames (override default of 4 frames)
    optix.set_param(max_accumulation_frames=30)

    print("done")

if __name__ == '__main__':
    main()
