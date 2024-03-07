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

    # Create plots:

    rt = TkOptiX() # create and configure, show the window later

    # accumulate up to 30 frames (override default of 4 frames)
    rt.set_param(max_accumulation_frames=30)

    # white background
    rt.set_background(0.99)

    # add plots, ParticleSet geometry with variable radius is default
    rt.set_data("particles", pos=particles, r=rp, c=cp)
    #rt.set_data("particles", pos=particles, r=0.02, c=cp, geom="ParticleSetConstSize")
    
    # and use geom parameter to specify cubes geometry;
    # Parallelepipeds can be described precisely with U, V, W vectors,
    # but here we only provide the r parameter - this results with
    # randomly rotated cubes of U, V, W lenghts equal to r 
    rt.set_data("cubes", pos=cubes, r=rc, c=cc, geom="Parallelepipeds")

    # tetrahedrons look good as well, and they are really fast on RTX devices:
    #rt.set_data("tetras", pos=cubes, r=rc, c=cc, geom="Tetrahedrons")

    # if you prefer cubes aligned with xyz:
    #rt.set_data("cubes", pos=cubes, r=rc, c=cc, geom="Parallelepipeds", rnd=False)

    # or if you'd like to fix some edges:
    #v = np.zeros((rc.shape[0], 3)); v[:,1] = rc[:]
    #rt.set_data("cubes", pos=cubes, u=[0.05,0,0], v=v, w=[0,0,0.05], c=cc, geom="Parallelepipeds")
    
    # or maybe fix geometry of all primitives (note "ConstSize" in geom name):
    #rt.set_data("cubes", pos=cubes, u=[0.02,0,0], v=[0,0.05,0], w=[0,0,0.08], c=cc, geom="ParallelepipedsConstSize")
    # or, if cubes are needed, simpy:
    #rt.set_data("cubes", pos=cubes, r=0.05, c=cc, geom="ParallelepipedsConstSize")

    # show coordinates box
    rt.set_coordinates()

    # show the UI window here - this method is calling some default
    # initialization for us, e.g. creates camera, so any modification
    # of these defaults should come below (or we provide on_initialization
    # callback)
    rt.show()

    # camera and lighting configured by hand
    rt.update_camera(eye=[5, 0, -8])
    rt.setup_light("light1", color=10*np.array([0.99, 0.9, 0.7]), radius=2)

    print("done")

if __name__ == '__main__':
    main()
