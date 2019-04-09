"""
Parametric line plot 3D.

This example shows how to:
   - create a line plot using bezier curves
   - present data feature as a line thickness
   - setup 2-color lighting
"""

import math
import numpy as np
from plotoptix import TkOptiX
from plotoptix.utils import map_to_colors  # feature to color conversion


def main():

    # Make some data first:

    n = 3600
    s = 50

    m =  -0.97
    # try various m: 0.4, 2.48, -1.383, -2.03, ...

    m = math.exp(m)
    k = 1

    xyz = [np.array([0, s * (math.e - 2), 0])]

    t = 0
    while t < n:
        rad = math.pi * t / 180
        f = s * (math.exp(math.cos(rad)) - 2 * math.cos(4 * rad) - math.pow(math.sin(rad/m), 5))
        xyz.append(np.array([f * math.sin(rad), f * math.cos(rad), f * math.sin(2*rad)]))
        t += k

    r = np.linalg.norm(xyz, axis=1)
    r = 5 - (5 / r.max()) * r + 0.02

    # Create the plots:

    optix = TkOptiX() # create and configure, show the window later

    optix.set_background(0.99) # white background
    optix.set_ambient(0.2)     # dim ambient light

    # add plot, BezierChain geometry makes a smooth line interpolating data points
    optix.set_data("curve", pos=xyz, r=r, c=0.9, geom="BezierChain")

    # show the UI window here - this method is calling some default
    # initialization for us, e.g. creates camera, so any modification
    # of these defaults should come below (or we provide on_initialization
    # callback)
    optix.show()

    # camera auto-configured to fit the plot
    optix.camera_fit()

    # 2 spherical light sources, warm and cool, fit positions with respect to
    # the current camera plane: 45 deg right/left and 25 deg up;
    # do not include lights in geometry, so they do not appear in the image
    optix.setup_light("light1", color=20*np.array([0.99, 0.9, 0.7]), radius=250, in_geometry=False)
    optix.light_fit("light1", horizontal_rot=45, vertical_rot=25, dist_scale=1.1)
    optix.setup_light("light2", color=25*np.array([0.7, 0.9, 0.99]), radius=200, in_geometry=False)
    optix.light_fit("light2", horizontal_rot=-45, vertical_rot=25, dist_scale=1.1)

    # accumulate up to 30 frames (override default of 4 frames)
    optix.set_param(max_accumulation_frames=200)

    print("done")

if __name__ == '__main__':
    main()
