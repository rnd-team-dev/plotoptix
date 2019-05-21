"""
Bar plots 1D.

This example shows how to create bar plots from 1D data.

Parallelepipeds are used to build the plots. In future releases
a convenience method can handle some of the preparatory work.
"""

import numpy as np
from plotoptix import TkOptiX
from plotoptix.utils import map_to_colors  # feature to color conversion


def main():

    # Make some data first (arbitrary scles):

    data = np.random.normal(2, 0.8, 1000)
    h, x = np.histogram(data, bins=15, range=(0, 4))
    s = np.cumsum(h) / np.sum(h)
    e = 6 * np.sqrt(h) / np.sum(h)
    h = 3 * h / np.sum(h)
    bin_size = x[1] - x[0]

    # Create the plot:
    optix = TkOptiX() # create and configure, show the window later

    # accumulate more frames (override default of step=1 and max=4 frames)
    optix.set_param(min_accumulation_step=2, max_accumulation_frames=100)

    # Add data:
    # - pos argument is used for bar positions
    # - u and w vectors are used to set base x and z sizes
    # - v is used to set bar heights.

    ps = np.zeros((h.shape[0], 3)); ps[:,0] = x[:-1]
    vs = np.zeros((h.shape[0], 3)); vs[:,1] = s
    optix.set_data("cumulative", pos=ps, u=[0.9*bin_size, 0, 0], v=vs, w=[0, 0, 0.8*bin_size], c=[0.6, 0, 0], geom="Parallelepipeds")

    ph = np.zeros((h.shape[0], 3)); ph[:,0] = x[:-1]; ph[:,2] = bin_size
    vh = np.zeros((h.shape[0], 3)); vh[:,1] = h
    optix.set_data("density", pos=ph, u=[0.9*bin_size, 0, 0], v=vh, w=[0, 0, 0.8*bin_size], c=0.95, geom="Parallelepipeds")

    pe = np.zeros((e.shape[0], 3)); pe[:,0] = x[:-1]; pe[:,1] = h; pe[:,2] = bin_size
    ve = np.zeros((e.shape[0], 3)); ve[:,1] = e
    optix.set_data("error", pos=pe, u=[0.9*bin_size, 0, 0], v=ve, w=[0, 0, 0.8*bin_size], c=map_to_colors(e, "Blues"), geom="Parallelepipeds")

    # Setup camera and light:
    optix.setup_camera("cam", eye=[x[0], 1.5*np.amax((h,s+e)), x[-1]-x[0]], target=[(x[0]+x[-1])/2, 0, 0])
    optix.setup_light("l1", color=10)

    # show the UI window
    optix.show()

    print("done")

if __name__ == '__main__':
    main()
