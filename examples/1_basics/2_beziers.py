"""
With ``Beziers`` geometry you can setup all control points and segments
can be placed independently in contrast to other curve geometries where
all data points are connected or interpolated with a single curve.

This example shows how to:
   - create a large set of independent bezier segments
   - assign individual colors and radii to each segment
"""

import math
import numpy as np
from plotoptix import TkOptiX

def random3d(n):
    phi = np.random.uniform(0, 2 * np.pi, size=n)
    costheta = np.random.uniform(-1, 1, size=n)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack([x, y, z], axis=1)

def main():

    # create some data

    n = 150000

    # origins at (0,0,0)
    xyz = np.zeros((n, 4, 3))

    # control points #1 randomly distributed on the sphere
    xyz[:,1,:] = random3d(n)

    # endpoints follow the above direction, with some randomization
    xyz[:,3,:] = 2 * xyz[:,1,:] + np.array([0, -0.75, 0])
    xyz[:,3,:] += 0.75 * (np.random.rand(n, 3) - 0.5)

    # control points #2 placed ~over the endpoints, so the curves look like falling down 
    xyz[:,2,:] = xyz[:,3,:] + np.array([0, 0.5, 0])
    xyz[:,2,:] += 0.2 * (np.random.rand(n, 3) - 0.5)

    # reshape to have a single sequence of origin-ctrl#1-ctrl#2-endpoint for all segments
    xyz = xyz.reshape((4 * n, 3))

    c = np.array([
        [0.4, 0.1, 0.1],
        [0.7, 0.1, 0.1],
        [0.9, 0.7, 0.9],
        [0.93, 0.93, 0.93],
    ])
    c = np.tile(c, (n, 1))
    c += 0.3 * (np.random.rand(4 * n, 3) - 0.5)
    c = np.clip(c, 0, 1)

    r = np.array([0.003, 0.008, 0.002, 0.0002])
    r = np.tile(r, n)
    print(xyz.shape, c.shape, r.shape)

    # setup ray tracing

    rt = TkOptiX()

    rt.set_uint("path_seg_range", 8, 24)    # more segments to let the light enter between strands
    rt.set_background(0.99)                 # white background
    rt.set_ambient(0.15)                    # dim ambient light

    rt.set_data("curve", pos=xyz, r=r, c=c, geom="Beziers")

    rt.setup_camera("cam1", # cam_type="DoF",
                    eye=[-6, 4, 5], target=[0, -0.6, 0],
                    glock=True
    )

    rt.setup_light("light1", pos=[-6, 10, 10], color=10*np.array([0.99, 0.9, 0.7]), radius=3, in_geometry=True)
    rt.setup_light("light2", pos=[-6, -10, 5], color=20*np.array([0.7, 0.9, 0.99]), radius=2, in_geometry=True)

    rt.set_param(max_accumulation_frames=256)

    rt.show()

    print("done")

if __name__ == '__main__':
    main()
