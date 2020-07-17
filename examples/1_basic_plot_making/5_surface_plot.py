"""
Surface plot.
"""

import numpy as np
from plotoptix import TkOptiX
from plotoptix.utils import map_to_colors


def f(x, z): return np.sin(np.sqrt(x ** 2 + z ** 2))

def main():

    # make some data first:
    rx = (-15, 5)
    rz = (-10, 10)
    n = 50

    x = np.linspace(rx[0], rx[1], n)
    z = np.linspace(rz[0], rz[1], n)

    X, Z = np.meshgrid(x, z)
    Y = f(X, Z)

    optix = TkOptiX() # create and configure, show the window later

    optix.set_param(max_accumulation_frames=50)  # accumulate up to 50 frames
    optix.set_background(0) # black background
    optix.set_ambient(0.25) # some ambient light


    # try commenting out optional arguments
    optix.set_data_2d("surface", Y,
                      range_x=rx, range_z=rz,
                      c=map_to_colors(Y, "OrRd"),
                      floor_c=[0.05, 0.12, 0.38],
                      floor_y=-1.75,
                      make_normals=True
                     )


    # set camera and light position to fit the scene
    optix.setup_camera("cam1")
    eye = optix.get_camera_eye()
    eye[1] = 0.5 * eye[2]
    optix.update_camera("cam1", eye=eye)

    d = np.linalg.norm(optix.get_camera_target() - eye)
    optix.setup_light("light1", color=8, radius=0.3 * d)

    optix.start()
    print("done")


if __name__ == '__main__':
    main()
