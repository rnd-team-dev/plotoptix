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

    rt = TkOptiX() # create and configure, show the window later

    rt.set_param(max_accumulation_frames=50)  # accumulate up to 50 frames
    rt.set_background(0) # black background
    rt.set_ambient(0.25) # some ambient light


    # try commenting out optional arguments
    rt.set_data_2d("surface", Y,
                   range_x=rx, range_z=rz,
                   c=map_to_colors(Y, "OrRd"),
                   floor_c=[0.05, 0.12, 0.38],
                   floor_y=-1.75,
                   make_normals=True
                  )

    # add wireframe above the surface
    rt.set_data_2d("wireframe", Y + 3,
                   range_x=rx, range_z=rz,
                   r=0.06 * np.abs(Y),
                   geom="Graph",
                   c=0.92
                  )

    # set camera and light position to fit the scene
    rt.setup_camera("cam1")
    eye = rt.get_camera_eye()
    eye[1] = 0.5 * eye[2]
    rt.update_camera("cam1", eye=eye)

    d = np.linalg.norm(rt.get_camera_target() - eye)
    rt.setup_light("light1", color=8, radius=0.3 * d)

    rt.start()

    print("done")

if __name__ == '__main__':
    main()
