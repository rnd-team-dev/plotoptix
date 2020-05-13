"""
Create simple mesh in code.
"""

import numpy as np
from plotoptix import TkOptiX


def main():

    # tetrahedron vertices and faces:
    pt = 1/np.sqrt(2)

    points = [
        [1, 0, -pt],
        [-1, 0, -pt],
        [0, 1, pt],
        [0, -1, pt]
    ]

    faces = [
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3]
    ]

    # colors assigned to vertices
    colors = [
        [0.7, 0, 0],
        [0, 0.7, 0],
        [0, 0, 0.7],
        [1, 1, 1],
    ]

    rt = TkOptiX() # create and configure, show the window later

    rt.set_param(min_accumulation_step=2, max_accumulation_frames=100)
    rt.setup_camera("cam1", cam_type="DoF", eye=[1, -6, 4], focal_scale=0.9, fov=25)
    rt.setup_light("light1", pos=[10, -9, -8], color=[14, 13, 12], radius=4)
    rt.set_ambient([0.2, 0.3, 0.4])

    # add mesh geometry to the scene
    rt.set_mesh("m", points, faces, c=colors)

    rt.start()
    print("done")


if __name__ == '__main__':
    main()
