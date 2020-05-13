"""
Generate mesh with pygmsh.
See https://github.com/nschloe/pygmsh for installation.
Note: any package with meshio (https://github.com/nschloe/meshio) interface is fine.

Hint: double-click an object to select it, then drag with mouse to rotate it. Double-click
again in the empty area to select camera. Drag with mouse to rotate camera. ;)
"""

import pygmsh
import numpy as np
from plotoptix import TkOptiX

def main():

    # a mesh from pygmsh example:
    geom = pygmsh.built_in.Geometry()

    # Draw a cross.
    poly = geom.add_polygon([
        [ 0.0,  0.5, 0.0],
        [-0.1,  0.1, 0.0],
        [-0.5,  0.0, 0.0],
        [-0.1, -0.1, 0.0],
        [ 0.0, -0.5, 0.0],
        [ 0.1, -0.1, 0.0],
        [ 0.5,  0.0, 0.0],
        [ 0.1,  0.1, 0.0]
        ],
        lcar=0.05
    )

    axis = [0, 0, 1]

    geom.extrude(
        poly,
        translation_axis=axis,
        rotation_axis=axis,
        point_on_axis=[0, 0, 0],
        angle=2.0 / 6.0 * np.pi
    )

    mesh = pygmsh.generate_mesh(geom)

    rt = TkOptiX() # create and configure, show the window later

    rt.set_param(min_accumulation_step=2, max_accumulation_frames=100)
    rt.setup_camera("cam1", cam_type="DoF",
                    eye=[0.5, -3, 2], target=[0, 0, 0.5],
                    focal_scale=0.9, fov=25)
    rt.setup_light("light1", pos=[15, 0, 10], color=[14, 13, 12], radius=4)
    rt.set_ambient([0.1, 0.15, 0.2])

    # add mesh geometry to the scene
    rt.set_mesh("m", mesh.points, mesh.cells_dict['triangle'])

    rt.start()
    print("done")


if __name__ == '__main__':
    main()
