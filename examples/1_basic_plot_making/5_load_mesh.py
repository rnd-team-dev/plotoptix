"""
Load mesh from Wavefront .obj file.
"""

import numpy as np
from plotoptix import TkOptiX


def main():

    optix = TkOptiX() # create and configure, show the window later

    optix.set_param(max_accumulation_frames=100)  # accumulate up to 100 frames
    optix.set_background(0) # black background
    optix.set_ambient(0.25) # some ambient light


    # original Stanford Utah teapot (try also with default value of make_normals=False):
    #optix.load_mesh_obj("https://graphics.stanford.edu/courses/cs148-10-summer/as3/code/as3/teapot.obj",
    #                    "teapot", c=0.92, make_normals=True)

    # another public-domain version with normals included:
    optix.load_mesh_obj("data/utah-teapot.obj", "teapot", c=0.92)


    # camera and light position auto-fit the scene geometry
    optix.setup_camera("cam1")
    d = np.linalg.norm(optix.get_camera_target() - optix.get_camera_eye())
    optix.setup_light("light1", color=10, radius=0.3 * d)

    optix.start()
    print("done")


if __name__ == '__main__':
    main()
