"""
Load mesh from Wavefront .obj file.
"""

import numpy as np
from plotoptix import TkOptiX


def main():

    rt = TkOptiX() # create and configure, show the window later

    rt.set_param(max_accumulation_frames=100)  # accumulate up to 100 frames
    rt.set_background(0) # black background
    rt.set_ambient(0.25) # some ambient light


    # original Stanford Utah teapot (try also with default value of make_normals=False):
    # note: this file has no named objects specified, and you have to use load_merged_mesh_obj
    # method
    #rt.load_merged_mesh_obj("https://graphics.stanford.edu/courses/cs148-10-summer/as3/code/as3/teapot.obj",
    #                        "teapot", c=0.92, make_normals=True)

    # another public-domain version with normals included:
    rt.load_mesh_obj("data/utah-teapot.obj", c=0.92)

    # camera and light position auto-fit the scene geometry
    rt.setup_camera("cam1")
    d = np.linalg.norm(rt.get_camera_target() - rt.get_camera_eye())
    rt.setup_light("light1", color=8, radius=0.3 * d)

    rt.start()
    print("done")


if __name__ == '__main__':
    main()
