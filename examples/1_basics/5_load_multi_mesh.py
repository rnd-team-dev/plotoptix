"""
Load multiple mesh from Wavefront .obj file, assign materials
by mesh names, set parent for all meshes.
Select (double-click) parent object (head) to rotate, scale, etc,
all meshes tohether. Select any other object to manipulate it
individually.
"""

import numpy as np
from plotoptix import TkOptiX
from plotoptix.materials import m_diffuse, m_transparent_plastic, m_matt_plastic


def main():

    rt = TkOptiX() # create and configure, show the window later

    rt.set_param(max_accumulation_frames=100)  # accumulate up to 100 frames
    rt.set_background(0.99) # white background
    rt.set_ambient(0.25) # some ambient light

    # setup materials:
    m_diffuse["VarFloat3"] = { "base_color": [ 0.15, 0.17, 0.2 ] }
    rt.update_material("diffuse", m_diffuse)

    m_matt_plastic["VarFloat3"]["base_color"] = [ 0.5, 0.1, 0.05 ]
    rt.setup_material("plastic", m_matt_plastic)

    rt.load_texture("wing", "data/wing.png")

    m_transparent_plastic["ColorTextures"] = [ "wing" ]
    rt.setup_material("transparent", m_transparent_plastic)

    rt.load_normal_tilt("transparent", "data/wing.png", prescale=0.002)

    # prepare dictionary and load meshes; note that both eyes and wings
    # are assigned with single material by providing only a part of
    # the mesh name:
    materials = { "eye": "plastic", "wing": "transparent" }
    rt.load_multiple_mesh_obj("data/fly.obj", materials, parent="head_Icosphere")

    # camera and light position auto-fit the scene geometry
    rt.setup_camera("cam1")
    d = np.linalg.norm(rt.get_camera_target() - rt.get_camera_eye())
    rt.setup_light("light1", color=10, radius=0.3 * d)

    rt.start()
    print("done")


if __name__ == '__main__':
    main()
