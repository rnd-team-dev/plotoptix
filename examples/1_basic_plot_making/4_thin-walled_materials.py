"""
This example shows how to apply thin-walled materials, colorized and textured.
"""

import numpy as np
from plotoptix import TkOptiX
from plotoptix.materials import m_clear_glass, m_thin_walled
from plotoptix.utils import map_to_colors, make_color_2d, read_image
from plotoptix.enums import RtFormat

def main():

    # Setup the raytracer:

    rt = TkOptiX()

    rt.set_param(min_accumulation_step=2,      # update image every 2 frames
                 max_accumulation_frames=250)  # accumulate 250 frames
    rt.set_uint("path_seg_range", 5, 15)       # allow many reflections/refractions

    exposure = 1.5; gamma = 2.2
    rt.set_float("tonemap_exposure", exposure)
    rt.set_float("tonemap_gamma", gamma)
    rt.add_postproc("Denoiser")                # AI denoiser, or the gamma correction only.
    #rt.add_postproc("Gamma")                  #    *** but not both together ***

    rt.set_background(0)
    rt.set_ambient(0)

    # Setup texture:

    # in one go, with gamma and exposure parameters saved to the scene:
    rt.load_texture("rainbow", "data/rainbow.jpg", prescale=0.3, baseline=0.7, gamma=2.2)

    # or through ndarray, allowing for custom processing:
    #rainbow = 0.7 + 0.3 * read_image("data/rainbow.jpg", normalized=True)
    #rainbow = make_color_2d(rainbow, gamma=2.2, channel_order="RGBA")
    #rt.set_texture_2d("rainbow", rainbow)

    # Setup materials:

    m_thin2 = m_thin_walled.copy()             # textured material based on the predefined thin-walled material
    m_thin2["ColorTextures"] = [ "rainbow" ]   # reference texture by name

    rt.setup_material("glass", m_clear_glass)
    rt.setup_material("thin", m_thin_walled)
    rt.setup_material("thin2", m_thin2)

    # Prepare a simple scene objects, camera and lights:

    rt.set_data("plane", geom="Parallelograms", pos=[[-15, 0, -15]], u=[30, 0, 0], v=[0, 0, 30], c=0.94)
    rt.set_data("block1", geom="Parallelepipeds", pos=[[-6, -0.07, -1]], u=[12, 0, 0], v=[0, 0.1, 0], w=[0, 0, 3], c=0.94)
    rt.set_data("block2", geom="Parallelepipeds", pos=[[-6, 0, -1]], u=[12, 0, 0], v=[0, 4, 0], w=[0, 0, 0.1], c=0.94)

    # Setup lights and the camera:

    rt.setup_light("light1", light_type="Parallelogram", pos=[-2.5, 2.5, 3], u=[0.8, 0, 0], v=[0, -0.8, 0], color=[6, 5.7, 5.4])
    rt.setup_light("light2", light_type="Parallelogram", pos=[-0.5, 3.2, 0], u=[0.8, 0, 0], v=[0, 0, 0.8], color=[8, 7.6, 7.2])
    rt.setup_light("light3", light_type="Parallelogram", pos=[1.5, 2.5, 3], u=[0.8, 0, 0], v=[0, -0.8, 0], color=[6, 5.7, 5.4])
    rt.setup_camera("cam1", cam_type="DoF", eye=[0, 0.4, 6], target=[0, 1, 0], aperture_radius=0.025, fov=35, focal_scale=0.9)

    # Make some bubbles:

    n = 20
    x = np.linspace(-3, 3, n)
    r = 0.3*np.cos(0.4*x)

    y = 0.8*np.sin(x) + 1.2
    z = 0.8*np.cos(x) + 0.4
    b1 = np.stack((x, y, z)).T
    rt.set_data("bubbles1", mat="thin", pos=b1, r=r, c=0.5+0.5*map_to_colors(x, "rainbow"))

    y = 0.8*np.sin(x + 3) + 1.2
    z = 0.8*np.cos(x + 3) + 0.4
    b2 = np.stack((x, y, z)).T
    rt.set_data("bubbles2", geom="ParticleSetTextured", mat="thin2", pos=b2, r=r)

    y = np.sin(x + 2) + 1.2
    z = np.cos(x + 2) + 0.4
    b3 = np.stack((x, y, z)).T
    rt.set_data("beads", mat="glass", pos=b3, r=0.75*r, c=10)

    y = np.sin(x + 5) + 1.3
    z = np.cos(x + 5) + 0.3
    b4 = np.stack((x, y, z)).T
    rt.set_data("balls", pos=b4, r=0.75*r, c=0.92)

    # Let's start:
    rt.start()

    print("done")

if __name__ == '__main__':
    main()
