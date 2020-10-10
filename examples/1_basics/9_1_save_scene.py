"""
This example shows how to save scene to JSON file. Use 9_2_load_scene.py or
9_3_load_scene.py to restore the scane saved in this script.
"""

import numpy as np
from plotoptix import TkOptiX


def main():

    # Setup the raytracer:

    rt = TkOptiX()

    rt.set_param(min_accumulation_step=2,      # update image every 2 frames
                 max_accumulation_frames=250)  # accumulate 250 frames

    exposure = 1.1; gamma = 2.2
    rt.set_float("tonemap_exposure", exposure)
    rt.set_float("tonemap_gamma", gamma)
    rt.add_postproc("Denoiser")                # Gamma correction, or use AI denoiser.
    #rt.add_postproc("Gamma")                  #    *** but not both together ***

    rt.set_background(0)
    rt.set_ambient(0)

    # Prepare a simple scene objects, camera and lights:

    rt.set_data("plane", geom="Parallelograms", pos=[[-15, 0, -15]], u=[30, 0, 0], v=[0, 0, 30], c=0.94)
    rt.set_data("block1", geom="Parallelepipeds", pos=[[-6, -0.07, -1]], u=[12, 0, 0], v=[0, 0.1, 0], w=[0, 0, 3], c=0.94)
    rt.set_data("block2", geom="Parallelepipeds", pos=[[-6, 0, -1]], u=[12, 0, 0], v=[0, 4, 0], w=[0, 0, 0.1], c=0.94)

    # Setup lights and the camera:

    rt.setup_light("light1", light_type="Parallelogram", pos=[-2.5, 2.6, 3], u=[0.8, 0, 0], v=[0, -0.8, 0], color=[4, 4.7, 5])
    rt.setup_light("light2", light_type="Parallelogram", pos=[-0.5, 3.2, 0], u=[0.8, 0, 0], v=[0, 0, 0.8], color=[6, 5.6, 5.2])
    rt.setup_light("light3", light_type="Parallelogram", pos=[1.5, 2.6, 3], u=[0.8, 0, 0], v=[0, -0.8, 0], color=[4, 4.7, 5])
    rt.setup_camera("cam1", cam_type="DoF", eye=[0, 0.4, 6], target=[0, 1.2, 0], aperture_radius=0.025, fov=35, focal_scale=0.9)

    # Make some data:

    n = 20
    x = np.linspace(-3, 3, n)
    r = 0.8*np.cos(0.4*x)
    y = np.sin(x + 5) + 1.3
    z = np.cos(x + 5) + 0.3
    data = np.stack((x, y, z)).T

    # Add object to scene:
    rt.set_data("balls", pos=data, r=r, c=0.93)
    # or use another geometry and save it in the second scene file:
    #rt.set_data("cubes", geom="Parallelepipeds", pos=data, r=r, c=0.92)

    # Let's start:
    rt.start()

    # Save the scene. This actually can be called also before starting.
    rt.save_scene("strange_object.json")
    #rt.save_scene("strange_object_2.json") # the second file will be usefull in the scene loading example

    print("done")

if __name__ == '__main__':
    main()
