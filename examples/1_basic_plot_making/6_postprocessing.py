"""
2D postprocessing algorithms.
"""

import numpy as np
from plotoptix import TkOptiX


def main():

    # make some data first:
    rx = (-1, 1.8)
    rz = (-1, 1.8)
    n = 18

    x = np.linspace(rx[0], rx[1], n)
    z = np.linspace(rz[0], rz[1], n)

    X, Z = np.meshgrid(x, z)
    Y = np.sqrt(X ** 2 + Z ** 2)

    # positions of blocks
    data = np.stack((X.flatten(), np.zeros(n*n), Z.flatten())).T
    # heights of blocks
    v = np.zeros(data.shape); v[:,1] = (Y.flatten() + 0.3 * np.random.rand(n*n))[:]

    optix = TkOptiX() # create and configure, show the window later

    optix.set_param(max_accumulation_frames=50)  # accumulate up to 50 frames
    optix.set_background(0)          # black background
    optix.set_ambient([0, 0.2, 0.4]) # cold ambient light

    # add plot geometry
    size_u = 0.98 * (rx[1] - rx[0]) / (n - 1)
    size_w = 0.98 * (rz[1] - rz[0]) / (n - 1)
    optix.set_data("blocks", pos=data,
                   u=[size_u, 0, 0], v=v, w=[0, 0, size_w],
                   c = np.random.rand(n*n),
                   geom="Parallelepipeds")

    # set camera and light position
    optix.setup_camera("cam1", cam_type="DoF",
                       eye=[-0.3, 2, -0.3], target=[1, 1, 1],
                       fov=60, focal_scale=0.85)
    optix.setup_light("light1", pos=[3, 4.5, 1], color=[6, 5, 4.5], radius=2)


    # Postprocessing configuration - uncomment what you'd like to include
    # or comment out all stages to see raw image. Try also combining
    # the mask overlay with one of tonal or levels corrections.

    # 1. Levels adjustment.
    #optix.set_float("levels_low_range", 0.1, 0.05, -0.05)
    #optix.set_float("levels_high_range", 0.9, 0.85, 0.8)
    #optix.add_postproc("Levels")

    # 2. Gamma correction.
    #optix.set_float("tonemap_exposure", 0.8)
    #optix.set_float("tonemap_igamma", 1 / 1.0)
    #optix.add_postproc("Gamma")

    # 3. Tonal correction with a custom curve.
    optix.set_float("tonemap_exposure", 0.8)
    # correction curve set explicitly:
    optix.set_texture_1d("tone_curve_gray", np.sqrt(np.linspace(0, 1, 64)))
    # correction curve calculated from control input/output values
    # (convenient if the curve was prepared in an image editor)
    #optix.set_correction_curve([[13,51], [54, 127], [170, 192]])
    optix.add_postproc("GrayCurve")

    # 4. Tonal correction with a custom RGB curves.
    #optix.set_float("tonemap_exposure", 0.8)
    #optix.set_texture_1d("tone_curve_r", np.sqrt(np.linspace(0.1, 1, 64)))
    #optix.set_texture_1d("tone_curve_g", np.sqrt(np.linspace(0, 1, 64)))
    #optix.set_texture_1d("tone_curve_b", np.sqrt(np.linspace(0, 1, 64)))
    #optix.add_postproc("RgbCurve")

    # 5. Overlay with a mask.
    #mx = (-1, 1)
    #mz = (-1, 1)
    #nm = 20

    #x = np.linspace(mx[0], mx[1], nm)
    #z = np.linspace(mz[0], mz[1], nm)

    #Mx, Mz = np.meshgrid(x, z)
    #M = np.abs(Mx) ** 3 + np.abs(Mz) ** 3
    #M = 1 - (0.6 / np.max(M)) * M

    #optix.set_texture_2d("frame_mask", M)
    #optix.add_postproc("Mask")


    optix.start()
    print("done")


if __name__ == '__main__':
    main()
