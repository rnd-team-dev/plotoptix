"""
2D postprocessing algorithms.
"""

import numpy as np
from plotoptix import TkOptiX
from plotoptix.utils import make_color_2d

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

    rt = TkOptiX() # create and configure, show the window later

    rt.set_param(max_accumulation_frames=50)  # accumulate up to 50 frames
    rt.set_background(0)          # black background
    rt.set_ambient([0, 0.2, 0.4]) # cold ambient light

    # add plot geometry
    size_u = 0.98 * (rx[1] - rx[0]) / (n - 1)
    size_w = 0.98 * (rz[1] - rz[0]) / (n - 1)
    rt.set_data("blocks", pos=data,
                u=[size_u, 0, 0], v=v, w=[0, 0, size_w],
                c = np.random.rand(n*n),
                geom="Parallelepipeds"
    )

    # set camera and light position
    rt.setup_camera("cam1", cam_type="DoF",
                    eye=[-0.3, 2, -0.3], target=[1, 1, 1],
                    fov=60, focal_scale=0.85
    )
    rt.setup_light("light1", pos=[3, 4.5, 1], color=[6, 5, 4.5], radius=2)


    # Postprocessing configuration - uncomment what you'd like to include
    # or comment out all stages to see raw image. Try also combining
    # the mask overlay with one of tonal or levels corrections.

    # 1. Levels adjustment.
    #rt.set_float("levels_low_range", 0.1, 0.05, -0.05)
    #rt.set_float("levels_high_range", 0.9, 0.85, 0.8)
    #rt.add_postproc("Levels")

    # 2. Gamma correction.
    #rt.set_float("tonemap_exposure", 0.8)
    #rt.set_float("tonemap_gamma", 2.2)
    #rt.add_postproc("Gamma")

    # 3. Tonal correction with a custom curve.
    rt.set_float("tonemap_exposure", 0.8)
    # correction curve set explicitly:
    rt.set_texture_1d("tone_curve_gray", np.sqrt(np.linspace(0, 1, 64)))
    # correction curve calculated from control input/output values
    # (convenient if the curve was prepared eg in an image editor)
    #rt.set_correction_curve([[13,51], [54, 127], [170, 192]])
    rt.add_postproc("GrayCurve")

    # 4. Tonal correction with a custom RGB curves.
    #rt.set_float("tonemap_exposure", 0.8)
    #rt.set_texture_1d("tone_curve_r", np.sqrt(np.linspace(0.15, 1, 64)))
    #rt.set_texture_1d("tone_curve_g", np.sqrt(np.linspace(0, 1, 64)))
    #rt.set_texture_1d("tone_curve_b", np.sqrt(np.linspace(0, 1, 64)))
    #rt.add_postproc("RgbCurve")

    # 5. Overlay with a mask.
    #mx = (-1, 1)
    #mz = (-1, 1)
    #nm = 20

    #x = np.linspace(mx[0], mx[1], nm)
    #z = np.linspace(mz[0], mz[1], nm)

    #Mx, Mz = np.meshgrid(x, z)
    #M = np.abs(Mx) ** 3 + np.abs(Mz) ** 3
    #M = 1 - (0.9 / np.max(M)) * M

    #M = make_color_2d(M, channel_order="RGBA")

    #rt.set_texture_2d("frame_mask", M)
    #rt.add_postproc("Mask")


    rt.start()
    print("done")


if __name__ == '__main__':
    main()
