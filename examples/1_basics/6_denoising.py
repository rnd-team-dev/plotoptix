"""
AI denoising applied in 2D postprocessing stage.

Note: denoiser binaries are not installed by default, run::

   python -m plotoptix.install denoiser

to download denoiser.
"""

import numpy as np
from plotoptix import TkOptiX
from plotoptix.enums import DenoiserKind


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
    rt.set_background(0)             # black background
    rt.set_ambient([0.1, 0.15, 0.2]) # cold ambient light

    # add plot geometry
    size_u = 0.98 * (rx[1] - rx[0]) / (n - 1)
    size_w = 0.98 * (rz[1] - rz[0]) / (n - 1)
    rt.set_data("blocks", pos=data,
                u=[size_u, 0, 0], v=v, w=[0, 0, size_w],
                c = np.random.rand(n*n),
                geom="Parallelepipeds")

    # set camera and light position
    rt.setup_camera("cam1", cam_type="DoF",
                    eye=[-0.3, 2, -0.3], target=[1, 1, 1],
                    fov=60, focal_scale=0.85)
    rt.setup_light("light1", pos=[3, 5.5, 1], color=[6.5, 5, 4.5], radius=2)


    # AI denoiser includes exposure and gamma corection, configured with the
    # same variables as the Gamma postprocessing algorithm.
    rt.set_float("tonemap_exposure", 0.5)
    rt.set_float("tonemap_gamma", 2.2)

    # Denoiser blend allows for different mixing with the raw image. Its value
    # can be modified also during the ray tracing.
    rt.set_float("denoiser_blend", 0.0)

    # Denoising is applied by default after the 4th accumulation frames is completed.
    # You can change the starting frame with the following variable:
    rt.set_uint("denoiser_start", 12)

    # Denoiser can use various inputs. By default it is raw RGB and surface
    # albedo, but not always it results with the optimal output quality.
    # Try one of the below settings and find best configuration for your scene. 
    #rt.set_int("denoiser_kind", DenoiserKind.Rgb.value)
    #rt.set_int("denoiser_kind", DenoiserKind.RgbAlbedo.value)
    rt.set_int("denoiser_kind", DenoiserKind.RgbAlbedoNormal.value)

    #rt.add_postproc("Denoiser")
    #rt.add_postproc("DenoiserHDR")
    #rt.add_postproc("DenoiserUp2x")
    rt.add_postproc("OIDenoiser")
    #rt.add_postproc("OIDenoiserHDR")

    # Postprocessing stages are applied after AI denoiser (even if configured
    # in a different order).

    rt.set_float("levels_low_range", 0.1, 0.05, -0.05)
    rt.set_float("levels_high_range", 0.95, 0.85, 0.8)
    rt.add_postproc("Levels")

    rt.start()
    print("done")


if __name__ == '__main__':
    main()
