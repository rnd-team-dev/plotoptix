"""
Direct update of geometry data on device.

This example uses numpy arrays, but pytorch tensors are supported as well,
just note that rt.enable_torch() call is required before using pytorch.
"""

import numpy as np

from plotoptix import TkOptiX
from plotoptix.utils import simplex


class params():
    n = 100
    rx = (-20, 20)
    r = 0.85 * 0.5 * (rx[1] - rx[0]) / (n - 1)

    x = np.linspace(rx[0], rx[1], n)
    z = np.linspace(rx[0], rx[1], n)
    X, Z = np.meshgrid(x, z)

    data = np.stack([X.flatten(), np.zeros(n*n), Z.flatten()], axis=1)
    t = 0
    
# Compute updated geometry data.
def compute(rt, delta):
    row = np.ones((params.data.shape[0], 1))
    xn = simplex(np.concatenate([params.data, params.t * row], axis=1))
    yn = simplex(np.concatenate([params.data, (params.t + 20) * row], axis=1))
    zn = simplex(np.concatenate([params.data, (params.t - 20) * row], axis=1))
    dv = np.stack([xn, yn, zn], axis=1)
    params.data += 0.02 * dv
    params.t += 0.05
    
# Fast copy to geometry buffer on device, without making a host copy.
def update_data(rt):
    rt.update_raw_data("balls", pos=params.data)


def main():
    rt = TkOptiX(
        on_scene_compute=compute,
        on_rt_completed=update_data
    )
    rt.set_param(min_accumulation_step=10, max_accumulation_frames=16)
    rt.set_background(0)
    rt.set_ambient(0)

    rt.setup_camera("cam1", cam_type="ThinLens", eye=[3.5, 1.27, 3.5], target=[0, 0, 0], fov=30, glock=True)
    rt.setup_light("light1", pos=[4, 5, 5], color=18, radius=1.0)

    exposure = 1.0; gamma = 2.2
    rt.set_float("tonemap_exposure", exposure)
    rt.set_float("tonemap_gamma", gamma)
    rt.add_postproc("Gamma")

    rt.set_data("balls", pos=params.data, c=0.82, r=params.r)

    rt.start()

if __name__ == '__main__':
    main()

