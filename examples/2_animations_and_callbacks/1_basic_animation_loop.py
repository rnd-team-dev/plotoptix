"""
Animation pattern.
"""

import numpy as np
from plotoptix import TkOptiX

class params:
    eye = [0, 0, 9]
    t = 0

# Perform any calculations here, including CPU-extensive work. But do
# not access ray tracing buffers, and do not update anything in the
# scene (geometry or plots data, lights, cameras, materials).
# Code in this function runs in parallel to the ray tracing.
def compute_changes(rt: TkOptiX, delta: int) -> None:
    params.eye = [9*np.sin(params.t), 3, 9*np.cos(params.t)]
    params.t += 0.05

# Access ray tracing buffers, modify/update scene, but do not launch
# time consuming work here as this will delay the next frame ray tracing.
# Code in this function runs synchronously with the ray tracing launches.
def update_scene(rt: TkOptiX) -> None:
    rt.update_camera(eye=params.eye)

def main():
    rt = TkOptiX(on_scene_compute=compute_changes, on_rt_completed=update_scene)

    # 4 accumulation passes on each compute-update cycle:
    rt.set_param(min_accumulation_step=4)

    n = 1000000
    xyz = 3 * (np.random.random((n, 3)) - 0.5)
    r = 0.02 * np.random.random(n) + 0.002

    rt.set_data("plot", xyz, r=r)
    rt.set_ambient(0.9);

    rt.setup_camera("cam", eye=params.eye)

    rt.start()

if __name__ == '__main__':
    main()
