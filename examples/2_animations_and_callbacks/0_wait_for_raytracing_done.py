import numpy as np
from plotoptix import NpOptiX

import threading

class params:
    done = threading.Event()
    k = 0

def accum_done(rt: NpOptiX) -> None:
    print("callback")
    params.k += 1
    params.done.set()

def main():
    rt = NpOptiX(on_rt_accum_done=accum_done, width=800, height=500)
    rt.set_param(min_accumulation_step=16, max_accumulation_frames=16)

    n = 1000000
    xyz = 3 * (np.random.random((n, 3)) - 0.5)
    r = 0.02 * np.random.random(n) + 0.002

    rt.set_data("plot", xyz, r=r)
    rt.set_ambient(0.9);

    rt.start()

    print("working...")

    # Wait for a signal from the callback function.
    if params.done.wait(10):
        print("frame 1 done")
    else:
        print("timeout")

    rt.save_image("frame_1.jpg")

    # Average of RGB pixel values (skip alpha), just to illustrate access to the image data.
    print(np.sum(rt._img_rgba[..., :3]) / (rt._img_rgba.shape[0]*rt._img_rgba.shape[1]))

    params.done.clear()

    # Any update to the scene will trigger ray tracing.
    rt.update_camera(eye=[5, 0, -8])

    print("working...")

    if params.done.wait(10):
        print("frame 2 done")
    else:
        print("timeout")

    rt.save_image("frame_2.jpg")

    print(np.sum(rt._img_rgba[..., :3]) / (rt._img_rgba.shape[0]*rt._img_rgba.shape[1]))

    print("Callback executed %s times." % params.k)

    rt.close()

if __name__ == '__main__':
    main()
