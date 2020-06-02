import numpy as np

# docs, examples: https://plotoptix.rnd.team
from plotoptix import TkOptiX
from plotoptix.materials import m_plastic
from plotoptix.utils import simplex

def main():
    b = 7000   # number of curves
    n = 60     # number pf nodes per curve
    dt = 0.08  # nodes distance

    ofs = 50 * np.random.rand(3)
    inp = 5 * np.random.rand(b, 3, 4) - 2.5
    for c in range(b):
        inp[c,1:,:3] = inp[c,0,:3]
        inp[c,:,0] *= 1.75            # more spread in X
        inp[c,:,3] = ofs              # sync the 4'th dim of the noise
    #print(inp)

    pos = np.zeros((b, n, 3), dtype=np.float32)
    r = np.zeros((b, n), dtype=np.float32)

    rnd = simplex(inp)
    #print(rnd)

    for t in range(n):
        rt = 2.0 * (t+1) / (n+2) - 1
        rt = 1 - rt*rt
        r[:,t] = 0.07 * rt * rt
        for c in range(b):
            mag = np.linalg.norm(rnd[c])
            r[c,t] *= 0.2 + 0.8 * mag      # modulate thickness
        
            rnd[c] = (dt/mag) * rnd[c]     # normalize and scale the step size
            inp[c,:,:3] += rnd[c]          # step in the field direction
            pos[c,t] = inp[c,0,:3]
        
        rnd = simplex(inp, rnd)            # calculate noise at the next pos

    rt = TkOptiX(start_now=False)
    rt.set_param(
        min_accumulation_step=1,
        max_accumulation_frames=200,
        rt_timeout=100000                  # accept lower fps
    )

    exposure = 1.2; gamma = 1.8
    rt.set_float("tonemap_exposure", exposure)
    rt.set_float("tonemap_gamma", gamma)
    rt.set_float("denoiser_blend", 0.25)
    rt.add_postproc("Denoiser")

    rt.setup_material("plastic", m_plastic)

    for c in range(b):
        if np.random.uniform() < 0.05:
            rt.set_data("c"+str(c), pos=pos[c], r=1.1*r[c], c=[0.4, 0, 0], geom="BezierChain")
        else:
            rt.set_data("c"+str(c), pos=pos[c], r=r[c], c=0.94, geom="BezierChain", mat="plastic")
        
    rt.setup_camera("dof_cam", eye=[0, 0, 12], target=[0, 0, 0], fov=57, focal_scale=0.7, cam_type="DoF")

    rt.setup_light("l1", pos=[8, -3, 13], color=1.5*np.array([0.99, 0.97, 0.93]), radius=5)
    rt.setup_light("l2", pos=[-17, -7, 5], u=[0, 0, -10], v=[0, 14, 0], color=1*np.array([0.25, 0.28, 0.35]), light_type="Parallelogram")
    rt.set_ambient([0.05, 0.07, 0.09])
    rt.set_background(0)
    rt.show()

    print("done")

if __name__ == '__main__':
    main()
