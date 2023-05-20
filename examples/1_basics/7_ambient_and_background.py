"""
Various modes of background and ambient light.
"""

import numpy as np
from plotoptix import TkOptiX
from plotoptix.materials import m_plastic

def main():

    # make some data first:
    b = 4
    n = 200
    k = 9.1
    l = 1.1
    m = 7.1

    r0 = 0.06
    q = 11

    x = np.sin(np.linspace(0, 2*b*k*np.pi, b*n))
    y = np.cos(np.linspace(0, 2*b*l*np.pi, b*n))
    z = np.sin(np.linspace(0, 2*b*m*np.pi, b*n))
    pos = np.stack((x,y,z)).T

    r = r0 * 0.5 * (1 + np.sin(np.linspace(0, 2*b*q*np.pi, b*n))) + 0.002

    # create and configure, show the window later
    rt = TkOptiX()

    rt.set_param(min_accumulation_step=2, max_accumulation_frames=64)
    rt.setup_material("plastic", m_plastic)

    rt.setup_camera("dof_cam", cam_type="DoF",
                    eye=[-4, 3, 4], target=[0, 0, 0], up=[0.21, 0.86, -0.46],
                    focal_scale=0.75
    )

    # some exposure and gamma adjutments
    rt.set_float("tonemap_exposure", 0.9)
    rt.set_float("tonemap_gamma", 1.4)
    rt.add_postproc("Gamma")

    ################################################################################
    # Try one of the background / lighting settings:

    # ------------------------------------------------------------------------------
    # 1. Ray tracer is initialized in `AmbientLight` mode. There is a constant
    #    background color and an omnidirectional light with the color independent
    #    from the background:

    #rt.setup_light("l1", color=np.array([0.99, 0.95, 0.9]), radius=3)
    #rt.set_background([0, 0.02, 0.1]) # ambient and background colors can be 
    #rt.set_ambient(0.4)               # RGB array or a gray level


    # ------------------------------------------------------------------------------
    # 2. `Default` (although not set by default initialization) mode uses
    #    background color to paint the background and as the ambient light,
    #    so a brighter one is better looking here:

    #rt.set_background_mode("Default")
    #rt.setup_light("l1", color=np.array([0.99, 0.95, 0.9]), radius=3) # just a little light from the side
    #rt.set_background(0.94)


    # ------------------------------------------------------------------------------
    # 3. Environment map. Background texture is projected on the sphere with
    #    infinite radius, and it is also the source of the ambient light. 

    # make a small RGB texture with a vertical gradient
    a = np.linspace(0.94, 0, 10)
    b = np.zeros((10, 2, 3))
    for i in range(10):
        b[i,0]=np.full(3, a[i])
        b[i,1]=np.full(3, a[i])
    b[:,:] *= [0.8, 0.87, 1.0] # bluish tone

    rt.set_background_mode("TextureEnvironment")
    rt.setup_light("l1", color=np.array([0.99, 0.95, 0.9]), radius=3) # just a little light from the side
    rt.set_background(b)


    # ------------------------------------------------------------------------------
    # 4. Fixed background texture. Background is just a wallpaper, you need to setup
    #    ambient light and/or other lights.

    # make a small RGB texture with a vertical gradient
    #a = np.linspace(0.94, 0, 10)
    #b = np.zeros((10, 2, 3))
    #for i in range(10):
    #    b[i,0]=np.full(3, a[i])
    #    b[i,1]=np.full(3, a[i])

    #rt.set_background_mode("TextureFixed")
    #rt.setup_light("l1", color=4*np.array([0.99, 0.95, 0.9]), radius=3)
    #rt.set_background(b)
    #rt.set_ambient(0.4)


    # ------------------------------------------------------------------------------
    # Note: background mode, lights, colors and textures can be all updated also
    # after the window is open, while the ray tracing running.
    ################################################################################

    # create a plot of parametric curve calculated above, and open the window
    rt.set_data("plot", pos=pos, r=r, c=0.94, geom="BezierChain", mat="plastic")
    #rt.set_data("plot", pos=pos, r=r, c=0.94, geom="BSplineQuad", mat="plastic")
    #rt.set_data("plot", pos=pos, r=r, c=0.94, geom="BSplineCubic", mat="plastic")
    #rt.set_data("plot", pos=pos, r=r, c=0.94, geom="SegmentChain", mat="plastic")

    rt.start()

    print(rt.get_background_mode())
    
    print("done")


if __name__ == '__main__':
    main()
