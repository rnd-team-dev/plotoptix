import numpy as np
from plotoptix import TkOptiX
from plotoptix.utils import make_color
from plotoptix.materials import m_plastic
from plotoptix.geometry import PinnedBuffer
from plotoptix.enums import GeomBuffer

def torus(u, v, r, R):
    #v *= -1 # makes normals pointing outside the torus 
    x = r * np.sin(v)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = (R + r * np.cos(v)) * np.cos(u)
    return np.array([x, y, z], dtype=np.float32)

def main():

    ru = rv = (0, 2*np.pi)

    n = 50

    i = np.linspace(ru[0], ru[1], 2*n)[:-1]
    j = np.linspace(rv[0], rv[1], n)[:-1]

    U, V = np.meshgrid(i, j)
    S = torus(U, V, 3, 5)

    S = np.swapaxes(S, 0, 2)

    rt = TkOptiX()

    rt.set_param(min_accumulation_step=2, max_accumulation_frames=500, light_shading="Hard")
    rt.set_uint("path_seg_range", 6, 15)

    rt.setup_material("plastic", m_plastic)

    exposure = 0.8; gamma = 2.2
    rt.set_float("tonemap_exposure", exposure)
    rt.set_float("tonemap_gamma", gamma)
    rt.add_postproc("Gamma")

    rt.set_background(0)
    rt.set_ambient(0.0)

    rt.set_surface("surface", S, c=0.95,
                   wrap_u=True, wrap_v=True,
                   mat="plastic",
                   make_normals=True
                   )

    # make data for a bigger torus using normals calculated automatically
    # for the firts object; note that this surface has normals oriented
    # inwards - that's not a problem unless you'd like to have a refractive
    # glass-like material, then use the code below that will flip normals
    # (or simply change the surface formula, but this  would not illustrate
    # how to access buffers)
    #with PinnedBuffer(rt.geometry_data["surface"], "Vectors") as N:
    #    print(S.shape, N.shape) # note that internal buffers are flat arrays of mesh vertex properties 
    #    S -= N.reshape(S.shape)

    # flip normals and update data on gpu, note that both normal vectors
    # and vertex order should be inverted for correct shading
    with PinnedBuffer(rt.geometry_data["surface"], "Vectors") as N:
        with PinnedBuffer(rt.geometry_data["surface"], "FaceIdx") as F:
            N *= -1                  # flip shading normals
            F[:,[0,1]] = F[:,[1,0]]  # invert vertex order in faces, so geometry normals are consistent with shading normals

            rt.update_geom_buffers("surface", GeomBuffer.Vectors | GeomBuffer.FaceIdx, forced=True) # update data on device

            S += 0.5 * N.reshape(S.shape) # make data for a bigger torus

    rt.set_surface("wireframe", S, c=[0.4, 0.01, 0],
                   r=0.015, geom="Graph",
                   wrap_u=True, wrap_v=True,
                   )

    rt.set_data("plane", geom="Parallelograms",
                pos=[[-100, -14, -100]], u=[200, 0, 0], v=[0, 0, 200],
                c=make_color([0.1, 0.2, 0.3], exposure=exposure, gamma=gamma)[0])

    rt.setup_camera("cam1", cam_type="DoF",
                    eye=[-50, -7, -15], target=[0, 0, -1], up=[0, 1, 0],
                    aperture_radius=0.4, aperture_fract=0.2,
                    focal_scale=0.92, fov=25)

    rt.setup_light("light1", pos=[-15, 20, 15], color=5, radius=6)

    rt.start()

    print("done")

if __name__ == '__main__':
    main()
