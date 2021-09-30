import numpy as np
from plotoptix import TkOptiX
from plotoptix.utils import make_color
from plotoptix.materials import m_plastic

def trefoil(u, v, r):
    x = r * np.sin(3 * u) / (2 + np.cos(v))
    y = r * (np.sin(u) + 2 * np.sin(2 * u)) / (2 + np.cos(v + np.pi * 2 / 3))
    z = r / 2 * (np.cos(u) - 2 * np.cos(2 * u)) * (2 + np.cos(v)) * (2 + np.cos(v + np.pi * 2 / 3)) / 4
    return np.array([x, y, z], dtype=np.float32)

def sphere(u, v, r):
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    return r * np.array([x, y, z], dtype=np.float32)

def torus(u, v, r, R):
    x = r * np.sin(v)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = (R + r * np.cos(v)) * np.cos(u)
    return np.array([x, y, z], dtype=np.float32)

def main():

    ru = rv = (-np.pi, 3*np.pi) # use with trefoil()
    #ru = rv = (0, 2*np.pi)     # use with torus()

    #ru = (0, np.pi)          # use with sphere()
    #rv = (0, 2*np.pi)

    n = 500

    i = np.linspace(ru[0], ru[1], n)
    j = np.linspace(rv[0], rv[1], n)

    U, V = np.meshgrid(i, j)
    S = trefoil(U, V, 5)
    #S = sphere(U, V, 7)
    #S = torus(U, V, 3, 5)

    S = np.swapaxes(S, 0, 2)

    rt = TkOptiX(width=750, height=900)

    rt.set_param(min_accumulation_step=2, max_accumulation_frames=500, light_shading="Hard")
    rt.set_uint("path_seg_range", 6, 15)

    rt.setup_material("plastic", m_plastic)

    exposure = 0.8; gamma = 2.2
    rt.set_float("tonemap_exposure", exposure)
    rt.set_float("tonemap_gamma", gamma)
    rt.add_postproc("Gamma")

    rt.set_background(0)
    rt.set_ambient(0.15)

    rt.set_surface("surface", S, c=0.94,
                   #wrap_u=True, wrap_v=True,        # use wrapping to close a gap that can appear on u or v edges
                   make_normals=True, mat="plastic")

    rt.set_data("plane", geom="Parallelograms",
                pos=[[-100, -14, -100]], u=[200, 0, 0], v=[0, 0, 200],
                c=make_color([0.1, 0.2, 0.3], exposure=exposure, gamma=gamma)[0])

    rt.setup_camera("cam1", cam_type="DoF",
                    eye=[-50, -7, -15], target=[0, 0, -1], up=[0, 1, 0],
                    aperture_radius=0.4, aperture_fract=0.2,
                    focal_scale=0.92, fov=35, glock=True)

    rt.setup_light("light1", pos=[-15, 20, 15], color=5, radius=6)

    rt.start()

    print("done")

if __name__ == '__main__':
    main()
