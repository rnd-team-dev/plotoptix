import numpy as np
from plotoptix import TkOptiX
from plotoptix.materials import m_shadow_catcher

def main():

    n = 100
    xyz = 3 * (np.random.random((n, 3)) - 0.5)
    r = 0.4 * np.random.random(n) + 0.002

    rt = TkOptiX()
    rt.set_param(min_accumulation_step=4, max_accumulation_frames=500)

    rt.setup_light("light1", pos=[4, 5, 5], color=[8, 7, 6], radius=1.5)
    rt.setup_light("light2", pos=[4, 15, -5], color=[8, 7, 6], radius=1.5)
    rt.set_ambient([0.1, 0.2, 0.3]);
    rt.set_background(1)
    
    rt.set_data("plot", xyz, r=r, c=0.8)

    # shadow catcher makes the object transparent except regions where a light casts shadows,
    # this can be useful e.g. for making packshot style images
    rt.setup_material("shadow", m_shadow_catcher)
    rt.set_data("plane", pos=[-50, -2, -50], u=[100, 0, 0], v=[0, 0, 100], c=1, geom="Parallelograms", mat="shadow")

    rt.start()

    print("done")

if __name__ == '__main__':
    main()