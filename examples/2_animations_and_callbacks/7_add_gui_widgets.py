"""
Custom widgets added to TkOptiX GUI.
"""

import copy
import numpy as np
import tkinter as tk

from plotoptix import TkOptiX
from plotoptix.materials import m_clear_glass


def autoexposure():
    q = np.quantile(rt._raw_rgba[...,:3], 0.9)
    rt.set_float("tonemap_exposure", 1 / q)
    
def slide(event=None):
    att_len = 0.1 * int(slider.get())
    m_clear_glass["VarFloat3"]["base_color"] = [ att_len, att_len, 100 ]
    rt.setup_material("glass", m_clear_glass)



rt = TkOptiX(start_now=False)

rt.set_param(min_accumulation_step=8, max_accumulation_frames=500)

rt.set_uint("path_seg_range", 8, 16)

rt.set_background(0)
rt.set_ambient(0.1)

rt.set_float("tonemap_exposure", 1.0)
rt.set_float("tonemap_gamma", 2.2)
rt.add_postproc("Gamma")

rt.setup_camera("cam", cam_type="Pinhole",
                #work_distribution="RelNoiseBalanced", # use that in 0.17.1
                eye=[-7, 8, 11], target=[0, 2, 0],
                fov=22, glock=True
               )
rt.setup_light("light", pos=[4, 7, -1], color=20, radius=0.5)
    
rt.set_data("plane", geom="Parallelograms",
            pos=[-100, 0, -100], u=[200, 0, 0], v=[0, 0, 200],
            c=0.95
           )

m_clear_glass["VarFloat3"]["base_color"] = [ 0.3, 0.3, 100 ]
rt.setup_material("glass", m_clear_glass)
rt.set_data("cube", geom="Parallelepipeds",
            pos=[-1, 3, -1], u=[2, 0, 0], v=[0, 0, 2], w=[0, 2, 0],
            mat="glass"
           )
    
rt.start()
    
### Add a panel with custom widgets ###

# First, reconfigure the ray tracing output to leave a column for the new panel:
rt._canvas.grid(column=0, row=0, columnspan=2, sticky="nsew")

# ...then insert the panel:
p1 = tk.PanedWindow()
p1.grid(column=2, row=0, sticky="ns")

# ...and add widgets, bind to handlers:
btn = tk.Button(p1, text="Autoexposure", command=autoexposure)
btn.grid(column=0, row=0, sticky="new", padx=8, pady=4)

slider = tk.Scale(p1, from_=0, to=100, orient="horizontal")
slider.set(int(10 * m_clear_glass["VarFloat3"]["base_color"][0]))
slider.bind("<B1-Motion>", slide)
slider.grid(column=0, row=1, sticky="new")
