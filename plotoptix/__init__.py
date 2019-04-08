"""
3D raytracing package for Python. Based on NVIDIA OptiX framework,
and RnD.SharpOptiX C#/C++/CUDA libraries by R&D Team.

Copyright (C) 2019 R&D Team. All Rights Reserved.

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
"""

from plotoptix.enums import *
from plotoptix.npoptix import NpOptiX
from plotoptix.tkoptix import TkOptiX

__all__ = ["enums", "materials", "utils", "npoptix", "tkoptix"]

__author__  = "Robert Sulej, R&D Team <dev@rnd.team>"
__status__  = "beta"
__version__ = "0.1.1.5"
__date__    = "08 April 2019"
