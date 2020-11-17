"""
3D ray tracing package for Python. Based on NVIDIA OptiX 7 framework,
wrapped in RnD.SharpOptiX C#/C++/CUDA libraries by R&D Team.

https://github.com/rnd-team-dev/plotoptix/blob/master/LICENSE.txt

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
Documentation: https://plotoptix.rnd.team
"""

__all__ = ["enums", "materials", "utils", "npoptix", "tkoptix"]

__author__  = "Robert Sulej, R&D Team <dev@rnd.team>"
__status__  = "beta"
__version__ = "0.12.0"
__date__    = "17 November 2020"

import logging

logging.basicConfig(level=logging.WARN, format='[%(levelname)s] (%(threadName)-10s) %(message)s')

# import PlotOptiX modules ###############################################
from plotoptix.enums import *
from plotoptix.npoptix import NpOptiX
from plotoptix.tkoptix import TkOptiX

# check PlotOptiX updates ################################################
import json
import urllib.request
from packaging import version

try:
    url = "https://pypi.python.org/pypi/plotoptix/json"
    webURL = urllib.request.urlopen(url, timeout=3)
    data = webURL.read()
    encoding = webURL.info().get_content_charset('utf-8')
    data_dict = json.loads(data.decode(encoding))
    versions = list(data_dict["releases"].keys())
    versions.sort(key=version.parse)

    if version.parse(__version__) < version.parse(versions[-1]):
        print(80 * "*")
        print("PlotOptiX newer version is available:", versions[-1])
        print("to update your release use:")
        print("      pip install plotoptix --upgrade")
        print(80 * "*")

except: pass
