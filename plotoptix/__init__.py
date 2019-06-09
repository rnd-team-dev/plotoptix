"""
3D ray tracing package for Python. Based on NVIDIA OptiX framework,
wrapped in RnD.SharpOptiX C#/C++/CUDA libraries by R&D Team.

Copyright (C) 2019 R&D Team. All Rights Reserved.

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix
Documentation: https://plotoptix.rnd.team
"""

from plotoptix.enums import *
from plotoptix.npoptix import NpOptiX
from plotoptix.tkoptix import TkOptiX

__all__ = ["enums", "materials", "utils", "npoptix", "tkoptix"]

__author__  = "Robert Sulej, R&D Team <dev@rnd.team>"
__status__  = "beta"
__version__ = "0.3.0"
__date__    = "9 June 2019"


import logging

logging.basicConfig(level=logging.WARN, format='[%(levelname)s] (%(threadName)-10s) %(message)s')

# verify Python is 64-bit ################################################
import struct

if struct.calcsize("P") * 8 != 64:
    logging.error(80 * "*"); logging.error(80 * "*")
    logging.error("Python 64-bit is required.")
    logging.error(80 * "*"); logging.error(80 * "*")
    raise ImportError

# verify CUDA release ####################################################
import subprocess

_rel_required = "10." # accept any minor number
try:
    _outp = subprocess.check_output(["nvcc", "--version"]).decode("utf-8").split(" ")
    try:
        _idx = _outp.index("release")
        if _idx + 1 < len(_outp):
            _rel = _outp[_idx + 1].strip(" ,")
            if _rel.startswith(_rel_required):
                logging.info("OK: found CUDA %s", _rel)
            else:
                logging.error(80 * "*"); logging.error(80 * "*")
                logging.error("Found CUDA release %s. This PlotOptiX release requires CUDA %s,", _rel, _rel_required)
                logging.error("available at: https://developer.nvidia.com/cuda-downloads")
                logging.error(80 * "*"); logging.error(80 * "*")
                raise ImportError
        else: raise ValueError
    except:
        logging.error(80 * "*"); logging.error(80 * "*")
        logging.error("CUDA release not recognized. This PlotOptiX release requires CUDA %s,", _rel_required)
        logging.error("available at: https://developer.nvidia.com/cuda-downloads")
        logging.error(80 * "*"); logging.error(80 * "*")
        raise ImportError

except FileNotFoundError:
    logging.error(80 * "*"); logging.error(80 * "*")
    logging.error("Cannot access nvcc. Please check your CUDA installation.")
    logging.error("This PlotOptiX release requires CUDA %s, available at:", _rel_required)
    logging.error("     https://developer.nvidia.com/cuda-downloads")
    logging.error(80 * "*"); logging.error(80 * "*")
    raise ImportError

except ImportError: raise ImportError

except Exception as e:
    logging.error("Cannot verify CUDA installation: " + str(e))
    raise ImportError

# check PlotOptiX updates ################################################
import json
import urllib.request
from packaging import version

try:
    url = "https://pypi.python.org/pypi/plotoptix/json"
    webURL = urllib.request.urlopen(url)
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
