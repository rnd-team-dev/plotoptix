"""
3D ray tracing package for Python based on NVIDIA OptiX framework,
wrapped in RnD.SharpOptiX C#/C++/CUDA libraries by R&D Team.

https://github.com/rnd-team-dev/plotoptix/blob/master/LICENSE.txt

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
Documentation: https://plotoptix.rnd.team
"""

__all__ = ["enums", "materials", "utils", "npoptix", "tkoptix"]

__author__  = "Robert Sulej, R&D Team <dev@rnd.team>"
__status__  = "beta"
__version__ = "0.19.2"
__date__    = "1 June 2026"

import logging

logging.basicConfig(level=logging.WARN, format='[%(levelname)s] (%(threadName)-10s) %(message)s')

import os
import sys
import ctypes

def _bootstrap_ffmpeg_dlls():
    if sys.platform != "win32" or sys.version_info < (3, 8):
        return

    # Shared FFmpeg DLL names to search for:
    target_dlls = ["avcodec-62.dll", "avformat-62.dll", "avutil-60.dll"]
    ffmpeg_dir = None

    path_env = os.environ.get("PATH", "")
    for directory in path_env.split(os.pathsep):
        clean_dir = directory.strip('"')
        if not os.path.isdir(clean_dir):
            continue
        
        if all(os.path.isfile(os.path.join(clean_dir, dll)) for dll in target_dlls):
            ffmpeg_dir = clean_dir
            break

    if not ffmpeg_dir:
        print(f"[{__name__}] Warning: No FFmpeg DLLs found in PATH.")
        return

    try:
        _ = os.add_dll_directory(ffmpeg_dir)
        
        for dll in target_dlls:
            dll_path = os.path.join(ffmpeg_dir, dll)
            if os.path.isfile(dll_path):
                # Using ALTER_RE_SEARCH_PATH helps resolve dependencies in the same folder
                ctypes.CDLL(dll_path, winmode=0x00000008) 
                
    except Exception as e:
        # Sandbox denied read permissions entirely
        print(f"[{__name__}] Warning: Failed to map FFmpeg DLL directory: {e}", sys.stderr)

_bootstrap_ffmpeg_dlls()

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
        print(f"[{__name__}] PlotOptiX newer version is available:", versions[-1])
        print(f"[{__name__}] to update your release use:")
        print(f"[{__name__}]       pip install plotoptix --upgrade")
        print(80 * "*")

except: pass
