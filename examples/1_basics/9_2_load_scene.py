"""
This example shows how to load scene from JSON file. Scene is loaded
in the constructor here, see 9_3_load_scene.py for more options.

Note: if your secene has textures saved in files, be sure to cd to
the JSON file before loading the scene. Texture paths are relatve
to the scene file.
"""

import numpy as np
from plotoptix import TkOptiX


def main():

    # That's really simple:

    rt = TkOptiX("strange_object.json", start_now=True)

    print("done")

if __name__ == '__main__':
    main()
