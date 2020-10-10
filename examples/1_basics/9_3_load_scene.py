"""
This example shows how to load scene from JSON file. Scene is loaded
in the constructor here, see 9_3_load_scene.py for more options.

Note: if your secene has textures saved in files, be sure to cd to
the JSON file before loading the scene. Texture paths are relatve
to the scene file.
"""

import os
import numpy as np
from plotoptix import TkOptiX


def main():

    if not os.path.isfile("strange_object.json") or not os.path.isfile("strange_object.json"):
        print("Prepare two files: strange_object.json and strange_object_2.json with the scene saving example 9_1_save_scene.py.")

    # Load scene, but don't start yet:
    rt = TkOptiX("strange_object.json")

    # Apply some modification:
    if "balls" in rt.get_geometry_names():
        rt.scale_geometry("balls", 0.5)

    rt.start()

    input("Press Enter to continue...")

    # Now, let's save the current scene to a dictionary and load
    # the second scene from the file made with 9_1_save_scene.py:

    d = rt.get_scene()

    rt.load_scene("strange_object_2.json")

    # Apply some modification:
    if "block2" in rt.get_geometry_names():
        rt.update_data("block2", c=[0.5, 0, 0.1])

    input("Press Enter to continue...")

    # Manipulate something in the scene saved in the dictionary,
    # and load it back:

    if "light2" in d["Lights"]:
        d["Lights"]["light2"]["Color"] = [10, 5, 1.5]

    rt.set_scene(d)

    print("done")

if __name__ == '__main__':
    main()
