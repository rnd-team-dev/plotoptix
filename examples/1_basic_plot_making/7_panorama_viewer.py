"""
This example uses `TextureEnvironment` background mode to view 360deg panoramas.
Just run it with the image file as an argument. Google 'environment map', you
can find plenty of free images.
"""

import sys
import numpy as np
from plotoptix import TkOptiX
from plotoptix.materials import m_plastic

def main():

    if len(sys.argv) < 2:
        print("Usage: python 7_panorama_viewer.py image_file.ext")
        print("ext: jpg, png, tif and bmp files are accepted.")
        exit(0)
    
    fname = str(sys.argv[1])
    # just in case you have spaces in the path...
    for s in sys.argv[2:]: fname += " " + str(s)
    print("Reading:", fname)

    # create and configure, show the window later
    optix = TkOptiX()

    optix.set_background_mode("TextureEnvironment")

    optix.set_background(fname)

    optix.start()    
    print("done")


if __name__ == '__main__':
    main()
