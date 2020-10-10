import numpy as np
from plotoptix import TkOptiX


def main():
    rt = TkOptiX()

    n = 1000000 # 1M data points

    xyz = 3 * (np.random.random((n, 3)) - 0.5)
    r = 0.02 * np.random.random(n) + 0.002

    rt.set_data("plot", xyz, r=r)

    rt.set_ambient(0.9); # set ambient light, brighter than default

    rt.show() # note: non-blocking in the interactive shell

    print("done")

if __name__ == '__main__':
    main()
