import numpy as np
from plotoptix import TkOptiX


def main():
    optix = TkOptiX()

    n = 1000000 # 1M data points

    xyz = 3 * (np.random.random((n, 3)) - 0.5)
    r = 0.02 * np.random.random(n) + 0.002

    optix.set_data("plot", xyz, r=r)
    optix.show() # note: non-blocking in the interactive shell

    print("done")

if __name__ == '__main__':
    main()
