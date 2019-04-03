from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# expect this to be quite slow!

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100000 # 100k data points

    x, y, z = 3 * (np.random.random((3, n)) - 0.5)
    r = 0.1 * np.random.random(n) + 0.01

    ax.scatter(x, y, z, c='r', marker='.', s=r)
    plt.show() # note: blocking in the interactive shell

    print("done")


if __name__ == '__main__':
    main()
