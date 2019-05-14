from unittest import TestCase

from plotoptix.utils import *

import numpy as np

class TestScene(TestCase):

    @classmethod
    def setUpClass(cls):
        print("################   Test 040: Utility and convenience functions.   ################")

    def test010_map_to_colors(self):

        x = np.linspace(-1, 2, 10)
        c = map_to_colors(x, "Greys")

        self.assertTrue(c.shape == x.shape + (3,), msg="Expected RGB values for each input x value.")
        self.assertTrue(np.array_equal(c[0], np.array([1, 1, 1])), msg="Expected white color.")
        self.assertTrue(np.array_equal(c[-1], np.array([0, 0, 0])), msg="Expected black color.")

    def test020_simplex(self):

        n = 10

        # 2D algorithm / 2D noise
        x = np.linspace(1, 10, n)
        y = np.linspace(1, 10, n)
        xm, ym = np.meshgrid(x, y)
        xy = np.stack((xm.flatten(), ym.flatten())).T.reshape(n, n, 2)
        z = simplex(xy)

        self.assertTrue(z.shape == (n, n), msg="Expected %d x %d noise output." % (n, n))
        self.assertTrue(np.min(z) >= -1 and np.max(z) <= 1, msg="Noise value out of <-1, 1> range.")
        self.assertFalse(np.isnan(z).any() or np.isinf(z).any(), msg="Noise contains NaNs or Infs.")

        # 3D algorithm / 2D noise
        xy = np.stack((xm.flatten(), ym.flatten(), np.full(n*n, 1.0))).T.reshape(n, n, 3)
        z = simplex(xy)

        self.assertTrue(z.shape == (n, n), msg="Expected %d x %d noise output." % (n, n))
        self.assertTrue(np.min(z) >= -1 and np.max(z) <= 1, msg="Noise value out of <-1, 1> range.")
        self.assertFalse(np.isnan(z).any() or np.isinf(z).any(), msg="Noise contains NaNs or Infs.")

        # in-place generation
        z1 = simplex(xy, z)

        self.assertTrue(z1 is z, msg="Did not generate in place.")
        self.assertTrue(z.shape == (n, n), msg="Expected %d x %d noise output." % (n, n))
        self.assertTrue(np.min(z) >= -1 and np.max(z) <= 1, msg="Noise value out of <-1, 1> range.")
        self.assertFalse(np.isnan(z).any() or np.isinf(z).any(), msg="Noise contains NaNs or Infs.")

        # 4D algorithm / 3D noise
        z = np.linspace(1, 10, n)
        xm, ym, zm = np.meshgrid(x, y, z, indexing='ij')
        xyz = np.stack((xm.flatten(), ym.flatten(), zm.flatten(), np.full(n*n*n, 1.0))).T.reshape(n, n, n, 4)
        w = simplex(xyz)

        self.assertTrue(w.shape == (n, n, n), msg="Expected %d x %d x %d noise output." % (n, n, n))
        self.assertTrue(np.min(w) >= -1 and np.max(w) <= 1, msg="Noise value out of <-1, 1> range.")
        self.assertFalse(np.isnan(w).any() or np.isinf(w).any(), msg="Noise contains NaNs or Infs.")

    @classmethod
    def tearDownClass(cls):
        print("Test 040: completed.")
