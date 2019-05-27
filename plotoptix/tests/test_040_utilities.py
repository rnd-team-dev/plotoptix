from unittest import TestCase

from plotoptix.utils import *
from plotoptix.utils import _make_contiguous_vector, _make_contiguous_3d

import numpy as np

class TestScene(TestCase):

    @classmethod
    def setUpClass(cls):
        print("################   Test 040: Utility and convenience functions.   ################")

    def test010_make_contiguous(self):
        
        x = 1
        a = _make_contiguous_vector(x, 3)
        self.assertTrue(a.shape == (3,), msg="Output shape is wrong.")
        self.assertTrue(a.dtype == np.float32, msg="Data type is not np.float32.")
        self.assertTrue(np.array_equal(a, [x, x, x]), msg="Output content is wrong.")
        self.assertTrue(a.flags['C_CONTIGUOUS'], msg="Output array is not c-contiguous.")

        y = [1, 2, 3]
        a = _make_contiguous_vector(y, len(y))
        self.assertTrue(a.shape == (len(y),), msg="Output shape is wrong.")
        self.assertTrue(a.dtype == np.float32, msg="Data type is not np.float32.")
        self.assertTrue(np.array_equal(a, y), msg="Output content is wrong.")
        self.assertTrue(a.flags['C_CONTIGUOUS'], msg="Output array is not c-contiguous.")

        z = [[1, 2], [3, 4]]
        a = _make_contiguous_vector(z, 4)
        self.assertTrue(a.shape == (4,), msg="Output shape is wrong.")
        self.assertTrue(a.dtype == np.float32, msg="Data type is not np.float32.")
        self.assertTrue(np.array_equal(a, [1, 2, 3, 4]), msg="Output content is wrong.")
        self.assertTrue(a.flags['C_CONTIGUOUS'], msg="Output array is not c-contiguous.")

        x = 1
        a = _make_contiguous_3d(x, extend_scalars=True)
        self.assertTrue(a.shape == (1, 3), msg="Output shape is wrong.")
        self.assertTrue(a.dtype == np.float32, msg="Data type is not np.float32.")
        self.assertTrue(np.array_equal(a, [[x, x, x]]), msg="Output content is wrong.")
        self.assertTrue(a.flags['C_CONTIGUOUS'], msg="Output array is not c-contiguous.")

        a = _make_contiguous_3d(x, extend_scalars=False)
        self.assertTrue(a.shape == (1, 3), msg="Output shape is wrong.")
        self.assertTrue(a.dtype == np.float32, msg="Data type is not np.float32.")
        self.assertTrue(np.array_equal(a, [[x, x, x]]), msg="Output content is wrong.")
        self.assertTrue(a.flags['C_CONTIGUOUS'], msg="Output array is not c-contiguous.")

        x = [1, 2, 3]
        a = _make_contiguous_3d(x, extend_scalars=True)
        self.assertTrue(a.shape == (1, 3), msg="Output shape is wrong.")
        self.assertTrue(a.dtype == np.float32, msg="Data type is not np.float32.")
        self.assertTrue(np.array_equal(a, [x]), msg="Output content is wrong.")
        self.assertTrue(a.flags['C_CONTIGUOUS'], msg="Output array is not c-contiguous.")

        a = _make_contiguous_3d(x, extend_scalars=False)
        self.assertTrue(a.shape == (1, 3), msg="Output shape is wrong.")
        self.assertTrue(a.dtype == np.float32, msg="Data type is not np.float32.")
        self.assertTrue(np.array_equal(a, [x]), msg="Output content is wrong.")
        self.assertTrue(a.flags['C_CONTIGUOUS'], msg="Output array is not c-contiguous.")

        y = [1, 2, 3, 4]
        a = _make_contiguous_3d(y, extend_scalars=True)
        self.assertTrue(a.shape == (4, 3), msg="Output shape is wrong.")
        self.assertTrue(a.dtype == np.float32, msg="Data type is not np.float32.")
        self.assertTrue(np.array_equal(a, [
            [y[0], y[0], y[0]],
            [y[1], y[1], y[1]],
            [y[2], y[2], y[2]],
            [y[3], y[3], y[3]]]),
                        msg="Output content is wrong.")
        self.assertTrue(a.flags['C_CONTIGUOUS'], msg="Output array is not c-contiguous.")

        a = _make_contiguous_3d(y, extend_scalars=False)
        self.assertTrue(a.shape == (4, 3), msg="Output shape is wrong.")
        self.assertTrue(a.dtype == np.float32, msg="Data type is not np.float32.")
        self.assertTrue(np.array_equal(a, [
            [y[0], 0, 0],
            [y[1], 0, 0],
            [y[2], 0, 0],
            [y[3], 0, 0]]),
                        msg="Output content is wrong.")

        self.assertTrue(a.flags['C_CONTIGUOUS'], msg="Output array is not c-contiguous.")

        z = [[1, 2, 3], [4, 5, 6]]
        a = _make_contiguous_3d(z, n=2)
        self.assertTrue(a.shape == (2, 3), msg="Output shape is wrong.")
        self.assertTrue(a.dtype == np.float32, msg="Data type is not np.float32.")
        self.assertTrue(np.array_equal(a, z), msg="Output content is wrong.")
        self.assertTrue(a.flags['C_CONTIGUOUS'], msg="Output array is not c-contiguous.")

    def test020_make_color(self):

        e = 0.5; g = 2.2; r = 255
        x = [255, 127, 0]
        c = make_color(x, exposure=e, gamma=g, input_range=r)
        t = r * np.power(e * c, 1 / g)

        self.assertTrue(c.dtype == np.float32, msg="Data type is not np.float32.")
        self.assertTrue(c.flags['C_CONTIGUOUS'], msg="Output array is not c-contiguous.")
        self.assertTrue(np.allclose(x, t), msg="Output content is wrong.")

    def test030_map_to_colors(self):

        x = np.linspace(-1, 2, 10)
        c = map_to_colors(x, "Greys")

        self.assertTrue(c.shape == x.shape + (3,), msg="Expected RGB values for each input x value.")
        self.assertTrue(np.array_equal(c[0], np.array([1, 1, 1])), msg="Expected white color.")
        self.assertTrue(np.array_equal(c[-1], np.array([0, 0, 0])), msg="Expected black color.")

    def test040_simplex(self):

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
