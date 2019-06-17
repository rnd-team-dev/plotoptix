from unittest import TestCase

from plotoptix import NpOptiX # no UI, this was tested earlier
from plotoptix.materials import *
from plotoptix.enums import *

import numpy as np

class TestScene(TestCase):

    scene = None
    is_alive = False

    @classmethod
    def setUpClass(cls):
        print("################    Test 040: 2D postprocessing configuration.    ################")
        cls.scene = NpOptiX(width=128, height=64, start_now=False, log_level='INFO')

    def test010_default_init_values(self):
        self.assertTrue(TestScene.scene is not None, msg="Wrong state of the test class.")

        state = TestScene.scene._raise_on_error
        TestScene.scene._raise_on_error = True

        # levels adjustment
        levels_low = (0.1, 0.2, 0.3)
        levels_high = (0.9, 0.8, 0.7)

        TestScene.scene.set_float("levels_low_range", levels_low[0], levels_low[1], levels_low[2])
        TestScene.scene.set_float("levels_high_range", levels_high[0], levels_high[1], levels_high[2])

        test_levels_low = TestScene.scene.get_float3("levels_low_range")
        self.assertTrue(np.float32(test_levels_low[0]) == np.float32(levels_low[0]),
                        msg="Value set: %f, readback: %f." % (levels_low[0], test_levels_low[0]))
        self.assertTrue(np.float32(test_levels_low[1]) == np.float32(levels_low[1]),
                        msg="Value set: %f, readback: %f." % (levels_low[1], test_levels_low[1]))
        self.assertTrue(np.float32(test_levels_low[2]) == np.float32(levels_low[2]),
                        msg="Value set: %f, readback: %f." % (levels_low[2], test_levels_low[2]))

        test_levels_high = TestScene.scene.get_float3("levels_high_range")
        self.assertTrue(np.float32(test_levels_high[0]) == np.float32(levels_high[0]),
                        msg="Value set: %f, readback: %f." % (levels_high[0], test_levels_high[0]))
        self.assertTrue(np.float32(test_levels_high[1]) == np.float32(levels_high[1]),
                        msg="Value set: %f, readback: %f." % (levels_high[1], test_levels_high[1]))
        self.assertTrue(np.float32(test_levels_high[2]) == np.float32(levels_high[2]),
                        msg="Value set: %f, readback: %f." % (levels_high[2], test_levels_high[2]))

        TestScene.scene.add_postproc("Levels")


        # gamma correction
        exposure = 0.8
        igamma = 1 / 1.9

        TestScene.scene.set_float("tonemap_exposure", exposure)
        TestScene.scene.set_float("tonemap_igamma", igamma)

        test_exposure = TestScene.scene.get_float("tonemap_exposure")
        self.assertTrue(test_exposure is not None and np.float32(test_exposure) == np.float32(exposure),
                        msg="Value set: %f, readback: %f." % (exposure, test_exposure))
        test_igamma = TestScene.scene.get_float("tonemap_igamma")
        self.assertTrue(test_igamma is not None and np.float32(test_igamma) == np.float32(igamma),
                        msg="Value set: %f, readback: %f." % (igamma, test_igamma))

        TestScene.scene.add_postproc("Gamma")


        # tonal corrections with custom Gray/RGB curves
        TestScene.scene.set_texture_1d("tone_curve_gray", np.sqrt(np.linspace(0, 1, 32)))
        TestScene.scene.add_postproc("GrayCurve")

        TestScene.scene.set_texture_1d("tone_curve_r", np.sqrt(np.linspace(0, 1, 32)))
        TestScene.scene.set_texture_1d("tone_curve_g", np.sqrt(np.linspace(0, 1, 32)))
        TestScene.scene.set_texture_1d("tone_curve_b", np.sqrt(np.linspace(0, 1, 32)))
        TestScene.scene.add_postproc("RgbCurve")


        # mask overlay
        TestScene.scene.set_texture_2d("frame_mask", np.full((10, 10), 0.5))
        TestScene.scene.add_postproc("Mask")

        TestScene.scene._raise_on_error = state


    def test998_start_rt(self):
        self.assertTrue(TestScene.scene is not None, msg="Wrong state of the test class.")

        TestScene.scene.start()
        self.assertTrue(TestScene.scene.is_started(), msg="Scene did not flip to _is_started=True state.")
        self.assertTrue(TestScene.scene.isAlive(), msg="Raytracing thread is not alive.")
        TestScene.is_alive = True


    def test999_close(self):
        self.assertTrue(TestScene.scene is not None and TestScene.is_alive, msg="Wrong state of the test class.")

        TestScene.scene.close()
        TestScene.scene.join(10)
        self.assertTrue(TestScene.scene.is_closed(), msg="Scene did not flip to _is_closed=True state.")
        self.assertFalse(TestScene.scene.isAlive(), msg="Raytracing thread closing timed out.")
        TestScene.is_alive = False

    @classmethod
    def tearDownClass(cls):
        cls.assertFalse(cls, cls.is_alive, msg="Wrong state of the test class.")
        print("Test 040: completed.")
