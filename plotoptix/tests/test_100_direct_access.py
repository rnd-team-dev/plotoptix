from unittest import TestCase

from plotoptix import NpOptiX

import numpy as np

class AccTestOptiX(NpOptiX):

    def __init__(self):

        super().__init__(
            width=128, height=96,
            start_now=False,
            log_level='INFO'
        )

class TestAccess(TestCase):

    scene = None
    data = None
    r = None
    is_alive = False

    @classmethod
    def setUpClass(cls):
        print("################    Test 100: direct access.   ###################################")
        n = 100
        rx = (-10, 10)

        TestAccess.r = 0.85 * 0.5 * (rx[1] - rx[0]) / (n - 1)

        x = np.linspace(rx[0], rx[1], n)
        z = np.linspace(rx[0], rx[1], n)
        X, Z = np.meshgrid(x, z)

        TestAccess.data = np.stack([X.flatten(), np.zeros(n*n), Z.flatten()], axis=1)

    def test010_setup_and_start(self):
        TestAccess.scene = AccTestOptiX()
        TestAccess.scene.set_param(min_accumulation_step=2, max_accumulation_frames=6)

        TestAccess.scene.set_data("balls", pos=TestAccess.data, c=0.82, r=TestAccess.r)

        TestAccess.scene.setup_camera("cam1")
        TestAccess.scene.setup_light("light1", color=10, radius=3)
        TestAccess.scene.set_background(0)
        TestAccess.scene.set_ambient(0)

        TestAccess.scene.start()
        self.assertTrue(TestAccess.scene.is_started(), msg="Scene did not flip to _is_started=True state.")
        self.assertTrue(TestAccess.scene.is_alive(), msg="Raytracing thread is not alive.")
        TestAccess.is_alive = True

    def test020_write_read_geom(self):
        rb_data = TestAccess.scene.get_data("balls", "Positions")
        self.assertTrue(np.allclose(TestAccess.data, rb_data), msg="Incorrect values in data readback.")

        shift = np.linspace(0, 1, TestAccess.data.shape[0])
        mod_data = TestAccess.data.copy()
        mod_data[:,1] += shift
        TestAccess.scene.update_raw_data("balls", pos=mod_data)
        rb_data = TestAccess.scene.get_data("balls", "Positions")
        self.assertTrue(np.allclose(mod_data, rb_data), msg="Incorrect values in modified data readback.")

    def test999_close(self):
        self.assertTrue(TestAccess.scene is not None and TestAccess.is_alive, msg="Wrong state of the test class.")

        TestAccess.scene.close()
        TestAccess.scene.join(10)
        self.assertTrue(TestAccess.scene.is_closed(), msg="Scene did not flip to _is_closed=True state.")
        self.assertFalse(TestAccess.scene.is_alive(), msg="Raytracing thread closing timed out.")
        TestAccess.is_alive = False

    @classmethod
    def tearDownClass(cls):
        cls.assertFalse(cls, cls.is_alive, msg="Wrong state of the test class.")
        print("Test 100: completed.")

