from unittest import TestCase

from plotoptix import NpOptiX
from plotoptix.materials import m_plastic, m_clear_glass

import numpy as np

class SceneIOTestOptiX(NpOptiX):

    def __init__(self,
                 on_scene_compute = None,
                 on_rt_completed = None,
                 on_launch_finished = None):

        super().__init__(
            on_scene_compute=on_scene_compute,
            on_rt_completed=on_rt_completed,
            width=320, height=200,
            start_now=False,
            log_level='INFO')

class TestOutput(TestCase):

    scene = None
    is_alive = False

    @classmethod
    def setUpClass(cls):
        print("################    Test 080: mesh ops.   ########################################")

    def test010_setup_and_start(self):
        TestOutput.scene = SceneIOTestOptiX()

        TestOutput.scene.set_param(min_accumulation_step=2, max_accumulation_frames=6)

        TestOutput.scene.setup_material("plastic", m_plastic)
        TestOutput.scene.setup_material("glass", m_clear_glass)
        materials = { "Cube": "plastic", "Cone": "glass" }

        TestOutput.scene.load_multiple_mesh_obj("tests/data/two_mesh.obj", materials, parent="Cube")
        n_obj = len(TestOutput.scene.geometry_handles)
        self.assertTrue(n_obj == 2, msg="Expected 2 objects, %d loaded." % n_obj)
        self.assertTrue("Cube" in TestOutput.scene.geometry_handles, msg="Cube not loaded.")
        self.assertTrue("Cone" in TestOutput.scene.geometry_handles, msg="Cone not loaded.")

        TestOutput.scene.setup_camera("cam1")
        TestOutput.scene.setup_light("light1", color=10, radius=3)
        TestOutput.scene.set_background(0)
        TestOutput.scene.set_ambient(0.25)

        TestOutput.scene.start()
        self.assertTrue(TestOutput.scene.is_started(), msg="Scene did not flip to _is_started=True state.")
        self.assertTrue(TestOutput.scene.isAlive(), msg="Raytracing thread is not alive.")
        TestOutput.is_alive = True

    def test999_close(self):
        self.assertTrue(TestOutput.scene is not None and TestOutput.is_alive, msg="Wrong state of the test class.")

        TestOutput.scene.close()
        TestOutput.scene.join(10)
        self.assertTrue(TestOutput.scene.is_closed(), msg="Scene did not flip to _is_closed=True state.")
        self.assertFalse(TestOutput.scene.isAlive(), msg="Raytracing thread closing timed out.")
        TestOutput.is_alive = False

    @classmethod
    def tearDownClass(cls):
        cls.assertFalse(cls, cls.is_alive, msg="Wrong state of the test class.")
        print("Test 080: completed.")

