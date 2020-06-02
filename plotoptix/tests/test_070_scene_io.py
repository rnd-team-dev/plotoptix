from unittest import TestCase

from plotoptix import NpOptiX

import json, os
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
        print("################    Test 070: scene IO.   ########################################")

    def test010_setup_and_start(self):
        TestOutput.scene = SceneIOTestOptiX()

        TestOutput.scene.set_param(min_accumulation_step=2, max_accumulation_frames=6)

        TestOutput.scene.load_mesh_obj("tests/data/two_mesh.obj", parent="Cube", c=0.9)
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

    def test020_save_scene(self):

        TestOutput.scene.save_scene("scene_1.json")

        self.assertTrue(os.path.isfile("scene_1.json"), msg="Scene 1 file not created.")

        d = TestOutput.scene.get_scene()
        self.assertTrue("GeometryObjects" in d, msg="Scene description is missing geometries.")
        self.assertTrue("Cone" in d["GeometryObjects"], msg="Cone not found in geometries.")

        d["GeometryObjects"].pop("Cone")
        with open("scene_2.json", "w") as f: json.dump(d, f)
        self.assertTrue(os.path.isfile("scene_2.json"), msg="Scene 2 file not created.")

        d["GeometryObjects"].pop("Cube")
        with open("scene_3.json", "w") as f: json.dump(d, f)
        self.assertTrue(os.path.isfile("scene_3.json"), msg="Scene 3 file not created.")

    def test030_load_scene(self):

        with open("scene_2.json") as f: d = json.loads(f.read())

        TestOutput.scene.set_scene(d)

        self.assertTrue("Cube" in TestOutput.scene.geometry_handles, msg="Cube not found in geometries.")
        self.assertTrue("Cone" not in TestOutput.scene.geometry_handles, msg="Cone is found while it sould be removed.")

        TestOutput.scene.load_scene("scene_1.json")

        self.assertTrue("Cube" in TestOutput.scene.geometry_handles, msg="Cube not found in geometries.")
        self.assertTrue("Cone" in TestOutput.scene.geometry_handles, msg="Cone not found in geometries.")

        TestOutput.scene.load_scene("scene_3.json")

        self.assertTrue(len(TestOutput.scene.geometry_handles) == 0, msg="Scene 3 should have no geometries.")

        TestOutput.scene.load_merged_mesh_obj("tests/data/two_mesh.obj", "meshes", c=0.9)

        self.assertTrue("meshes" in TestOutput.scene.geometry_handles, msg="Meshes not found in geometries.")

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
        if os.path.isfile("scene_1.json"): os.remove("scene_1.json")
        if os.path.isfile("scene_2.json"): os.remove("scene_2.json")
        if os.path.isfile("scene_3.json"): os.remove("scene_3.json")
        print("Test 070: completed.")

