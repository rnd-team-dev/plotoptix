from unittest import TestCase
from plotoptix import TkOptiX

class TestScene(TestCase):

    scene = None
    is_alive = False

    @classmethod
    def setUpClass(cls):
        cls.scene = TkOptiX(width=128, height=64, start_now=True, log_level='INFO')

    def test000_isAlive(self):
        self.assertTrue(TestScene.scene is not None, msg="Wrong state of the test class.")

        self.assertTrue(TestScene.scene.is_started(), msg="Scene did not flip to _is_started=True state.")
        self.assertTrue(TestScene.scene.isAlive(), msg="Raytracing thread is not alive.")
        TestScene.is_alive = True

    def test010_default_init_values(self):
        self.assertTrue(TestScene.scene is not None and TestScene.is_alive, msg="Wrong state of the test class.")

        cam, handle = TestScene.scene.get_camera()
        self.assertFalse((cam is None) or (handle is None), msg="Could not get default camera.")
        self.assertTrue((cam == "default") and (handle == 1), msg="Wrong name/handle of the default camera: %s / %d." % (cam, handle))

        mat = TestScene.scene.get_material("diffuse")
        self.assertFalse(mat is None, msg="Could not read diffuse material.")
        self.assertTrue(("ClosestHitPrograms" in mat) and ("AnyHitPrograms" in mat), msg="Default material data is incomplete.")

    def test999_close(self):
        self.assertTrue(TestScene.scene is not None and TestScene.is_alive, msg="Wrong state of the test class.")

        TestScene.scene.close()
        TestScene.scene.join(10)
        self.assertTrue(TestScene.scene.is_closed(), msg="Scene did not flip to _is_closed=True state.")
        self.assertFalse(TestScene.scene.isAlive(), msg="Raytracing thread closing timed out.")
        TestScene.is_alive = False

    @classmethod
    def tearDownClass(cls):
        if cls.is_alive: raise ValueError

