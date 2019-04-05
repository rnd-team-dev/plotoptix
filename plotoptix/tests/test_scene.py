from unittest import TestCase

class TestScene(TestCase):

    def test_import(self):
        import plotoptix

    def test_empty_scene(self):
        from plotoptix import TkOptiX
        scene = TkOptiX(width=128, height=64, start_now=True, log_level='INFO')
        self.assertTrue(scene.is_started(), msg="Scene did not flip to _is_started=True state.")
        self.assertTrue(scene.isAlive(), msg="Raytracing thread is not alive.")
        scene.close()
        scene.join(10)
        self.assertTrue(scene.is_closed(), msg="Scene did not flip to _is_closed=True state.")
        self.assertFalse(scene.isAlive(), msg="Raytracing thread closing timed out.")
