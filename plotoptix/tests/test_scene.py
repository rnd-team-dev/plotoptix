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

        # test for failures

        fx = TestScene.scene.get_float("non_existing_float")
        self.assertTrue(fx is None, msg="Should return None")
        fx, fy = TestScene.scene.get_float2("non_existing_float2")
        self.assertTrue(fx is None and fy is None, msg="Should return Nones")
        fx, fy, fz = TestScene.scene.get_float3("non_existing_float3")
        self.assertTrue(fx is None and fy is None and fz is None, msg="Should return Nones")

        ux = TestScene.scene.get_uint("non_existing_uint")
        self.assertTrue(ux is None, msg="Should return None")
        ux, uy = TestScene.scene.get_uint2("non_existing_uint2")
        self.assertTrue(ux is None and uy is None, msg="Should return Nones")

        ix = TestScene.scene.get_int("non_existing_int")
        self.assertTrue(ix is None, msg="Should return None")

        # test for required values

        eps = TestScene.scene.get_float("scene_epsilon")
        self.assertTrue(eps is not None and eps > 0 and eps < 0.05, msg="Unreasonable scene epsilon value: %f." % eps)

        ray_type_0 = TestScene.scene.get_uint("radiance_ray_type")
        self.assertTrue(ray_type_0 is not None and ray_type_0 == 0, msg="Radiance ray type not 0: %d." % ray_type_0)
        ray_type_1 = TestScene.scene.get_uint("shadow_ray_type")
        self.assertTrue(ray_type_1 is not None and ray_type_1 == 1, msg="Shadow ray type not 1: %d." % ray_type_1)

        n_lights = TestScene.scene.get_int("num_lights")
        self.assertTrue(n_lights is not None and n_lights == 0, msg="Lights count not 0: %d." % n_lights)

        seg_min, seg_max = TestScene.scene.get_uint2("path_seg_range")
        self.assertTrue(seg_min is not None and seg_min > 0, msg="Unreasonable min traced segments: %d." % seg_min)
        self.assertTrue(seg_max is not None and seg_max >= seg_min, msg="Unreasonable max traced segments: %d." % seg_max)

        cx, cy, cz = TestScene.scene.get_float3("bad_color")
        self.assertTrue(cx is not None and cx >= 0, msg="Unreasonable bad color r: %f." % cx)
        self.assertTrue(cy is not None and cy >= 0, msg="Unreasonable bad color g: %f." % cy)
        self.assertTrue(cz is not None and cz >= 0, msg="Unreasonable bad color b: %f." % cz)

        exposure = TestScene.scene.get_float("tonemap_exposure")
        self.assertTrue(exposure is not None and exposure > 0, msg="Unreasonable exposure value: %f." % exposure)
        igamma = TestScene.scene.get_float("tonemap_igamma")
        self.assertTrue(igamma is not None and igamma > 0, msg="Unreasonable igamma value: %f." % igamma)

        cx, cy, cz = TestScene.scene.get_background()
        self.assertTrue(cx is not None and cx >= 0, msg="Unreasonable background color r: %f." % cx)
        self.assertTrue(cy is not None and cy >= 0, msg="Unreasonable background color g: %f." % cy)
        self.assertTrue(cz is not None and cz >= 0, msg="Unreasonable background color b: %f." % cz)

        cx, cy, cz = TestScene.scene.get_ambient()
        self.assertTrue(cx is not None and cx >= 0, msg="Unreasonable ambient color r: %f." % cx)
        self.assertTrue(cy is not None and cy >= 0, msg="Unreasonable ambient color g: %f." % cy)
        self.assertTrue(cz is not None and cz >= 0, msg="Unreasonable ambient color b: %f." % cz)

        cam, handle = TestScene.scene.get_camera_name_handle()
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
        cls.assertFalse(cls, cls.is_alive, msg="Wrong state of the test class.")

