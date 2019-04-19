from unittest import TestCase

from plotoptix import TkOptiX
from plotoptix.materials import *
from plotoptix.enums import *

import numpy as np

class TestScene(TestCase):

    scene = None
    is_alive = False

    @classmethod
    def setUpClass(cls):
        cls.scene = TkOptiX(width=128, height=64, start_now=False, log_level='INFO')

    def test010_default_init_values(self):
        self.assertTrue(TestScene.scene is not None, msg="Wrong state of the test class.")

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

        mat = TestScene.scene.get_material("diffuse")
        self.assertFalse(mat is None, msg="Could not read diffuse material.")
        self.assertTrue(("ClosestHitPrograms" in mat) and ("AnyHitPrograms" in mat), msg="Default material data is incomplete.")

        frames_min = TestScene.scene.get_param("min_accumulation_step")
        self.assertTrue(frames_min is not None and frames_min > 0, msg="Unreasonable min accumulation step value: %d." % frames_min)
        frames_max = TestScene.scene.get_param("max_accumulation_frames")
        self.assertTrue(frames_max is not None and frames_max >= frames_min, msg="Unreasonable max accumulation frames value: %d." % frames_max)

    def test020_rt_parameters(self):
        self.assertTrue(TestScene.scene is not None, msg="Wrong state of the test class.")

        frames_min = 2
        frames_max = 4
        TestScene.scene.set_param(
            min_accumulation_step=frames_min,
            max_accumulation_frames=frames_max
        )
        self.assertTrue(TestScene.scene.get_param("min_accumulation_step") == frames_min, msg="Min accumulation step did not match.")
        self.assertTrue(TestScene.scene.get_param("max_accumulation_frames") == frames_max, msg="Max accumulation frames did not match.")

    def test030_predefined_materials(self):
        self.assertTrue(TestScene.scene is not None, msg="Wrong state of the test class.")

        m_list = ["m_flat", "m_eye_normal_cos", "m_diffuse", "m_mirror", "m_metalic", "m_plastic", "m_clear_glass"]

        for m in m_list:
            TestScene.scene.setup_material(m, globals()[m])
            mat = TestScene.scene.get_material(m)
            self.assertFalse(mat is None, msg="Could not read back %s material." % m)

    def test040_light_shading(self):
        self.assertTrue(TestScene.scene is not None, msg="Wrong state of the test class.")

        TestScene.scene.set_light_shading(LightShading.Hard)
        m = TestScene.scene.get_light_shading()
        self.assertTrue(m is not None and m == LightShading.Hard, msg="Returned light shading mode different than value set.")

    def test050_light(self):
        self.assertTrue(TestScene.scene is not None, msg="Wrong state of the test class.")

        pos1=[20, 10, 10]
        col1=[1, 0.5, 0.5]
        r=2
        TestScene.scene.setup_light("test_light1", light_type=Light.Spherical,
                    pos=pos1, color=col1, radius=r, in_geometry=True)

        pos2=[-20, 10, 10]
        col2=[0.5, 0.5, 1]
        u=[1, 0, 0]
        v=[0, 1, 0]
        TestScene.scene.setup_light("test_light2", light_type=Light.Parallelogram,
                    pos=pos2, color=col2, u=u, v=v, in_geometry=True)

        self.assertTrue(np.array_equal(TestScene.scene.get_light_pos("test_light1"), pos1), msg="Light 1 position did not match.")
        self.assertTrue(np.array_equal(TestScene.scene.get_light_pos("test_light2"), pos2), msg="Light 2 position did not match.")

        self.assertTrue(np.array_equal(TestScene.scene.get_light_color("test_light1"), col1), msg="Light 1 color did not match.")
        self.assertTrue(np.array_equal(TestScene.scene.get_light_color("test_light2"), col2), msg="Light 2 color did not match.")

        self.assertTrue(TestScene.scene.get_light_r("test_light1") == r, msg="Light 1 radius did not match.")

        self.assertTrue(np.array_equal(TestScene.scene.get_light_u("test_light2"), u), msg="Light 2 U did not match.")
        self.assertTrue(np.array_equal(TestScene.scene.get_light_v("test_light2"), v), msg="Light 2 V did not match.")

    #todo test new geometry

    def test060_start_rt(self):
        self.assertTrue(TestScene.scene is not None, msg="Wrong state of the test class.")

        TestScene.scene.start()
        self.assertTrue(TestScene.scene.is_started(), msg="Scene did not flip to _is_started=True state.")
        self.assertTrue(TestScene.scene.isAlive(), msg="Raytracing thread is not alive.")
        TestScene.is_alive = True

    def test070_camera(self):
        self.assertTrue(TestScene.scene is not None and TestScene.is_alive, msg="Wrong state of the test class.")

        cam, handle = TestScene.scene.get_camera_name_handle()
        self.assertFalse((cam is None) or (handle is None), msg="Could not get default camera.")
        self.assertTrue((cam == "default") and (handle == 1), msg="Wrong name/handle of the default camera: %s / %d." % (cam, handle))

        eye=[10, 10, 10]
        target=[1, 1, 1]
        up=[0, 0, 1]
        cam_type=Camera.DoF
        aperture_radius=0.2
        aperture_fract=0.3
        focal_scale=0.9
        fov=35
        blur=0.5
        make_current=True
        TestScene.scene.setup_camera("test_cam1",
                                     eye=eye, target=target, up=up,
                                     cam_type=cam_type,
                                     aperture_radius=aperture_radius,
                                     aperture_fract=aperture_fract,
                                     focal_scale=focal_scale,
                                     fov=fov, blur=blur,
                                     make_current=make_current)
        cam, handle = TestScene.scene.get_camera_name_handle()
        self.assertFalse((cam is None) or (handle is None), msg="Could not get back the new camera.")
        self.assertTrue((cam == "test_cam1") and (handle > 1), msg="Wrong name/handle of the new camera: %s / %d." % (cam, handle))
        self.assertTrue(np.array_equal(TestScene.scene.get_camera_eye(cam), eye), msg="Camera eye did not match.")
        self.assertTrue(np.array_equal(TestScene.scene.get_camera_target(cam), target), msg="Camera target did not match.")

        cam_params = TestScene.scene.get_camera("test_cam1")
        self.assertFalse(cam_params is None, msg="Could not get back parameters dictionary of the new camera.")

        #todo test camera with default values

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

