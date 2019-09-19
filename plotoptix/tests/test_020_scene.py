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
        print("################  Test 020: scene basic config and GUI startup.   ################")
        cls.scene = TkOptiX(width=128, height=64, start_now=False, log_level='INFO')

    def test010_default_init_values(self):
        self.assertTrue(TestScene.scene is not None, msg="Wrong state of the test class.")

        # test for failures
        print("---- Now you'll see logger error messages, this is OK.")

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

        print("---- End of error messages.")

        # test for required values

        self.assertFalse(TestScene.scene._optix.is_defined("non_existing_var"), msg="Non-existing variable should not be reported as defined.")
        self.assertTrue(TestScene.scene._optix.is_defined("scene_epsilon"), msg="Not defined variable scene_epsilon.")

        self.assertTrue(TestScene.scene._optix.is_defined("radiance_ray_type"), msg="Not defined variable radiance_ray_type.")
        self.assertTrue(TestScene.scene._optix.is_defined("shadow_ray_type"), msg="Not defined variable shadow_ray_type.")

        self.assertTrue(TestScene.scene._optix.is_defined("num_lights"), msg="Not defined variable num_lights.")

        seg_min, seg_max = TestScene.scene.get_uint2("path_seg_range")
        self.assertTrue(seg_min is not None and seg_min > 0, msg="Unreasonable min traced segments: %d." % seg_min)
        self.assertTrue(seg_max is not None and seg_max >= seg_min, msg="Unreasonable max traced segments: %d." % seg_max)

        self.assertTrue(TestScene.scene._optix.is_defined("bad_color"), msg="Not defined variable bad_color.")

        self.assertTrue(TestScene.scene._optix.is_defined("tonemap_exposure"), msg="Not defined variable tonemap_exposure.")
        self.assertTrue(TestScene.scene._optix.is_defined("tonemap_igamma"), msg="Not defined variable tonemap_igamma.")

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

    def test040_background_modes(self):
        self.assertTrue(TestScene.scene is not None, msg="Wrong state of the test class.")

        self.assertTrue(TestScene.scene._optix.is_defined("ambient_color"), msg="Not defined float3 ambient_color.")
        self.assertTrue(TestScene.scene._optix.is_defined("bg_color"), msg="Not defined float3 bg_color.")
        self.assertTrue(TestScene.scene._optix.is_defined("bg_texture"), msg="Not defined texture sampler bg_texture.")
        self.assertTrue(TestScene.scene.get_background_mode() == MissProgram.AmbientLight, msg="Initial miss program should be AmbientLight.")

        state = TestScene.scene._raise_on_error
        TestScene.scene._raise_on_error = True

        gray = 0.5
        TestScene.scene.set_background(gray)
        cx, cy, cz = TestScene.scene.get_background()
        self.assertTrue(cx == gray and cy == gray and cz == gray, msg="Background gray level readback does not match set value %f." % gray)

        color = [0.5, 0.4, 0.3]
        TestScene.scene.set_background(color)
        cx, cy, cz = TestScene.scene.get_background()
        self.assertTrue(np.allclose(color, [cx, cy, cz]), msg="Background color readback does not match set value [%f, %f, %f]." % (color[0], color[1], color[2]))

        h = 10; w = 4
        a = np.linspace(0.05, 0.95, h)
        rgb = np.zeros((h, w, 3))
        for i in range(h):
            rgb[i,0::2]=np.full(3, a[i])
            rgb[i,1::2]=np.full(3, 0)

        rgba = np.zeros((h, w, 4))
        rgba[:,:,:-1] = rgb

        TestScene.scene.set_background(rgb)   # TODO: test readback of background textures
        TestScene.scene.set_background(rgba)  #

        TestScene.scene._raise_on_error = state

        bg_list = ["Default", "TextureFixed", "TextureEnvironment"]

        for m in bg_list:
            TestScene.scene.set_background_mode(m)
            self.assertTrue(TestScene.scene.get_background_mode() == MissProgram[m], msg="Miss program not updated (%s)." % m)

    def test050_light_shading(self):
        self.assertTrue(TestScene.scene is not None, msg="Wrong state of the test class.")

        TestScene.scene.set_light_shading(LightShading.Hard)
        m = TestScene.scene.get_light_shading()
        self.assertTrue(m is not None and m == LightShading.Hard, msg="Returned light shading mode different than value set.")

    def test060_light(self):
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

        l1 = TestScene.scene.get_light("test_light1")
        self.assertTrue(l1["Type"] == Light.Spherical.value, msg="Light 1 type did not match.")

        l2 = TestScene.scene.get_light("test_light2")
        self.assertTrue(l2["Type"] == Light.Parallelogram.value, msg="Light 1 type did not match.")

    #todo test new geometry

    def test070_start_rt(self):
        self.assertTrue(TestScene.scene is not None, msg="Wrong state of the test class.")

        TestScene.scene.start()
        self.assertTrue(TestScene.scene.is_started(), msg="Scene did not flip to _is_started=True state.")
        self.assertTrue(TestScene.scene.isAlive(), msg="Raytracing thread is not alive.")
        TestScene.is_alive = True

    def test080_camera(self):
        self.assertTrue(TestScene.scene is not None and TestScene.is_alive, msg="Wrong state of the test class.")

        cam, handle1 = TestScene.scene.get_camera_name_handle()
        self.assertFalse((cam is None) or (handle1 is None), msg="Could not get default camera.")
        self.assertTrue((cam == "default") and (handle1 > 0), msg="Wrong name / handle of the default camera: %s / %d." % (cam, handle1))

        name = "test_cam1"
        eye=[10, 10, 0]
        target=[1, 1, 0]
        up=[-1, 1, 0]
        cam_type=Camera.DoF
        aperture_radius=0.2
        aperture_fract=0.3
        focal_scale=0.9
        fov=35
        blur=0.5
        make_current=True
        TestScene.scene.setup_camera(name,
                                     eye=eye, target=target, up=up,
                                     cam_type=cam_type,
                                     aperture_radius=aperture_radius,
                                     aperture_fract=aperture_fract,
                                     focal_scale=focal_scale,
                                     fov=fov, blur=blur,
                                     make_current=make_current)
        cam, handle2 = TestScene.scene.get_camera_name_handle()
        self.assertFalse((cam is None) or (handle2 is None), msg="Could not get back the new camera.")
        self.assertTrue((cam == name) and (handle2 > handle1), msg="Wrong name/handle of the new camera: %s / %d." % (cam, handle2))
        self.assertTrue(np.array_equal(TestScene.scene.get_camera_eye(cam), eye), msg="Camera eye did not match.")
        self.assertTrue(np.array_equal(TestScene.scene.get_camera_target(cam), target), msg="Camera target did not match.")

        current_cam = TestScene.scene.get_current_camera()
        self.assertTrue(current_cam == name, msg="Wrong current camera name: %s." % current_cam)

        eye_array = np.array(eye)
        tgt_array = np.array(target)
        TestScene.scene.camera_move_by(tuple(-tgt_array))
        TestScene.scene.camera_move_by_local((0, 0, -np.linalg.norm(eye_array - tgt_array)))
        TestScene.scene.camera_rotate_target((0, 0, -np.pi/4))
        TestScene.scene.camera_rotate_target_local((0, -np.pi/2, 0))
        TestScene.scene.camera_rotate_eye((-np.pi/2, 0, 0))
        TestScene.scene.camera_rotate_eye_local((0, 0, -np.pi/2))
        TestScene.scene.camera_rotate_by((0, -np.pi/2, 0), (0, 0, 0))
        print(TestScene.scene.get_camera())

        atol = 0.0001
        rtol = 0.0001
        cam_params = TestScene.scene.get_camera(name)
        eye_dst = cam_params["Eye"]
        tgt_dst = cam_params["Target"]
        up_dst = cam_params["Up"]
        self.assertFalse(cam_params is None, msg="Could not get back parameters dictionary of the new camera.")
        self.assertTrue(np.allclose(eye_dst, np.array([12.72792, 12.72792, 0]), rtol=rtol, atol=atol), msg="Move/rotate result wrong (eye).")
        self.assertTrue(np.allclose(tgt_dst, np.array([12.72792, 0, 0]), rtol=rtol, atol=atol), msg="Move/rotate result wrong (target).")
        self.assertTrue(np.allclose(up_dst, np.array([0.0, 0.0, 1.414214]), rtol=rtol, atol=atol), msg="Move/rotate result wrong (up).")

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
        print("Test 020: completed.")

