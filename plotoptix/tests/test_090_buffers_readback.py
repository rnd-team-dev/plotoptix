from unittest import TestCase

from plotoptix import NpOptiX

import numpy as np
import time

class RbTestOptiX(NpOptiX):

    def __init__(self, on_rt_accum_done = None):

        super().__init__(
            on_rt_accum_done=on_rt_accum_done,
            width=128, height=96,
            start_now=False,
            log_level='INFO')


class TestCallbacks(TestCase):

    scene = None
    is_alive = False

    accumulation_done_trigs = 0

    tex_rb_ok = False
    hit_rb_ok = False
    geo_rb_ok = False

    max_rt_time = 30
    rt_dt = 0.1

    def cb_accum_done(self, rt):
        tex_rb = TestCallbacks.scene._raw_rgba[:,:,0]
        TestCallbacks.tex_rb_ok = np.allclose(TestCallbacks.tex, tex_rb)

        hit_test = np.zeros((TestCallbacks.scene._height, TestCallbacks.scene._width, 4), dtype=np.float32)
        hit_test[:,:,:3] = 0.1
        hit_test[:,:,3] = 0.5
        TestCallbacks.hit_rb_ok = np.allclose(TestCallbacks.scene._hit_pos, hit_test)

        geo_test = np.full((TestCallbacks.scene._height, TestCallbacks.scene._width, 2), 0x12345678, dtype=np.uint32)
        TestCallbacks.geo_rb_ok = (TestCallbacks.scene._geo_id == geo_test).all()

        TestCallbacks.accumulation_done_trigs += 1
        print("... Accumulation done callback.")

    @classmethod
    def setUpClass(cls):
        print("################    Test 090: read device buffers.    ############################")

    def test010_setup_and_start(self):
        TestCallbacks.scene = RbTestOptiX(on_rt_accum_done=self.cb_accum_done)

        TestCallbacks.scene.set_param(min_accumulation_step=1,
                                      max_accumulation_frames=1)
        
        w = TestCallbacks.scene._width
        h = TestCallbacks.scene._height
        TestCallbacks.tex = np.zeros((h, w))
        for j in range(h):
            for i in range(w):
                TestCallbacks.tex[j,i] = ((h - j)/h) * ((w - i)/w)

        TestCallbacks.scene.set_texture_2d("cam_tex", TestCallbacks.tex)
        TestCallbacks.scene.setup_camera("cam1", cam_type="TexTest", eye=[0, 0, 10], target=[0, 0, 0], textures=["cam_tex"])

        TestCallbacks.scene.start()
        self.assertTrue(TestCallbacks.scene.is_started(), msg="Scene did not flip to _is_started=True state.")
        self.assertTrue(TestCallbacks.scene.is_alive(), msg="Raytracing thread is not alive.")
        TestCallbacks.is_alive = True

        t = 0;
        while (t < TestCallbacks.max_rt_time) and (TestCallbacks.accumulation_done_trigs) < 1:
            time.sleep(TestCallbacks.rt_dt)
            t += TestCallbacks.rt_dt
        self.assertTrue(t < TestCallbacks.max_rt_time, msg="Scene raytracing time exceeded the limit of %f." % (TestCallbacks.max_rt_time,))
        print("*** Scene raytraced in ~%.1fs." % (t,))

        self.assertTrue(TestCallbacks.tex_rb_ok, msg="Camera texture not passed properly.")
        self.assertTrue(TestCallbacks.hit_rb_ok, msg="Hit info not passed properly.")
        self.assertTrue(TestCallbacks.geo_rb_ok, msg="Geometry id's not passed properly.")

    def test999_close(self):
        self.assertTrue(TestCallbacks.scene is not None and TestCallbacks.is_alive, msg="Wrong state of the test class.")

        TestCallbacks.scene.close()
        TestCallbacks.scene.join(10)
        self.assertTrue(TestCallbacks.scene.is_closed(), msg="Scene did not flip to _is_closed=True state.")
        self.assertFalse(TestCallbacks.scene.is_alive(), msg="Raytracing thread closing timed out.")
        TestCallbacks.is_alive = False

    @classmethod
    def tearDownClass(cls):
        cls.assertFalse(cls, cls.is_alive, msg="Wrong state of the test class.")
        print("Test 090: completed.")

