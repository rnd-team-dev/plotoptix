from unittest import TestCase

from plotoptix import NpOptiX

import os
import numpy as np
import time

class SaveTestOptiX(NpOptiX):

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

    frame = 0

    def cb_start_compute(self, rt, delta):
        rt.rotate_geometry("plot", (0, np.pi/180, 0))

    def cb_rt_completed(self, rt):
        rt.update_geom_buffers("plot", "Positions")

    @classmethod
    def setUpClass(cls):
        print("################    Test 050: save output.    ####################################")

    def test010_setup_and_start(self):
        TestOutput.scene = SaveTestOptiX(on_scene_compute=self.cb_start_compute,
                                         on_rt_completed=self.cb_rt_completed)

        TestOutput.scene.set_param(min_accumulation_step=4, max_accumulation_frames=8)
        TestOutput.scene.encoder_create(fps=20, bitrate=1, profile="High")

        n = 1000 # 1k data points, to have something in view
        xyz = 3 * (np.random.random((n, 3)) - 0.5)
        TestOutput.scene.set_data("plot", xyz, r=0.1)
        TestOutput.scene.setup_camera("cam1")
        TestOutput.scene.set_background(0)
        TestOutput.scene.set_ambient(0.75)

        TestOutput.scene.start()
        self.assertTrue(TestOutput.scene.is_started(), msg="Scene did not flip to _is_started=True state.")
        self.assertTrue(TestOutput.scene.isAlive(), msg="Raytracing thread is not alive.")
        TestOutput.is_alive = True

        self.assertFalse(TestOutput.scene.encoder_is_open(), msg="Encoder is_open is True on startup.")
        self.assertTrue(TestOutput.scene.encoded_frames() == 0, msg="Encoded frames on startup should be 0.")

    def test020_encode(self):
        frames_to_encode = 20 * 2

        fname = "test_encode.mp4"
        TestOutput.scene.encoder_start(fname, frames_to_encode)

        self.assertTrue(TestOutput.scene.encoder_is_open(), msg="Encoder is_open is True on startup.")

        t = 0; dt = 0.5; max_rt_time = 20
        while (t < max_rt_time) and TestOutput.scene.encoder_is_open():
            time.sleep(dt)
            t += dt
        self.assertTrue(t < max_rt_time, msg="Encoding time exceeded the limit of %f." % (max_rt_time,))

        encoded_frames = TestOutput.scene.encoded_frames()
        self.assertTrue(encoded_frames == frames_to_encode,
                        msg="Encoded frames: %d, expected: %d." % (encoded_frames, frames_to_encode))

        self.assertTrue(os.path.isfile(fname), msg="Video file %s not created." % (fname,))
        os.remove(fname)

    def test030_image(self):
        fname = "test_img.png"
        TestOutput.scene.save_image(fname)
        self.assertTrue(os.path.isfile(fname), msg="Image file %s not created." % (fname,))
        os.remove(fname)

        fname = "test_img.jpg"
        TestOutput.scene.save_image(fname)
        self.assertTrue(os.path.isfile(fname), msg="Image file %s not created." % (fname,))
        os.remove(fname)

        fname = "test_img.tif"
        TestOutput.scene.save_image(fname)
        self.assertTrue(os.path.isfile(fname), msg="Image file %s not created." % (fname,))
        os.remove(fname)
        print("*** python test test030_image done")

    def test999_close(self):
        self.assertTrue(TestOutput.scene is not None and TestOutput.is_alive, msg="Wrong state of the test class.")

        print("*** python test close")
        TestOutput.scene.close()
        TestOutput.scene.join(10)
        self.assertTrue(TestOutput.scene.is_closed(), msg="Scene did not flip to _is_closed=True state.")
        self.assertFalse(TestOutput.scene.isAlive(), msg="Raytracing thread closing timed out.")
        TestOutput.is_alive = False

    @classmethod
    def tearDownClass(cls):
        cls.assertFalse(cls, cls.is_alive, msg="Wrong state of the test class.")
        print("Test 050: completed.")

