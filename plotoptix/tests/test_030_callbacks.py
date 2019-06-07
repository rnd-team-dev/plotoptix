from unittest import TestCase

from plotoptix import NpOptiX

import numpy as np
import time

class CbTestOptiX(NpOptiX):

    def __init__(self,
                 on_initialization = None,
                 on_scene_compute = None,
                 on_rt_completed = None,
                 on_launch_finished = None,
                 on_rt_accum_done = None):

        super().__init__(
            on_initialization=on_initialization,
            on_scene_compute=on_scene_compute,
            on_rt_completed=on_rt_completed,
            on_launch_finished=on_launch_finished,
            on_rt_accum_done=on_rt_accum_done,
            width=128, height=96,
            start_now=False,
            log_level='INFO')


class TestCallbacks(TestCase):

    scene = None
    is_alive = False

    initialization_trigs = 0
    launch_finished_trigs = 0
    accumulation_done_trigs = 0
    start_compute_trigs = 0
    rt_completed_trigs = 0

    accumulation_step = 3
    expected_launches = 4

    max_rt_time = 30
    rt_dt = 0.1

    def cb_init(self, rt):
        TestCallbacks.initialization_trigs += 1
        print("... Initialization callback.")

    def cb_launch_finished(self, rt):
        TestCallbacks.launch_finished_trigs += 1
        print("... Launch finished callback.")

    def cb_accum_done(self, rt):
        TestCallbacks.accumulation_done_trigs += 1
        print("... Accumulation done callback.")

    def cb_start_compute(self, rt, delta):
        TestCallbacks.start_compute_trigs += 1
        print("+++ Start compute callback.")

    def cb_rt_completed(self, rt):
        TestCallbacks.rt_completed_trigs += 1
        print("+++ RT completed callback.")

    @classmethod
    def setUpClass(cls):
        print("################    Test 030: callbacks.    ######################################")

    def test010_setup_and_start(self):
        TestCallbacks.scene = CbTestOptiX(on_initialization=self.cb_init,
                                          on_scene_compute=self.cb_start_compute,
                                          on_rt_completed=self.cb_rt_completed,
                                          on_launch_finished=self.cb_launch_finished,
                                          on_rt_accum_done=self.cb_accum_done)

        TestCallbacks.scene.set_param(min_accumulation_step=TestCallbacks.accumulation_step,
                                      max_accumulation_frames=TestCallbacks.accumulation_step * TestCallbacks.expected_launches)

        n = 1000 # 1k data points, in order to have something in view
        xyz = 3 * (np.random.random((n, 3)) - 0.5)
        TestCallbacks.scene.set_data("plot", xyz, r=0.1)
        TestCallbacks.scene.set_background(0.99)
        TestCallbacks.scene.set_ambient(0.33)
        TestCallbacks.scene.setup_camera("cam1")

        TestCallbacks.scene.start()
        self.assertTrue(TestCallbacks.scene.is_started(), msg="Scene did not flip to _is_started=True state.")
        self.assertTrue(TestCallbacks.scene.isAlive(), msg="Raytracing thread is not alive.")
        TestCallbacks.is_alive = True

        self.assertTrue(TestCallbacks.initialization_trigs == 1,
                        msg="Initialization triggers: %d, expected: %d." % (TestCallbacks.initialization_trigs, 1))

        t = 0;
        while (t < TestCallbacks.max_rt_time) and (TestCallbacks.accumulation_done_trigs) < 1:
            time.sleep(TestCallbacks.rt_dt)
            t += TestCallbacks.rt_dt
        self.assertTrue(t < TestCallbacks.max_rt_time, msg="Scene raytracing time exceeded the limit of %f." % (TestCallbacks.max_rt_time,))
        self.assertTrue(TestCallbacks.launch_finished_trigs == TestCallbacks.expected_launches,
                        msg="Launch finished triggers: %d, expected: %d." % (TestCallbacks.launch_finished_trigs, TestCallbacks.expected_launches))
        self.assertTrue(TestCallbacks.accumulation_done_trigs == 1,
                        msg="Accumulation done triggers: %d, expected: %d." % (TestCallbacks.accumulation_done_trigs, 1))
        print("*** Scene raytraced in ~%.1fs." % (t,))

    def test020_compute_thread(self):

        TestCallbacks.scene.pause_compute()

        f = TestCallbacks.accumulation_done_trigs
        k = TestCallbacks.start_compute_trigs

        TestCallbacks.scene.refresh_scene()

        t = 0;
        while (t < TestCallbacks.max_rt_time) and (TestCallbacks.accumulation_done_trigs) < f + 1:
            time.sleep(TestCallbacks.rt_dt)
            t += TestCallbacks.rt_dt

        self.assertTrue(TestCallbacks.start_compute_trigs == k,
                        msg="Start compute triggers increased during pause (%d -> %d)." % (k, TestCallbacks.start_compute_trigs))

        f = TestCallbacks.accumulation_done_trigs
        k = TestCallbacks.start_compute_trigs

        TestCallbacks.scene.resume_compute()
        TestCallbacks.scene.refresh_scene()

        t = 0;
        while (t < TestCallbacks.max_rt_time) and (TestCallbacks.accumulation_done_trigs) < f + 1:
            time.sleep(TestCallbacks.rt_dt)
            t += TestCallbacks.rt_dt

        self.assertTrue(TestCallbacks.start_compute_trigs > k,
                        msg="Start compute triggers not increasing after resume.")

    def test999_close(self):
        self.assertTrue(TestCallbacks.scene is not None and TestCallbacks.is_alive, msg="Wrong state of the test class.")

        self.assertTrue(TestCallbacks.initialization_trigs == 1,
                        msg="Initialization triggers: %d, expected: %d." % (TestCallbacks.initialization_trigs, 1))

        self.assertTrue(TestCallbacks.start_compute_trigs == TestCallbacks.rt_completed_trigs,
                        msg="Start compute triggers (%d) not equal to RT finished triggers (%d)." % (TestCallbacks.start_compute_trigs, TestCallbacks.rt_completed_trigs))

        TestCallbacks.scene.close()
        TestCallbacks.scene.join(10)
        self.assertTrue(TestCallbacks.scene.is_closed(), msg="Scene did not flip to _is_closed=True state.")
        self.assertFalse(TestCallbacks.scene.isAlive(), msg="Raytracing thread closing timed out.")
        TestCallbacks.is_alive = False

    @classmethod
    def tearDownClass(cls):
        cls.assertFalse(cls, cls.is_alive, msg="Wrong state of the test class.")
        print("Test 030: completed.")

