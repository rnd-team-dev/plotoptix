"""
Tkinter UI for PlotOptiX raytracer.

Copyright (C) 2019 R&D Team. All Rights Reserved.

Have a look at examples on GitHub: https://github.com/rnd_team/plotoptix.
"""

import sys, os, math, platform, logging, operator, functools, threading
import numpy as np
import tkinter as tk

from PIL import Image, ImageTk
from ctypes import cdll, CFUNCTYPE, POINTER, byref, c_float, c_uint, c_int, c_bool, c_char_p, c_wchar_p, c_void_p
from typing import List, Callable, Optional, Union, Any

from plotoptix.singleton import Singleton
from plotoptix.enums import *

logging.basicConfig(level=logging.WARN, format='[%(levelname)s] (%(threadName)-10s) %(message)s')

PARAM_NONE_CALLBACK = CFUNCTYPE(None)
PARAM_INT_CALLBACK = CFUNCTYPE(None, c_int)

PLATFORM = platform.system()
if PLATFORM == "Windows":
    BIN_PATH = "bin\\win"
    LIB_EXT = ".dll"
elif PLATFORM == "Linux":
    BIN_PATH = "bin\\linux"
    LIB_EXT = ".so"
elif PLATFORM == "Darwin":
    BIN_PATH = "bin\\mac"
    LIB_EXT = ".so"
else:
    BIN_PATH = ""
    LIB_EXT == ""

# development shortcut
#BIN_PATH = "..\\..\\..\\OptiX\\libSharpOptiX\\bin\\Debug"

## NOTE the naming: functions with the names _gui_* can be used from the GUI thread (Tk event loop) only.

class TkOptiX(threading.Thread, metaclass=Singleton):

    def __init__(self,
                 on_initialization = None,
                 on_scene_compute = None,
                 on_rt_completed = None,
                 on_rt_accum_done = None,
                 width: int = -1,
                 height: int = -1,
                 show_on_start: bool = False,
                 log_level: Union[int, str] = logging.WARN) -> None:

        self._logger = logging.getLogger(__name__ + "-TkOptiX")
        self._logger.setLevel(log_level)
        self._package_dir = os.path.dirname(__file__)
        self._started_event = threading.Event()
        self._padlock = threading.Lock()
        self._is_started = False
        self._is_closed = False

        self._ini_width = width
        self._ini_height = height

        # load SharpOptiX library, setup arguments and return types ###
        self._optix = cdll.LoadLibrary(os.path.join(self._package_dir, BIN_PATH, "RnD.SharpOptiX" + LIB_EXT))

        self._optix.create_empty_scene.argtypes = [c_int, c_int, c_void_p, c_int]
        self._optix.create_empty_scene.restype = c_bool

        self._optix.create_scene_from_json.argtypes = [c_wchar_p, c_int, c_int, c_void_p, c_int]
        self._optix.create_scene_from_json.restype = c_bool

        self._optix.load_scene_from_json.argtypes = [c_wchar_p]
        self._optix.load_scene_from_json.restype = c_bool

        self._optix.load_scene_from_file.argtypes = [c_wchar_p]
        self._optix.load_scene_from_file.restype = c_bool

        self._optix.save_scene_to_file.argtypes = [c_wchar_p]
        self._optix.save_scene_to_file.restype = c_bool

        self._optix.start_rt.restype = c_bool
        self._optix.stop_rt.restype = c_bool

        self._optix.set_int.argtypes = [c_wchar_p, c_int, c_bool]
        self._optix.set_int.restype = c_bool

        self._optix.set_uint.argtypes = [c_wchar_p, c_uint, c_bool]
        self._optix.set_uint.restype = c_bool

        self._optix.set_uint2.argtypes = [c_wchar_p, c_uint, c_uint, c_bool]
        self._optix.set_uint2.restype = c_bool

        self._optix.set_float.argtypes = [c_wchar_p, c_float, c_bool]
        self._optix.set_float.restype = c_bool

        self._optix.set_float2.argtypes = [c_wchar_p, c_float, c_float, c_bool]
        self._optix.set_float2.restype = c_bool

        self._optix.set_float3.argtypes = [c_wchar_p, c_float, c_float, c_float, c_bool]
        self._optix.set_float3.restype = c_bool

        self._optix.resize_scene.argtypes = [c_int, c_int, c_void_p, c_int]
        self._optix.resize_scene.restype = c_bool

        self._optix.setup_geometry.argtypes = [c_int, c_wchar_p, c_wchar_p, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
        self._optix.setup_geometry.restype = c_uint

        self._optix.update_geometry.argtypes = [c_wchar_p, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
        self._optix.update_geometry.restype = c_uint

        self._optix.set_coordinates_geom.argtypes = [c_int, c_float]
        self._optix.set_coordinates_geom.restype = c_bool

        self._optix.setup_camera.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_float, c_float, c_bool]
        self._optix.setup_camera.restype = c_int

        self._optix.update_camera.argtypes = [c_uint, c_void_p, c_void_p, c_void_p, c_float, c_float, c_float]
        self._optix.update_camera.restype = c_bool

        self._optix.fit_camera.argtypes = [c_uint, c_wchar_p, c_float]
        self._optix.fit_camera.restype = c_bool

        self._optix.get_current_camera.restype = c_uint

        self._optix.set_current_camera.argtypes = [c_uint]
        self._optix.set_current_camera.restype = c_bool

        self._optix.rotate_camera_eye.argtypes = [c_int, c_int, c_int, c_int]
        self._optix.rotate_camera_eye.restype = c_bool

        self._optix.rotate_camera_tgt.argtypes = [c_int, c_int, c_int, c_int]
        self._optix.rotate_camera_tgt.restype = c_bool

        self._optix.get_camera_focal_scale.argtypes = [c_uint]
        self._optix.get_camera_focal_scale.restype = c_float

        self._optix.set_camera_focal_scale.argtypes = [c_float]
        self._optix.set_camera_focal_scale.restype = c_bool

        self._optix.set_camera_focal_length.argtypes = [c_float]
        self._optix.set_camera_focal_length.restype = c_bool

        self._optix.get_camera_fov.argtypes = [c_uint]
        self._optix.get_camera_fov.restype = c_float

        self._optix.set_camera_fov.argtypes = [c_float]
        self._optix.set_camera_fov.restype = c_bool

        self._optix.get_camera_aperture.argtypes = [c_uint]
        self._optix.get_camera_aperture.restype = c_float

        self._optix.set_camera_aperture.argtypes = [c_float]
        self._optix.set_camera_aperture.restype = c_bool

        self._optix.get_camera_eye.argtypes = [c_uint, c_void_p]
        self._optix.get_camera_eye.restype = c_bool

        self._optix.set_camera_eye.argtypes = [c_void_p]
        self._optix.set_camera_eye.restype = c_bool

        self._optix.get_camera_target.argtypes = [c_uint, c_void_p]
        self._optix.get_camera_target.restype = c_bool

        self._optix.set_camera_target.argtypes = [c_void_p]
        self._optix.set_camera_target.restype = c_bool

        self._optix.setup_spherical_light.argtypes = [c_void_p, c_void_p, c_float, c_bool]
        self._optix.setup_spherical_light.restype = c_int

        self._optix.setup_parallelogram_light.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_bool]
        self._optix.setup_parallelogram_light.restype = c_int

        self._optix.update_light.argtypes = [c_int, c_void_p, c_void_p, c_float, c_void_p, c_void_p]
        self._optix.update_light.restype = c_bool

        self._optix.fit_light.argtypes = [c_int, c_uint, c_float, c_float, c_float]
        self._optix.fit_light.restype = c_bool

        self._optix.get_object_at.argtypes = [c_int, c_int, POINTER(c_uint), POINTER(c_uint)]
        self._optix.get_object_at.restype = c_bool

        self._optix.get_hit_at.argtypes = [c_int, c_int, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float)]
        self._optix.get_hit_at.restype = c_bool

        self._optix.register_launch_finished_callback.argtypes = [PARAM_INT_CALLBACK]
        self._optix.register_launch_finished_callback.restype = c_bool

        self._optix.register_scene_rt_starting_callback.argtypes = [PARAM_NONE_CALLBACK]
        self._optix.register_scene_rt_starting_callback.restype = c_bool

        self._optix.register_start_scene_compute_callback.argtypes = [PARAM_INT_CALLBACK]
        self._optix.register_start_scene_compute_callback.restype = c_bool

        self._optix.register_scene_rt_completed_callback.argtypes = [PARAM_INT_CALLBACK]
        self._optix.register_scene_rt_completed_callback.restype = c_bool

        self._optix.set_min_accumulation_step.argtypes = [c_int]
        self._optix.set_min_accumulation_step.restype = c_bool

        self._optix.set_max_accumulation_frames.argtypes = [c_int]
        self._optix.set_max_accumulation_frames.restype = c_bool

        self._optix.set_gpu_architecture.argtypes = [c_int]

        self._optix.set_library_dir.argtypes = [c_wchar_p]
        self._optix.set_library_dir(os.path.join(self._package_dir, BIN_PATH))

        self._optix.set_include_dir.argtypes = [c_wchar_p]
        self._optix.set_include_dir(os.path.join(self._package_dir, BIN_PATH, "cuda"))

        if PLATFORM == "Windows":
            self._optix.get_display_scaling.restype = c_float

        self._logger.info("RnD.SharpOptiX library configured.")
        ###############################################################

        if PLATFORM == "Windows":
            dpi_scale = self._optix.get_display_scaling()
            self._logger.info("DPI scaling: %d", dpi_scale)
            if dpi_scale != 1:
                self._logger.warn("DPI setting may cause blurred raytracing output, see this answer")
                self._logger.warn("for the solution https://stackoverflow.com/a/52599951/10037996:")
                self._logger.warn("set python.exe and pythonw.exe files Properties -> Compatibility")
                self._logger.warn("-> Change high DPI settings -> check Override high DPI scaling")
                self._logger.warn("behaviour, select Application in the drop-down menu.")

        # setup SharpOptiX interface ##################################
        self._logger.info("Preparing empty scene...")
        
        self._width = 16
        self._height = 16

        self._img_rgba = np.ascontiguousarray(np.zeros((self._height, self._width, 4), dtype=np.uint8))

        self.geometry_handles = {} # geometry name to handle dictionary
        self.geometry_names = {}   # geometry handle to name dictionary
        self.geometry_sizes = {}   # geometry name to size dictionary
        self.camera_handles = {}   # camera name to handle dictionary
        self.light_handles = {}    # light name to handle dictionary


        if on_initialization is not None: self._initialization_cb = self._make_list_of_callable(on_initialization)
        else: self._initialization_cb = [self._default_initialization]

        if on_scene_compute is not None: self._scene_compute_cb = self._make_list_of_callable(on_scene_compute)
        else: self._scene_compute_cb = []

        if on_rt_completed is not None: self._rt_completed_cb = self._make_list_of_callable(on_rt_completed)
        else: self._rt_completed_cb = []

        if on_rt_accum_done is not None: self._rt_accum_done_cb = self._make_list_of_callable(on_rt_accum_done)
        else: self._rt_accum_done_cb = []

        self._is_scene_created = self._optix.create_empty_scene(self._width, self._height, self._img_rgba.ctypes.data, 4 * self._img_rgba.size)
        self._logger.info("Scene ready: %r", self._is_scene_created)

        if self._is_scene_created:

            super(TkOptiX, self).__init__()

            if show_on_start:
                self.start()
                if not self._started_event.wait(10):
                    self._logger.error("Raytracing output startup timed out.")
                    self._is_started = False
            else:
                self._logger.info("Use show() or start() to open the raytracing output.")
        else:
            self._logger.error("Initial setup failed, see errors above.")
        ###############################################################

    def _make_list_of_callable(self, items) -> List[Callable[["TkOptiX"], None]]:
        if callable(items): return [items]
        else:
            for item in items:
                assert callable(item), "Expected callable or list of callable items."
            return items

    # For matplotlib users convinience.
    def show(self) -> None:
        if not self._is_started:
            self.start()
            if not self._started_event.wait(10):
                self._logger.error("Raytracing output startup timed out.")
                self._is_started = False
        else:
            self._logger.warn("Raytracing output already running.")

    def close(self) -> None:
        if self._is_started: self._canvas.event_generate("<<CloseScene>>", when="tail")
        else: self._logger.warn("Raytracing output not running.")

    def is_started(self) -> bool: return self._is_started
    def is_closed(self) -> bool: return self._is_closed

    def get_rt_output(self) -> np.ndarray:
        assert self._is_started, "Raytracing output not running."
        self._padlock.acquire()
        a = self._img_rgba.copy()
        self._padlock.release()
        return a

    def _gui_quit_callback(self, *args):
        assert self._is_started, "Raytracing output not running."

        self._logger.info("Closing raytracing window.")
        self._is_scene_created = False
        self._is_started = False
        self._optix.stop_rt()
        self._optix.destroy_scene()
        self._root.quit()
        self._is_closed = True


    def _gui_motion_left(self, event):
        self._mouse_to_x, self._mouse_to_y = event.x, event.y

    def _gui_motion_right(self, event):
        self._mouse_to_x, self._mouse_to_y = event.x, event.y

    def _gui_pressed_left(self, event):
        self._mouse_from_x, self._mouse_from_y = event.x, event.y
        self._mouse_to_x = self._mouse_from_x
        self._mouse_to_y = self._mouse_from_y
        self._left_mouse = True

    def _gui_pressed_right(self, event):
        self._mouse_from_x, self._mouse_from_y = event.x, event.y
        self._mouse_to_x = self._mouse_from_x
        self._mouse_to_y = self._mouse_from_y
        self._right_mouse = True

    def _gui_released_left(self, event):
        self._mouse_to_x, self._mouse_to_y = event.x, event.y
        self._mouse_from_x = self._mouse_to_x
        self._mouse_from_y = self._mouse_to_y
        self._left_mouse = False

    def _gui_released_right(self, event):
        self._mouse_to_x, self._mouse_to_y = event.x, event.y
        self._mouse_from_x = self._mouse_to_x
        self._mouse_from_y = self._mouse_to_y
        self._right_mouse = False

    def _gui_doubleclick_left(self, event):
        assert self._is_started, "Raytracing output not running."

        x, y = event.x, event.y
        c_handle = c_uint()
        c_index = c_uint()
        if self._optix.get_object_at(x, y, byref(c_handle), byref(c_index)):
            handle = c_handle.value
            index = c_index.value
            if (handle != 0xFFFFFFFF) and (handle in self.geometry_names):
                if not self._any_key:
                    self._logger.info("Selected geometry: %s, primitive index %d", self.geometry_names[handle], index)
                elif self._ctrl_key:
                    c_x = c_float()
                    c_y = c_float()
                    c_z = c_float()
                    c_d = c_float()
                    if self._optix.get_hit_at(x, y, c_x, c_y, c_z, c_d):
                        hx = c_x.value
                        hy = c_y.value
                        hz = c_z.value
                        hd = c_d.value
                        if hd > 0:
                            self._logger.info("Hit 3D coordinates: [%f %f %f], at focal distance %f", hx, hy, hz, hd)
                            _ = self._optix.set_camera_focal_length(hd)
            else:
                self._logger.info("No object at [%d %d]", x, y)

    def _gui_key_pressed(self, event):
        if event.keysym == "Control_L":
            self._ctrl_key = True
            self._any_key = True
        elif event.keysym == "Shift_L":
            self._shift_key = True
            self._any_key = True
        else:
            self._any_key = False

    def _gui_key_released(self, event):
        if event.keysym == "Control_L":
            self._ctrl_key = False
        elif event.keysym == "Shift_L":
            self._shift_key = False
        self._any_key = False

    def _gui_configure(self, event):
        assert self._is_started, "Raytracing output not running."

        if not self._started_event.is_set(): self._started_event.set()

        self._width, self._height = self._canvas.winfo_width(), self._canvas.winfo_height()

        self._logger.info("New size: %d x %d", self._width, self._height)

        self._padlock.acquire()
        # allocate new buffer
        img_buffer = np.ascontiguousarray(np.zeros((self._height, self._width, 4), dtype=np.uint8))
        pil_img = Image.fromarray(img_buffer, mode="RGBX")
        tk_img = ImageTk.PhotoImage(pil_img)
        # update buffer pointer and size in the raytracer
        self._optix.resize_scene(self._width, self._height, img_buffer.ctypes.data, 4 * img_buffer.size)
        # swap references stored in the window instance
        self._img_rgba = img_buffer
        self._tk_img = tk_img
        # update image on canvas
        self._canvas.itemconfig(self._img_id, image=self._tk_img)
        # no redraws until the next launch
        self._update_req = False
        self._padlock.release()

    # update raytraced image in Tk window #############################
    def _gui_update_content(self, *args):
        assert self._is_started, "Raytracing output not running."

        if self._update_req:
            self._padlock.acquire()
            self._update_req = False
            pil_img = Image.fromarray(self._img_rgba, mode="RGBX")
            tk_img = ImageTk.PhotoImage(pil_img)
            self._canvas.itemconfig(self._img_id, image=tk_img)
            self._tk_img = tk_img
            self._padlock.release()

    def _launch_finished_callback(self, rt_result : int):
        if self._is_started and rt_result != RtResult.NoUpdates.value:
            self._update_req = True
            self._canvas.event_generate("<<LaunchFinished>>", when="tail")
    def _get_launch_finished_callback(self):
        def func(rt_result : int): self._launch_finished_callback(rt_result)
        return PARAM_INT_CALLBACK(func)
    ###################################################################

    # apply scene edits made in ui ####################################
    def _gui_apply_edits(self, *args):
        assert self._is_started, "Raytracing output not running."

        if (self._mouse_from_x != self._mouse_to_x) or (self._mouse_from_y != self._mouse_to_y):

            # manipulate camera:
            if self._left_mouse:
                if not self._any_key:
                    self._optix.rotate_camera_eye(self._mouse_from_x, self._mouse_from_y, self._mouse_to_x, self._mouse_to_y)
                elif self._ctrl_key:
                    df = 1 + 0.01 * (self._mouse_from_y - self._mouse_to_y)
                    f = self._optix.get_camera_focal_scale(0) # 0 is current cam
                    self._optix.set_camera_focal_scale(df * f)
                elif self._shift_key:
                    df = 1 + 0.005 * (self._mouse_from_y - self._mouse_to_y)
                    f = self._optix.get_camera_fov(0) # 0 is current cam
                    self._optix.set_camera_fov(df * f)

            elif self._right_mouse:
                if not self._any_key:
                    self._optix.rotate_camera_tgt(self._mouse_from_x, self._mouse_from_y, self._mouse_to_x, self._mouse_to_y)
                elif self._ctrl_key:
                    da = 1 + 0.01 * (self._mouse_from_y - self._mouse_to_y)
                    a = self._optix.get_camera_aperture(0) # 0 is current cam
                    self._optix.set_camera_aperture(da * a)
                elif self._shift_key:
                    target = np.ascontiguousarray([0, 0, 0], dtype=np.float32)
                    self._optix.get_camera_target(0, target.ctypes.data) # 0 is current cam
                    eye = np.ascontiguousarray([0, 0, 0], dtype=np.float32)
                    self._optix.get_camera_eye(0, eye.ctypes.data) # 0 is current cam
                    dl = 0.01 * (self._mouse_from_y - self._mouse_to_y)
                    eye = eye - dl * (target - eye)
                    self._optix.set_camera_eye(eye.ctypes.data)
                    pass

            # ... or manipulate other ogjects (need to save selected object, to be implemented)

            self._mouse_from_x = self._mouse_to_x
            self._mouse_from_y = self._mouse_to_y

    def _scene_rt_starting_callback(self):
        if self._is_started: self._canvas.event_generate("<<SceneRtStarting>>", when="tail")
    def _get_scene_rt_starting_callback(self):
        def func(): self._scene_rt_starting_callback()
        return PARAM_NONE_CALLBACK(func)
    ###################################################################

    # rt-synced scene computation and data uploads to gpu #############
    def _start_scene_compute_callback(self, n_frames : int):
        if self._is_started:
            self._logger.info("Compute, delta %d frames.", n_frames)
            for c in self._scene_compute_cb: c(self, n_frames)
    def _get_start_scene_compute_callback(self):
        def func(n_frames : int): self._start_scene_compute_callback(n_frames)
        return PARAM_INT_CALLBACK(func)

    def _scene_rt_completed_callback(self, rt_result : int):
        if self._is_started:
            self._logger.info("RT completed, result %d.", rt_result)
            for c in self._rt_completed_cb: c(self)
            if rt_result == RtResult.AccumDone.value:
                self._logger.info("RT accumulation finished.")
                for c in self._rt_accum_done_cb: c(self)
    def _get_scene_rt_completed_callback(self):
        def func(rt_result : int): self._scene_rt_completed_callback(rt_result)
        return PARAM_INT_CALLBACK(func)
    ###################################################################

    def run(self):
        assert self._is_scene_created, "Scene is not ready, see initialization messages."
        assert not self._is_started, "Raytracing output already running."

        # setup Tk window #############################################
        self._padlock.acquire()
        self._root = tk.Tk()

        screen_width = self._root.winfo_screenwidth()
        screen_height = self._root.winfo_screenheight()

        if self._ini_width > 0: self._width = self._ini_width
        else: self._width = int(screen_width / 2)
        if self._ini_height > 0: self._height = self._ini_height
        else: self._height = int(screen_height / 2)
        self._update_req = False

        self._mouse_from_x = 0
        self._mouse_from_y = 0
        self._mouse_to_x = 0
        self._mouse_to_y = 0
        self._left_mouse = False
        self._right_mouse = False
        self._ctrl_key = False
        self._shift_key = False
        self._any_key = False

        self._root.title("R&D PlotOptiX")
        self._root.protocol("WM_DELETE_WINDOW", self._gui_quit_callback)

        self._canvas = tk.Canvas(self._root, width=self._width, height=self._height)
        self._canvas.pack(side="top", fill=tk.BOTH, expand=True)
        self._canvas.pack_propagate(0)
        self._canvas.bind("<Configure>", self._gui_configure)
        self._canvas.bind('<B1-Motion>', self._gui_motion_left)
        self._canvas.bind('<B3-Motion>', self._gui_motion_right)
        self._canvas.bind("<Button-1>", self._gui_pressed_left)
        self._canvas.bind("<Button-3>", self._gui_pressed_right)
        self._canvas.bind("<ButtonRelease-1>", self._gui_released_left)
        self._canvas.bind("<ButtonRelease-3>", self._gui_released_right)
        self._canvas.bind("<Double-Button-1>", self._gui_doubleclick_left)
        self._root.bind_all("<KeyPress>", self._gui_key_pressed)
        self._root.bind_all("<KeyRelease>", self._gui_key_released)
        self._canvas.bind("<<LaunchFinished>>", self._gui_update_content)
        self._canvas.bind("<<SceneRtStarting>>", self._gui_apply_edits)
        self._canvas.bind("<<CloseScene>>", self._gui_quit_callback)
        self._logger.info("Tkinter widgets ready.")
        ###############################################################

        # setup SharpOptiX interface ##################################
        self._logger.info("Couple scene to the output window...")

        img_buffer = np.ascontiguousarray(np.zeros((self._height, self._width, 4), dtype=np.uint8))
        pil_img = Image.fromarray(img_buffer, mode="RGBX")
        self._tk_img = ImageTk.PhotoImage(pil_img)

        result = self._optix.resize_scene(self._width, self._height, img_buffer.ctypes.data, 4 * img_buffer.size)
        self._img_id = self._canvas.create_image(0, 0, image=self._tk_img, anchor=tk.NW)
        self._img_rgba = img_buffer

        for c in self._initialization_cb: c(self)

        if result:
            c1_ptr = self._get_launch_finished_callback()
            r1 = self._optix.register_launch_finished_callback(c1_ptr)
            c2_ptr = self._get_scene_rt_starting_callback()
            r2 = self._optix.register_scene_rt_starting_callback(c2_ptr)
            c3_ptr = self._get_start_scene_compute_callback()
            r3 = self._optix.register_start_scene_compute_callback(c3_ptr)
            c4_ptr = self._get_scene_rt_completed_callback()
            r4 = self._optix.register_scene_rt_completed_callback(c4_ptr)
            self._logger.info("Callbacks registered: %r", r1 & r2 & r3 & r4)
            result &= r1 & r2 & r3 & r4
            if result: self._logger.info("...coupling and initialization done.")
        ###############################################################

        self._padlock.release()

        # start event loop ############################################
        if result:
            self._is_started = True
            self._gui_update_content()
            self._optix.start_rt()
            self._root.mainloop()
        else:
            self._logger.error("Did not start PlotOptiX window, check errors above.")
            self._started_event.set()
        ###############################################################

    @staticmethod
    def _default_initialization(wnd) -> None:
        wnd._logger.info("Default scene initialization.")
        if wnd._optix.get_current_camera() == 0:
            wnd.setup_camera("default", [0, 0, 10], [0, 0, 0])

    def refresh_scene(self) -> None:
        self._optix.refresh_scene()

    def set_float(self, name: str, x: float, y: Optional[float], z: Optional[float], refresh: bool = False) -> None:
        if not isinstance(name, str): name = str(name)
        if not isinstance(x, float): x = float(x)

        if z is not None: # expect float3
            if not isinstance(z, float): z = float(z)
            if not isinstance(y, float): y = float(y)

            self._optix.set_float3(name, x, y, z, refresh)
            return

        if y is not None: # expect float2
            if not isinstance(y, float): y = float(y)

            self._optix.set_float2(name, x, y, refresh)
            return

        self._optix.set_float(name, x, refresh)


    def set_uint(self, name: str, x: int, y: Optional[int], refresh: bool = False) -> None:
        if not isinstance(name, str): name = str(name)
        if not isinstance(x, int): x = int(x)

        if y is not None: # expect uint2
            if not isinstance(y, int): y = int(y)

            self._optix.set_uint2(name, x, y, refresh)
            return

        self._optix.set_uint(name, x, refresh)


    def set_int(self, name: str, x: int, refresh: bool = False) -> None:
        if not isinstance(name, str): name = str(name)
        if not isinstance(x, int): x = int(x)

        self._optix.set_int(name, x, refresh)


    def set_background(self, color: Any, refresh: bool = False) -> None:
        if isinstance(color, float) or isinstance(color, int):
            x = float(color)
            y = float(color)
            z = float(color)
        elif not isinstance(color, np.ndarray):
            color = np.asarray(color, dtype=np.float32)
            if (len(color.shape) != 1) or (color.shape[0] != 3):
                self._logger.error("Color should be a single value or 3-element array/list/tupe.")
                return
            x = color[0]
            y = color[1]
            z = color[2]

        self._optix.set_float3("bg_color", x, y, z, refresh)
        self._logger.info("Background color updated.")

    def set_ambient(self, color: Any, refresh: bool = False) -> None:
        if isinstance(color, float) or isinstance(color, int):
            x = float(color)
            y = float(color)
            z = float(color)
        elif not isinstance(color, np.ndarray):
            color = np.asarray(color, dtype=np.float32)
            if (len(color.shape) != 1) or (color.shape[0] != 3):
                self._logger.error("Color should be a single value or 3-element array/list/tupe.")
                return
            x = color[0]
            y = color[1]
            z = color[2]

        self._optix.set_float3("ambient_color", x, y, z, refresh)
        self._logger.info("Ambient color updated.")


    def set_param(self, **kwargs) -> None:
        try:
            self._padlock.acquire()
            for key, value in kwargs.items():
                self._logger.info("Set %s to %s", key, value)

                if key == 'min_accumulation_step':
                    self._optix.set_min_accumulation_step(int(value))
                elif key == 'max_accumulation_frames':
                    self._optix.set_max_accumulation_frames(int(value))
                else:
                    self._logger.error('Unknown parameter ' + key)

        except Exception as e:
            self._logger.error(str(e))

        finally:
            self._padlock.release()


    @staticmethod
    def _make_contiguous_vector(a: Optional[Any], n_dim: int) -> Optional[np.ndarray]:
        if a is None: return None

        if not isinstance(a, np.ndarray) or (a.dtype != np.float32):
            a = np.ascontiguousarray(a, dtype=np.float32)
        if len(a.shape) > 1: a = a.flatten()
        if a.shape[0] > n_dim: a = a[:n_dim]
        if a.shape[0] == 1: a = np.full(n_dim, a[0], dtype=np.float32)
        if a.shape[0] < n_dim: return None

        if not a.flags['C_CONTIGUOUS']: a = np.ascontiguousarray(a, dtype=np.float32)

        return a

    def get_camera(self, name: Optional[str] = None) -> (Optional[str], Optional[int]):
        cam_handle = 0
        if name is None: # try current camera
            cam_handle = self._optix.get_current_camera()
            if cam_handle == 0:
                self._logger.error("Current camera is not set.")
                return None, None

            for n, h in self.camera_handles.items():
                if h == cam_handle:
                    name = n
                    break

        else: # try camera by name
           if not isinstance(name, str): name = str(name)

           if name in self.camera_handles:
               cam_handle = self.camera_handles[name]
           else:
               self._logger.error("Camera %s does not exists.")
               return None, None

        return name, cam_handle

    def setup_camera(self, name: str,
                     eye: Optional[Any] = None,
                     target: Optional[Any] = None,
                     up: Any = np.ascontiguousarray([0, 1, 0], dtype=np.float32),
                     cam_type: Union[Camera, str] = Camera.Pinhole,
                     aperture_radius: float = 0.1,
                     aperture_fract: float = 0.15,
                     focal_scale: float = 1.0,
                     fov: float = 35.0,
                     blur: float = 1,
                     make_current: bool = True) -> None:
        
        if not isinstance(name, str): name = str(name)
        if isinstance(cam_type, str): cam_type = Camera[cam_type]

        if name in self.camera_handles:
            self._logger.error("Camera %s already exists.")
            return

        eye_ptr = 0
        eye = self._make_contiguous_vector(eye, 3)
        if eye is not None: eye_ptr = eye.ctypes.data

        target_ptr = 0
        target = self._make_contiguous_vector(target, 3)
        if target is not None: target_ptr = target.ctypes.data

        up = self._make_contiguous_vector(up, 3)
        if up is None:
            self._logger.error("Need 3D camera up vector.")
            return

        h = self._optix.setup_camera(cam_type.value,
                                     eye_ptr, target_ptr, up.ctypes.data,
                                     aperture_radius, aperture_fract,
                                     focal_scale, fov, blur,
                                     make_current)
        if h > 0:
            self._logger.info("Camera %s handle: %d.", name, h)
            self.camera_handles[name] = h
        else:
            self._logger.error("Camera setup failed.")

    def update_camera(self, name: Optional[str] = None,
                      eye: Optional[Any] = None,
                      target: Optional[Any] = None,
                      up: Optional[Any] = None,
                      aperture_radius: float = -1.0,
                      focal_scale: float = -1.0,
                      fov: float = -1.0) -> None:
        
        name, cam_handle = self.get_camera(name)
        if (name is None) or (cam_handle == 0): return

        eye = self._make_contiguous_vector(eye, 3)
        if eye is not None: eye_ptr = eye.ctypes.data
        else:               eye_ptr = 0

        target = self._make_contiguous_vector(target, 3)
        if target is not None: target_ptr = target.ctypes.data
        else:                  target_ptr = 0

        up = self._make_contiguous_vector(up, 3)
        if up is not None: up_ptr = up.ctypes.data
        else:              up_ptr = 0

        if self._optix.update_camera(cam_handle, eye_ptr, target_ptr, up_ptr,
                                     aperture_radius, focal_scale, fov):
            self._logger.info("Camera %s updated.", name)
        else:
            self._logger.error("Camera %s update failed.", name)

    def set_current_camera(self, name: str) -> None:
        
        if not isinstance(name, str): name = str(name)

        if name not in self.camera_handles:
            self._logger.error("Camera %s does not exists.")
            return

        if self._optix.set_current_camera(self.camera_handles[name]):
            self._logger.info("Current camera: %s", name)
        else:
            self._logger.error("Current camera not changed.")


    def camera_fit(self,
                   camera: Optional[str] = None,
                   geometry: Optional[str] = None,
                   scale: float = 2.5) -> None:

        cam_handle = 0
        if camera is not None:
            if not isinstance(camera, str): camera = str(camera)
            if camera in self.camera_handles:
                cam_handle = self.camera_handles[camera]

        if geometry is not None:
           if not isinstance(geometry, str): geometry = str(geometry)
        else: geometry = ""

        self._optix.fit_camera(cam_handle, geometry, scale)

    def setup_spherical_light(self, name: str, pos: Optional[Any] = None,
                              autofit_camera: Optional[str] = None,
                              color: Any = 10 * np.ascontiguousarray([1, 1, 1], dtype=np.float32),
                              radius: float = 1.0, in_geometry: bool = True) -> None:
        
        if not isinstance(name, str): name = str(name)

        if name in self.light_handles:
            self._logger.error("Light %s already exists.")
            return

        autofit = False
        pos = self._make_contiguous_vector(pos, 3)
        if pos is None:
            cam_name, _ = self.get_camera(autofit_camera)
            if cam_name is None:
                self._logger.error("Need 3D coordinates for the new light.")
                return

            pos = np.ascontiguousarray([0, 0, 0])
            autofit = True

        color = self._make_contiguous_vector(color, 3)
        if color is None:
            self._logger.error("Need color (single value or 3-element array/list/tuple).")
            return

        h = self._optix.setup_spherical_light(pos.ctypes.data, color.ctypes.data,
                                              radius, in_geometry)
        if h >= 0:
            self._logger.info("Light %s handle: %d.", name, h)
            self.light_handles[name] = h

            if autofit:
                self.light_fit(name, camera=cam_name)
        else:
            self._logger.error("Light setup failed.")

    def setup_parallelogram_light(self, name: str, pos: Optional[Any] = None,
                                  autofit_camera: Optional[str] = None,
                                  color: Any = 10 * np.ascontiguousarray([1, 1, 1], dtype=np.float32),
                                  u: Any = np.ascontiguousarray([0, 1, 0], dtype=np.float32),
                                  v: Any = np.ascontiguousarray([-1, 0, 0], dtype=np.float32),
                                  in_geometry: bool = True) -> None:
        
        if not isinstance(name, str): name = str(name)

        if name in self.light_handles:
            self._logger.error("Light %s already exists.")
            return

        autofit = False
        pos = self._make_contiguous_vector(pos, 3)
        if pos is None:
            cam_name, _ = self.get_camera(autofit_camera)
            if cam_name is None:
                self._logger.error("Need 3D coordinates for the new light.")
                return

            pos = np.ascontiguousarray([0, 0, 0])
            autofit = True

        color = self._make_contiguous_vector(color, 3)
        if color is None:
            self._logger.error("Need color (single value or 3-element array/list/tuple).")
            return

        u = self._make_contiguous_vector(u, 3)
        if u is None:
            self._logger.error("Need 3D vector U.")
            return

        v = self._make_contiguous_vector(v, 3)
        if v is None:
            self._logger.error("Need 3D vector V.")
            return

        h = self._optix.setup_parallelogram_light(pos.ctypes.data, color.ctypes.data,
                                                  u.ctypes.data, v.ctypes.data, in_geometry)
        if h >= 0:
            self._logger.info("Light %s handle: %d.", name, h)
            self.light_handles[name] = h

            if autofit:
                self.light_fit(name, camera=cam_name)
        else:
            self._logger.error("Light setup failed.")

    def setup_light(self, name: str,
                    light_type: Union[Light, str] = Light.Spherical,
                    pos: Optional[Any] = None,
                    autofit_camera: Optional[str] = None,
                    color: Any = 10 * np.ascontiguousarray([1, 1, 1], dtype=np.float32),
                    u: Any = np.ascontiguousarray([0, 1, 0], dtype=np.float32),
                    v: Any = np.ascontiguousarray([1, 0, 0], dtype=np.float32),
                    radius: float = 1.0, in_geometry: bool = True) -> None:

        if isinstance(light_type, str): light_type = Light[light_type]

        if light_type == Light.Spherical:
            self.setup_spherical_light(name, pos=pos,
                                       autofit_camera=autofit_camera,
                                       color=color, radius=radius,
                                       in_geometry=in_geometry)
        elif light_type == Light.Parallelogram:
            self.setup_parallelogram_light(name, pos=pos,
                                  autofit_camera=autofit_camera,
                                  color=color, u=u, v=v,
                                  in_geometry=in_geometry)

    def update_light(self, name: str,
                     pos: Optional[Any] = None,
                     color: Optional[Any] = None,
                     radius: float = -1,
                     u: Optional[Any] = None,
                     v: Optional[Any] = None) -> None:
        
        if not isinstance(name, str): name = str(name)

        if name not in self.light_handles:
            self._logger.error("Light %s does not exists.")
            return

        pos = self._make_contiguous_vector(pos, 3)
        if pos is not None: pos_ptr = pos.ctypes.data
        else:               pos_ptr = 0

        color = self._make_contiguous_vector(color, 3)
        if color is not None: color_ptr = color.ctypes.data
        else:                 color_ptr = 0

        u = self._make_contiguous_vector(u, 3)
        if u is not None: u_ptr = u.ctypes.data
        else:             u_ptr = 0

        v = self._make_contiguous_vector(v, 3)
        if v is not None: v_ptr = v.ctypes.data
        else:             v_ptr = 0

        if self._optix.update_light(self.light_handles[name],
                                    pos_ptr, color_ptr,
                                    radius, u_ptr, v_ptr):
            self._logger.info("Light %s updated.", name)
        else:
            self._logger.error("Light %s update failed.", name)

    def light_fit(self, light: str,
                  camera: Optional[str] = None,
                  horizontal_rot: Optional[float] = 45,
                  vertical_rot: Optional[float] = 25,
                  dist_scale: Optional[float] = 1.5) -> None:

        if not isinstance(light, str): light = str(light)
        light_handle = self.light_handles[light]

        cam_handle = 0
        if camera is not None:
            if not isinstance(camera, str): camera = str(camera)
            if camera in self.camera_handles:
                cam_handle = self.camera_handles[camera]

        horizontal_rot = math.pi * horizontal_rot / 180.0
        vertical_rot = math.pi * vertical_rot / 180.0

        self._optix.fit_light(light_handle, cam_handle, horizontal_rot, vertical_rot, dist_scale)


    def _make_contiguous_3d(self, a: Optional[Any], n: int = -1, extend_scalars = False) -> Optional[np.ndarray]:
        if a is None: return None

        if not isinstance(a, np.ndarray): a = np.ascontiguousarray(a, dtype=np.float32)

        if a.dtype != np.float32: a = np.ascontiguousarray(a, dtype=np.float32)

        if len(a.shape) == 1:
            if a.shape[0] == 1: a = np.full((1, 3), a[0], dtype=np.float32)
            elif a.shape[0] == n: a = np.reshape(a, (n, 1))
            elif a.shape[0] == 3: a = np.reshape(a, (1, 3))
            else:
                self._logger.error("Input shape not matching single 3D vector nor desired array length.")
                return None

        if len(a.shape) > 2:
            m = functools.reduce(operator.mul, a.shape[:-1], 1)
            if (n >= 0) and (n != m):
                self._logger.error("Input shape not matching desired array length.")
                return None
            a = np.reshape(a, (m, a.shape[-1]))

        if n >= 0:
            if (a.shape[0] == 1) and (n != a.shape[0]):
                _a = np.zeros((n, a.shape[-1]), dtype=np.float32)
                _a[:] = a[0]
                a = _a
            if n != a.shape[0]:
                self._logger.error("Input shape not matching desired array length.")
                return None

        if a.shape[-1] != 3:
            _a = np.zeros((a.shape[0], 3), dtype=np.float32)
            if a.shape[-1] == 1:
                if extend_scalars:
                    _a[:,0] = a[:,0]
                    _a[:,1] = a[:,0]
                    _a[:,2] = a[:,0]
                else:
                    _a[:,0] = a[:,0]
            elif a.shape[-1] == 2: _a[:,[0,1]] = a[:,[0,1]]
            else: _a[:,[0,1,2]] = a[:,[0,1,2]]
            a = _a

        if not a.flags['C_CONTIGUOUS']: a = np.ascontiguousarray(a, dtype=np.float32)

        return a

    def set_data(self, name: str, pos: Any,
                 c: Any = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32),
                 r: Any = np.ascontiguousarray([0.05], dtype=np.float32),
                 u = None, v = None, w = None,
                 geom: Union[Geometry, str] = Geometry.ParticleSet,
                 mat: str = "diffuse") -> None:

        if not isinstance(name, str): name = str(name)
        if isinstance(geom, str): geom = Geometry[geom]

        if name in self.geometry_handles:
            self._logger.error("Geometry %s already exists, use update_data() instead.", name)
            return

        n_primitives = -1

        # Prepare positions data
        pos = self._make_contiguous_3d(pos)
        if pos is None:
            self._logger.error("Positions (pos) are required for the new instances and cannot be left as None.")
            return
        if (len(pos.shape) != 2) or (pos.shape[0] < 1) or (pos.shape[1] != 3):
            self._logger.error("Positions (pos) should be an array of shape (n, 3).")
            return
        n_primitives = pos.shape[0]
        pos_ptr = pos.ctypes.data

        # Prepare colors data
        c = self._make_contiguous_3d(c, n=n_primitives, extend_scalars=True)
        if c is not None: col_ptr = c.ctypes.data
        else: col_ptr = 0

        # Prepare radii data
        if r is not None:
            if not isinstance(r, np.ndarray): r = np.ascontiguousarray(r, dtype=np.float32)
            if r.dtype != np.float32: r = np.ascontiguousarray(r, dtype=np.float32)
            if len(r.shape) > 1: r = r.flatten()
            if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
        if r is not None:
            if r.shape[0] == 1:
                if n_primitives > 0: r = np.full(n_primitives, r[0], dtype=np.float32)
                else:
                    self._logger.error("Cannot resolve proper radii (r) shape from preceding data arguments.")
                    return
                if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
            if (n_primitives > 0) and (n_primitives != r.shape[0]):
                self._logger.error("Radii (r) shape does not match shape of preceding data arguments.")
                return
            n_primitives = r.shape[0]
            radii_ptr = r.ctypes.data
        else: radii_ptr = 0

        # Prepare U vectors
        u = self._make_contiguous_3d(u, n=n_primitives)
        u_ptr = 0
        if u is not None:
            u_ptr = u.ctypes.data

        # Prepare V vectors
        v = self._make_contiguous_3d(v, n=n_primitives)
        v_ptr = 0
        if v is not None:
            v_ptr = v.ctypes.data

        # Prepare W vectors
        w = self._make_contiguous_3d(w, n=n_primitives)
        w_ptr = 0
        if w is not None:
            w_ptr = w.ctypes.data

        if n_primitives == -1:
            self._logger.error("Could not figure out proper data shapes.")
            return

        # Configure according to selected geometry
        is_ok = True
        if geom == Geometry.ParticleSet:
            if c is None:
                self._logger.error("ParticleSet setup failed, colors data is missing.")
                is_ok = False

            if r is None:
                self._logger.error("ParticleSet setup failed, radii data is missing.")
                is_ok = False

        elif geom == Geometry.Parallelepipeds:
            if c is None:
                self._logger.error("Parallelepipeds setup failed, colors data is missing.")
                is_ok = False

            if (u is None) or (v is None) or (w is None):
                if r is None:
                    self._logger.error("Parallelepipeds setup failed, need U, V, W vectors or radii data.")
                    is_ok = False

        elif geom == Geometry.BezierChain:
            if c is None:
                self._logger.error("BezierChain setup failed, colors data is missing.")
                is_ok = False

            if r is None:
                self._logger.error("BezierChain setup failed, radii data is missing.")
                is_ok = False

        else:
            self._logger.error("Unknown geometry")
            is_ok = False

        if is_ok:
            try:
                self._padlock.acquire()

                self._logger.info("Create %s %s, %d primitives...", geom.name, name, n_primitives)
                g_handle = self._optix.setup_geometry(geom.value, name, mat, n_primitives,
                                                      pos_ptr, col_ptr, radii_ptr, u_ptr, v_ptr, w_ptr)

                if g_handle > 0:
                    self._logger.info("...done, handle: %d", g_handle)
                    self.geometry_names[g_handle] = name
                    self.geometry_handles[name] = g_handle
                    self.geometry_sizes[name] = n_primitives
                else:
                    self._logger.error("Geometry setup failed.")
                
            except Exception as e:
                self._logger.error(str(e))
            finally:
                self._padlock.release()


    def update_data(self, name: str,
                    pos: Optional[Any] = None, c: Optional[Any] = None, r: Optional[Any] = None,
                    u: Optional[Any] = None, v: Optional[Any] = None, w: Optional[Any] = None) -> None:

        if not isinstance(name, str): name = str(name)

        if not name in self.geometry_handles:
            self._logger.error("Geometry %s does not exists yet, use setup_data() instead.", name)
            return

        n_primitives = self.geometry_sizes[name]
        size_changed = False

        # Prepare positions data
        pos = self._make_contiguous_3d(pos)
        pos_ptr = 0
        if pos is not None:
            if (len(pos.shape) != 2) or (pos.shape[0] < 1) or (pos.shape[1] != 3):
                self._logger.error("Positions (pos) should be an array of shape (n, 3).")
                return
            n_primitives = pos.shape[0]
            size_changed = (n_primitives != self.geometry_sizes[name])
            pos_ptr = pos.ctypes.data

        # Prepare colors data
        if size_changed and c is None:
            c = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32)
        c = self._make_contiguous_3d(c, n=n_primitives, extend_scalars=True)
        col_ptr = 0
        if c is not None:
            col_ptr = c.ctypes.data

        # Prepare radii data
        if size_changed and r is None:
            r = np.ascontiguousarray([0.05], dtype=np.float32)
        if r is not None:
            if not isinstance(r, np.ndarray): r = np.ascontiguousarray(r, dtype=np.float32)
            if r.dtype != np.float32: r = np.ascontiguousarray(r, dtype=np.float32)
            if len(r.shape) > 1: r = r.flatten()
            if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
        radii_ptr = 0
        if r is not None:
            if r.shape[0] == 1:
                r = np.full(n_primitives, r[0], dtype=np.float32)
                if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
            if n_primitives != r.shape[0]:
                self._logger.error("Radii (r) shape does not match shape of preceding data arguments.")
                return
            radii_ptr = r.ctypes.data

        # Prepare U vectors
        u = self._make_contiguous_3d(u, n=n_primitives)
        u_ptr = 0
        if u is not None: u_ptr = u.ctypes.data

        # Prepare V vectors
        v = self._make_contiguous_3d(v, n=n_primitives)
        v_ptr = 0
        if v is not None: v_ptr = v.ctypes.data

        # Prepare W vectors
        w = self._make_contiguous_3d(w, n=n_primitives)
        w_ptr = 0
        if w is not None: w_ptr = w.ctypes.data

        try:
            self._padlock.acquire()
            self._logger.info("Update %s, %d primitives...", name, n_primitives)
            g_handle = self._optix.update_geometry(name, n_primitives,
                                                   pos_ptr, col_ptr, radii_ptr,
                                                   u_ptr, v_ptr, w_ptr)

            if (g_handle > 0) and (g_handle == self.geometry_handles[name]):
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_sizes[name] = n_primitives
            else:
                self._logger.error("Geometry update failed.")
                
        except Exception as e:
            self._logger.error(str(e))
        finally:
            self._padlock.release()

    def set_coordinates(self, mode: Union[Coordinates, str] = Coordinates.Box, thickness: float = 1.0) -> None:

        if isinstance(mode, str): mode = Coordinates[mode]

        if self._optix.set_coordinates_geom(mode.value, thickness):
            self._logger.info("Coordinate system mode set to: %s.", mode.name)
        else:
            self._logger.error("Coordinate system mode not changed.")
