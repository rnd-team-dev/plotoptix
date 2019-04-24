"""
Tkinter UI for PlotOptiX raytracer.

Copyright (C) 2019 R&D Team. All Rights Reserved.

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
"""

import logging
import numpy as np
import tkinter as tk

from PIL import Image, ImageTk
from ctypes import byref, c_float, c_uint
from typing import Union

from plotoptix.enums import *
from plotoptix._load_lib import PLATFORM
from plotoptix.npoptix import NpOptiX

class TkOptiX(NpOptiX):
    """Tkinter based UI for PlotOptiX. Derived from :class:`plotoptix.NpOptiX`.

    Summary of mouse and keys actions:

    - rotate camera eye around the target: hold and drag left mouse button
    - rotate camera target around the eye: hold and drag right mouse button
    - zoom out/in (change camera field of view): hold shift + left mouse button and drag up/down
    - move camera eye backward/forward: hold shift + right mouse button and drag up/down
    - change focus distance in "depth of field" cameras: hold ctrl + left mouse button and drag up/down
    - change aperture radius in "depth of field" cameras: hold ctrl + right mouse button and drag up/down
    - focus at an object: hold ctrl + double-click left mouse button
    - select an object: double-click left mouse button (info on terminal output)

    Note: functions with the names ``_gui_*`` can be used from the
    GUI thread (Tk event loop) only.

    Parameters
    ----------
    on_initialization : callable or list, optional
        Callable or list of callables to execute upon starting the raytracing
        thread. These callbacks are executed on the main thread.
    on_scene_compute : callable or list, optional
        Callable or list of callables to execute upon starting the new frame.
        Callbacks are executed in a thread parallel to the raytracing.
    on_rt_completed : callable or list, optional
        Callable or list of callables to execute when the frame raytracing
        is completed (execution may be paused with pause_compute() method).
        Callbacks are executed in a thread parallel to the raytracing.
    on_launch_finished : callable or list, optional
        Callable or list of callables to execute when the frame raytracing
        is completed. These callbacks are executed on the raytracing thread.
    on_rt_accum_done : callable or list, optional
        Callable or list of callables to execute when the last accumulation
        frame is finished. These callbacks are executed on the raytracing thread.
    width : int, optional
        Pixel width of the raytracing output. Default value is half of the
        screen width.
    height : int, optional
        Pixel height of the raytracing output. Default value is half of the
        screen height.
    start_now : bool, optional
        Open the GUI window and start raytracing thread immediately. If set
        to False, then user should call start() or show() method. Default is
        False.
    log_level : int or string, optional
        Log output level. Default is ``WARN``.
    """

    def __init__(self,
                 on_initialization = None,
                 on_scene_compute = None,
                 on_rt_completed = None,
                 on_launch_finished = None,
                 on_rt_accum_done = None,
                 width: int = -1,
                 height: int = -1,
                 start_now: bool = False,
                 log_level: Union[int, str] = logging.WARN) -> None:
        """TkOptiX constructor
        """

        # pass all arguments, except start_now - we'll do that later
        super().__init__(
            on_initialization=on_initialization,
            on_scene_compute=on_scene_compute,
            on_rt_completed=on_rt_completed,
            on_launch_finished=on_launch_finished,
            on_rt_accum_done=on_rt_accum_done,
            width=width, height=height,
            start_now=False, # do not start yet
            log_level=log_level)

        # save initial values to set size of Tk window on startup 
        self._ini_width = width
        self._ini_height = height

        if PLATFORM == "Windows":
            dpi_scale = self._optix.get_display_scaling()
            self._logger.info("DPI scaling: %d", dpi_scale)
            if dpi_scale != 1:
                self._logger.warn("DPI setting may cause blurred raytracing output, see this answer")
                self._logger.warn("for the solution https://stackoverflow.com/a/52599951/10037996:")
                self._logger.warn("set python.exe and pythonw.exe files Properties -> Compatibility")
                self._logger.warn("-> Change high DPI settings -> check Override high DPI scaling")
                self._logger.warn("behaviour, select Application in the drop-down menu.")

        if self._is_scene_created and start_now:
                self._logger.info("Starting TkOptiX window and raytracing thread.")
                self.start()
        ###############################################################

    # For matplotlib users convenience.
    def show(self) -> None:
        """Start raytracing thread and open the GUI window.
        
        Convenience method to call :meth:`plotoptix.NpOptiX.start`.

        Actions provided with ``on_initialization`` parameter of TkOptiX
        constructor are executed by this method on the main thread, before
        the ratracing thread is started and GUI window open.
        """
        self.start()

    def _run_event_loop(self):
        """Override NpOptiX's method for running the UI event loop.

        Configure the GUI window properties and events, prepare image
        to display raytracing output.
        """
        # setup Tk window #############################################
        self._root = tk.Tk()

        screen_width = self._root.winfo_screenwidth()
        screen_height = self._root.winfo_screenheight()

        if self._ini_width <= 0: self._ini_width = int(screen_width / 2)
        else: self._ini_width = None
        if self._ini_height <= 0: self._ini_height = int(screen_height / 2)
        else: self._ini_height = None
        self.resize(self._ini_width, self._ini_height)

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
        self._canvas.bind("<<ApplyUiEdits>>", self._gui_apply_scene_edits)
        self._canvas.bind("<<CloseScene>>", self._gui_quit_callback)
        self._logger.info("Tkinter widgets ready.")

        self._logger.info("Couple scene to the output window...")
        with self._padlock:
            pil_img = Image.fromarray(self._img_rgba, mode="RGBX")
            self._tk_img = ImageTk.PhotoImage(pil_img)
            self._img_id = self._canvas.create_image(0, 0, image=self._tk_img, anchor=tk.NW)
        ###############################################################

        # start event loop ############################################
        self._logger.info("Start UI event loop...")
        self._is_started = True
        self._update_req = False
        self._root.mainloop()
        ###############################################################

    def close(self) -> None:
        """Stop the raytracing thread, release resources.

        Raytracing cannot be restarted after this method is called.

        See Also
        --------
        :meth:`plotoptix.NpOptiX.close`
        """
        if not self._is_closed:
            self._canvas.event_generate("<<CloseScene>>", when="head")
        else:
            self._logger.warn("UI already closed.")

    def _gui_quit_callback(self, *args):
        super().close()
        self._root.quit()

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
        assert self._is_started, "Raytracing thread not running."

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
                    if self._optix.get_hit_at(x, y, byref(c_x), byref(c_y), byref(c_z), byref(c_d)):
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

    def _gui_internal_image_update(self):
        pil_img = Image.fromarray(self._img_rgba, mode="RGBX")
        tk_img = ImageTk.PhotoImage(pil_img)
        # update image on canvas
        self._canvas.itemconfig(self._img_id, image=tk_img)
        # swap reference stored in the window instance
        self._tk_img = tk_img
        # no redraws until the next launch
        self._update_req = False

    def _gui_configure(self, event):
        assert self._is_started, "Raytracing thread not running."

        if not self._started_event.is_set():
            self._started_event.set()

        w, h = self._canvas.winfo_width(), self._canvas.winfo_height()
        if (w == self._width) and (h == self._height): return

        with self._padlock:
            self._logger.info("Resize to: %d x %d", w, h)
            self.resize(width=w, height=h)
            self._gui_internal_image_update()

    ###########################################################################
    # update raytraced image in Tk window                              ########
    def _gui_update_content(self, *args):
        assert self._is_started, "Raytracing thread not running."

        if self._update_req:
            with self._padlock:
                self._gui_internal_image_update()

    def _launch_finished_callback(self, rt_result: int):
        super()._launch_finished_callback(rt_result)
        if self._is_started and rt_result != RtResult.NoUpdates.value:
            self._update_req = True
            self._canvas.event_generate("<<LaunchFinished>>", when="now")
    ###########################################################################

    ###########################################################################
    # apply manual scene edits made in ui                              ########
    def _gui_apply_scene_edits(self, *args):
        if (self._mouse_from_x != self._mouse_to_x) or (self._mouse_from_y != self._mouse_to_y):

            # manipulate camera:
            if self._left_mouse:
                if not self._any_key:
                    self._optix.rotate_camera_eye(
                        self._mouse_from_x, self._mouse_from_y,
                        self._mouse_to_x, self._mouse_to_y)
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
                    self._optix.rotate_camera_tgt(
                        self._mouse_from_x, self._mouse_from_y,
                        self._mouse_to_x, self._mouse_to_y)
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
        if self._is_started:
            self._canvas.event_generate("<<ApplyUiEdits>>", when="now")
    ###########################################################################
