"""
Tkinter UI for PlotOptiX raytracer.

https://github.com/rnd-team-dev/plotoptix/blob/master/LICENSE.txt

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
"""

import logging
import numpy as np
import tkinter as tk

from PIL import Image, ImageTk
from ctypes import byref, c_float, c_uint
from typing import List, Tuple, Optional, Union

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
    src : string or dict, optional
        Scene description, file name or dictionary. Empty scene is prepared
        if the default ``None`` value is used.
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
        to ``False``, then user should call ``start()`` or ``show()`` method.
        Default is ``False``.
    devices : list, optional
        List of selected devices, with the primary device at index 0. Empty list
        is default, resulting with all compatible devices selected for processing.
    log_level : int or string, optional
        Log output level. Default is ``WARN``.
    """

    def __init__(self,
                 src: Optional[Union[str, dict]] = None,
                 on_initialization = None,
                 on_scene_compute = None,
                 on_rt_completed = None,
                 on_launch_finished = None,
                 on_rt_accum_done = None,
                 width: int = -1,
                 height: int = -1,
                 start_now: bool = False,
                 devices: List = [],
                 log_level: Union[int, str] = logging.WARN) -> None:
        """TkOptiX constructor
        """

        # pass all arguments, except start_now - we'll do that later
        super().__init__(
            src=src,
            on_initialization=on_initialization,
            on_scene_compute=on_scene_compute,
            on_rt_completed=on_rt_completed,
            on_launch_finished=on_launch_finished,
            on_rt_accum_done=on_rt_accum_done,
            width=width, height=height,
            start_now=False, # do not start yet
            devices=devices,
            log_level=log_level)

        # save initial values to set size of Tk window on startup 
        self._ini_width = width
        self._ini_height = height

        self._dummy_rgba = np.ascontiguousarray(np.zeros((8, 8, 4), dtype=np.uint8))

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
        self._any_mouse = False
        self._ctrl_key = False
        self._shift_key = False
        self._any_key = False

        self._selection_handle = -1
        self._selection_index = -1

        self._fixed_size = None
        self._image_scale = 1.0
        self._image_at = (0, 0)

        self._root.title("R&D PlotOptiX")
        self._root.protocol("WM_DELETE_WINDOW", self._gui_quit_callback)

        self._canvas = tk.Canvas(self._root, width=self._width, height=self._height)
        self._canvas.grid(column=0, row=0, columnspan=3, sticky="nsew")
        self._canvas.pack_propagate(0)
        self._canvas.bind("<Configure>", self._gui_configure)
        self._canvas.bind('<Motion>', self._gui_motion)
        self._canvas.bind('<B1-Motion>', self._gui_motion_pressed)
        self._canvas.bind('<B3-Motion>', self._gui_motion_pressed)
        self._canvas.bind("<Button-1>", self._gui_pressed_left)
        self._canvas.bind("<Button-3>", self._gui_pressed_right)
        self._canvas.bind("<ButtonRelease-1>", self._gui_released_left)
        self._canvas.bind("<ButtonRelease-3>", self._gui_released_right)
        self._canvas.bind("<Double-Button-1>", self._gui_doubleclick_left)
        self._canvas.bind("<Double-Button-3>", self._gui_doubleclick_right)
        self._root.bind_all("<KeyPress>", self._gui_key_pressed)
        self._root.bind_all("<KeyRelease>", self._gui_key_released)
        self._canvas.bind("<<LaunchFinished>>", self._gui_update_content)
        self._canvas.bind("<<ApplyUiEdits>>", self._gui_apply_scene_edits)
        self._canvas.bind("<<CloseScene>>", self._gui_quit_callback)

        self._status_main_text = tk.StringVar(value="Selection: camera")
        self._status_main = tk.Label(self._root, textvariable=self._status_main_text, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self._status_main.grid(column=0, row=1, sticky="ew")

        self._status_action_text = tk.StringVar(value="")
        self._status_action = tk.Label(self._root, textvariable=self._status_action_text, width=70, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self._status_action.grid(column=1, row=1, sticky="ew")

        self._status_fps_text = tk.StringVar(value="FPS")
        self._status_fps = tk.Label(self._root, textvariable=self._status_fps_text, width=16, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self._status_fps.grid(column=2, row=1, sticky="ew")

        self._root.rowconfigure(0, weight=1)
        self._root.columnconfigure(0, weight=1)

        self._logger.info("Tkinter widgets ready.")

        self._logger.info("Couple scene to the output window...")
        with self._padlock:
            if self._img_rgba is not None:
                pil_img = Image.fromarray(self._img_rgba, mode="RGBX")
            else:
                pil_img = Image.fromarray(self._dummy_rgba, mode="RGBX")
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
            self._optix.break_launch()
            self._canvas.event_generate("<<CloseScene>>", when="head")
        else:
            self._logger.warn("UI already closed.")

    def _gui_quit_callback(self, *args):
        super().close()
        self._root.quit()

    def _get_image_xy(self, wnd_x, wnd_y):
        if self._fixed_size is None: return wnd_x, wnd_y
        else:
            x = int((wnd_x - self._image_at[0]) / self._image_scale)
            y = int((wnd_y - self._image_at[1]) / self._image_scale)
            return x, y

    def _get_hit_at(self, x, y):
        c_x = c_float()
        c_y = c_float()
        c_z = c_float()
        c_d = c_float()
        if self._optix.get_hit_at(x, y, byref(c_x), byref(c_y), byref(c_z), byref(c_d)):
            return c_x.value, c_y.value, c_z.value, c_d.value
        else: return 0, 0, 0, 0

    def _gui_get_object_at(self, x, y):
        c_handle = c_uint()
        c_index = c_uint()
        c_face = c_uint()
        if self._optix.get_object_at(x, y, byref(c_handle), byref(c_index), byref(c_face)):
            return c_handle.value, c_index.value, c_face.value
        else:
            return None, None

    def _gui_motion(self, event):
        if not (self._any_mouse or self._any_key):
            x, y = self._get_image_xy(event.x, event.y)

            handle, index, face = self._gui_get_object_at(x, y)
            if (handle != 0x3FFFFFFF):
                hx, hy, hz, hd = self._get_hit_at(x, y)
                if handle in self.geometry_names:
                    if (face != 0xFFFFFFFF):
                        self._status_action_text.set("%s[f:%d; vtx:%d]: 2D (%d %d), 3D (%f %f %f), at dist.: %f" % (self.geometry_names[handle], face, index, x, y, hx, hy, hz, hd))
                    else:
                        self._status_action_text.set("%s[%d]: 2D (%d %d), 3D (%f %f %f), at dist.: %f" % (self.geometry_names[handle], index, x, y, hx, hy, hz, hd))
                else:
                    lh = self._optix.get_light_handle(handle, index)
                    if lh in self.light_names:
                        self._status_action_text.set("%s: 2D (%d %d), 3D (%f %f %f), at dist.: %f" % (self.light_names[lh], x, y, hx, hy, hz, hd))
                    else:
                        self._status_action_text.set("unknown: 2D (%d %d), 3D (%f %f %f), at dist.: %f" % (x, y, hx, hy, hz, hd))
            else:
                self._status_action_text.set("empty area")

    def _gui_motion_pressed(self, event):
        self._mouse_to_x, self._mouse_to_y = self._get_image_xy(event.x, event.y)
        self._optix.break_launch()

    def _gui_pressed_left(self, event):
        self._mouse_from_x, self._mouse_from_y = self._get_image_xy(event.x, event.y)
        self._mouse_to_x = self._mouse_from_x
        self._mouse_to_y = self._mouse_from_y
        self._left_mouse = True
        self._any_mouse = True

    def _gui_pressed_right(self, event):
        self._mouse_from_x, self._mouse_from_y = self._get_image_xy(event.x, event.y)
        self._mouse_to_x = self._mouse_from_x
        self._mouse_to_y = self._mouse_from_y
        self._right_mouse = True
        self._any_mouse = True

    def _gui_released_left(self, event):
        self._mouse_to_x, self._mouse_to_y = self._get_image_xy(event.x, event.y)
        self._mouse_from_x = self._mouse_to_x
        self._mouse_from_y = self._mouse_to_y
        self._left_mouse = False
        self._any_mouse = False

    def _gui_released_right(self, event):
        self._mouse_to_x, self._mouse_to_y = self._get_image_xy(event.x, event.y)
        self._mouse_from_x = self._mouse_to_x
        self._mouse_from_y = self._mouse_to_y
        self._right_mouse = False
        self._any_mouse = False

    def _gui_doubleclick_left(self, event):
        assert self._is_started, "Raytracing thread not running."

        x, y = self._get_image_xy(event.x, event.y)
        handle, index, _ = self._gui_get_object_at(x, y)

        if (handle != 0xFFFFFFFF):

            if handle in self.geometry_names:
                # switch selection: primitive / whole geom
                if self._ctrl_key or (self._selection_handle == handle and self._selection_index == -1):
                    self._status_main_text.set("Selection: %s[%d]" % (self.geometry_names[handle], index))
                    self._selection_index = index
                else:
                    self._status_main_text.set("Selection: %s" % self.geometry_names[handle])
                    self._selection_handle = handle
                    self._selection_index = -1
                    
                if self._ctrl_key:
                    hx, hy, hz, hd = self._get_hit_at(x, y)
                    if hd > 0:
                        self._status_action_text.set("Focused at (%f %f %f), distance %f" % (hx, hy, hz, hd))
                        cam_info = self.get_camera()
                        if "fisheye" in cam_info["RayGeneration"]:
                            w = np.array([hx, hy, hz], dtype=np.float32) - np.array(cam_info["Eye"], dtype=np.float32)
                            _ = self._optix.set_camera_focal_length(np.linalg.norm(w))
                        else:
                            _ = self._optix.set_camera_focal_length(hd)

                return

            else:
                lh = self._optix.get_light_handle(handle, index)
                if lh in self.light_names:
                    self._status_main_text.set("Selection: %s" % self.light_names[lh])
                    self._selection_handle = -2
                    self._selection_index = lh

                    return

        self._status_main_text.set("Selection: camera")
        self._selection_handle = -1
        self._selection_index = -1

    def select(self, name: Optional[str] = None, index: int = -1):
        """Select geometry, light or camera.

        Select object for manual manipulations (rotations, shifts, etc). Geometry or light
        is selected by its name. If ``name`` is not provided, then active camera is selected.
        Optional ``index`` allows selection of a primitive within the geometry.

        Parameters
        ----------
        name : string, optional
            Name of the geometry or light to select. If ``None`` then active camera is selected.
        index : int, optional
            Primitive index to select. Entire geometry is selected if ``index`` is out of primitives range. 
        """
        if name is None:
            self._status_main_text.set("Selection: camera")
            self._selection_handle = -1
            self._selection_index = -1
            return

        if name in self.geometry_data:
            self._selection_handle = self.geometry_data[name]._handle
            if index >= 0 and index < self.geometry_data[name]._size:
                self._status_main_text.set("Selection: %s[%d]" % (name, index))
                self._selection_index = index
            else:
                self._status_main_text.set("Selection: %s" % name)
                self._selection_index = -1
            return

        if name in self.light_handles:
            self._status_main_text.set("Selection: %s" % name)
            self._selection_handle = -2
            self._selection_index = self.light_handles[name]
            return


    def _gui_doubleclick_right(self, event):
        self._status_main_text.set("Selection: camera")
        self._selection_handle = -1
        self._selection_index = -1


    def _gui_key_pressed(self, event):
        if event.keysym == "Control_L":
            self._ctrl_key = True
            self._any_key = True
        elif event.keysym == "Shift_L":
            self._shift_key = True
            self._any_key = True
            self._any_key = True
        else:
            self._any_key = False

    def _gui_key_released(self, event):
        if event.keysym == "Control_L":
            self._ctrl_key = False
        elif event.keysym == "Shift_L":
            self._shift_key = False
        self._any_key = False


    def get_rt_size(self) -> Tuple[int, int]:
        """Get size of ray-tracing output image.

        Get fixed dimensions of the output image or ``None`` if the
        image is fit to the GUI window size.

        Returns
        -------
        out : tuple (int, int)
            Output image size or ``None`` if set auto-fit mode.
        """
        return self._fixed_size

    def set_rt_size(self, size: Tuple[int, int]) -> None:
        """Set fixed / free size of ray-tracing output image.

        Set fixed dimensions of the output image or allow automatic fit to the
        GUI window size. Fixed size image updates are slower, but allow ray tracing
        of any size. Default mode is fit to the GUI window size.

        Parameters
        ----------
        size : tuple (int, int)
            Output image size or ``None`` to set auto-fit mode.
        """
        assert self._is_started, "Raytracing thread not running."

        if self._fixed_size == size: return

        self._fixed_size = size
        with self._padlock:
            if self._fixed_size is None:
                w, h = self._canvas.winfo_width(), self._canvas.winfo_height()
            else:
                w, h = self._fixed_size
            self.resize(width=w, height=h)

    def _gui_internal_image_update(self):
        if self._img_rgba is not None:
            pil_img = Image.fromarray(self._img_rgba, mode="RGBX")
        else:
            pil_img = Image.fromarray(self._dummy_rgba, mode="RGBX")

        move_to = (0, 0)
        self._image_scale = 1.0
        if self._fixed_size is not None:
            wc, hc = self._canvas.winfo_width(), self._canvas.winfo_height()
            if self._width / wc > self._height / hc:
                self._image_scale = wc / self._width
                hnew = int(self._height * self._image_scale)
                pil_img = pil_img.resize((wc, hnew), Image.ANTIALIAS)
                move_to = (0, (hc - hnew) // 2)
            else:
                self._image_scale = hc / self._height
                wnew = int(self._width * self._image_scale)
                pil_img = pil_img.resize((wnew, hc), Image.ANTIALIAS)
                move_to = ((wc - wnew) // 2, 0)

        tk_img = ImageTk.PhotoImage(pil_img)
        # update image on canvas
        self._canvas.itemconfig(self._img_id, image=tk_img)
        if self._image_at != move_to:
            self._canvas.move(self._img_id, -self._image_at[0], -self._image_at[1])
            self._canvas.move(self._img_id, move_to[0], move_to[1])
            self._image_at = move_to
        # swap reference stored in the window instance
        self._tk_img = tk_img
        # no redraws until the next launch
        self._update_req = False

    def _gui_configure(self, event):
        assert self._is_started, "Raytracing thread not running."

        if not self._started_event.is_set():
            self._started_event.set()

        with self._padlock:
            if self._fixed_size is None:
                w, h = self._canvas.winfo_width(), self._canvas.winfo_height()
                if (w == self._width) and (h == self._height): return
                self._logger.info("Resize to: %d x %d", w, h)
                self.resize(width=w, height=h)

            self._gui_internal_image_update()

    ###########################################################################
    # update raytraced image in Tk window                              ########
    def _gui_update_content(self, *args):
        assert self._is_started, "Raytracing thread not running."

        if self._update_req:
            self._status_fps_text.set("FPS: %.3f" % self._optix.get_fps())
            with self._padlock:
                self._gui_internal_image_update()

    def _launch_finished_callback(self, rt_result: int):
        super()._launch_finished_callback(rt_result)
        if self._is_started and rt_result < RtResult.NoUpdates.value:
            self._update_req = True
            self._canvas.event_generate("<<LaunchFinished>>", when="now")
    ###########################################################################

    ###########################################################################
    # apply manual scene edits made in ui                              ########
    def _gui_apply_scene_edits(self, *args):
        if (self._mouse_from_x == self._mouse_to_x) and (self._mouse_from_y == self._mouse_to_y): return

        if self._selection_handle == -1:
            # manipulate camera:
            if self._left_mouse:
                if not self._any_key:
                    self._status_action_text.set("rotate camera eye XZ")
                    self._optix.rotate_camera_eye(
                        self._mouse_from_x, self._mouse_from_y,
                        self._mouse_to_x, self._mouse_to_y)
                elif self._ctrl_key:
                    self._status_action_text.set("change camera focus")
                    df = 1 + 0.01 * (self._mouse_from_y - self._mouse_to_y)
                    f = self._optix.get_camera_focal_scale(0) # 0 is current cam
                    self._optix.set_camera_focal_scale(df * f)
                elif self._shift_key:
                    self._status_action_text.set("change camera FoV")
                    df = 1 + 0.005 * (self._mouse_from_y - self._mouse_to_y)
                    f = self._optix.get_camera_fov(0) # 0 is current cam
                    self._optix.set_camera_fov(df * f)

            elif self._right_mouse:
                if not self._any_key:
                    self._status_action_text.set("camera pan/tilt")
                    self._optix.rotate_camera_tgt(
                        self._mouse_from_x, self._mouse_from_y,
                        self._mouse_to_x, self._mouse_to_y)
                elif self._ctrl_key:
                    self._status_action_text.set("change camera aperture")
                    da = 1 + 0.01 * (self._mouse_from_y - self._mouse_to_y)
                    a = self._optix.get_camera_aperture(0) # 0 is current cam
                    self._optix.set_camera_aperture(da * a)
                elif self._shift_key:
                    self._status_action_text.set("camera dolly")
                    target = np.ascontiguousarray([0, 0, 0], dtype=np.float32)
                    self._optix.get_camera_target(0, target.ctypes.data) # 0 is current cam
                    eye = np.ascontiguousarray([0, 0, 0], dtype=np.float32)
                    self._optix.get_camera_eye(0, eye.ctypes.data) # 0 is current cam
                    dl = 0.01 * (self._mouse_from_y - self._mouse_to_y)
                    eye = eye - dl * (target - eye)
                    self._optix.set_camera_eye(eye.ctypes.data)
            
        elif self._selection_handle == -2:
            # manipulate light:
            if self._selection_index in self.light_names:
                name = self.light_names[self._selection_index]
                if self._left_mouse:
                    if not self._any_key:
                        rx = np.pi * (self._mouse_to_y - self._mouse_from_y) / self._height
                        ry = np.pi * (self._mouse_to_x - self._mouse_from_x) / self._width
                        self._status_action_text.set("rotate light in camera XY")
                        self._optix.rotate_light_in_view(name, rx, ry, 0)

                    elif self._ctrl_key and self._shift_key:
                        s = 1 - (self._mouse_to_y - self._mouse_from_y) / self._height
                        self._status_action_text.set("scale light")
                        self._optix.scale_light(name, s)

                    elif self._ctrl_key:
                        rx = np.pi * (self._mouse_to_y - self._mouse_from_y) / self._height
                        rz = np.pi * (self._mouse_from_x - self._mouse_to_x) / self._width
                        self._status_action_text.set("rotate light in camera XZ")
                        self._optix.rotate_light_in_view(name, rx, 0, rz)

                    elif self._shift_key:
                        dx = (self._mouse_to_x - self._mouse_from_x) / self._width
                        dy = (self._mouse_from_y - self._mouse_to_y) / self._height
                        self._status_action_text.set("move light in camera XY")
                        self._optix.move_light_in_view(name, dx, dy, 0)

                elif self._right_mouse:
                    if not self._any_key:
                        dx = (self._mouse_to_x - self._mouse_from_x) / self._width
                        dz = (self._mouse_to_y - self._mouse_from_y) / self._height
                        self._status_action_text.set("move light in camera XZ")
                        self._optix.move_light_in_view(name, dx, 0, dz)

                    elif self._shift_key:
                        dx = (self._mouse_from_y - self._mouse_to_y) / self._height
                        self._status_action_text.set("move light in normal direction")
                        self._optix.dolly_light(name, dx)

        else:
            # manipulate selected ogject
            name = self.geometry_names[self._selection_handle]
            if self._left_mouse:
                if not self._any_key:
                    rx = np.pi * (self._mouse_to_y - self._mouse_from_y) / self._height
                    ry = np.pi * (self._mouse_to_x - self._mouse_from_x) / self._width
                    if self._selection_index == -1:
                        self._status_action_text.set("rotate geometry in camera XY")
                        self._optix.rotate_geometry_in_view(name, rx, ry, 0, True)
                    else:
                        self._status_action_text.set("rotate primitive in camera XY")
                        self._optix.rotate_primitive_in_view(name, self._selection_index, rx, ry, 0, True)

                elif self._ctrl_key and self._shift_key:
                    s = 1 - (self._mouse_to_y - self._mouse_from_y) / self._height
                    if self._selection_index == -1:
                        self._status_action_text.set("scale geometry")
                        self._optix.scale_geometry(name, s, True)
                    else:
                        self._status_action_text.set("scale primitive")
                        self._optix.scale_primitive(name, self._selection_index, s, True)

                elif self._ctrl_key:
                    rx = np.pi * (self._mouse_to_y - self._mouse_from_y) / self._height
                    rz = np.pi * (self._mouse_from_x - self._mouse_to_x) / self._width
                    if self._selection_index == -1:
                        self._status_action_text.set("rotate geometry in camera XZ")
                        self._optix.rotate_geometry_in_view(name, rx, 0, rz, True)
                    else:
                        self._status_action_text.set("rotate primitive in camera XY")
                        self._optix.rotate_primitive_in_view(name, self._selection_index, rx, 0, rz, True)

                elif self._shift_key:
                    dx = (self._mouse_to_x - self._mouse_from_x) / self._width
                    dy = (self._mouse_from_y - self._mouse_to_y) / self._height
                    if self._selection_index == -1:
                        self._status_action_text.set("move geometry in camera XY")
                        self._optix.move_geometry_in_view(name, dx, dy, 0, True)
                    else:
                        self._status_action_text.set("move primitive in camera XY")
                        self._optix.move_primitive_in_view(name, self._selection_index, dx, dy, 0, True)

            elif self._right_mouse:
                if not self._any_key:
                    dx = (self._mouse_to_x - self._mouse_from_x) / self._width
                    dz = (self._mouse_to_y - self._mouse_from_y) / self._height
                    if self._selection_index == -1:
                        self._status_action_text.set("move geometry in camera XZ")
                        self._optix.move_geometry_in_view(name, dx, 0, dz, True)
                    else:
                        self._status_action_text.set("move primitive in camera XZ")
                        self._optix.move_primitive_in_view(name, self._selection_index, dx, 0, dz, True)

        self._mouse_from_x = self._mouse_to_x
        self._mouse_from_y = self._mouse_to_y

    def _scene_rt_starting_callback(self):
        if self._is_started:
            super()._scene_rt_starting_callback()
            self._canvas.event_generate("<<ApplyUiEdits>>", when="now")
    ###########################################################################
