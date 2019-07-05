"""
No-UI PlotOptiX raytracer (output to numpy array only).

Copyright (C) 2019 R&D Team. All Rights Reserved.

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
"""

import json, math, logging, threading, time
import numpy as np

from ctypes import byref, c_float, c_uint, c_int
from typing import List, Tuple, Callable, Optional, Union, Any

from plotoptix.singleton import Singleton
from plotoptix._load_lib import load_optix, load_denoiser, PARAM_NONE_CALLBACK, PARAM_INT_CALLBACK
from plotoptix.utils import _make_contiguous_vector, _make_contiguous_3d
from plotoptix.enums import *


class NpOptiX(threading.Thread, metaclass=Singleton):
    """No-UI raytracer, output to numpy array only.

    Base, headless interface to the `RnD.SharpOptiX` raytracing engine. Provides
    infrastructure for running the raytracing and compute threads and exposes
    their callbacks to the user. Outputs raytraced image to numpy array.

    In derived UI classes, implement in overriden methods:

    - start and run UI event loop in:  :meth:`plotoptix.NpOptiX._run_event_loop`
    - raise UI close event in:         :meth:`plotoptix.NpOptiX.close`
    - update image in UI in:           :meth:`plotoptix.NpOptiX._launch_finished_callback`
    - optionally apply UI edits in:    :meth:`plotoptix.NpOptiX._scene_rt_starting_callback`

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
        Pixel width of the raytracing output. Default value is 16.
    height : int, optional
        Pixel height of the raytracing output. Default value is 16.
    start_now : bool, optional
        Start raytracing thread immediately. If set to False, then user should
        call start() method. Default is ``False``.
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
        """NpOptiX constructor.
        """

        super().__init__()

        self._raise_on_error = False
        self._logger = logging.getLogger(__name__ + "-NpOptiX")
        self._logger.setLevel(log_level)
        self._started_event = threading.Event()
        self._padlock = threading.RLock()
        self._is_scene_created = False
        self._is_started = False
        self._is_closed = False

        # load SharpOptiX library, configure paths ####################
        self._logger.info("Configure RnD.SharpOptiX library...")
        self._optix = load_optix()
        self._logger.info("...done.")
        ###############################################################

        # setup SharpOptiX interface ##################################
        self._logger.info("Preparing empty scene...")
        
        self._width = 0
        self._height = 0
        if width < 16: width = 16
        if height < 16: height = 16
        self.resize(width, height)

        self.geometry_handles = {} # geometry name to handle dictionary
        self.geometry_names = {}   # geometry handle to name dictionary
        self.geometry_sizes = {}   # geometry name to size dictionary
        self.camera_handles = {}   # camera name to handle dictionary
        self.light_handles = {}    # light name to handle dictionary

        # scene initialization / compute / upload / accumulation done callbacks:
        if on_initialization is not None: self._initialization_cb = self._make_list_of_callable(on_initialization)
        else: self._initialization_cb = [self._default_initialization]

        if on_scene_compute is not None: self._scene_compute_cb = self._make_list_of_callable(on_scene_compute)
        else: self._scene_compute_cb = []

        if on_rt_completed is not None: self._rt_completed_cb = self._make_list_of_callable(on_rt_completed)
        else: self._rt_completed_cb = []

        if on_launch_finished is not None: self._launch_finished_cb = self._make_list_of_callable(on_launch_finished)
        else: self._launch_finished_cb = []

        if on_rt_accum_done is not None: self._rt_accum_done_cb = self._make_list_of_callable(on_rt_accum_done)
        else: self._rt_accum_done_cb = []

        # create empty scene / optionally start raytracing thread:
        self._is_scene_created = self._optix.create_empty_scene(self._width, self._height, self._img_rgba.ctypes.data, self._img_rgba.size)
        if self._is_scene_created:
            self._logger.info("Empty scene ready.")

            if start_now: self.start()
            else: self._logger.info("Use start() to start raytracing.")

        else:
            msg = "Initial setup failed, see errors above."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)
        ###############################################################

    def __del__(self):
        """Release resources if destructed before starting the ray tracing thread.
        """
        if self._is_scene_created and not self._is_closed:
            self._optix.destroy_scene()

    def get_gpu_architecture(self, ordinal: int) -> Optional[GpuArchitecture]:
        """Get SM architecture of selected GPU.

        Returns architecture of selected GPU.

        Parameters
        ----------
        ordinal : int
            CUDA ordinal of the GPU.

        Returns
        -------
        out : GpuArchitecture or None
            SM architecture or ``None`` if not recognized.

        See Also
        --------
        :py:mod:`plotoptix.enums.GpuArchitecture`
        """
        cfg = self._optix.get_n_gpu_architecture(ordinal)
        if cfg >= 0: return GpuArchitecture(cfg)
        else: return None


    def _make_list_of_callable(self, items) -> List[Callable[["NpOptiX"], None]]:
        if callable(items): return [items]
        else:
            for item in items:
                assert callable(item), "Expected callable or list of callable items."
            return items

    def start(self) -> None:
        """Start the raytracing, compute, and UI threads.

        Actions provided with ``on_initialization`` parameter of NpOptiX
        constructor are executed by this method on the main thread,
        before starting the ratracing thread.
        """
        if self._is_closed:
            self._logger.warn("Raytracing output was closed, cannot re-open.")
            return

        if self._is_started:
            self._logger.warn("Raytracing output already running.")
            return

        for c in self._initialization_cb: c(self)
        self._logger.info("Initialization done.")

        self._optix.start_rt()
        self._logger.info("RT loop ready.")

        super().start()
        if self._started_event.wait(10):
            self._logger.info("Raytracing started.")
            self._is_started = True
        else:
            msg = "Raytracing output startup timed out."
            self._logger.error(msg)
            self._is_started = False

            if self._raise_on_error: raise TimeoutError(msg)

    def run(self):
        """Starts UI event loop.

        Derived from `threading.Thread <https://docs.python.org/3/library/threading.html>`__.

        Use :meth:`plotoptix.NpOptiX.start` to perform complete initialization.

        **Do not override**, use :meth:`plotoptix.NpOptiX._run_event_loop` instead.
        """
        assert self._is_scene_created, "Scene is not ready, see initialization messages."

        c1_ptr = self._get_launch_finished_callback()
        r1 = self._optix.register_launch_finished_callback(c1_ptr)
        c2_ptr = self._get_accum_done_callback()
        r2 = self._optix.register_accum_done_callback(c2_ptr)
        c3_ptr = self._get_scene_rt_starting_callback()
        r3 = self._optix.register_scene_rt_starting_callback(c3_ptr)
        c4_ptr = self._get_start_scene_compute_callback()
        r4 = self._optix.register_start_scene_compute_callback(c4_ptr)
        c5_ptr = self._get_scene_rt_completed_callback()
        r5 = self._optix.register_scene_rt_completed_callback(c5_ptr)
        if r1 & r2 & r3 & r4 & r5: self._logger.info("Callbacks registered.")
        else:
            msg = "Callbacks setup failed."
            self._logger.error()
            if self._raise_on_error: raise RuntimeError(msg)

        self._run_event_loop()

    ###########################################################################
    def _run_event_loop(self):
        """Internal method for running the UI event loop.

        This method should be overriden in derived UI class (but **do not call
        this base implementation**).
        
        Remember to set self._started_event after all your UI initialization.
        """
        self._started_event.set()
        while not self._is_closed: time.sleep(0.5)
    ###########################################################################

    ###########################################################################
    def close(self) -> None:
        """Stop the raytracing thread, release resources.

        Raytracing cannot be restarted after this method is called.

        Override in UI class, call this base implementation (or raise a close
        event for your UI and call this base implementation there).
        """
        assert not self._is_closed, "Raytracing output already closed."
        assert self._is_started, "Raytracing output not yet running."

        with self._padlock:
            self._logger.info("Stopping raytracing output.")
            self._is_scene_created = False
            self._is_started = False
            self._optix.stop_rt()
            self._optix.destroy_scene()
            self._is_closed = True
    ###########################################################################

    def is_started(self) -> bool: return self._is_started
    def is_closed(self) -> bool: return self._is_closed

    def get_rt_output(self) -> np.ndarray:
        """Return a copy of the output image.
        
        Safe to call at any time, from any thread.

        Returns
        -------
        out : ndarray
            RGBA array of shape (height, width, 4) and type ``numpy.uint8``.
        """
        assert self._is_started, "Raytracing output not running."
        with self._padlock:
            a = self._img_rgba.copy()
        return a

    def resize(self, width: Optional[int] = None, height: Optional[int] = None) -> None:
        """Change dimensions of the raytracing output.
        
        Both or one of the dimensions may be provided. No effect if width and height
        is same as of the current output.

        Parameters
        ----------
        width : int, optional
            New width of the raytracing output.
        height : int, optional
            New height of the raytracing output.
        """
        if width is None: width = self._width
        if height is None: height = self._height
        if (width == self._width) and (height == self._height): return

        with self._padlock:
            self._width = width
            self._height = height
            # allocate new buffer
            img_buffer = np.ascontiguousarray(np.zeros((self._height, self._width, 4), dtype=np.uint8))
            # update buffer pointer and size in the underlying library
            self._optix.resize_scene(self._width, self._height, img_buffer.ctypes.data, img_buffer.size)
            # swap references stored in the raytracer instance
            self._img_rgba = img_buffer

    @staticmethod
    def _default_initialization(wnd) -> None:
        wnd._logger.info("Default scene initialization.")
        if wnd._optix.get_max_accumulation_frames() < 4:
            wnd._optix.set_max_accumulation_frames(4)
        if wnd._optix.get_current_camera() == 0:
            wnd.setup_camera("default", [0, 0, 10], [0, 0, 0])

    ###########################################################################
    def _launch_finished_callback(self, rt_result: int) -> None:
        """
        Callback executed after each finished frame (``min_accumulation_step``
        accumulation frames are raytraced together). This callback is
        executed in the raytracing thread and should not compute extensively
        (make a copy of the image data and process it in another thread).

        Override this method in the UI class, call this base implementation
        and update image in UI (or raise an event to do so).
        
        Actions provided with ``on_launch_finished`` parameter of NpOptiX
        constructor are executed here.

        Parameters
        ----------
        rt_result : int
            Raytracing result code corresponding to :class:`plotoptix.enums.RtResult`.
        """
        if self._is_started and rt_result != RtResult.NoUpdates.value:
            for c in self._launch_finished_cb: c(self)
    def _get_launch_finished_callback(self):
        def func(rt_result: int): self._launch_finished_callback(rt_result)
        return PARAM_INT_CALLBACK(func)
    ###########################################################################

    ###########################################################################
    def _scene_rt_starting_callback(self) -> None:
        """
        Callback executed before starting frame raytracing. Appropriate to
        override in UI class and apply scene edits (or raise an event to do
        so) like camera rotations, etc. made by a user in UI.
        
        This callback is executed in the raytracing thread and should not
        compute extensively.
        """
        pass
    def _get_scene_rt_starting_callback(self):
        def func(): self._scene_rt_starting_callback()
        return PARAM_NONE_CALLBACK(func)
    ###########################################################################

    ###########################################################################
    def _accum_done_callback(self) -> None:
        """
        Callback executed when all accumulation frames are completed.
        
        **Do not override**, it is intended to launch ``on_rt_accum_done``
        actions provided with NpOptiX constructor parameters.
        
        Executed in the raytracing thread, so do not compute or write files
        (make a copy of the image data and process it in another thread).
        """
        if self._is_started:
            self._logger.info("RT accumulation finished.")
            for c in self._rt_accum_done_cb: c(self)
    def _get_accum_done_callback(self):
        def func(): self._accum_done_callback()
        return PARAM_NONE_CALLBACK(func)
    ###########################################################################

    ###########################################################################
    def _start_scene_compute_callback(self, n_frames : int) -> None:
        """
        Compute callback executed together with the start of each frame raytracing.

        This callback is executed in parallel to the raytracing and is intended
        for CPU intensive computations. Do not set, update data, cameras, lights,
        etc. here, as it will block until the end of raytracing in the parallel
        thread.

        Callback execution can be suspended / resumed with :meth:`plotoptix.NpOptiX.pause_compute` /
        :meth:`plotoptix.NpOptiX.resume_compute` methods.

        **Do not override**, this method is intended to launch ``on_scene_compute``
        actions provided with NpOptiX constructor parameters.

        Parameters
        ----------
        n_frames : int
            Number of the raytraced frames since the last call (excluding paused
            cycles).
        """
        if self._is_started:
            self._logger.info("Compute, delta %d frames.", n_frames)
            for c in self._scene_compute_cb: c(self, n_frames)
    def _get_start_scene_compute_callback(self):
        def func(n_frames : int): self._start_scene_compute_callback(n_frames)
        return PARAM_INT_CALLBACK(func)

    def _scene_rt_completed_callback(self, rt_result : int) -> None:
        """
        Callback executed in the same thread as _start_scene_compute_callback,
        after it finishes computations.
        
        This callback is synchronized also with the raytracing thread and should
        be used for any uploads of the updated scene to GPU: data, cameras, lights
        setup or updates. Image updates in UI are also possible here, but note that
        callback execution can be suspended / resumed with pause_compute() /
        resume_compute() methods.

        **Do not override**, this method is intended to launch on_rt_completed
        actions provided with __init__ method parameters.

        Parameters
        ----------
        rt_result : int
            Raytracing result code corresponding to RtResult enum.
        """
        if self._is_started:
            self._logger.info("RT completed, result %d.", rt_result)
            for c in self._rt_completed_cb: c(self)
    def _get_scene_rt_completed_callback(self):
        def func(rt_result : int): self._scene_rt_completed_callback(rt_result)
        return PARAM_INT_CALLBACK(func)
    ###########################################################################

    def pause_compute(self) -> None:
        """Suspend execution of ``on_scene_compute`` / ``on_rt_completed`` actions.
        """
        if self._optix.set_compute_paused(True):
            self._logger.info("Compute thread paused.")
        else:
            self._logger.warn("Pausing compute thread had no effect.")

    def resume_compute(self) -> None:
        """Resume execution of ``on_scene_compute`` / ``on_rt_completed actions``.
        """
        if self._optix.set_compute_paused(False):
            self._logger.info("Compute thread resumed.")
        else:
            msg = "Resuming compute thread had no effect."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def refresh_scene(self) -> None:
        """Refresh scene
        
        Starts raytracing accumulation from scratch.
        """
        self._optix.refresh_scene()

    def get_float(self, name: str) -> Optional[float]:
        """Get shader ``float`` variable with given ``name``.

        Parameters
        ----------
        name : string
            Varable name.

        Returns
        -------
        out : float
            Value of the variable or ``None`` if variable not found.
        """
        if not isinstance(name, str): name = str(name)

        c_x = c_float()
        if self._optix.get_float(name, byref(c_x)):
            self._logger.info("Variable float %s = %f", name, c_x.value)
            return c_x.value
        else:
            msg = "Variable float %s not found." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None

    def get_float2(self, name: str) -> (Optional[float], Optional[float]):
        """Get shader ``float2`` variable with given ``name``.

        Parameters
        ----------
        name : string
            Varable name.

        Returns
        -------
        out : tuple (float, float)
            Value (x, y) of the variable or ``(None, None)`` if variable not found.
        """
        if not isinstance(name, str): name = str(name)

        c_x = c_float()
        c_y = c_float()
        if self._optix.get_float2(name, byref(c_x), byref(c_y)):
            self._logger.info("Variable float2 %s = (%f, %f)", name, c_x.value, c_y.value)
            return c_x.value, c_y.value
        else:
            msg = "Variable float2 %s not found." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None, None

    def get_float3(self, name: str) -> (Optional[float], Optional[float], Optional[float]):
        """Get shader ``float3`` variable with given ``name``.

        Parameters
        ----------
        name : string
            Varable name.

        Returns
        -------
        out : tuple (float, float, float)
            Value (x, y, z) of the variable or ``(None, None, None)`` if variable not found.
        """
        if not isinstance(name, str): name = str(name)

        c_x = c_float()
        c_y = c_float()
        c_z = c_float()
        if self._optix.get_float3(name, byref(c_x), byref(  c_y), byref(c_z)):
            self._logger.info("Variable float3 %s = (%f, %f, %f)", name, c_x.value, c_y.value, c_z.value)
            return c_x.value, c_y.value, c_z.value
        else:
            msg = "Variable float3 %s not found." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None, None, None

    def set_float(self, name: str, x: float, y: Optional[float] = None, z: Optional[float] = None, refresh: bool = False) -> None:
        """Set shader variable.

        Set shader variable with given ``name`` and of the type ``float``, ``float2``
        (if y provided), or ``float3`` (if y and z provided). Raytrace the whole
        scene if refresh is set to ``True``.

        Parameters
        ----------
        name : string
            Varable name.
        x : float
            Variable value (x component in case of ``float2`` and ``float3``).
        y : float, optional
            Y component value for ``float2`` and ``float3`` variables.
        z : float, optional
            Z component value for ``float3`` variables.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.

        Examples
        --------
        >>> optix = TkOptiX()
        >>> optix.set_float("tonemap_exposure", 0.8)
        >>> optix.set_float("tonemap_igamma", 1/2.2) # set sRGB gamma of 2.2
        """
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

    def get_uint(self, name: str) -> Optional[int]:
        """Get shader ``uint`` variable with given ``name``.

        Parameters
        ----------
        name : string
            Varable name.

        Returns
        -------
        out : int
            Value of the variable or ``None`` if variable not found.
        """
        if not isinstance(name, str): name = str(name)

        c_x = c_uint()
        if self._optix.get_uint(name, byref(c_x)):
            self._logger.info("Variable uint %s = %d", name, c_x.value)
            return c_x.value
        else:
            msg = "Variable uint %s not found." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None

    def get_uint2(self, name: str) -> (Optional[int], Optional[int]):
        """Get shader ``uint2`` variable with given ``name``.

        Parameters
        ----------
        name : string
            Varable name.

        Returns
        -------
        out : tuple (int, int)
            Value (x, y) of the variable or ``(None, None)`` if variable not found.
        """
        if not isinstance(name, str): name = str(name)

        c_x = c_uint()
        c_y = c_uint()
        if self._optix.get_uint2(name, byref(c_x), byref(c_y)):
            self._logger.info("Variable uint2 %s = (%d, %d)", name, c_x.value, c_y.value)
            return c_x.value, c_y.value
        else:
            msg = "Variable uint2 %s not found." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None, None


    def set_uint(self, name: str, x: int, y: Optional[int] = None, refresh: bool = False) -> None:
        """Set shader variable.

        Set shader variable with given ``name`` and of the type ``uint`` or ``uint2``
        (if y provided). Raytrace the whole scene if refresh is set to ``True``.
        Note, shader variables distinguish ``int`` and ``uint``, while the type
        provided by Python methods is ``int`` in both cases.

        Parameters
        ----------
        name : string
            Varable name.
        x : int
            Variable value (x component in case of ``uint2``).
        y : int, optional
            Y component value for ``uint2`` variable.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.

        Examples
        --------
        >>> optix = TkOptiX()
        >>> optix.set_uint("path_seg_range", 4, 16) # set longer range of traced path segments
        """
        if not isinstance(name, str): name = str(name)
        if not isinstance(x, int): x = int(x)

        if y is not None: # expect uint2
            if not isinstance(y, int): y = int(y)

            self._optix.set_uint2(name, x, y, refresh)
            return

        self._optix.set_uint(name, x, refresh)


    def get_int(self, name: str) -> Optional[int]:
        """Get shader ``int`` variable with given ``name``.

        Parameters
        ----------
        name : string
            Varable name.

        Returns
        -------
        out : int
            Value of the variable or ``None`` if variable not found.
        """
        if not isinstance(name, str): name = str(name)

        c_x = c_int()
        if self._optix.get_int(name, byref(c_x)):
            self._logger.info("Variable int %s = %d", name, c_x.value)
            return c_x.value
        else:
            msg = "Variable int %s not found." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None

    def set_int(self, name: str, x: int, refresh: bool = False) -> None:
        """Set shader variable.

        Set shader variable with given ``name`` and of the type ``int``. Raytrace
        the whole scene if refresh is set to ``True``.
        Note, shader variables distinguish ``int`` and ``uint``, while the type
        provided by Python methods is ``int`` in both cases.

        Parameters
        ----------
        name : string
            Varable name.
        x : int
            Variable value.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.
        """
        if not isinstance(name, str): name = str(name)
        if not isinstance(x, int): x = int(x)

        self._optix.set_int(name, x, refresh)


    def set_texture_1d(self, name: str, data: Any, keep_on_host: bool = False, refresh: bool = False) -> None:
        """Set texture data.

        Set data of the shader texture with given ``name``. Texture format
        (float, float2 or float4) and lenght are deduced from the ``data`` array
        shape. Use ``keep_on_host=True`` to make a copy of data in the host memory
        (in addition to GPU memory), this option is required when (small) textures
        are saved to JSON description of the scene.

        Parameters
        ----------
        name : string
            Varable name.
        data : array_like
            Texture data.
        keep_on_host : bool, optional
            Store texture data copy in the host memory.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.
        """
        if not isinstance(name, str): name = str(name)
        if not isinstance(data, np.ndarray): data = np.ascontiguousarray(data, dtype=np.float32)

        if len(data.shape) == 1:     rt_format = RtFormat.Float
        elif len(data.shape) == 2:
            if data.shape[1] == 1:   rt_format = RtFormat.Float
            elif data.shape[1] == 2: rt_format = RtFormat.Float2
            elif data.shape[1] == 4: rt_format = RtFormat.Float4
            else:
                msg = "Texture 1D shape should be (length,n), where n=1,2,4."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                return
        else:
            msg = "Texture 1D shape should be (length,) or (length,n), where n=1,2,4."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return
            
        if data.dtype != np.float32: data = np.ascontiguousarray(data, dtype=np.float32)
        if not data.flags['C_CONTIGUOUS']: data = np.ascontiguousarray(data, dtype=np.float32)

        self._logger.info("Set texture 1D %s: length=%d, format=%s.", name, data.shape[0], rt_format.name)
        if not self._optix.set_texture_1d(name, data.ctypes.data, data.shape[0], rt_format.value, keep_on_host, refresh):
            msg = "Texture 1D %s not uploaded." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def set_texture_2d(self, name: str, data: Any, keep_on_host: bool = False, refresh: bool = False) -> None:
        """Set texture data.

        Set data of the shader texture with given ``name``. Texture format
        (float, float2 or float4) and width/height are deduced from the ``data``
        array shape. Use ``keep_on_host=True`` to make a copy of data in the
        host memory (in addition to GPU memory), this option is required when
        (small) textures are saved to JSON description of the scene.

        Parameters
        ----------
        name : string
            Varable name.
        data : array_like
            Texture data.
        keep_on_host : bool, optional
            Store texture data copy in the host memory.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.
        """
        if not isinstance(name, str): name = str(name)
        if not isinstance(data, np.ndarray): data = np.ascontiguousarray(data, dtype=np.float32)

        if len(data.shape) == 2:     rt_format = RtFormat.Float
        elif len(data.shape) == 3:
            if data.shape[2] == 1:   rt_format = RtFormat.Float
            elif data.shape[2] == 2: rt_format = RtFormat.Float2
            elif data.shape[2] == 4: rt_format = RtFormat.Float4
            else:
                msg = "Texture 2D shape should be (height,width,n), where n=1,2,4."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                return
        else:
            msg = "Texture 2D shape should be (height,width) or (height,width,n), where n=1,2,4."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if data.dtype != np.float32: data = np.ascontiguousarray(data, dtype=np.float32)
        if not data.flags['C_CONTIGUOUS']: data = np.ascontiguousarray(data, dtype=np.float32)

        self._logger.info("Set texture 2D %s: %d x %d, format=%s.", name, data.shape[1], data.shape[0], rt_format.name)
        if not self._optix.set_texture_2d(name, data.ctypes.data, data.shape[1], data.shape[0], rt_format.value, keep_on_host, refresh):
            msg = "Texture 2D %s not uploaded." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def get_background_mode(self) -> Optional[MissProgram]:
        """Get currently configured miss program.

        Returns
        -------
        out : MissProgram or None
            Miss program, see :py:mod:`plotoptix.enums.MissProgram`, or
            `None` if reading the mode failed.

        See Also
        --------
        :py:mod:`plotoptix.enums.MissProgram`
        """
        miss = self._optix.get_miss_program()
        if miss >= 0:
            mode = MissProgram(miss)
            self._logger.info("Current miss program is: %s", mode.name)
            return mode
        else:
            msg = "Failed on reading the miss program."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)
            return None

    def set_background_mode(self, mode: Union[MissProgram, str], refresh: bool = False) -> None:
        """Set miss program.

        Parameters
        ----------
        mode : MissProgram enum or string
            Miss program, see :py:mod:`plotoptix.enums.MissProgram`.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.

        See Also
        --------
        :py:mod:`plotoptix.enums.MissProgram`
        """
        if isinstance(mode, str): mode = MissProgram[mode]

        if self._optix.set_miss_program(mode.value, refresh):
            self._logger.info("Miss program %s is selected.", mode.name)
        else:
            msg = "Miss program setup failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def get_background(self) -> (float, float, float):
        """Get background color.

        **Note**, currently returns constant background color also in texture
        based background modes.

        Returns
        -------
        out : tuple (float, float, float)
            Color values (r, g, b) of the background color.
        """
        return self.get_float3("bg_color")

    def set_background(self, bg: Any,
                       exposure: float = 1.0,
                       gamma: float = 1.0,
                       refresh: bool = False) -> None:
        """Set background color.

        Set background color or texture (shader variable ``bg_color`` or
        texture ``bg_texture``). Raytrace the whole scene if refresh is set
        to ``True``. Texture should be provided as an array of shape ``(height, width, n)``,
        where ``n`` is 3 or 4. 3-component RGB arrays are extended to 4-component
        RGBA mode (alpha channel is reserved for future implementation).
        Function attempts to load texture from file if ``bg`` is a string.

        Color values are corrected to account for the postprocessing tone
        mapping if ``exposure`` and ``gamma`` values are provided.

        Note, color components range is <0; 1>.

        Parameters
        ----------
        bg : Any
            New backgroud color or texture data; single value is a grayscale level,
            RGB color components can be provided as an array-like values, texture
            is provided as an array of shape ``(height, width, n)`` or string
            with the source image file path.
        exposure : float, optional
            Exposure value used in the postprocessing.
        gamma : float, optional
            Gamma value used in the postprocessing.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.

        Examples
        --------
        >>> optix = TkOptiX()
        >>> optix.set_background(0.5) # set gray background
        >>> optix.set_background([0.5, 0.7, 0.9]) # set light bluish background
        """
        if isinstance(bg, str):
            if self._optix.load_texture_2d("bg_texture", bg, exposure, gamma, RtFormat.Float4.value, refresh):
                self._logger.info("Background texture loaded from file.")
            else:
                msg = "Failed on reading background texture."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
            return

        e = 1 / exposure

        if isinstance(bg, float) or isinstance(bg, int):
            x = float(bg); x = e * np.power(x, gamma)
            y = float(bg); y = e * np.power(y, gamma)
            z = float(bg); z = e * np.power(z, gamma)
            if self._optix.set_float3("bg_color", x, y, z, refresh):
                self._logger.info("Background constant gray level updated.")
            else:
                msg = "Failed on updating background color."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
            return

        if not isinstance(bg, np.ndarray):
            bg = np.ascontiguousarray(bg, dtype=np.float32)

        if (len(bg.shape) == 1) and (bg.shape[0] == 3):
            x = e * np.power(bg[0], gamma)
            y = e * np.power(bg[1], gamma)
            z = e * np.power(bg[2], gamma)
            if self._optix.set_float3("bg_color", x, y, z, refresh):
                self._logger.info("Background constant color updated.")
            else:
                msg = "Failed on updating background color."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
            return

        if len(bg.shape) == 3:
            if bg.shape[-1] == 3:
                b = np.zeros((bg.shape[0], bg.shape[1], 4), dtype=np.float32)
                b[:,:,:-1] = bg
                bg = b

            if bg.shape[-1] == 4:
                if gamma != 1: bg = np.power(bg, gamma)
                if e != 1: bg = e * bg
                self.set_texture_2d("bg_texture", bg, keep_on_host=False, refresh=refresh)
                return

        msg = "Background should be a single gray level or [r,g,b] array_like or 2D array_like of [r,g,b]/[r,g,b,a] values."
        self._logger.error(msg)
        if self._raise_on_error: raise ValueError(msg)

    def get_ambient(self) -> (float, float, float):
        """Get ambient color.

        Returns
        -------
        out : tuple (float, float, float)
            Color values (r, g, b) of the ambient light color.
        """
        return self.get_float3("ambient_color")

    def set_ambient(self, color: Any, refresh: bool = False) -> None:
        """Set ambient light color.

        Set ambient light color of the scene (shader variable ``ambient_color``,
        default value is [0.86, 0.89, 0.94]). Raytrace the whole scene if
        refresh is set to ``True``.
        Note, color components range is <0; 1>.

        Parameters
        ----------
        color : Any
            New ambient light color value; single value is a grayscale level,
            RGB color components can be provided as array-like values.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.

        Examples
        --------
        >>> optix = TkOptiX()
        >>> optix.set_ambient(0.5) # set dim gray light
        >>> optix.set_ambient([0.1, 0.2, 0.3]) # set dim bluish light
        """
        if isinstance(color, float) or isinstance(color, int):
            x = float(color)
            y = float(color)
            z = float(color)
        else:
            if not isinstance(color, np.ndarray):
                color = np.asarray(color, dtype=np.float32)
                if (len(color.shape) != 1) or (color.shape[0] != 3):
                    msg = "Color should be a single value or 3-element array/list/tupe."
                    self._logger.error(msg)
                    if self._raise_on_error: raise ValueError(msg)
                    return
            x = color[0]
            y = color[1]
            z = color[2]

        self._optix.set_float3("ambient_color", x, y, z, refresh)
        self._logger.info("Ambient color updated.")


    def get_param(self, name: str) -> None:
        """Get raytracing parameter.

        Parameters
        ----------
        name : string
            Parameter name.

        Returns
        -------
        out : Any, optional
            Value of the parameter or ``None`` if parameter not found.

        Examples
        --------
        >>> optix = TkOptiX()
        >>> print(optix.get_param("max_accumulation_frames"))
        """
        try:
            v = None
            self._padlock.acquire()
            if name == "min_accumulation_step":
                v = self._optix.get_min_accumulation_step()
            elif name == "max_accumulation_frames":
                v = self._optix.get_max_accumulation_frames()
            else:
                msg = "Unknown parameter " + name
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise

        finally:
            self._padlock.release()

        self._logger.info("Value of %s is %s", name, v)
        return v

    def set_param(self, **kwargs) -> None:
        """Set raytracing parameter(s).

        Set raytracing parameters (one or more) and start raytracing of the scene.

        Parameters
        ----------
        kwargs : Any
            Values of parameters corresponding to provided names.

        Examples
        --------
        >>> optix = TkOptiX()
        >>> optix.set_param(min_accumulation_step=4, max_accumulation_frames=200)
        """
        try:
            self._padlock.acquire()
            for key, value in kwargs.items():
                self._logger.info("Set %s to %s", key, value)

                if key == "min_accumulation_step":
                    self._optix.set_min_accumulation_step(int(value))
                elif key == "max_accumulation_frames":
                    self._optix.set_max_accumulation_frames(int(value))
                else:
                    msg = "Unknown parameter " + key
                    self._logger.error(msg)
                    if self._raise_on_error: raise ValueError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise

        finally:
            self._padlock.release()


    def save_image(self, file_name: str) -> None:
        """Save current image to file.

        Save current content of the image buffer to file. Accepted formats,
        recognized by the extension used in the ``file_name``, are bmp, gif,
        png, jpg, and tif. Existing files are overwritten.

        Parameters
        ----------
        file_name : str
            Output file name.
        """
        try:
            self._padlock.acquire()

            if not self._optix.save_image_to_file(file_name):
                msg = "Image not saveed."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise

        finally:
            self._padlock.release()


    def encoder_create(self, fps: int, bitrate: float = 2,
                       idrrate: Optional[int] = None,
                       profile: Union[NvEncProfile, str] = NvEncProfile.Default,
                       preset: Union[NvEncPreset, str] = NvEncPreset.Default) -> None:
        """Create video encoder.

        Create and configure video encoder for this raytracer instance. Only one encoder
        per raytracer instance is supported now. Specifying ``preset`` overrides ``bitrate``
        settings. Beware that some combinations are not supported by all players
        (e.g. lossless encoding is not playable in Windows Media Player).

        Parameters
        ----------
        fps : int
            Frames per second assumed in the output file.
        bitrate : float, optional
            Constant bitrate of the encoded stream, in Mbits to save you typing 0's.
        idrrate : int, optional
            Instantaneous Decode Refresh frame interval. 2 seconds interval is used if
            ``idrrate`` is not provided.
        profile : NvEncProfile enum or string, optional
            H.264 encoding profile.
        preset : NvEncPreset enum or string, optional
            H.264 encoding preset,  overrides ``bitrate`` settings.

        See Also
        --------
        :class:`plotoptix.enums.NvEncProfile`, :class:`plotoptix.enums.NvEncPreset`
        """
        if idrrate is None: idrrate = 2 * fps
        if isinstance(profile, str): profile = NvEncProfile[profile]
        if isinstance(preset, str): preset = NvEncPreset[preset]

        try:
            self._padlock.acquire()

            if not self._optix.encoder_create(fps, int(1000000 * bitrate), idrrate, profile.value, preset.value):
                msg = "Encoder not created."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise

        finally:
            self._padlock.release()

    def encoder_start(self, out_name: str, n_frames: int = 0) -> None:
        """Start video encoding.

        Start encoding to MP4 file with provided name. Total number of frames
        can be optionally limited. Output file is overwritten if it already exists.
        New file is created and encoding is restarted if method is launched
        during previously started encoding.

        Parameters
        ----------
        out_name : str
            Output file name.
        n_frames : int, optional
            Maximum number of frames to encode if ``n_frames`` or unlimited
            encoding when default value is used.
        """
        try:
            self._padlock.acquire()

            if not self._optix.encoder_start(out_name, n_frames):
                msg = "Encoder not started."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise

        finally:
            self._padlock.release()

    def encoder_stop(self) -> None:
        """Stop video encoding.

        Stop encoding and close the output file (can happen before configured
        total number of frames to encode).
        """
        try:
            self._padlock.acquire()

            if not self._optix.encoder_stop():
                msg = "Encoder not stopped."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise

        finally:
            self._padlock.release()

    def encoder_is_open(self) -> bool:
        """Encoder is encoding.

        Returns
        -------
        out : bool
            ``True`` if encoder is encoding.
        """
        return self._optix.encoder_is_open()

    def encoded_frames(self) -> int:
        """Number of encoded video frames.

        Returns
        -------
        out : int
            Number of frames.
        """
        n = self._optix.encoded_frames()
        if n < 0:
            msg = "Number of encoded frames unavailable."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)

        return n

    def encoding_frames(self) -> int:
        """Number of frames to encode.

        Returns
        -------
        out : int
            Number of frames.
        """
        n = self._optix.encoding_frames()
        if n < 0:
            msg = "Number of frames to encode unavailable."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)

        return n


    def get_camera_name_handle(self, name: Optional[str] = None) -> (Optional[str], Optional[int]):
        """Get camera name and handle.

         Mostly for the internal use.

        Parameters
        ----------
        name : string, optional
            Camera name; current camera is used if name not provided.

        Returns
        -------
        out : tuple (name, handle)
            Name and handle of the camera or ``(None, None)`` if camera not found.
        """
        cam_handle = 0
        if name is None: # try current camera
            cam_handle = self._optix.get_current_camera()
            if cam_handle == 0:
                msg = "Current camera is not set."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
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
               msg = "Camera %s does not exists." % name
               self._logger.error(msg)
               if self._raise_on_error: raise ValueError(msg)
               return None, None

        return name, cam_handle

    def get_camera(self, name: Optional[str] = None) -> Optional[dict]:
        """Get camera parameters.

        Parameters
        ----------
        name : string, optional
            Name of the camera, use current camera if name not provided.

        Returns
        -------
        out : dict, optional
            Dictionary of the camera parameters or ``None`` if failed on
            accessing camera data.
        """
        name, cam_handle = self.get_camera_name_handle(name)
        if name is None: return None

        s = self._optix.get_camera(cam_handle)
        if len(s) > 2: return json.loads(s)
        else:
            msg = "Failed on reading camera %s." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None

    def get_camera_eye(self, name: Optional[str] = None) -> Optional[np.ndarray]:
        """Get camera eye coordinates.

        Parameters
        ----------
        name : string, optional
            Name of the camera, use current camera if name not provided.

        Returns
        -------
        out : np.ndarray, optional
            3D coordinates of the camera eye or None if failed on
            accessing camera data.
        """
        if name is not None and not isinstance(name, str): name = str(name)

        name, cam_handle = self.get_camera_name_handle(name)
        if name is None: return None

        eye = np.ascontiguousarray([0, 0, 0], dtype=np.float32)
        self._optix.get_camera_eye(cam_handle, eye.ctypes.data)
        return eye

    def get_camera_target(self, name: Optional[str] = None) -> Optional[np.ndarray]:
        """Get camera target coordinates.

        Parameters
        ----------
        name : string, optional
            Name of the camera, use current camera if name not provided.

        Returns
        -------
        out : np.ndarray, optional
            3D coordinates of the camera target or ``None`` if failed on
            accessing camera data.
        """
        if name is not None and not isinstance(name, str): name = str(name)

        name, cam_handle = self.get_camera_name_handle(name)
        if name is None: return None

        target = np.ascontiguousarray([0, 0, 0], dtype=np.float32)
        self._optix.get_camera_target(cam_handle, target.ctypes.data)
        return target

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
        """Setup new camera with given name.

        Parameters
        ----------
        name : string
            Name of the new camera.
        eye : array_like, optional
            Eye 3D position. Best fit for the current scene is computed if
            argument is not provided.
        target : array_like, optional
            Target 3D position. Center of all geometries if argument not provided.
        up : array_like, optional
            Up (vertical) direction. Y axis if argument not provided.
        cam_type : Camera enum or string, optional
            Type (pinhole, depth of field, ...), see :class:`plotoptix.enums.Camera`.
            Cannot be changed after construction.
        aperture_radius : float, optional
            Aperture radius (increases focus blur for depth of field cameras).
        aperture_fract : float, optional
            Fraction of blind central spot of the aperture (results with ring-like
            bokeh if > 0). Cannot be changed after construction.
        focal_scale : float, optional
            Focus distance / (eye - target).length.
        fov : float, optional
            Field of view in degrees.
        blur : float, optional
            Weight of the new frame in averaging with already accumulated frames.
            Range is (0; 1>, lower values result with a higher motion blur, value
            1.0 turns off the blur (default). Cannot be changed after construction.
        make_current : bool, optional
            Automatically switch to this camera if set to ``True``.
        """
        if name is None: raise ValueError()
        
        if not isinstance(name, str): name = str(name)
        if isinstance(cam_type, str): cam_type = Camera[cam_type]

        if name in self.camera_handles:
            msg = "Camera %s already exists." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        eye_ptr = 0
        eye = _make_contiguous_vector(eye, 3)
        if eye is not None: eye_ptr = eye.ctypes.data

        target_ptr = 0
        target = _make_contiguous_vector(target, 3)
        if target is not None: target_ptr = target.ctypes.data

        up = _make_contiguous_vector(up, 3)
        if up is None:
            msg = "Need 3D camera up vector."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
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
            msg = "Camera setup failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def update_camera(self, name: Optional[str] = None,
                      eye: Optional[Any] = None,
                      target: Optional[Any] = None,
                      up: Optional[Any] = None,
                      aperture_radius: float = -1.0,
                      focal_scale: float = -1.0,
                      fov: float = -1.0) -> None:
        """Update camera parameters.

        Parameters
        ----------
        name : string
            Name of the camera to update.
        eye : array_like, optional
            Eye 3D position.
        target : array_like, optional
            Target 3D position.
        up : array_like, optional
            Up (vertical) direction.
        aperture_radius : float, optional
            Aperture radius (increases focus blur for depth of field cameras).
        focal_scale : float, optional
            Focus distance / (eye - target).length.
        fov : float, optional
            Field of view in degrees.
        """
        name, cam_handle = self.get_camera_name_handle(name)
        if (name is None) or (cam_handle == 0): return

        eye = _make_contiguous_vector(eye, 3)
        if eye is not None: eye_ptr = eye.ctypes.data
        else:               eye_ptr = 0

        target = _make_contiguous_vector(target, 3)
        if target is not None: target_ptr = target.ctypes.data
        else:                  target_ptr = 0

        up = _make_contiguous_vector(up, 3)
        if up is not None: up_ptr = up.ctypes.data
        else:              up_ptr = 0

        if self._optix.update_camera(cam_handle, eye_ptr, target_ptr, up_ptr,
                                     aperture_radius, focal_scale, fov):
            self._logger.info("Camera %s updated.", name)
        else:
            msg = "Camera %s update failed." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def set_current_camera(self, name: str) -> None:
        """Switch to another camera.

        Parameters
        ----------
        name : string
            Name of the new current camera.
        """
        if name is None: raise ValueError()
        
        if not isinstance(name, str): name = str(name)

        if name not in self.camera_handles:
            msg = "Camera %s does not exists." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if self._optix.set_current_camera(self.camera_handles[name]):
            self._logger.info("Current camera: %s", name)
        else:
            msg = "Current camera not changed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def camera_fit(self,
                   camera: Optional[str] = None,
                   geometry: Optional[str] = None,
                   scale: float = 2.5) -> None:
        """Fit the camera eye and target to contain geometry in the field of view.

        Parameters
        ----------
        camera : string, optional
            Name of the new camera to fit; current camera if name not provided.
        geometry : string, optional
            Name of the geometry to fit in view; all geometries if not provided.
        scale : float, optional
            Adjustment of the prefered distance (useful for wide angle cameras).
        """
        camera, cam_handle = self.get_camera_name_handle(camera)
        if camera is None: return

        if geometry is not None:
           if not isinstance(geometry, str): geometry = str(geometry)
        else: geometry = ""

        self._optix.fit_camera(cam_handle, geometry, scale)


    def get_light_shading(self) -> Optional[LightShading]:
        """Get light shading mode.

        Returns
        ----------
        out : LightShading or None
            Light shading mode. ``None`` is returned if function could
            not read the mode from the raytracer.
        """
        shading = self._optix.get_light_shading()
        if shading >= 0:
            mode = LightShading(shading)
            self._logger.info("Current light shading is: %s", mode.name)
            return mode
        else:
            msg = "Failed on reading the light shading mode."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)
            return None

    def set_light_shading(self, mode: Union[LightShading, str]) -> None:
        """Set light shading mode.

        Use :attr:`plotoptix.enums.LightShading.Hard` for best caustics or
        :attr:`plotoptix.enums.LightShading.Soft` for fast convergence.
        
        Set mode before adding lights.

        Parameters
        ----------
        mode : LightShading or string
            Light shading mode.
        """
        if isinstance(mode, str): mode = LightShading[mode]

        if len(self.light_handles) > 0:
            msg = "Light shading has to be selected before adding lights."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)
            return

        if self._optix.set_light_shading(mode.value):
            self._logger.info("Light shading %s is selected.", mode.name)
        else:
            msg = "Light shading setup failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def get_light_pos(self, name: Optional[str] = None) -> Optional[np.ndarray]:
        """Get light 3D position.

        Parameters
        ----------
        name : string, optional
            Name of the light (last added light if ``None``).

        Returns
        -------
        out : np.ndarray, optional
            3D of the light or ``None`` if failed on accessing light data.
        """
        if name is None:
            if len(self.light_handles) > 0: name = list(self.light_handles.keys())[-1]
            else: raise ValueError()

        if not isinstance(name, str): name = str(name)

        if name not in self.light_handles:
            msg = "Light %s does not exists." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None

        pos = np.ascontiguousarray([0, 0, 0], dtype=np.float32)
        self._optix.get_light_pos(self.light_handles[name], pos.ctypes.data)
        return pos

    def get_light_color(self, name: Optional[str] = None) -> Optional[np.ndarray]:
        """Get light color.

        Parameters
        ----------
        name : string, optional
            Name of the light (last added light if ``None``).

        Returns
        -------
        out : np.ndarray, optional
            Light color RGB or ``None`` if failed on accessing light data.
        """
        if name is None:
            if len(self.light_handles) > 0: name = list(self.light_handles.keys())[-1]
            else: raise ValueError()

        if not isinstance(name, str): name = str(name)

        if name not in self.light_handles:
            msg = "Light %s does not exists." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None

        col = np.ascontiguousarray([0, 0, 0], dtype=np.float32)
        self._optix.get_light_color(self.light_handles[name], col.ctypes.data)
        return col

    def get_light_u(self, name: Optional[str] = None) -> Optional[np.ndarray]:
        """Get parallelogram light U vector.

        Parameters
        ----------
        name : string, optional
            Name of the light (last added light if ``None``).

        Returns
        -------
        out : np.ndarray, optional
            Light U vector or ``None`` if failed on accessing light data.
        """
        if name is None:
            if len(self.light_handles) > 0: name = list(self.light_handles.keys())[-1]
            else: raise ValueError()

        if not isinstance(name, str): name = str(name)

        if name not in self.light_handles:
            msg = "Light %s does not exists." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None

        u = np.ascontiguousarray([0, 0, 0], dtype=np.float32)
        self._optix.get_light_u(self.light_handles[name], u.ctypes.data)
        return u

    def get_light_v(self, name: Optional[str] = None) -> Optional[np.ndarray]:
        """Get parallelogram light V vector.

        Parameters
        ----------
        name : string, optional
            Name of the light (last added light if ``None``).

        Returns
        -------
        out : np.ndarray, optional
            Light V vector or ``None`` if failed on accessing light data.
        """
        if name is None:
            if len(self.light_handles) > 0: name = list(self.light_handles.keys())[-1]
            else: raise ValueError()

        if not isinstance(name, str): name = str(name)

        if name not in self.light_handles:
            msg = "Light %s does not exists." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None

        v = np.ascontiguousarray([0, 0, 0], dtype=np.float32)
        self._optix.get_light_v(self.light_handles[name], v.ctypes.data)
        return v

    def get_light_r(self, name: Optional[str] = None) -> Optional[float]:
        """Get spherical light radius.

        Parameters
        ----------
        name : string, optional
            Name of the light (last added light if ``None``).

        Returns
        -------
        out : float, optional
            Light readius or ``None`` if failed on accessing light data.
        """
        if name is None:
            if len(self.light_handles) > 0: name = list(self.light_handles.keys())[-1]
            else: raise ValueError()

        if not isinstance(name, str): name = str(name)

        if name not in self.light_handles:
            msg = "Light %s does not exists." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None

        return self._optix.get_light_r(self.light_handles[name])

    def setup_spherical_light(self, name: str, pos: Optional[Any] = None,
                              autofit_camera: Optional[str] = None,
                              color: Any = 10 * np.ascontiguousarray([1, 1, 1], dtype=np.float32),
                              radius: float = 1.0, in_geometry: bool = True) -> None:
        """Setup new spherical light.

        Parameters
        ----------
        name : string
            Name of the new light.
        pos : array_like, optional
            3D position.
        autofit_camera : string, optional
            Name of the camera used to compute light position automatically.
        color : Any, optional
            RGB color of the light; single value is gray, array_like is RGB
            color components. Color value range is (0; inf) as it means the
            light intensity.
        radius : float, optional
            Sphere radius.
        in_geometry: bool, optional
            Visible in the scene if set to ``True``.
        """
        if name is None: raise ValueError()

        if not isinstance(name, str): name = str(name)

        if name in self.light_handles:
            msg = "Light %s already exists." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        autofit = False
        pos = _make_contiguous_vector(pos, 3)
        if pos is None:
            cam_name, _ = self.get_camera_name_handle(autofit_camera)
            if cam_name is None:
                msg = "Need 3D coordinates for the new light."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                return

            pos = np.ascontiguousarray([0, 0, 0])
            autofit = True

        color = _make_contiguous_vector(color, 3)
        if color is None:
            msg = "Need color (single value or 3-element array/list/tuple)."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        h = self._optix.setup_spherical_light(pos.ctypes.data, color.ctypes.data,
                                              radius, in_geometry)
        if h >= 0:
            self._logger.info("Light %s handle: %d.", name, h)
            self.light_handles[name] = h

            if autofit:
                self.light_fit(name, camera=cam_name)
        else:
            msg = "Light setup failed."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)

    def setup_parallelogram_light(self, name: str, pos: Optional[Any] = None,
                                  autofit_camera: Optional[str] = None,
                                  color: Any = 10 * np.ascontiguousarray([1, 1, 1], dtype=np.float32),
                                  u: Any = np.ascontiguousarray([0, 1, 0], dtype=np.float32),
                                  v: Any = np.ascontiguousarray([-1, 0, 0], dtype=np.float32),
                                  in_geometry: bool = True) -> None:
        """Setup new parallelogram light.

        Note, the light direction is UxV, the back side is black.

        Parameters
        ----------
        name : string
            Name of the new light.
        pos : array_like, optional
            3D position.
        autofit_camera : string, optional
            Name of the camera used to compute light position automatically.
        color : Any, optional
            RGB color of the light; single value is gray, array_like is RGB
            color components. Color value range is (0; inf) as it means the
            light intensity.
        u : array_like, optional
            Parallelogram U vector.
        v : array_like, optional
            Parallelogram V vector.
        in_geometry: bool, optional
            Visible in the scene if set to ``True``.
        """
        if name is None: raise ValueError()

        if not isinstance(name, str): name = str(name)

        if name in self.light_handles:
            msg = "Light %s already exists." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        autofit = False
        pos = _make_contiguous_vector(pos, 3)
        if pos is None:
            cam_name, _ = self.get_camera_name_handle(autofit_camera)
            if cam_name is None:
                msg = "Need 3D coordinates for the new light."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                return

            pos = np.ascontiguousarray([0, 0, 0])
            autofit = True

        color = _make_contiguous_vector(color, 3)
        if color is None:
            msg = "Need color (single value or 3-element array/list/tuple)."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        u = _make_contiguous_vector(u, 3)
        if u is None:
            msg = "Need 3D vector U."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        v = _make_contiguous_vector(v, 3)
        if v is None:
            msg = "Need 3D vector V."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        h = self._optix.setup_parallelogram_light(pos.ctypes.data, color.ctypes.data,
                                                  u.ctypes.data, v.ctypes.data, in_geometry)
        if h >= 0:
            self._logger.info("Light %s handle: %d.", name, h)
            self.light_handles[name] = h

            if autofit:
                self.light_fit(name, camera=cam_name)
        else:
            msg = "Light setup failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def setup_light(self, name: str,
                    light_type: Union[Light, str] = Light.Spherical,
                    pos: Optional[Any] = None,
                    autofit_camera: Optional[str] = None,
                    color: Any = 10 * np.ascontiguousarray([1, 1, 1], dtype=np.float32),
                    u: Any = np.ascontiguousarray([0, 1, 0], dtype=np.float32),
                    v: Any = np.ascontiguousarray([-1, 0, 0], dtype=np.float32),
                    radius: float = 1.0, in_geometry: bool = True) -> None:
        """Setup new light.

        Note, the parallelogram light direction is UxV, the back side is black.

        Parameters
        ----------
        name : string
            Name of the new light.
        light_type : Light enum or string
            Light type (parallelogram, spherical, ...), see :class:`plotoptix.enums.Light` enum.
        pos : array_like, optional
            3D position.
        autofit_camera : string, optional
            Name of the camera used to compute light position automatically.
        color : Any, optional
            RGB color of the light; single value is gray, array_like is RGB
            color components. Color value range is (0; inf) as it means the
            light intensity.
        u : array_like, optional
            Parallelogram U vector.
        v : array_like, optional
            Parallelogram V vector.
        radius : float, optional
            Sphere radius.
        in_geometry: bool, optional
            Visible in the scene if set to ``True``.
        """
        if name is None: raise ValueError()

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
        """Update light parameters.

        Note, the parallelogram light direction is UxV, the back side is black.

        Parameters
        ----------
        name : string, optional
            Name of the light.
        pos : array_like, optional
            3D position.
        color : Any, optional
            RGB color of the light; single value is gray, array_like is RGB
            color components. Color value range is (0; inf) as it means the
            light intensity.
        radius : float, optional
            Sphere radius.
        u : array_like, optional
            Parallelogram U vector.
        v : array_like, optional
            Parallelogram V vector.
        """
        if name is None: raise ValueError()

        if not isinstance(name, str): name = str(name)

        if name not in self.light_handles:
            msg = "Light %s does not exists." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        pos = _make_contiguous_vector(pos, 3)
        if pos is not None: pos_ptr = pos.ctypes.data
        else:               pos_ptr = 0

        color = _make_contiguous_vector(color, 3)
        if color is not None: color_ptr = color.ctypes.data
        else:                 color_ptr = 0

        u = _make_contiguous_vector(u, 3)
        if u is not None: u_ptr = u.ctypes.data
        else:             u_ptr = 0

        v = _make_contiguous_vector(v, 3)
        if v is not None: v_ptr = v.ctypes.data
        else:             v_ptr = 0

        if self._optix.update_light(self.light_handles[name],
                                    pos_ptr, color_ptr,
                                    radius, u_ptr, v_ptr):
            self._logger.info("Light %s updated.", name)
        else:
            msg = "Light %s update failed." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def light_fit(self, light: str,
                  camera: Optional[str] = None,
                  horizontal_rot: float = 45,
                  vertical_rot: float = 25,
                  dist_scale: float = 1.5) -> None:
        """Fit light position and direction to the camera.

        Parameters
        ----------
        name : string
            Name of the light.
        camera : string, optional
            Name of the camera; current camera is used if not provided.
        horizontal_rot : float, optional
            Angle: eye - target - light in the camera horizontal plane.
        vertical_rot : float, optional
            Angle: eye - target - light in the camera vertical plane.
        dist_scale : float, optional
            Light to target distance with reespect to the eye to target distance.
        """
        if light is None: raise ValueError()

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


    def get_material(self, name: str) -> Optional[dict]:
        """Get material parameters.

        Parameters
        ----------
        name : string
            Name of the material.

        Returns
        -------
        out : dict, optional
            Dictionary of the material parameters or ``None`` if failed on
            accessing material data.
        """
        if name is None: raise ValueError()

        if not isinstance(name, str): name = str(name)

        s = self._optix.get_material(name)
        if len(s) > 2: return json.loads(s)
        else:
            msg = "Failed on reading material %s." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)
            return None

    def setup_material(self, name: str, data: dict) -> None:
        """Setup new material.

        Note: in order to keep maximum performance, setup only those materials
        you need in the plot.

        Parameters
        ----------
        name : string
            Name of the material.
        data : dict
            Parameters of the material.

        See Also
        --------
        :py:mod:`plotoptix.materials`
        """
        if name is None or data is None: raise ValueError()

        if self._optix.setup_material(name, json.dumps(data)):
            self._logger.info("Configured material %s.", name)
        else:
            msg = "Material %s not configured." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def update_material(self, name: str, data: dict, refresh: bool = False) -> None:
        """Update material properties.

        Update material properties and optionally refresh the scene.

        Parameters
        ----------
        name : string
            Name of the material.
        data : dict
            Parameters of the material.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.

        See Also
        --------
        :py:mod:`plotoptix.materials`
        """
        self.setup_material(name, data)
        if refresh: self._optix.refresh_scene()

    def set_correction_curve(self, ctrl_points: Any,
                             channel: Union[Channel, str] = Channel.Gray,
                             n_points: int = 256,
                             range: float = 255,
                             refresh: bool = False) -> None:
        """Set correction curve.

        Calculate and setup a color correction curve using control points provided with
        ``ctrl_points``. Curve is applied in 2D postprocessing stage to the selected
        ``channel``. Control points should be an array_like set of input-output values
        (array shape is ``(m,2)``). Control point input and output maximum value can be
        provided with the ``range`` parameter. Control points are scaled to the range
        <0;1>, extreme values (0,0) and (1,1) are added if not present in ``ctrl_points``
        (use :meth:`plotoptix.NpOptiX.set_texture_1d` if custom correction curve should
        e.g. start above 0 or saturate at a level lower than 1).
        Smooth bezier curve is calculated from the control points and stored in 1D texture
        with ``n_points`` length.

        Parameters
        ----------
        ctrl_points : array_like
            Control points to construct curve.
        channel : Channel or string, optional
            Destination color for the correction curve.
        n_points : int, optional
            Number of curve points to be stored in texture.
        range : float, optional
            Maximum input / output value corresponding to provided ``ctrl_points``.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.

        See Also
        --------
        :py:mod:`plotoptix.enums.Postprocessing`
        :py:mod:`plotoptix.enums.Channel`
        """
        if isinstance(channel, str): channel = Channel[channel]

        if not isinstance(ctrl_points, np.ndarray): ctrl_points = np.ascontiguousarray(ctrl_points, dtype=np.float32)

        if len(ctrl_points.shape) != 2 or ctrl_points.shape[1] != 2:
            msg = "Control points shape should be (n,2)."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if ctrl_points.dtype != np.float32: ctrl_points = np.ascontiguousarray(ctrl_points, dtype=np.float32)
        if not ctrl_points.flags['C_CONTIGUOUS']: ctrl_points = np.ascontiguousarray(ctrl_points, dtype=np.float32)

        self._logger.info("Set correction curve in %s channel.", channel.name)
        if not self._optix.set_correction_curve(ctrl_points.ctypes.data, ctrl_points.shape[0], n_points, channel.value, range, refresh):
            msg = "Correction curve setup failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def add_postproc(self, stage: Union[Postprocessing, str], refresh: bool = False) -> None:
        """Add 2D postprocessing stage.

        Stages are applied to image in the order they are added with this
        method. Each stage algorithm has its own variables that should be
        configured before adding the postprocessing stage. Configuration
        can be updated at any time, but stages cannot be disabled after
        adding. See :py:mod:`plotoptix.enums.Postprocessing` for algorithms
        configuration examples.

        Parameters
        ----------
        stage : Postprocessing or string
            Postprocessing algorithm to add.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.

        See Also
        --------
        :py:mod:`plotoptix.enums.Postprocessing`
        """
        if isinstance(stage, str): stage = Postprocessing[stage]
        self._logger.info("Add postprocessing stage: %s.", stage.name)
        if not self._optix.add_postproc(stage.value, refresh):
            msg = "Configuration of postprocessing stage %s failed." % stage.name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def setup_denoiser(self, blend: float = 0.5,
                       exposure: Optional[float] = None,
                       gamma: Optional[float] = None,
                       refresh: bool = False) -> None:
        """Add AI denoiser to 2D postprocessing stages.

        Configure AI denoiser as the first stage in 2D postprocessing.
        Denoiser is applied after min. 4 accumulation frames and uses
        a partially converged image, albedo, and normals of objects in
        the scene to predict the final ray tracing result. Output of the
        denoiser can be mixed with the raw image to improve quality in
        the early accumulation stages using ``blend`` parameter.

        Denoiser is trained on images prepared with gamma correction.
        In order to match the training data characteristics, tone mapping
        is applied before denoising, using ``exposure`` and ``gamma`` values
        configured as for the :py:attr:`plotoptix.enums.Postprocessing.Gamma`
        algorithm. For convenience, these values can be also provided as
        parameters of this method.
        
        *Note:* additional binaries are required to dowload, please run
        ``install_denoser.py`` script to enable denoiser features.

        Parameters
        ----------
        blend : float, optional
            Blend with the raw input, ``0`` means only denoiser output is
            passing, ``1`` means only raw input is passing. Default value
            of ``0.5`` is averaging raw image and denoiser output with
            equal weights.
        exposure : float or ``None``, optional
            Set ``tonemap_exposure`` value if not ``None``.
        gamma : float or ``None``, optional
            Set ``tonemap_igamma`` value to ``1/gamma`` if not ``None``.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.

        See Also
        --------
        :py:mod:`plotoptix.enums.Postprocessing`
        """
        if exposure is not None:
            self._logger.info("Set tonemap_exposure to %f.", exposure)
            self._optix.set_float("tonemap_exposure", exposure)

        if gamma is not None:
            self._logger.info("Set tonemap_igamma to 1/%f.", gamma)
            self._optix.set_float("tonemap_igamma", 1/gamma)

        self._logger.info("Configure AI denoiser.")

        if not load_denoiser() or not self._optix.setup_denoiser(blend, refresh):
            msg = "AI denoiser setup failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)


    def set_data(self, name: str, pos: Any,
                 c: Any = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32),
                 r: Any = np.ascontiguousarray([0.05], dtype=np.float32),
                 u: Optional[Any] = None, v: Optional[Any] = None, w: Optional[Any] = None,
                 geom: Union[Geometry, str] = Geometry.ParticleSet,
                 mat: str = "diffuse",
                 rnd: bool = True) -> None:
        """Create new geometry for the dataset.

        Data is provided as an array of 3D positions of data points, with the shape ``(n, 3)``.
        Additional features can be visualized as a color and size/thickness of the primitives.

        Parameters
        ----------
        name : string
            Name of the geometry.
        pos : array_like
            Positions of data points.
        c : Any, optional
            Colors of the primitives. Single value means a constant gray level.
            3-component array means constant RGB color. Array with the shape[0]
            equal to the number of primitives will set individual gray/color for
            each primitive.
        r : Any, optional
            Radii of particles / bezier primitives or U / V / W lengths of
            parallelograms / parallelepipeds / tetrahedrons (if u / v / w not provided).
            Single value sets const. size for all primitives.
        u : array_like, optional
            U vector(s) of parallelograms / parallelepipeds / tetrahedrons. Single
            vector sets const. value for all primitives.
        v : array_like, optional
            V vector(s) of parallelograms / parallelepipeds / tetrahedrons. Single
            vector sets const. value for all primitives.
        w : array_like, optional
            W vector(s) of parallelepipeds / tetrahedrons. Single vector sets const.
            value for all primitives.
        geom : Geometry enum or string, optional
            Geometry of primitives (ParticleSet, Tetrahedrons, ...). See Geometry
            enum.
        mat : string, optional
            Material name.
        rnd : bool, optional
            Randomize not provided U / V / W vectors so regular but randomly rotated
            primitives are generated using available vectors (default). If set to
            ``False`` all primitives are aligned in the same direction.

        See Also
        --------
        :class:`plotoptix.enums.Geometry`
        """
        if name is None: raise ValueError()

        if not isinstance(name, str): name = str(name)
        if isinstance(geom, str): geom = Geometry[geom]

        if name in self.geometry_handles:
            msg = "Geometry %s already exists, use update_data() instead." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        n_primitives = -1

        # Prepare positions data
        pos = _make_contiguous_3d(pos)
        if pos is None:
            msg = "Positions (pos) are required for the new instances and cannot be left as None."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return
        if (len(pos.shape) != 2) or (pos.shape[0] < 1) or (pos.shape[1] != 3):
            msg = "Positions (pos) should be an array of shape (n, 3)."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return
        n_primitives = pos.shape[0]
        pos_ptr = pos.ctypes.data

        # Prepare colors data
        c = _make_contiguous_3d(c, n=n_primitives, extend_scalars=True)
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
                    msg = "Cannot resolve proper radii (r) shape from preceding data arguments."
                    self._logger.error(msg)
                    if self._raise_on_error: raise ValueError(msg)
                    return
                if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
            if (n_primitives > 0) and (n_primitives != r.shape[0]):
                msg = "Radii (r) shape does not match shape of preceding data arguments."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                return
            n_primitives = r.shape[0]
            radii_ptr = r.ctypes.data
        else: radii_ptr = 0

        # Prepare U vectors
        u = _make_contiguous_3d(u, n=n_primitives)
        u_ptr = 0
        if u is not None:
            u_ptr = u.ctypes.data

        # Prepare V vectors
        v = _make_contiguous_3d(v, n=n_primitives)
        v_ptr = 0
        if v is not None:
            v_ptr = v.ctypes.data

        # Prepare W vectors
        w = _make_contiguous_3d(w, n=n_primitives)
        w_ptr = 0
        if w is not None:
            w_ptr = w.ctypes.data

        if n_primitives == -1:
            msg = "Could not figure out proper data shapes."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        # Configure according to selected geometry
        is_ok = True
        if geom == Geometry.ParticleSet:
            if c is None:
                msg = "ParticleSet setup failed, colors data is missing."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

            if r is None:
                msg = "ParticleSet setup failed, radii data is missing."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

        elif geom == Geometry.Parallelograms:
            if c is None:
                msg = "Parallelograms setup failed, colors data is missing."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

            if (u is None) or (v is None):
                if r is None:
                    msg = "Parallelograms setup failed, need U / V vectors or radii data."
                    self._logger.error(msg)
                    if self._raise_on_error: raise ValueError(msg)
                    is_ok = False

        elif (geom == Geometry.Parallelepipeds) or (geom == Geometry.Tetrahedrons):
            if c is None:
                msg = "Plot setup failed, colors data is missing."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

            if (u is None) or (v is None) or (w is None):
                if r is None:
                    msg = "Plot setup failed, need U, V, W vectors or radii data."
                    self._logger.error(msg)
                    if self._raise_on_error: raise ValueError(msg)
                    is_ok = False

        elif geom == Geometry.BezierChain:
            if c is None:
                msg = "BezierChain setup failed, colors data is missing."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

            if r is None:
                msg = "BezierChain setup failed, radii data is missing."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

        else:
            msg = "Unknown geometry"
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            is_ok = False

        if is_ok:
            try:
                self._padlock.acquire()

                self._logger.info("Create %s %s, %d primitives...", geom.name, name, n_primitives)
                g_handle = self._optix.setup_geometry(geom.value, name, mat, rnd, n_primitives,
                                                      pos_ptr, col_ptr, radii_ptr, u_ptr, v_ptr, w_ptr)

                if g_handle > 0:
                    self._logger.info("...done, handle: %d", g_handle)
                    self.geometry_names[g_handle] = name
                    self.geometry_handles[name] = g_handle
                    self.geometry_sizes[name] = n_primitives
                else:
                    msg = "Geometry setup failed."
                    self._logger.error(msg)
                    if self._raise_on_error: raise RuntimeError(msg)
                
            except Exception as e:
                self._logger.error(str(e))
                if self._raise_on_error: raise
            finally:
                self._padlock.release()


    def update_data(self, name: str,
                    pos: Optional[Any] = None, c: Optional[Any] = None, r: Optional[Any] = None,
                    u: Optional[Any] = None, v: Optional[Any] = None, w: Optional[Any] = None) -> None:
        """Update data of an existing geometry.

        Note that on data size changes (``pos`` array size different than provided on :meth:`plotoptix.NpOptiX.set_data`)
        also other properties must be provided matching the new size, otherwise default values are used.

        Parameters
        ----------
        name : string
            Name of the geometry.
        pos : array_like
            Positions of data points.
        c : Any, optional
            Colors of the primitives. Single value means a constant gray level.
            3-component array means constant RGB color. Array with the shape[0]
            equal to the number of primitives will set individual grey/color for
            each primitive.
        r : Any, optional
            Radii of particles / bezier primitives. Single value sets constant
            radius for all primitives.
        u : array_like, optional
            U vector(s) of parallelograms / parallelepipeds / tetrahedrons. Single
            vector sets const. value for all primitives.
        v : array_like, optional
            V vector(s) of parallelograms / parallelepipeds / tetrahedrons. Single
            vector sets const. value for all primitives.
        w : array_like, optional
            W vector(s) of parallelepipeds / tetrahedrons. Single vector sets const.
            value for all primitives.
        """
        if name is None: raise ValueError()

        if not isinstance(name, str): name = str(name)

        if not name in self.geometry_handles:
            msg = "Geometry %s does not exists yet, use set_data() instead." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        n_primitives = self.geometry_sizes[name]
        size_changed = False

        # Prepare positions data
        pos = _make_contiguous_3d(pos)
        pos_ptr = 0
        if pos is not None:
            if (len(pos.shape) != 2) or (pos.shape[0] < 1) or (pos.shape[1] != 3):
                msg = "Positions (pos) should be an array of shape (n, 3)."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                return
            n_primitives = pos.shape[0]
            size_changed = (n_primitives != self.geometry_sizes[name])
            pos_ptr = pos.ctypes.data

        # Prepare colors data
        if size_changed and c is None:
            c = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32)
        c = _make_contiguous_3d(c, n=n_primitives, extend_scalars=True)
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
                msg = "Radii (r) shape does not match shape of preceding data arguments."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                return
            radii_ptr = r.ctypes.data

        # Prepare U vectors
        u = _make_contiguous_3d(u, n=n_primitives)
        u_ptr = 0
        if u is not None: u_ptr = u.ctypes.data

        # Prepare V vectors
        v = _make_contiguous_3d(v, n=n_primitives)
        v_ptr = 0
        if v is not None: v_ptr = v.ctypes.data

        # Prepare W vectors
        w = _make_contiguous_3d(w, n=n_primitives)
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
                msg = "Geometry update failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)
                
        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise
        finally:
            self._padlock.release()


    def set_data_2d(self, name: str, pos: Any,
                    c: Any = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32),
                    normals: Optional[Any] = None,
                    range_x: Optional[Tuple[float, float]] = None,
                    range_z: Optional[Tuple[float, float]] = None,
                    floor_y: Optional[float] = None,
                    floor_c: Optional[Any] = None,
                    mat: str = "diffuse",
                    make_normals: bool = False) -> None:
        """Create new surface geometry for the 2D dataset.

        Data is provided as 2D array of :math:`z = f(x, y)` values, with the shape ``(n, m)``,
        where ``n`` and ``m`` are at least 2. Additional data features can be
        visualized with color (array of RGB values, shape ``(n, m, 3)``).
        
        Currently, convention of vertical Y and horizontal XZ plane is adopted.

        Parameters
        ----------
        name : string
            Name of the new surface geometry.
        pos : array_like
            Z values of data points.
        c : Any, optional
            Colors of data points. Single value means a constant gray level.
            3-component array means a constant RGB color. Array of the shape
            ``(n, m, 3)`` will set individual color for each data point,
            interpolated between points; ``n`` and ``m`` have to be the same
            as in data points shape.
        normals : array_like, optional
            Surface normal vectors at data points. Array shape has to be ``(n, m, 3)``,
            with ``n`` and ``m`` the same as in data points shape.
        range_x : tuple (float, float), optional
            Data range along X axis. Data array indexes are used if range is
            not provided.
        range_z : tuple (float, float), optional
            Data range along Z axis. Data array indexes are used if range is
            not provided.
        floor_y : float, optional
            Y level of XZ plane forming the base of the new geometry. Surface
            only is created if ``floor_y`` is not provided.
        floor_c: Any, optional
            Color of the base volume. Single value or array_like RGB color values.
        mat : string, optional
            Material name.
        make_normals : bool, optional
            Calculate normals for data points, if not provided with ``normals``
            argument. Normals of all triangles attached to the point are averaged.
        """
        if name is None: raise ValueError()
        if not isinstance(name, str): name = str(name)

        if name in self.geometry_handles:
            msg = "Geometry %s already exists, use update_data_2d() instead." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if not isinstance(pos, np.ndarray): pos = np.ascontiguousarray(pos, dtype=np.float32)
        assert len(pos.shape) == 2 and pos.shape[0] > 1 and pos.shape[1] > 1, "Required vertex data shape is (z,x), where z >= 2 and x >= 2."
        if pos.dtype != np.float32: pos = np.ascontiguousarray(pos, dtype=np.float32)
        if not pos.flags['C_CONTIGUOUS']: pos = np.ascontiguousarray(pos, dtype=np.float32)
        pos_ptr = pos.ctypes.data

        n_ptr = 0
        if normals is not None:
            if not isinstance(normals, np.ndarray): normals = np.ascontiguousarray(normals, dtype=np.float32)
            assert len(normals.shape) == 3 and normals.shape == pos.shape + (3,), "Normals shape must be (z,x,3), where (z,x) id the vertex data shape."
            if normals.dtype != np.float32: normals = np.ascontiguousarray(normals, dtype=np.float32)
            if not normals.flags['C_CONTIGUOUS']: normals = np.ascontiguousarray(normals, dtype=np.float32)
            n_ptr = normals.ctypes.data
            make_normals = False

        c_ptr = 0
        c_const = None
        if c is not None:
            if isinstance(c, float) or isinstance(c, int): c = np.full(3, c, dtype=np.float32)
            if not isinstance(c, np.ndarray): c = np.ascontiguousarray(c, dtype=np.float32)
            if len(c.shape) == 1 and c.shape[0] == 3:
                c_const = c
                cm = np.zeros(pos.shape + (3,), dtype=np.float32)
                cm[:,:] = c
                c = cm
            assert len(c.shape) == 3 and c.shape == pos.shape + (3,), "Colors shape must be (m,n,3), where (m,n) id the vertex data shape."
            if c.dtype != np.float32: c = np.ascontiguousarray(c, dtype=np.float32)
            if not c.flags['C_CONTIGUOUS']: c = np.ascontiguousarray(c, dtype=np.float32)
            c_ptr = c.ctypes.data

        make_floor = floor_y is not None
        if not make_floor: floor_y = np.float32(np.nan)

        cl_ptr = 0
        if make_floor:
            if floor_c is not None:
                if isinstance(floor_c, float) or isinstance(floor_c, int): floor_c = np.full(3, floor_c, dtype=np.float32)
                if not isinstance(floor_c, np.ndarray): floor_c = np.ascontiguousarray(floor_c, dtype=np.float32)
                if len(floor_c.shape) == 1 and floor_c.shape[0] == 3:
                    if floor_c.dtype != np.float32: floor_c = np.ascontiguousarray(floor_c, dtype=np.float32)
                    if not floor_c.flags['C_CONTIGUOUS']: floor_c = np.ascontiguousarray(floor_c, dtype=np.float32)
                    cl_ptr = floor_c.ctypes.data
                else:
                    self._logger.warn("Floor color should be a single value or RGB array.")
            elif c_const is not None:
                floor_c = np.ascontiguousarray(c_const, dtype=np.float32)
                cl_ptr = floor_c.ctypes.data

        if range_x is None: range_x = (np.float32(np.nan), np.float32(np.nan))
        if range_z is None: range_z = (np.float32(np.nan), np.float32(np.nan))

        try:
            self._padlock.acquire()
            self._logger.info("Setup surface %s...", name)
            g_handle = self._optix.setup_surface(name, mat, pos.shape[1], pos.shape[0], pos_ptr, n_ptr, c_ptr, cl_ptr,
                                                 range_x[0], range_x[1], range_z[0], range_z[1], floor_y, make_normals)

            if g_handle > 0:
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_names[g_handle] = name
                self.geometry_handles[name] = g_handle
                self.geometry_sizes[name] = pos.shape[0] * pos.shape[1]
            else:
                msg = "Surface setup failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise
        finally:
            self._padlock.release()

    def update_data_2d(self, name: str,
                       pos: Optional[Any] = None,
                       c: Optional[Any] = None,
                       normals: Optional[Any] = None,
                       range_x: Optional[Tuple[float, float]] = None,
                       range_z: Optional[Tuple[float, float]] = None,
                       floor_y: Optional[float] = None,
                       floor_c: Optional[Any] = None) -> None:
        """Update surface geometry data or properties.

        Parameters
        ----------
        name : string
            Name of the surface geometry.
        pos : array_like, optional
            Z values of data points.
        c : Any, optional
            Colors of data points. Single value means a constant gray level.
            3-component array means a constant RGB color. Array of the shape
            ``(n,m,3)`` will set individual color for each data point,
            interpolated between points; ``n`` and ``m`` have to be the same
            as in data points shape.
        normals : array_like, optional
            Surface normal vectors at data points. Array shape has to be
            ``(n,m,3)``, with ``n`` and``m`` the same as in data points shape.
        range_x : tuple (float, float), optional
            Data range along X axis.
        range_z : tuple (float, float), optional
            Data range along Z axis.
        floor_y : float, optional
            Y level of XZ plane forming the base of the geometry.
        floor_c: Any, optional
            Color of the base volume. Single value or array_like RGB color values.
        """
        if name is None: raise ValueError()
        if not isinstance(name, str): name = str(name)

        if not name in self.geometry_handles:
            msg = "Surface %s does not exists yet, use set_data_2d() instead." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        s_x = c_uint()
        s_z = c_uint()
        if not self._optix.get_surface_size(name, byref(s_x), byref(s_z)):
            msg = "Cannot get surface %s size." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return
        size_xz = (s_z.value, s_x.value)
        size_changed = False

        pos_ptr = 0
        if pos is not None:
            if not isinstance(pos, np.ndarray): pos = np.ascontiguousarray(pos, dtype=np.float32)
            assert len(pos.shape) == 2 and pos.shape[0] > 1 and pos.shape[1] > 1, "Required vertex data shape is (z,x), where z >= 2 and x >= 2."
            if pos.dtype != np.float32: pos = np.ascontiguousarray(pos, dtype=np.float32)
            if not pos.flags['C_CONTIGUOUS']: pos = np.ascontiguousarray(pos, dtype=np.float32)
            if pos.shape != size_xz: size_changed = True
            size_xz = pos.shape
            pos_ptr = pos.ctypes.data

        c_ptr = 0
        c_const = None
        if size_changed and c is None: c = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32)
        if c is not None:
            if isinstance(c, float) or isinstance(c, int): c = np.full(3, c, dtype=np.float32)
            if not isinstance(c, np.ndarray): c = np.ascontiguousarray(c, dtype=np.float32)
            if len(c.shape) == 1 and c.shape[0] == 3:
                c_const = c
                cm = np.zeros(size_xz + (3,), dtype=np.float32)
                cm[:,:] = c
                c = cm
            assert len(c.shape) == 3 and c.shape == size_xz + (3,), "Colors shape must be (m,n,3), where (m,n) id the vertex data shape."
            if c.dtype != np.float32: c = np.ascontiguousarray(c, dtype=np.float32)
            if not c.flags['C_CONTIGUOUS']: c = np.ascontiguousarray(c, dtype=np.float32)
            c_ptr = c.ctypes.data

        n_ptr = 0
        if normals is not None:
            if not isinstance(normals, np.ndarray): normals = np.ascontiguousarray(normals, dtype=np.float32)
            assert len(normals.shape) == 3 and normals.shape == size_xz + (3,), "Normals shape must be (z,x,3), where (z,x) id the vertex data shape."
            if normals.dtype != np.float32: normals = np.ascontiguousarray(normals, dtype=np.float32)
            if not normals.flags['C_CONTIGUOUS']: normals = np.ascontiguousarray(normals, dtype=np.float32)
            n_ptr = normals.ctypes.data

        cl_ptr = 0
        if floor_c is not None:
            if isinstance(floor_c, float) or isinstance(floor_c, int): floor_c = np.full(3, floor_c, dtype=np.float32)
            if not isinstance(floor_c, np.ndarray): floor_c = np.ascontiguousarray(floor_c, dtype=np.float32)
            if len(floor_c.shape) == 1 and floor_c.shape[0] == 3:
                if floor_c.dtype != np.float32: floor_c = np.ascontiguousarray(floor_c, dtype=np.float32)
                if not floor_c.flags['C_CONTIGUOUS']: floor_c = np.ascontiguousarray(floor_c, dtype=np.float32)
                cl_ptr = floor_c.ctypes.data
            else:
                self._logger.warn("Floor color should be a single value or RGB array.")

        if range_x is None: range_x = (np.float32(np.nan), np.float32(np.nan))
        if range_z is None: range_z = (np.float32(np.nan), np.float32(np.nan))

        if floor_y is None: floor_y = np.float32(np.nan)

        try:
            self._padlock.acquire()
            self._logger.info("Update surface %s, size (%d, %d)...", name, size_xz[1], size_xz[0])
            g_handle = self._optix.update_surface(name, size_xz[1], size_xz[0],
                                                  pos_ptr, n_ptr, c_ptr, cl_ptr,
                                                  range_x[0], range_x[1], range_z[0], range_z[1],
                                                  floor_y)

            if (g_handle > 0) and (g_handle == self.geometry_handles[name]):
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_sizes[name] = size_xz[0] * size_xz[1]
            else:
                msg = "Geometry update failed."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                
        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise
        finally:
            self._padlock.release()


    def set_surface(self, name: str, pos: Any,
                    c: Any = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32),
                    normals: Optional[Any] = None,
                    mat: str = "diffuse",
                    wrap_u: bool = False,
                    wrap_v: bool = False,
                    make_normals: bool = False) -> None:
        """Create new (parametric) surface geometry.

        Data is provided as 2D array of :math:`[x, y, z] = f(u, v)` values, with the shape
        ``(n, m, 3)``, where ``n`` and ``m`` are at least 2. Additional data features can be
        visualized with color (array of RGB values, shape ``(n, m, 3)``).
        
        Parameters
        ----------
        name : string
            Name of the new surface geometry.
        pos : array_like
            XYZ values of surface points.
        c : Any, optional
            Colors of surface points. Single value means a constant gray level.
            3-component array means a constant RGB color. Array of the shape
            ``(n, m, 3)`` will set individual color for each surface point,
            interpolated between points; ``n`` and ``m`` have to be the same
            as in the surface points shape.
        normals : array_like, optional
            Normal vectors at provided surface points. Array shape has to be ``(n, m, 3)``,
            with ``n`` and ``m`` the same as in the surface points shape.
        mat : string, optional
            Material name.
        wrap_u : bool, optional
            Stitch surface edges making U axis continuous.
        wrap_v : bool, optional
            Stitch surface edges making V axis continuous.
        make_normals : bool, optional
            Calculate normals for surface points, if not provided with ``normals``
            argument. Normals of all triangles attached to the point are averaged.
        """
        if name is None: raise ValueError()
        if not isinstance(name, str): name = str(name)

        if name in self.geometry_handles:
            msg = "Geometry %s already exists, use update_surface() instead." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if not isinstance(pos, np.ndarray): pos = np.ascontiguousarray(pos, dtype=np.float32)
        assert len(pos.shape) == 3 and pos.shape[0] > 1 and pos.shape[1] > 1 and pos.shape[2] == 3, "Required surface points shape is (v,u,3), where u >= 2 and v >= 2."
        if pos.dtype != np.float32: pos = np.ascontiguousarray(pos, dtype=np.float32)
        if not pos.flags['C_CONTIGUOUS']: pos = np.ascontiguousarray(pos, dtype=np.float32)
        pos_ptr = pos.ctypes.data

        n_ptr = 0
        if normals is not None:
            if not isinstance(normals, np.ndarray): normals = np.ascontiguousarray(normals, dtype=np.float32)
            assert normals.shape == pos.shape, "Normals shape must be (v,u,3), with u and v matching the surface points shape."
            if normals.dtype != np.float32: normals = np.ascontiguousarray(normals, dtype=np.float32)
            if not normals.flags['C_CONTIGUOUS']: normals = np.ascontiguousarray(normals, dtype=np.float32)
            n_ptr = normals.ctypes.data
            make_normals = False

        c_ptr = 0
        c_const = None
        if c is not None:
            if isinstance(c, float) or isinstance(c, int): c = np.full(3, c, dtype=np.float32)
            if not isinstance(c, np.ndarray): c = np.ascontiguousarray(c, dtype=np.float32)
            if len(c.shape) == 1 and c.shape[0] == 3:
                c_const = c
                cm = np.zeros(pos.shape, dtype=np.float32)
                cm[:,:] = c
                c = cm
            assert c.shape == pos.shape, "Colors shape must be (v,u,3), with u and v matching the surface points shape."
            if c.dtype != np.float32: c = np.ascontiguousarray(c, dtype=np.float32)
            if not c.flags['C_CONTIGUOUS']: c = np.ascontiguousarray(c, dtype=np.float32)
            c_ptr = c.ctypes.data

        try:
            self._padlock.acquire()
            self._logger.info("Setup surface %s...", name)
            g_handle = self._optix.setup_psurface(name, mat, pos.shape[1], pos.shape[0], pos_ptr, n_ptr, c_ptr, wrap_u, wrap_v, make_normals)

            if g_handle > 0:
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_names[g_handle] = name
                self.geometry_handles[name] = g_handle
                self.geometry_sizes[name] = pos.shape[0] * pos.shape[1]
            else:
                msg = "Surface setup failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise
        finally:
            self._padlock.release()

    def update_surface(self, name: str,
                       pos: Optional[Any] = None,
                       c: Optional[Any] = None,
                       normals: Optional[Any] = None) -> None:
        """Update surface geometry data or properties.

        Parameters
        ----------
        name : string
            Name of the surface geometry.
        pos : array_like, optional
            XYZ values of surface points.
        c : Any, optional
            Colors of surface points. Single value means a constant gray level.
            3-component array means a constant RGB color. Array of the shape
            ``(n, m, 3)`` will set individual color for each surface point,
            interpolated between points; ``n`` and ``m`` have to be the same
            as in the surface points shape.
        normals : array_like, optional
            Normal vectors at provided surface points. Array shape has to be ``(n, m, 3)``,
            with ``n`` and ``m`` the same as in the surface points shape.
        """
        if name is None: raise ValueError()
        if not isinstance(name, str): name = str(name)

        if not name in self.geometry_handles:
            msg = "Surface %s does not exists yet, use set_surface() instead." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        s_u = c_uint()
        s_v = c_uint()
        if not self._optix.get_surface_size(name, byref(s_u), byref(s_v)):
            msg = "Cannot get surface %s size." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return
        size_uv = (s_v.value, s_u.value, 3)
        size_changed = False

        pos_ptr = 0
        if pos is not None:
            if not isinstance(pos, np.ndarray): pos = np.ascontiguousarray(pos, dtype=np.float32)
            assert len(pos.shape) == 3 and pos.shape[0] > 1 and pos.shape[1] > 1 and pos.shape[2] == 3, "Required vertex data shape is (v,u,3), where u >= 2 and v >= 2."
            if pos.dtype != np.float32: pos = np.ascontiguousarray(pos, dtype=np.float32)
            if not pos.flags['C_CONTIGUOUS']: pos = np.ascontiguousarray(pos, dtype=np.float32)
            if pos.shape != size_uv: size_changed = True
            size_uv = pos.shape
            pos_ptr = pos.ctypes.data

        c_ptr = 0
        c_const = None
        if size_changed and c is None: c = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32)
        if c is not None:
            if isinstance(c, float) or isinstance(c, int): c = np.full(3, c, dtype=np.float32)
            if not isinstance(c, np.ndarray): c = np.ascontiguousarray(c, dtype=np.float32)
            if len(c.shape) == 1 and c.shape[0] == 3:
                c_const = c
                cm = np.zeros(size_uv, dtype=np.float32)
                cm[:,:] = c
                c = cm
            assert c.shape == size_uv, "Colors shape must be (m,n,3), with m and n matching the surface points shape."
            if c.dtype != np.float32: c = np.ascontiguousarray(c, dtype=np.float32)
            if not c.flags['C_CONTIGUOUS']: c = np.ascontiguousarray(c, dtype=np.float32)
            c_ptr = c.ctypes.data

        n_ptr = 0
        if normals is not None:
            if not isinstance(normals, np.ndarray): normals = np.ascontiguousarray(normals, dtype=np.float32)
            assert normals.shape == size_uv, "Normals shape must be (v,u,3), with u and v matching the surface points shape."
            if normals.dtype != np.float32: normals = np.ascontiguousarray(normals, dtype=np.float32)
            if not normals.flags['C_CONTIGUOUS']: normals = np.ascontiguousarray(normals, dtype=np.float32)
            n_ptr = normals.ctypes.data

        try:
            self._padlock.acquire()
            self._logger.info("Update surface %s, size (%d, %d)...", name, size_uv[1], size_uv[0])
            g_handle = self._optix.update_psurface(name, size_uv[1], size_uv[0], pos_ptr, n_ptr, c_ptr)

            if (g_handle > 0) and (g_handle == self.geometry_handles[name]):
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_sizes[name] = size_uv[0] * size_uv[1]
            else:
                msg = "Geometry update failed."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                
        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise
        finally:
            self._padlock.release()


    def set_mesh(self, name: str, pos: Any, pidx = None, normals = None, nidx = None,
                 c: Any = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32),
                 mat: str = "diffuse", make_normals: bool = False) -> None:

        if name is None: raise ValueError()

        if not isinstance(pos, np.ndarray): pos = np.ascontiguousarray(pos, dtype=np.float32)
        assert len(pos.shape) == 2 and pos.shape[0] > 2 and pos.shape[1] == 3, "Required vertex data shape is (n,3), where n >= 3."

        if pidx is not None and not isinstance(pidx, np.ndarray): pidx = np.ascontiguousarray(pidx, dtype=np.int32)
        pidx_ptr = 0
        if pidx is not None:
            assert (len(pidx.shape) == 2 and pidx.shape[0] > 0 and pos.shape[1] == 3) or (len(pidx.shape) == 1 and pidx.shape[0] > 3 and (pidx.shape[0] % 3) == 0), "Required index shape is (n,3) or (m), where m % 3 == 0."
            pidx_ptr = pidx.ctypes.data

        c = _make_contiguous_3d(c, n=pos.shape[0], extend_scalars=True)
        col_ptr = 0
        if c is not None:
            col_ptr = c.ctypes.data


    def load_mesh_obj(self, file_name: str, mesh_name: str,
                      c: Any = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32),
                      mat: str = "diffuse", make_normals: bool = False) -> None:
        """Load mesh geometry from Wavefront .obj file.

        Parameters
        ----------
        file_name : string
            File name (local file path or url) to read from.
        mesh_name : string
            Name of the new mesh geometry.
        c : Any, optional
            Color of the mesh. Single value means a constant gray level.
            3-component array means constant RGB color.
        mat : string, optional
            Material name.
        make_normals : bool, optional
            Calculate new normal for each vertex by averaging normals of connected
            mesh triangles. If set to ``False`` (default) then original normals from
            the .obj file are preserved or normals are not used (mesh triangles
            define normals).
        """
        if file_name is None or mesh_name is None: raise ValueError()

        if not isinstance(file_name, str): file_name = str(file_name)

        if not isinstance(mesh_name, str): mesh_name = str(mesh_name)

        if mesh_name in self.geometry_handles:
            msg = "Geometry %s already exists, use update_mesh() instead." % mesh_name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        c = _make_contiguous_vector(c, n_dim=3)
        if c is not None: col_ptr = c.ctypes.data
        else: col_ptr = 0

        try:
            self._padlock.acquire()
            self._logger.info("Load mesh %s, from file %s ...", mesh_name, file_name)
            g_handle = self._optix.load_mesh_obj(file_name, mesh_name, mat, col_ptr, make_normals)

            if g_handle > 0:
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_names[g_handle] = mesh_name
                self.geometry_handles[mesh_name] = g_handle
                self.geometry_sizes[mesh_name] = 1 # todo: read mesh size
            else:
                msg = "Mesh loading failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise
        finally:
            self._padlock.release()


    def move_geometry(self, name: str, v: Tuple[float, float, float],
                      update: bool = True) -> None:
        """Move all primitives by (x, y, z) vector.

        Updates GPU buffers immediately if update is set to ``True`` (default),
        otherwise update should be made using :meth:`plotoptix.NpOptiX.update_geom_buffers`
        after all geometry modifications are finished.

        Parameters
        ----------
        name : string
            Name of the geometry.
        v : tuple (float, float, float)
            (X, Y, Z) shift.
        update : bool, optional
            Update GPU buffer.
        """
        if name is None: raise ValueError()

        if not self._optix.move_geometry(name, v[0], v[1], v[2], update):
            msg = "Geometry move failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def move_primitive(self, name: str, idx: int, v: Tuple[float, float, float],
                       update: bool = True) -> None:
        """Move selected primitive by (x, y, z) vector.

        Updates GPU buffers immediately if update is set to ``True`` (default),
        otherwise update should be made using :meth:`plotoptix.NpOptiX.update_geom_buffers`
        after all geometry modifications are finished.

        Parameters
        ----------
        name : string
            Name of the geometry.
        idx : int
            Primitive index.
        v : tuple (float, float, float)
            (X, Y, Z) shift.
        update : bool, optional
            Update GPU buffer.
        """
        if name is None: raise ValueError()

        if not self._optix.move_primitive(name, idx, v[0], v[1], v[2], update):
            msg = "Primitive move failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def rotate_geometry(self, name: str, rot: Tuple[float, float, float],
                        center: Optional[Tuple[float, float, float]] = None,
                        update: bool = True) -> None:
        """Rotate all primitives by specified degrees.

        Rotate all primitives by specified degrees around x, y, z axis, with
        respect to the center of the geometry. Update GPU buffers immediately
        if update is set to ``True`` (default), otherwise update should be made using
        :meth:`plotoptix.NpOptiX.update_geom_buffers` after all geometry modifications
        are finished.

        Parameters
        ----------
        name : string
            Name of the geometry.
        rot : tuple (float, float, float)
            Rotation around (X, Y, Z) axis.
        center : tuple (float, float, float), optional
            Rotation center. If not provided, rotation is made about the geometry
            center.
        update : bool, optional
            Update GPU buffer.
        """
        if name is None: raise ValueError()

        if center is None:
            if not self._optix.rotate_geometry(name, rot[0], rot[1], rot[2], update):
                msg = "Geometry rotate failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)
        else:
            if not isinstance(center, tuple): center = tuple(center)
            if not self._optix.rotate_geometry_about(name, rot[0], rot[1], rot[2], center[0], center[1], center[2], update):
                msg = "Geometry rotate failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

    def rotate_primitive(self, name: str, idx: int, rot: Tuple[float, float, float],
                         center: Optional[Tuple[float, float, float]] = None,
                         update: bool = True) -> None:
        """Rotate selected primitive by specified degrees.

        Rotate selected primitive by specified degrees around x, y, z axis, with
        respect to the center of the selected primitive. Update GPU buffers
        immediately if update is set to ``True`` (default), otherwise update should be
        made using :meth:`plotoptix.NpOptiX.update_geom_buffers` after all geometry
        modifications are finished.

        Parameters
        ----------
        name : string
            Name of the geometry.
        idx : int
            Primitive index.
        rot : tuple (float, float, float)
            Rotation around (X, Y, Z) axis.
        center : tuple (float, float, float), optional
            Rotation center. If not provided, rotation is made about the primitive
            center.
        update : bool, optional
            Update GPU buffer.
        """
        if name is None: raise ValueError()

        if center is None:
            if not self._optix.rotate_primitive(name, idx, rot[0], rot[1], rot[2], update):
                msg = "Primitive rotate failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)
        else:
            if not isinstance(center, tuple): center = tuple(center)
            if not self._optix.rotate_primitive_about(name, idx, rot[0], rot[1], rot[2], center[0], center[1], center[2], update):
                msg = "Geometry rotate failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

    def scale_geometry(self, name: str, s: float,
                       update: bool = True) -> None:
        """Scale all primitive's positions and sizes.

        Scale all primitive's positions and sizes by specified factor, with respect
        to the center of the geometry. Update GPU buffers immediately if update is
        set to ``True`` (default), otherwise update should be made using
        :meth:`plotoptix.NpOptiX.update_geom_buffers` after all geometry modifications
        are finished.

        Parameters
        ----------
        name : string
            Name of the geometry.
        s : float
            Scaling factor.
        update : bool, optional
            Update GPU buffer.
        """
        if name is None: raise ValueError()

        if not self._optix.scale_geometry(name, s, update):
            msg = "Geometry scale failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def scale_primitive(self, name: str, idx: int, s: float,
                        update: bool = True) -> None:
        """Scale selected primitive.

        Scale selected primitive by specified factor, with respect to the center of
        the selected primitive. Update GPU buffers immediately if update is set to
        ``True`` (default), otherwise update should be made using
        :meth:`plotoptix.NpOptiX.update_geom_buffers` after all geometry modifications
        are finished.

        Parameters
        ----------
        name : string
            Name of the geometry.
        idx : int
            Primitive index.
        s : float
            Scaling factor.
        update : bool, optional
            Update GPU buffer.
        """
        if name is None: raise ValueError()

        if not self._optix.scale_primitive(name, idx, s, update):
            msg = "Primitive scale failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def update_geom_buffers(self, name: str,
                            mask: Union[GeomBuffer, str] = GeomBuffer.All) -> None:
        """Update geometry buffers.

        Update geometry buffers in GPU after modifications made with
        :meth:`plotoptix.NpOptiX.move_geometry` / :meth:`plotoptix.NpOptiX.move_primitive`
        and similar methods.

        Parameters
        ----------
        name : string
            Name of the geometry.
        mask : GeomBuffer or string, optional
            Which buffers to update. All buffers if not specified.
        """
        if name is None: raise ValueError()

        if isinstance(mask, str): mask = GeomBuffer[mask]

        if not self._optix.update_geom_buffers(name, mask.value):
            msg = "Geometry buffers update failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def set_coordinates(self, mode: Union[Coordinates, str] = Coordinates.Box, thickness: float = 1.0) -> None:
        """Set style of the coordinate system geometry (or hide it).

        Parameters
        ----------
        mode : Coordinates enum or string, optional
            Style of the coordinate system geometry.
        thickness : float, optional
            Thickness of lines.

        See Also
        --------
        :class:`plotoptix.enums.Coordinates`
        """
        if mode is None: raise ValueError()

        if isinstance(mode, str): mode = Coordinates[mode]

        if self._optix.set_coordinates_geom(mode.value, thickness):
            self._logger.info("Coordinate system mode set to: %s.", mode.name)
        else:
            msg = "Coordinate system mode not changed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)
