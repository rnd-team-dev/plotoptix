"""No-UI PlotOptiX raytracer (output to numpy array only).

https://github.com/rnd-team-dev/plotoptix/blob/master/LICENSE.txt

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
"""

import json, math, logging, os, threading, time
import numpy as np

from ctypes import byref, c_ubyte, c_float, c_uint, c_int, c_longlong
from typing import List, Tuple, Callable, Optional, Union, Any

from plotoptix.singleton import Singleton
from plotoptix.geometry import GeometryMeta
from plotoptix._load_lib import load_optix, PARAM_NONE_CALLBACK, PARAM_INT_CALLBACK
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
        Pixel width of the raytracing output. Default value is 16.
    height : int, optional
        Pixel height of the raytracing output. Default value is 16.
    start_now : bool, optional
        Start raytracing thread immediately. If set to ``False``, then user should
        call ``start()`` method. Default is ``False``.
    devices : list, optional
        List of selected devices, with the primary device at index 0. Empty list
        is default, resulting with all compatible devices selected for processing.
    log_level : int or string, optional
        Log output level. Default is ``WARN``.
    """

    _img_rgba = None
    """Ray-tracing output, 8bps color.

    Shape: ``(height, width, 4)``, ``dtype = np.uint8``, contains RGBA data
    (alpha channel is now constant, ``255``).

    A ndarray wrapped aroud the gpu bufffer. It enables reading the image with no
    additional memory copy. Access the buffer in the ``on_launch_finished`` callback
    or in/after the ``on_rt_accum_done`` callback to avoid reading while the buffer
    content is being updated.
    """

    _raw_rgba = None
    """Ray-tracing output, raw floating point data.

    Shape: ``(height, width, 4)``, ``dtype = np.float32``, contains RGBA data
    (alpha channel is now constant, ``1.0``).

    A ndarray wrapped aroud the gpu bufffer. It enables reading the image with no
    additional memory copy. Access the buffer in the ``on_launch_finished`` callback
    or in/after the ``on_rt_accum_done`` callback to avoid reading while the buffer
    content is being updated.
    """

    _hit_pos = None
    """Hit position.

    Shape: ``(height, width, 4)``, ``dtype = np.float32``, contains XYZD data, where
    XYZ is the hit 3D position and D channel is the hit distance to camera plane.
    """

    _geo_id = None
    """Object info.
    
    Encodes the object handle and primitive index (or vertex/face index for meshes)
    for each pixel in the output image.

    Shape: ``(height, width, 2)``, ``dtype = np.int32``, contains:
       - ``_geo_id[h, w, 0] = handle | (vtx_id << 30)``, where ``handle`` is the object
         handle, ``vtx_id`` is the vertex index for the triangular face that was hit
         (values are ``0``, ``1``, ``2``);
       - ``_geo_id[h, w, 1] = prim_idx``, where ``prim_idx`` is the primitive index in
         a data set, or face index of a mesh.
    """

    _albedo = None
    """Surface albedo.

    Shape: ``(height, width, 4)``, ``dtype = np.float32``, contains RGBA data
    (alpha channel is now constant, ``0.0``).

    Available only when the denoiser is enabled (:attr:`plotoptix.enums.Postprocessing.Denoiser`),
    and set to :attr:`plotoptix.enums.DenoiserKind.RgbAlbedo`
    or :attr:`plotoptix.enums.DenoiserKind.RgbAlbedoNormal` mode.
    """

    _normal = None
    """Surface normal.

    Shape: ``(height, width, 4)``, ``dtype = np.float32``, contains XYZ0 data
    (4'th channel is constant, ``0.0``).

    Surface normal vector in camera space. Available only when the denoiser is enabled
    (:attr:`plotoptix.enums.Postprocessing.Denoiser`), and set to
    :attr:`plotoptix.enums.DenoiserKind.RgbAlbedoNormal` mode.
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

        rt_log = 0
        if isinstance(log_level, int):
            if log_level == logging.ERROR: rt_log = 1
            elif log_level == logging.WARNING: rt_log = 2
            elif log_level == logging.INFO: rt_log = 3
            elif log_level == logging.DEBUG: rt_log = 4
        if isinstance(log_level, str):
            if log_level == 'ERROR': rt_log = 1
            elif log_level == 'WARNING': rt_log = 2
            elif log_level == 'WARN': rt_log = 2
            elif log_level == 'INFO': rt_log = 3
            elif log_level == 'DEBUG': rt_log = 4

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

        self.geometry_data = {}    # geometry name to metadata dictionary
        self.geometry_names = {}   # geometry handle to name dictionary
        self.camera_handles = {}   # camera name to handle dictionary
        self.camera_names = {}     # camera handle to name dictionary
        self.light_handles = {}    # light name to handle dictionary
        self.light_names = {}      # light handle to name dictionary

        # scene initialization / compute / upload / accumulation done callbacks:
        if on_initialization is not None: self._initialization_cb = self._make_list_of_callable(on_initialization)
        elif src is None: self._initialization_cb = [self._default_initialization]
        else: self._initialization_cb = []
        self.set_scene_compute_cb(on_scene_compute)
        self.set_rt_completed_cb(on_rt_completed)
        self.set_rt_starting_cb(cb=None)
        self.set_launch_finished_cb(on_launch_finished)
        self.set_accum_done_cb(on_rt_accum_done)

        device_ptr = 0
        device_count = 0
        if len(devices) > 0:
            self._logger.info("Configure selected devices.")
            device_idx = [int(d) for d in devices]
            device_idx = np.ascontiguousarray(device_idx, dtype=np.int32)
            device_ptr = device_idx.ctypes.data
            device_count = device_idx.shape[0]

        if src is None:                          # create empty scene
            self._logger.info("  - ray-tracer initialization")
            self._is_scene_created = self._optix.create_empty_scene(self._width, self._height, device_ptr, device_count, rt_log)
            if self._is_scene_created: self._logger.info("Empty scene ready.")

        elif isinstance(src, str):               # create scene from file
            if not os.path.isfile(src):
                msg = "File %s not found." % src
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                return

            wd = os.getcwd()
            if os.path.isabs(src):
                d, f = os.path.split(src)
                os.chdir(d)
            else: f = src

            self._is_scene_created = self._optix.create_scene_from_file(f, self._width, self._height, device_ptr, device_count)
            self._is_scene_created &= self._init_scene_metadata()
            if self._is_scene_created:
                self._logger.info("Scene loaded correctly.")

            os.chdir(wd)

        elif isinstance(src, dict):              # create scene from dictionary
            s = json.dumps(src)
            self._is_scene_created = self._optix.create_scene_from_json(s, self._width, self._height, device_ptr, device_count)
            self._is_scene_created &= self._init_scene_metadata()
            if self._is_scene_created: self._logger.info("Scene loaded correctly.")

        else:
            msg = "Scene source type not supported."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

        if self._is_scene_created:
            # optionally start raytracing thread:
            if start_now: self.start()
            else: self._logger.info("Use start() to start raytracing.")
        else:
            msg = "Initial setup failed, see errors above."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)
        ###############################################################

    def __del__(self):
        """Release resources.
        """
        if self._is_scene_created and not self._is_closed:
            if self._is_started: self.close()
            else: self._optix.destroy_scene()

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
        if self._started_event.wait(600):
            self._logger.info("Raytracing started.")
            self._is_started = True
        else:
            msg = "Raytracing output startup timed out."
            self._logger.error(msg)
            self._is_started = False

            if self._raise_on_error: raise TimeoutError(msg)

    def update_device_buffers(self):
        """Update buffer pointers.

        Use after changing denoiser mode since albedo and normal
        buffer wrappers are not updated automatically.
        """
        c_buf = c_longlong()
        c_len = c_int()
        r_buf = c_longlong()
        r_len = c_int()
        h_buf = c_longlong()
        h_len = c_int()
        g_buf = c_longlong()
        g_len = c_int()
        a_buf = c_longlong()
        a_len = c_int()
        n_buf = c_longlong()
        n_len = c_int()
        if self._optix.get_device_buffers(
                                    byref(c_buf), byref(c_len),
                                    byref(r_buf), byref(r_len),
                                    byref(h_buf), byref(h_len),
                                    byref(g_buf), byref(g_len),
                                    byref(a_buf), byref(a_len),
                                    byref(n_buf), byref(n_len)):
            buf = (((c_ubyte * 4) * self._width) * self._height).from_address(c_buf.value)
            self._img_rgba = np.ctypeslib.as_array(buf)
            buf = (((c_float * 4) * self._width) * self._height).from_address(r_buf.value)
            self._raw_rgba = np.ctypeslib.as_array(buf)
            buf = (((c_float * 4) * self._width) * self._height).from_address(h_buf.value)
            self._hit_pos = np.ctypeslib.as_array(buf)
            buf = (((c_uint * 2) * self._width) * self._height).from_address(g_buf.value)
            self._geo_id = np.ctypeslib.as_array(buf)
            if a_len.value > 0:
                buf = (((c_float * 4) * self._width) * self._height).from_address(a_buf.value)
                self._albedo = np.ctypeslib.as_array(buf)
            else: self._albedo = None
            if n_len.value > 0:
                buf = (((c_float * 4) * self._width) * self._height).from_address(n_buf.value)
                self._normal = np.ctypeslib.as_array(buf)
            else: self._normal = None
        else:
            msg = "Image buffers setup failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def run(self):
        """Starts UI event loop.

        Derived from `threading.Thread <https://docs.python.org/3/library/threading.html>`__.

        Use :meth:`plotoptix.NpOptiX.start` to perform complete initialization.

        **Do not override**, use :meth:`plotoptix.NpOptiX._run_event_loop` instead.
        """
        assert self._is_scene_created, "Scene is not ready, see initialization messages."

        c_buf = c_longlong()
        c_len = c_int()
        r_buf = c_longlong()
        r_len = c_int()
        h_buf = c_longlong()
        h_len = c_int()
        g_buf = c_longlong()
        g_len = c_int()
        a_buf = c_longlong()
        a_len = c_int()
        n_buf = c_longlong()
        n_len = c_int()
        if self._optix.resize_scene(self._width, self._height,
                                    byref(c_buf), byref(c_len),
                                    byref(r_buf), byref(r_len),
                                    byref(h_buf), byref(h_len),
                                    byref(g_buf), byref(g_len),
                                    byref(a_buf), byref(a_len),
                                    byref(n_buf), byref(n_len)):
            buf = (((c_ubyte * 4) * self._width) * self._height).from_address(c_buf.value)
            self._img_rgba = np.ctypeslib.as_array(buf)
            buf = (((c_float * 4) * self._width) * self._height).from_address(r_buf.value)
            self._raw_rgba = np.ctypeslib.as_array(buf)
            buf = (((c_float * 4) * self._width) * self._height).from_address(h_buf.value)
            self._hit_pos = np.ctypeslib.as_array(buf)
            buf = (((c_uint * 2) * self._width) * self._height).from_address(g_buf.value)
            self._geo_id = np.ctypeslib.as_array(buf)
            if a_len.value > 0:
                buf = (((c_float * 4) * self._width) * self._height).from_address(a_buf.value)
                self._albedo = np.ctypeslib.as_array(buf)
            else: self._albedo = None
            if n_len.value > 0:
                buf = (((c_float * 4) * self._width) * self._height).from_address(n_buf.value)
                self._normal = np.ctypeslib.as_array(buf)
            else: self._normal = None
        else:
            msg = "Image buffers setup failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

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
            self._logger.error(msg)
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

    def get_rt_output(self,
                      bps: Union[ChannelDepth, str] = ChannelDepth.Bps8,
                      channels: Union[ChannelOrder, str] = ChannelOrder.RGBA) -> Optional[np.ndarray]:
        """Return a copy of the output image.

        The image data type is specified with the ``bps`` argument. 8 bit per channel data,
        ``numpy.uint8``, is returned by default. Use ``Bps16`` value to read the image in
        16 bit per channel depth, ``numpy.uint16``. Use ``Bps32`` value to read the HDR image
        in 32 bit per channel format, ``numpy.float32``.

        If channels ordering includes alpha channel then it is a constant, 100% opaque value,
        to be used in the future releases.
        
        Safe to call at any time, from any thread.

        Parameters
        ----------
        bps : ChannelDepth enum or string, optional
            Color depth.
        channels : ChannelOrder enum or string, optional
            Color channels ordering.

        Returns
        -------
        out : ndarray
            RGB(A) array of shape (height, width, 3) or (height, width, 4) and type corresponding
            to ``bps`` argument. ``None`` in case of errors.

        See Also
        --------
        :class:`plotoptix.enums.ChannelDepth`, :class:`plotoptix.enums.ChannelOrder`
        """
        assert self._is_started, "Raytracing output not running."

        if isinstance(bps, str): bps = ChannelDepth[bps]
        if isinstance(channels, str): channels = ChannelOrder[channels]

        a = None

        try:
            self._padlock.acquire()

            if bps == ChannelDepth.Bps8 and channels == ChannelOrder.RGBA:
                if self._img_rgba is not None:
                    return self._img_rgba.copy()
                else:
                    a = np.ascontiguousarray(np.zeros((self._height, self._width, 4), dtype=np.uint8))

            if bps == ChannelDepth.Bps8:
                if channels == ChannelOrder.BGRA:
                    a = np.ascontiguousarray(np.zeros((self._height, self._width, 4), dtype=np.uint8))
                elif channels == ChannelOrder.RGB or channels == ChannelOrder.BGR:
                    a = np.ascontiguousarray(np.zeros((self._height, self._width, 3), dtype=np.uint8))

            elif bps == ChannelDepth.Bps16:
                if channels == ChannelOrder.RGBA or channels == ChannelOrder.BGRA:
                    a = np.ascontiguousarray(np.zeros((self._height, self._width, 4), dtype=np.uint16))
                elif channels == ChannelOrder.RGB or channels == ChannelOrder.BGR:
                    a = np.ascontiguousarray(np.zeros((self._height, self._width, 3), dtype=np.uint16))

            elif bps == ChannelDepth.Bps32:
                if channels == ChannelOrder.RGBA or channels == ChannelOrder.BGRA:
                    a = np.ascontiguousarray(np.zeros((self._height, self._width, 4), dtype=np.float32))
                elif channels == ChannelOrder.RGB or channels == ChannelOrder.BGR:
                    a = np.ascontiguousarray(np.zeros((self._height, self._width, 3), dtype=np.float32))

            else: return a

            if not self._optix.get_output(a.ctypes.data, a.nbytes, bps.value, channels.value):
                msg = "Image not copied."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise

        finally:
            self._padlock.release()

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

            # resize the scene, update gpu memory address
            c_buf = c_longlong()
            c_len = c_int()
            r_buf = c_longlong()
            r_len = c_int()
            h_buf = c_longlong()
            h_len = c_int()
            g_buf = c_longlong()
            g_len = c_int()
            a_buf = c_longlong()
            a_len = c_int()
            n_buf = c_longlong()
            n_len = c_int()
            if self._optix.resize_scene(self._width, self._height,
                                    byref(c_buf), byref(c_len),
                                    byref(r_buf), byref(r_len),
                                    byref(h_buf), byref(h_len),
                                    byref(g_buf), byref(g_len),
                                    byref(a_buf), byref(a_len),
                                    byref(n_buf), byref(n_len)):
                buf = (((c_ubyte * 4) * self._width) * self._height).from_address(c_buf.value)
                
                #buf_from_mem = ctypes.pythonapi.PyMemoryView_FromMemory
                #buf_from_mem.restype = ctypes.py_object
                #buf_from_mem.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
                #buf = buf_from_mem(c_buf.value, c_len.value, 0x100)

                self._img_rgba = np.ctypeslib.as_array(buf)
                #self._img_rgba = np.ndarray((height, width, 4), np.uint8, buf, order='C')

                #print(self._img_rgba.shape, self._img_rgba.__array_interface__)
                #print(self._img_rgba[int(height/2), int(width/2)])

                buf = (((c_float * 4) * self._width) * self._height).from_address(r_buf.value)
                self._raw_rgba = np.ctypeslib.as_array(buf)
                buf = (((c_float * 4) * self._width) * self._height).from_address(h_buf.value)
                self._hit_pos = np.ctypeslib.as_array(buf)
                buf = (((c_uint * 2) * self._width) * self._height).from_address(g_buf.value)
                self._geo_id = np.ctypeslib.as_array(buf)
                if a_len.value > 0:
                    buf = (((c_float * 4) * self._width) * self._height).from_address(a_buf.value)
                    self._albedo = np.ctypeslib.as_array(buf)
                else: self._albedo = None
                if n_len.value > 0:
                    buf = (((c_float * 4) * self._width) * self._height).from_address(n_buf.value)
                    self._normal = np.ctypeslib.as_array(buf)
                else: self._normal = None
            else:
                self._img_rgba = None
                self._hit_pos = None
                self._geo_id = None
                self._albedo = None
                self._normal = None

    @staticmethod
    def _default_initialization(wnd) -> None:
        wnd._logger.info("Default scene initialization.")
        if wnd._optix.get_current_camera() == 0:
            wnd.setup_camera("default", [0, 0, 10], [0, 0, 0])

    ###########################################################################
    def set_launch_finished_cb(self, cb) -> None:
        """Set callback function(s) executed after each finished frame.

        Parameters
        ----------
        cb : callable or list
            Callable or list of callables to set as the launch finished callback.
        """
        with self._padlock:
            if cb is not None: self._launch_finished_cb = self._make_list_of_callable(cb)
            else: self._launch_finished_cb = []

    def _launch_finished_callback(self, rt_result: int) -> None:
        """
        Callback executed after each finished frame (``min_accumulation_step``
        accumulation frames are raytraced together). This callback is
        executed in the raytracing thread and should not compute extensively
        (get/save the image data here but calculate scene etc in another thread).

        Override this method in the UI class, call this base implementation
        and update image in UI (or raise an event to do so).
        
        Actions provided with ``on_launch_finished`` parameter of NpOptiX
        constructor are executed here.

        Parameters
        ----------
        rt_result : int
            Raytracing result code corresponding to :class:`plotoptix.enums.RtResult`.
        """
        if self._is_started:
            if rt_result < RtResult.NoUpdates.value:
                #self._logger.info("Launch finished.")
                with self._padlock:
                    for c in self._launch_finished_cb: c(self)
    def _get_launch_finished_callback(self):
        def func(rt_result: int): self._launch_finished_callback(rt_result)
        return PARAM_INT_CALLBACK(func)
    ###########################################################################

    ###########################################################################
    def set_rt_starting_cb(self, cb) -> None:
        """Set callback function(s) executed before each frame raytracing.

        Parameters
        ----------
        cb : callable or list
            Callable or list of callables to set as the rt starting callback.
        """
        with self._padlock:
            if cb is not None: self._rt_starting_cb = self._make_list_of_callable(cb)
            else: self._rt_starting_cb = []

    def _scene_rt_starting_callback(self) -> None:
        """
        Callback executed before starting frame raytracing. Appropriate to
        override in UI class and apply scene edits (or raise an event to do
        so) like camera rotations, etc. made by a user in UI.
        
        This callback is executed in the raytracing thread and should not
        compute extensively.
        """
        for c in self._rt_starting_cb: c(self)
    def _get_scene_rt_starting_callback(self):
        def func(): self._scene_rt_starting_callback()
        return PARAM_NONE_CALLBACK(func)
    ###########################################################################

    ###########################################################################
    def set_accum_done_cb(self, cb) -> None:
        """Set callback function(s) executed when all accumulation frames
        are completed.

        Parameters
        ----------
        cb : callable or list
            Callable or list of callables to set as the accum done callback.
        """
        with self._padlock:
            if cb is not None: self._rt_accum_done_cb = self._make_list_of_callable(cb)
            else: self._rt_accum_done_cb = []

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
            with self._padlock:
                for c in self._rt_accum_done_cb: c(self)
    def _get_accum_done_callback(self):
        def func(): self._accum_done_callback()
        return PARAM_NONE_CALLBACK(func)
    ###########################################################################

    ###########################################################################
    def set_scene_compute_cb(self, cb) -> None:
        """Set callback function(s) executed on each frame ray tracing start.

        Callback(s) executed in parallel to the raytracing and intended for
        CPU intensive computations. Note, set ``compute_timeout`` to appropriate
        value if your computations are longer than single frame ray tracing, see
        :meth:`plotoptix.NpOptiX.set_param`.

        Parameters
        ----------
        cb : callable or list
            Callable or list of callables to set as the scene compute callback.
        """
        with self._padlock:
            if cb is not None: self._scene_compute_cb = self._make_list_of_callable(cb)
            else: self._scene_compute_cb = []

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

    def set_rt_completed_cb(self, cb) -> None:
        """Set callback function(s) executed on each frame ray tracing finished.

        Callback(s) executed in the same thread as the scene compute callback. Note,
        set ``compute_timeout`` to appropriate value if your computations are longer
        than single frame ray tracing, see :meth:`plotoptix.NpOptiX.set_param`.

        Parameters
        ----------
        cb : callable or list
            Callable or list of callables to set as the RT completed callback.
        """
        with self._padlock:
            if cb is not None: self._rt_completed_cb = self._make_list_of_callable(cb)
            else: self._rt_completed_cb = []

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
        if self._is_started and rt_result <= RtResult.NoUpdates.value:
            self._logger.info("RT completed, result %s.", RtResult(rt_result))
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
            Variable name.

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
            Variable name.

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
            Variable name.

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
            Vairable name.
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
        >>> optix.set_float("tonemap_gamma", 2.2)
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
            Variable name.

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
            Variable name.

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
        Note, shader variables distinguish ``int`` and ``uint`` while the type
        provided by Python methods is ``int`` in both cases.

        Parameters
        ----------
        name : string
            Variable name.
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
            Variable name.

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
        Note, shader variables distinguish ``int`` and ``uint`` while the type
        provided by Python methods is ``int`` in both cases.

        Parameters
        ----------
        name : string
            Variable name.
        x : int
            Variable value.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.
        """
        if not isinstance(name, str): name = str(name)
        if not isinstance(x, int): x = int(x)

        self._optix.set_int(name, x, refresh)


    def set_texture_1d(self, name: str, data: Any,
                       addr_mode: Union[TextureAddressMode, str] = TextureAddressMode.Clamp,
                       keep_on_host: bool = False,
                       refresh: bool = False) -> None:
        """Set texture data.

        Set texture ``name`` data. Texture format (float, float2 or float4) and
        length are deduced from the ``data`` array shape. Use ``keep_on_host=True``
        to make a copy of data in the host memory (in addition to GPU memory), this
        option is required when (small) textures are going to be saved to JSON description
        of the scene.

        Parameters
        ----------
        name : string
            Texture name.
        data : array_like
            Texture data.
        addr_mode : TextureAddressMode or string, optional
            Texture addressing mode on edge crossing.
        keep_on_host : bool, optional
            Store texture data copy in the host memory.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.
        """
        if not isinstance(name, str): name = str(name)
        if not isinstance(data, np.ndarray): data = np.ascontiguousarray(data, dtype=np.float32)

        if isinstance(addr_mode, str): addr_mode = TextureAddressMode[addr_mode]

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
        if not self._optix.set_texture_1d(name, data.ctypes.data, data.shape[0], rt_format.value, addr_mode.value, keep_on_host, refresh):
            msg = "Texture 1D %s not uploaded." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def set_texture_2d(self, name: str, data: Any,
                       addr_mode: Union[TextureAddressMode, str] = TextureAddressMode.Wrap,
                       keep_on_host: bool = False,
                       refresh: bool = False) -> None:
        """Set texture data.

        Set texture ``name`` data. Texture format (float, float2 or float4) and
        width/height are deduced from the ``data`` array shape. Use ``keep_on_host=True``
        to make a copy of data in the host memory (in addition to GPU memory), this
        option is required when (small) textures are going to be saved to JSON description
        of the scene.

        Parameters
        ----------
        name : string
            Texture name.
        data : array_like
            Texture data.
        addr_mode : TextureAddressMode or string, optional
            Texture addressing mode on edge crossing.
        keep_on_host : bool, optional
            Store texture data copy in the host memory.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.
        """
        if not isinstance(name, str): name = str(name)
        if not isinstance(data, np.ndarray): data = np.ascontiguousarray(data, dtype=np.float32)

        if isinstance(addr_mode, str): addr_mode = TextureAddressMode[addr_mode]

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
        if not self._optix.set_texture_2d(name, data.ctypes.data, data.shape[1], data.shape[0], rt_format.value, addr_mode.value, keep_on_host, refresh):
            msg = "Texture 2D %s not uploaded." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def load_texture(self, tex_name: str, file_name: str,
                     rt_format: RtFormat = RtFormat.Float4,
                     prescale: float = 1.0,
                     baseline: float = 0.0,
                     exposure: float = 1.0,
                     gamma: float = 1.0,
                     addr_mode: Union[TextureAddressMode, str] = TextureAddressMode.Wrap,
                     keep_on_host: bool = False,
                     refresh: bool = False) -> None:
        """Load texture from file.

        Parameters
        ----------
        tex_name : string
            Texture name.
        file_name : string
            Source image file.
        rt_format: RtFormat, optional
            Target format of the texture.
        prescale : float, optional
            Scaling factor for color values.
        baseline : float, optional
            Baseline added to color values.
        exposure : float, optional
            Exposure value used in the postprocessing.
        gamma : float, optional
            Gamma value used in the postprocessing.
        addr_mode : TextureAddressMode or string, optional
            Texture addressing mode on edge crossing.
        keep_on_host : bool, optional
            Store texture data copy in the host memory.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.

        Examples
        --------
        >>> optix = TkOptiX()
        >>> optix.load_texture("rainbow", "data/rainbow.jpg") # set gray background
        """
        if isinstance(rt_format, str): rt_format = RtFormat[rt_format]

        if isinstance(addr_mode, str): addr_mode = TextureAddressMode[addr_mode]

        if not self._optix.load_texture_2d(tex_name, file_name, prescale, baseline, exposure, gamma, rt_format.value, addr_mode.value, refresh):
            msg = "Failed on reading texture from file %s." % file_name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)

    def set_normal_tilt(self, name: str, data: Any,
                        mapping: Union[TextureMapping, str] = TextureMapping.Flat,
                        addr_mode: Union[TextureAddressMode, str] = TextureAddressMode.Wrap,
                        keep_on_host: bool = False,
                        refresh: bool = False) -> None:
        """Set normal tilt data.

        Set shading normal tilt according to displacement data for the material ``name``. The ``data``
        has to be a 2D array containing displacement mapping. ``mapping`` determines how the normal tilt
        is calculated from the displacement map (see :class:`plotoptix.enums.TextureMapping`).
        
        Use ``keep_on_host=True`` to make a copy of data in the host memory (in addition to GPU
        memory), this option is required when (small) arrays are going to be saved to JSON
        description of the scene.

        Parameters
        ----------
        name : string
            Object name.
        data : array_like
            Displacement map data.
        mapping : TextureMapping or string, optional
            Mapping mode (see :class:`plotoptix.enums.TextureMapping`).
        addr_mode : TextureAddressMode or string, optional
            Texture addressing mode on edge crossing.
        keep_on_host : bool, optional
            Store texture data copy in the host memory.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.
        """
        if not isinstance(name, str): name = str(name)
        if not isinstance(data, np.ndarray): data = np.ascontiguousarray(data, dtype=np.float32)

        if isinstance(mapping, str): mapping = TextureMapping[mapping]

        if isinstance(addr_mode, str): addr_mode = TextureAddressMode[addr_mode]

        if len(data.shape) != 2:
            msg = "Data shape should be (height,width)."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if data.dtype != np.float32: data = np.ascontiguousarray(data, dtype=np.float32)
        if not data.flags['C_CONTIGUOUS']: data = np.ascontiguousarray(data, dtype=np.float32)

        self._logger.info("Set shading normal tilt map for %s: %d x %d.", name, data.shape[1], data.shape[0])
        if not self._optix.set_normal_tilt(name, data.ctypes.data, data.shape[1], data.shape[0],
                                           mapping.value, addr_mode.value, keep_on_host, refresh):
            msg = "%s normal tilt map not uploaded." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def load_normal_tilt(self, name: str, file_name: str,
                         mapping: Union[TextureMapping, str] = TextureMapping.Flat,
                         addr_mode: Union[TextureAddressMode, str] = TextureAddressMode.Wrap,
                         prescale: float = 1.0,
                         baseline: float = 0.0,
                         refresh: bool = False) -> None:
        """Set normal tilt data.

        Set shading normal tilt according to displacement map loaded from an image file. ``mapping``
        determines how the normal tilt is calculated from the displacement data
        (see :class:`plotoptix.enums.TextureMapping`). Tilt data is stored in the device memory only
        (there is no host copy).

        Parameters
        ----------
        name : string
            Material name.
        file_name : string
            Image file name with the displacement data.
        mapping : TextureMapping or string, optional
            Mapping mode (see :class:`plotoptix.enums.TextureMapping`).
        addr_mode : TextureAddressMode or string, optional
            Texture addressing mode on edge crossing.
        prescale : float, optional
            Scaling factor for displacement values.
        baseline : float, optional
            Baseline added to displacement values.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.
        """
        if not isinstance(name, str): name = str(name)
        if not isinstance(file_name, str): name = str(file_name)

        if isinstance(mapping, str): mapping = TextureMapping[mapping]

        if isinstance(addr_mode, str): addr_mode = TextureAddressMode[addr_mode]

        self._logger.info("Set shading normal tilt map for %s using %s.", name, file_name)
        if not self._optix.load_normal_tilt(name, file_name, mapping.value, addr_mode.value, prescale, baseline, refresh):
            msg = "%s normal tilt map not uploaded." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def set_displacement(self, name: str, data: Any,
                         addr_mode: Union[TextureAddressMode, str] = TextureAddressMode.Wrap,
                         keep_on_host: bool = False,
                         refresh: bool = False) -> None:
        """Set surface displacement data.

        Set displacement data for the object ``name``. Geometry attribute program of the object
        has to be set to :attr:`plotoptix.enums.GeomAttributeProgram.DisplacedSurface`. The ``data``
        has to be a 2D array containing displacement map.

        Use ``keep_on_host=True`` to make a copy of data in the host memory (in addition to GPU
        memory), this option is required when (small) arrays are going to be saved to JSON
        description of the scene.

        Parameters
        ----------
        name : string
            Object name.
        data : array_like
            Displacement map data.
        addr_mode : TextureAddressMode or string, optional
            Texture addressing mode on edge crossing.
        keep_on_host : bool, optional
            Store texture data copy in the host memory.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.
        """
        if not isinstance(name, str): name = str(name)
        if not isinstance(data, np.ndarray): data = np.ascontiguousarray(data, dtype=np.float32)

        if isinstance(addr_mode, str): addr_mode = TextureAddressMode[addr_mode]

        if len(data.shape) != 2:
            msg = "Data shape should be (height,width)."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if data.dtype != np.float32: data = np.ascontiguousarray(data, dtype=np.float32)
        if not data.flags['C_CONTIGUOUS']: data = np.ascontiguousarray(data, dtype=np.float32)

        self._logger.info("Set displacement map for %s: %d x %d.", name, data.shape[1], data.shape[0])
        if not self._optix.set_displacement(name, data.ctypes.data, data.shape[1], data.shape[0],
                                            addr_mode, keep_on_host, refresh):
            msg = "%s displacement map not uploaded." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def load_displacement(self, name: str, file_name: str,
                          prescale: float = 1.0,
                          baseline: float = 0.0,
                          addr_mode: Union[TextureAddressMode, str] = TextureAddressMode.Wrap,
                          refresh: bool = False) -> None:
        """Load surface displacement data from file.

        Load displacement data for the object ``name`` from an image file. Geometry attribute
        program of the object has to be set to :attr:`plotoptix.enums.GeomAttributeProgram.DisplacedSurface`.
        Tilt data is stored in the device memory only (there is no host copy).

        Parameters
        ----------
        name : string
            Object name.
        file_name : string
            Image file name with the displacement data.
        prescale : float, optional
            Scaling factor for displacement values.
        baseline : float, optional
            Baseline added to displacement values.
        addr_mode : TextureAddressMode or string, optional
            Texture addressing mode on edge crossing.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.
        """
        if not isinstance(name, str): name = str(name)
        if not isinstance(file_name, str): name = str(file_name)

        if isinstance(addr_mode, str): addr_mode = TextureAddressMode[addr_mode]

        self._logger.info("Set displacement map for %s using %s.", name, file_name)
        if not self._optix.load_displacement(name, file_name, prescale, baseline, addr_mode, refresh):
            msg = "%s displacement map not uploaded." % name
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
                       prescale: float = 1.0,
                       baseline: float = 0.0,
                       exposure: float = 1.0,
                       gamma: float = 1.0,
                       keep_on_host: bool = False,
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

        Use ``keep_on_host=True`` to make a copy of data in the host memory (in addition
        to GPU memory), this option is required when (small) textures are going to be saved
        to JSON description of the scene.

        Note, color components range is <0; 1>.

        Parameters
        ----------
        bg : Any
            New backgroud color or texture data; single value is a grayscale level,
            RGB color components can be provided as an array-like values, texture
            is provided as an array of shape ``(height, width, n)`` or string
            with the source image file path.
        prescale : float, optional
            Scaling factor for color values.
        baseline : float, optional
            Baseline added to color values.
        exposure : float, optional
            Exposure value used in the postprocessing.
        gamma : float, optional
            Gamma value used in the postprocessing.
        keep_on_host : bool, optional
            Store texture data copy in the host memory.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.

        Examples
        --------
        >>> optix = TkOptiX()
        >>> optix.set_background(0.5) # set gray background
        >>> optix.set_background([0.5, 0.7, 0.9]) # set light bluish background
        """
        if isinstance(bg, str):
            if self._optix.load_texture_2d("bg_texture",
                                           bg, prescale, baseline, exposure, gamma,
                                           RtFormat.Float4.value,
                                           TextureAddressMode.Mirror.value,
                                           refresh):
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
                b[...,:-1] = bg
                bg = b

            if bg.shape[-1] == 4:
                if gamma != 1: bg = np.power(bg, gamma)
                if e != 1: bg = e * bg
                self.set_texture_2d("bg_texture", bg, addr_mode=TextureAddressMode.Mirror, keep_on_host=keep_on_host, refresh=refresh)
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


    def get_param(self, name: str) -> Optional[Any]:
        """Get raytracer parameter.

        Available parameters:

        - ``compute_timeout``
        - ``light_shading``
        - ``max_accumulation_frames``
        - ``min_accumulation_step``
        - ``rt_timeout``

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

        See Also
        --------
        :meth:`plotoptix.NpOptiX.set_param`
        """
        try:
            v = None
            self._padlock.acquire()
            if name == "min_accumulation_step":
                v = self._optix.get_min_accumulation_step()
            elif name == "max_accumulation_frames":
                v = self._optix.get_max_accumulation_frames()
            elif name == "light_shading":
                shading = self._optix.get_light_shading()
                if shading >= 0: v = LightShading(shading)
            elif name == "compute_timeout":
                v = self._optix.get_compute_timeout()
            elif name == "rt_timeout":
                v = self._optix.get_rt_timeout()
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
        """Set raytracer parameter(s).

        Available parameters:

        - ``compute_timeout``: timeout for the computation thread

          Set this parameter if the computations performed in the scene_compute
          callback are longer than the frame ray tracing. See also
          :meth:`plotoptix.NpOptiX.set_scene_compute_cb`.

        - ``light_shading``: light shading mode.

          Use :attr:`plotoptix.enums.LightShading.Hard` for best caustics or
          :attr:`plotoptix.enums.LightShading.Soft` for fast convergence. String
          names ``"Hard"`` and ``"Soft"`` are accepted.
        
          Set mode before adding lights.
        
        - ``max_accumulation_frames``
        
          Number of accumulation frames computed for the scene.
        
        - ``min_accumulation_step``

          Number of accumulation frames computed in a single step (before each
          image refresh).

        - ``rt_timeout``

          Ray tracing timeout. Default value is 30000 (30s).

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

                elif key == "light_shading":
                    if len(self.light_handles) > 0:
                        msg = "Light shading has to be selected before adding lights."
                        self._logger.error(msg)
                        if self._raise_on_error: raise RuntimeError(msg)
                        continue

                    if isinstance(value, str): mode = LightShading[value]
                    else: mode = value

                    self._optix.set_light_shading(mode.value)

                elif key == "compute_timeout":
                    self._optix.set_compute_timeout(int(value))

                elif key == "rt_timeout":
                    self._optix.set_rt_timeout(int(value))

                else:
                    msg = "Unknown parameter " + key
                    self._logger.error(msg)
                    if self._raise_on_error: raise ValueError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise

        finally:
            self._padlock.release()


    def get_scene(self) -> dict:
        """Get dictionary with the scene description.

        Returns a dictionary with the scene description. Geometry objects,
        materials, lights, texture data or file names, cameras, postprocessing
        and scene parameters are included. Callback functions and vieport dimensions
        are not saved. Existing files are overwritten.

        Returns
        -------
        out : dict, optional
            Dictionary with the scene description.
        """
        try:
            self._padlock.acquire()

            s = self._optix.save_scene_to_json()
            if len(s) > 2: return json.loads(s)
            else: return {}

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise

        finally:
            self._padlock.release()

    def _init_scene_metadata(self) -> bool:
        s = self._optix.get_scene_metadata()
        if len(s) > 2: meta = json.loads(s)
        else:
            self._logger.error("Scene loading failed.")
            return False

        self.geometry_data = {} # geometry name to handle dictionary
        self.geometry_names = {}   # geometry handle to name dictionary
        if "Geometry" in meta:
            for key, value in meta["Geometry"].items():
                self.geometry_data[key] = GeometryMeta(key, value["Handle"], value["Size"])
                self.geometry_names[value["Handle"]] = key
        else: return False

        self.camera_handles = {}   # camera name to handle dictionary
        self.camera_names = {}     # camera handle to name dictionary
        if "Cameras" in meta:
            for key, value in meta["Cameras"].items():
                self.camera_handles[key] = value
                self.camera_names[value] = key
        else: return False

        self.light_handles = {}    # light name to handle dictionary
        self.light_names = {}      # light handle to name dictionary
        if  "Lights" in meta:
            for key, value in meta["Lights"].items():
                self.light_handles[key] = value
                self.light_names[value] = key
        else: return False

        return True

    def set_scene(self, scene: dict) -> None:
        """Setup scene using description in provided dictionary.

        Set new scene using provided description (and destroy current scene). Geometry
        objects, materials, lights, texture data or file names, cameras, postprocessing
        and scene parameters are replaced. Callback functions and vieport dimensions are
        preserved.

        Note: locations of external resources loaded from files (e.g. textures) are saved
        as relative paths, ensure your working directory matches these locations.

        Parameters
        ----------
        scene : dict
            Dictionary with the scene description.
        """
        s = json.dumps(scene)
        with self._padlock:
            self._logger.info("Loading new scene from dictionary.")
            if self._optix.load_scene_from_json(s) and self._init_scene_metadata():
                self._logger.info("New scene ready.")
            else:
                msg = "Scene loading failed."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)

    def load_scene(self, file_name: str) -> None:
        """Load scene description from JSON file.

        Load new scene from JSON file (and destroy current scene). Geometry objects,
        materials, lights, texture data or file names, cameras, postprocessing and
        scene parameters are replaced. Callback functions and vieport dimensions are
        preserved.

        Parameters
        ----------
        file_name : str
            Input file name.
        """
        if not os.path.isfile(file_name):
            msg = "File %s not found." % file_name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        wd = os.getcwd()
        if os.path.isabs(file_name):
            d, f = os.path.split(file_name)
            os.chdir(d)
        else:
            f = file_name

        with self._padlock:
            self._logger.info("Loading new scene from file %s.", file_name)
            if self._optix.load_scene_from_file(f) and self._init_scene_metadata():
                self._logger.info("New scene ready.")
            else:
                msg = "Scene loading failed."
                self._logger.error(msg)
                if self._raise_on_error:
                    os.chdir(wd)
                    raise ValueError(msg)

        os.chdir(wd)

    def save_scene(self, file_name: str) -> None:
        """Save scene description to JSON file.

        Save description of the scene to file. Geometry objects, materials, lights,
        texture data or file names, cameras, postprocessing and scene parameters
        are included. Callback functions and vieport dimensions are not saved.
        Existing files are overwritten.

        Parameters
        ----------
        file_name : str
            Output file name.
        """
        try:
            self._padlock.acquire()

            if not self._optix.save_scene_to_file(file_name):
                msg = "Scene not saved."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise

        finally:
            self._padlock.release()


    def save_image(self, file_name: str,
                   bps: Union[ChannelDepth, str] = ChannelDepth.Bps8) -> None:
        """Save current image to file.

        Save current content of the image buffer to a file. Accepted formats,
        recognized by the extension used in the ``file_name``, are:

           - bmp, gif, png, jpg, and tif for 8bps color depth,
           - png (Windows only), and tif for 16bps color depth,
           - tif for 32bps hdr images.

        Existing files are overwritten.

        Parameters
        ----------
        file_name : str
            Output file name.
        bps : ChannelDepth enum or string, optional
            Color depth.

        See Also
        --------
        :class:`plotoptix.enums.ChannelDepth`
        """
        if isinstance(bps, str): bps = ChannelDepth[bps]

        try:
            self._padlock.acquire()

            if bps == ChannelDepth.Bps8:
                ok = self._optix.save_image_to_file(file_name)
            elif bps == ChannelDepth.Bps16:
                ok = self._optix.save_image_to_file_16bps(file_name)
            elif bps == ChannelDepth.Bps32:
                ok = self._optix.save_image_to_file_32bps(file_name)
            else:
                ok = False

            if not ok:
                msg = "Image not saved."
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


    def get_camera_names(self) -> list:
        """Return list of cameras' names.
        """
        return list(self.camera_handles.keys())

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

    def get_camera_glock(self, name: Optional[str] = None) -> Optional[bool]:
        """Get camera gimbal lock state.

        Parameters
        ----------
        name : string, optional
            Name of the camera, use current camera if name not provided.

        Returns
        -------
        out : bool, optional
            Gimbal lock state of the camera or ``None`` if failed on
            accessing camera data.
        """
        if name is not None and not isinstance(name, str): name = str(name)

        name, cam_handle = self.get_camera_name_handle(name)
        if name is None: return None

        return self._optix.get_camera_glock(cam_handle)

    def set_camera_glock(self, state: bool) -> None:
        """Set current camera's gimbal lock.

        Parameters
        ----------
        state : bool
            Gimbal lock state.
        """
        if not self._optix.set_camera_glock(state):
            msg = "Camera gimbal lock not set."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def setup_camera(self, name: str,
                     eye: Optional[Any] = None,
                     target: Optional[Any] = None,
                     up: Any = np.ascontiguousarray([0, 1, 0], dtype=np.float32),
                     cam_type: Union[Camera, str] = Camera.Pinhole,
                     aperture_radius: float = 0.1,
                     aperture_fract: float = 0.15,
                     focal_scale: float = 1.0,
                     chroma_l: float = 0.05,
                     chroma_t: float = 0.01,
                     fov: float = 35.0,
                     blur: float = 1,
                     glock: bool = False,
                     textures: Optional[Any] = None,
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
            Focusing distance, relative to ``eye - target`` length.
        chroma_l : float, optional
            Longitudinal chromatic aberration strength, relative variation of the focusing
            distance for different wavelengths. Use be a small positive value << 1.0. Default
            is ``0.05``, use ``0.0`` for no aberration.
        chroma_t : float, optional
            Transverse chromatic aberration strength, relative variation of the lens
            magnification for different wavelengths. Use be a small positive value << 1.0.
            Default is ``0.01``, use ``0.0`` for no aberration.
        fov : float, optional
            Field of view in degrees.
        blur : float, optional
            Weight of the new frame in averaging with already accumulated frames.
            Range is (0; 1>, lower values result with a higher motion blur, value
            1.0 turns off the blur (default). Cannot be changed after construction.
        glock : bool, optional
            Gimbal lock state of the new camera.
        textures : array_like, optional
            List of textures used by the camera ray generation program.
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

        tex_list = ""
        if textures is not None: tex_list = ";".join(textures)

        h = self._optix.setup_camera(name, cam_type.value,
                                     eye_ptr, target_ptr, up.ctypes.data,
                                     aperture_radius, aperture_fract,
                                     focal_scale, chroma_l, chroma_t,
                                     fov, blur, glock,
                                     tex_list, make_current)
        if h > 0:
            self._logger.info("Camera %s handle: %d.", name, h)
            self.camera_handles[name] = h
            self.camera_names[h] = name
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

        if self._optix.update_camera(name, eye_ptr, target_ptr, up_ptr,
                                     aperture_radius, focal_scale, fov):
            self._logger.info("Camera %s updated.", name)
        else:
            msg = "Camera %s update failed." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def get_current_camera(self) -> Optional[str]:
        """Get current camera name.

        Returns
        -------
        out : string, optional
            Name of the current camera or ``None`` if camera not set.
        """
        cam_handle = self._optix.get_current_camera()
        if cam_handle == 0:
            msg = "Current camera is not set."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None

        if cam_handle not in self.camera_names:
            msg = "Camera handle %d does not exists." % cam_handle
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None

        return self.camera_names[cam_handle]

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

        if self._optix.set_current_camera(name):
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

    def camera_move_by(self, shift: Tuple[float, float, float]) -> None:
        """Move current camera in the world coordinates.

        Parameters
        ----------
        shift : tuple (float, float, float)
            (X, Y, Z) shift vector.
        """
        if not self._optix.move_camera_by(shift[0], shift[1], shift[2]):
            msg = "Camera move failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def camera_move_by_local(self, shift: Tuple[float, float, float]) -> None:
        """Move current camera in the camera coordinates.

        Camera coordinates are: X to the right, Y up, Z towards camera.

        Parameters
        ----------
        shift : tuple (float, float, float)
            (X, Y, Z) shift vector.
        """
        if not self._optix.move_camera_by_local(shift[0], shift[1], shift[2]):
            msg = "Camera move failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def camera_rotate_by(self,
                         rot: Tuple[float, float, float],
                         center: Tuple[float, float, float]) -> None:
        """Rotate current camera in the world coordinates about the center.

        Rotation is done the world coordinates about Y, X, and then Z axis,
        by the angles provided with ``rot = (rx, ry, rz)`` parameter.

        Parameters
        ----------
        rot : tuple (float, float, float)
            Rotation around (X, Y, Z) axis.
        center : tuple (float, float, float)
            Rotation center.
        """
        if not self._optix.rotate_camera_by(rot[0], rot[1], rot[2], center[0], center[1], center[2]):
            msg = "Camera rotate failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def camera_rotate_by_local(self,
                               rot: Tuple[float, float, float],
                               center: Tuple[float, float, float]) -> None:
        """Rotate current camera in the camera coordinates about the center.

        Rotation is done the camera coordinates about Y (camera up, yaw),
        X (camera right, pitch), and then Z (towards camera, roll) axis,
        by the angles provided with ``rot = (rx, ry, rz)`` parameter.

        Parameters
        ----------
        rot : tuple (float, float, float)
            Rotation around (X, Y, Z) axis.
        center : tuple (float, float, float)
            Rotation center.
        """
        if not self._optix.rotate_camera_by_local(rot[0], rot[1], rot[2], center[0], center[1], center[2]):
            msg = "Camera rotate local failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def camera_rotate_eye(self, rot: Tuple[float, float, float]) -> None:
        """Rotate current camera eye about the target point in the world coordinates.

        Rotation is done the world coordinates about Y, X, and then Z axis,
        by the angles provided with ``rot = (rx, ry, rz)`` parameter.

        Parameters
        ----------
        rot : tuple (float, float, float)
            Rotation around (X, Y, Z) axis.
        """
        if not self._optix.rotate_camera_eye_by(rot[0], rot[1], rot[2]):
            msg = "Camera rotate eye failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def camera_rotate_eye_local(self, rot: Tuple[float, float, float]) -> None:
        """Rotate current camera eye about the target point in the camera coordinates.

        Rotation is done the camera coordinates about Y (camera up, yaw),
        X (camera right, pitch), and then Z (towards camera, roll) axis,
        by the angles provided with ``rot = (rx, ry, rz)`` parameter.

        Parameters
        ----------
        rot : tuple (float, float, float)
            Rotation around (X, Y, Z) axis.
        """
        if not self._optix.rotate_camera_eye_by_local(rot[0], rot[1], rot[2]):
            msg = "Camera rotate eye local failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def camera_rotate_target(self, rot: Tuple[float, float, float]) -> None:
        """Rotate current camera target about the eye point in the world coordinates.

        Rotation is done the world coordinates about Y, X, and then Z axis,
        by the angles provided with ``rot = (rx, ry, rz)`` parameter.

        Parameters
        ----------
        rot : tuple (float, float, float)
            Rotation around (X, Y, Z) axis.
        """
        if not self._optix.rotate_camera_tgt_by(rot[0], rot[1], rot[2]):
            msg = "Camera rotate target failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def camera_rotate_target_local(self, rot: Tuple[float, float, float]) -> None:
        """Rotate current camera target about the eye point in the camera coordinates.

        Rotation is done the camera coordinates about Y (camera up, yaw),
        X (camera right, pitch), and then Z (towards camera, roll) axis,
        by the angles provided with ``rot = (rx, ry, rz)`` parameter.

        Parameters
        ----------
        rot : tuple (float, float, float)
            Rotation around (X, Y, Z) axis.
        """
        if not self._optix.rotate_camera_tgt_by_local(rot[0], rot[1], rot[2]):
            msg = "Camera rotate target failed."
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)


    def get_light_names(self) -> list:
        """Return list of lights' names.
        """
        return list(self.light_handles.keys())

    def get_light_shading(self) -> Optional[LightShading]:
        """Get light shading mode.

        Deprecated, use ``get_param("light_shading")`` instead.

        Returns
        ----------
        out : LightShading or None
            Light shading mode. ``None`` is returned if function could
            not read the mode from the raytracer.

        See Also
        --------
        :meth:`plotoptix.NpOptiX.get_param`
        """
        self._logger.warn("Deprecated, use get_param(\"light_shading\") instead.")
        return self.get_param("light_shading")

    def set_light_shading(self, mode: Union[LightShading, str]) -> None:
        """Set light shading mode.

        Deprecated, use ``set_param(light_shading=mode)`` instead.

        See Also
        --------
        :meth:`plotoptix.NpOptiX.set_param`
        """
        self._logger.warn("Deprecated, use set_param(light_shading=mode) instead.")
        self.set_param(light_shading=mode)

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
        self._optix.get_light_pos(name, pos.ctypes.data)
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
        self._optix.get_light_color(name, col.ctypes.data)
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
        self._optix.get_light_u(name, u.ctypes.data)
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
        self._optix.get_light_v(name, v.ctypes.data)
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

        return self._optix.get_light_r(name)

    def get_light(self, name: str) -> Optional[dict]:
        """Get light source parameters.

        Parameters
        ----------
        name : string
            Name of the light source.

        Returns
        -------
        out : dict, optional
            Dictionary of the light source parameters or ``None`` if
            failed on accessing the data.
        """
        if name is None: raise ValueError()

        if not isinstance(name, str): name = str(name)

        if name not in self.light_handles:
            msg = "Light %s does not exists." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return None

        s = self._optix.get_light(name)
        if len(s) > 2: return json.loads(s)
        else:
            msg = "Failed on reading light %s." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)
            return None

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

        h = self._optix.setup_spherical_light(name, pos.ctypes.data, color.ctypes.data,
                                              radius, in_geometry)
        if h != 0:
            self._logger.info("Light %s handle: %d.", name, h)
            self.light_handles[name] = h
            self.light_names[h] = name

            if autofit:
                self.light_fit(name, camera=cam_name)
        else:
            msg = "Light %s setup failed." % name
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

        h = self._optix.setup_parallelogram_light(name, pos.ctypes.data, color.ctypes.data,
                                                  u.ctypes.data, v.ctypes.data, in_geometry)
        if h != 0:
            self._logger.info("Light %s handle: %d.", name, h)
            self.light_handles[name] = h
            self.light_names[h] = name

            if autofit:
                self.light_fit(name, camera=cam_name)
        else:
            msg = "Light %s setup failed." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def setup_area_light(self, name: str, center: Any, target: Any, u: float, v: float,
                         color: Any = 10 * np.ascontiguousarray([1, 1, 1], dtype=np.float32),
                         in_geometry: bool = True) -> None:
        """Setup new area (parallelogram) light.

        Convenience method to setup parallelogram light with ``center`` and ``target`` 3D points,
        and scalar lengths of sides ``u`` and ``v``.

        Parameters
        ----------
        name : string
            Name of the new light.
        center : array_like
            3D position of the light center.
        target : array_like
            3D position of the light target.
        u : float
            Horizontal side length.
        v : float
            Vertical side length.
        color : Any, optional
            RGB color of the light; single value is gray, array_like is RGB
            color components. Color value range is (0; inf) as it means the
            light intensity.
        in_geometry: bool, optional
            Visible in the scene if set to ``True``.
        """
        center = _make_contiguous_vector(center, 3)
        target = _make_contiguous_vector(target, 3)

        n = target - center
        n = n / np.linalg.norm(n)

        uvec = np.cross(n, [0, 1, 0])
        uvec = uvec / np.linalg.norm(uvec)

        vvec = np.cross(uvec, n)
        vvec = vvec / np.linalg.norm(vvec)

        uvec *= -u
        vvec *= v

        pos = center - 0.5 * (vvec + uvec)

        self.setup_parallelogram_light(name, pos=pos, color=color, u=uvec, v=vvec, in_geometry=in_geometry)

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
        name : string
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

        if self._optix.update_light(name,
                                    pos_ptr, color_ptr,
                                    radius, u_ptr, v_ptr):
            self._logger.info("Light %s updated.", name)
        else:
            msg = "Light %s update failed." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

    def update_area_light(self, name: str,
                          center: Optional[Any] = None, target: Optional[Any] = None,
                          u: Optional[float] = None, v: Optional[float] = None,
                          color: Optional[Any] = None) -> None:
        """Setup new area (parallelogram) light.

        Convenience method to update parallelogram light with ``center`` and ``target`` 3D points,
        and scalar lengths of sides ``u`` and ``v``.

        Parameters
        ----------
        name : string
            Name of the new light.
        center : array_like, optional
            3D position of the light center.
        target : array_like, optional
            3D position of the light target.
        u : float, optional
            Horizontal side length.
        v : float, optional
            Vertical side length.
        color : Any, optional
            RGB color of the light; single value is gray, array_like is RGB
            color components. Color value range is (0; inf) as it means the
            light intensity.
        """
        if name is None: raise ValueError()

        if not isinstance(name, str): name = str(name)

        if name not in self.light_handles:
            msg = "Light %s does not exists." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if u is None:
            u = np.linalg.norm(self.get_light_u(name))

        if v is None:
            v = np.linalg.norm(self.get_light_v(name))

        if center is None:
            center = self.get_light_pos(name) + 0.5 * (self.get_light_u(name) + self.get_light_v(name))

        if target is None:
            n = np.cross(self.get_light_u(name), self.get_light_v(name))
            target = center + 100 * n

        center = _make_contiguous_vector(center, 3)
        target = _make_contiguous_vector(target, 3)

        n = target - center
        n = n / np.linalg.norm(n)

        uvec = np.cross(n, [0, 1, 0])
        uvec = uvec / np.linalg.norm(uvec)

        vvec = np.cross(uvec, n)
        vvec = vvec / np.linalg.norm(vvec)

        uvec *= -u
        vvec *= v

        pos = center - 0.5 * (vvec + uvec)

        self.update_light(name, pos=pos, color=color, u=uvec, v=vvec)

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
        if not light in self.light_handles:
            msg = "Light %s not found." % light
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)

        cam_handle = 0
        if camera is not None:
            if not isinstance(camera, str): camera = str(camera)
            if camera in self.camera_handles:
                cam_handle = self.camera_handles[camera]

        horizontal_rot = math.pi * horizontal_rot / 180.0
        vertical_rot = math.pi * vertical_rot / 180.0

        self._optix.fit_light(light, cam_handle, horizontal_rot, vertical_rot, dist_scale)


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

        Note: for maximum performance, setup only those materials
        you need in the scene.

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

    def update_material_texture(self, name: str, data: Any, idx: int = 0, keep_on_host: bool = False, refresh: bool = False) -> None:
        """Update material texture data.

        Update texture content/size for material ``name`` data. Texture format has to be RGBA,
        width/height are deduced from the ``data`` array shape. Use ``keep_on_host=True``
        to make a copy of data in the host memory (in addition to GPU memory), this
        option is required when (small) textures are going to be saved to JSON description
        of the scene.

        Parameters
        ----------
        name : string
            Material name.
        data : array_like
            Texture data.
        idx : int, optional
            Texture index, the first texture if the default is left.
        keep_on_host : bool, optional
            Store texture data copy in the host memory.
        refresh : bool, optional
            Set to ``True`` if the image should be re-computed.
        """
        if not isinstance(name, str): name = str(name)
        if not isinstance(data, np.ndarray): data = np.ascontiguousarray(data, dtype=np.float32)

        if len(data.shape) != 3 or data.shape[-1] != 4:
            msg = "Material texture shape should be (height,width,4)."
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if data.dtype != np.float32: data = np.ascontiguousarray(data, dtype=np.float32)
        if not data.flags['C_CONTIGUOUS']: data = np.ascontiguousarray(data, dtype=np.float32)

        self._logger.info("Set material %s texture %d: %d x %d.", name, idx, data.shape[1], data.shape[0])
        if not self._optix.set_material_texture(name, idx, data.ctypes.data, data.shape[1], data.shape[0], RtFormat.Float4.value, keep_on_host, refresh):
            msg = "Material %s texture not uploaded." % name
            self._logger.error(msg)
            if self._raise_on_error: raise RuntimeError(msg)


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


    def set_data(self, name: str, pos: Any,
                 r: Any = np.ascontiguousarray([0.05], dtype=np.float32),
                 c: Any = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32),
                 u: Optional[Any] = None, v: Optional[Any] = None, w: Optional[Any] = None,
                 geom: Union[Geometry, str] = Geometry.ParticleSet,
                 geom_attr: Union[GeomAttributeProgram, str] = GeomAttributeProgram.Default,
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
            U vector(s) of parallelograms / parallelepipeds / tetrahedrons / textured particles.
            Single vector sets const. value for all primitives.
        v : array_like, optional
            V vector(s) of parallelograms / parallelepipeds / tetrahedrons / textured particles.
            Single vector sets const. value for all primitives.
        w : array_like, optional
            W vector(s) of parallelepipeds / tetrahedrons. Single vector sets const.
            value for all primitives.
        geom : Geometry enum or string, optional
            Geometry of primitives (ParticleSet, Tetrahedrons, ...). See :class:`plotoptix.enums.Geometry`
            enum.
        geom_attr : GeomAttributeProgram enum or string, optional
            Geometry attributes program. See :class:`plotoptix.enums.GeomAttributeProgram` enum.
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
        if isinstance(geom_attr, str): geom_attr = GeomAttributeProgram[geom_attr]

        if name in self.geometry_data:
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
        c = np.ascontiguousarray(c, dtype=np.float32)
        if c.shape == (1,):
            c = np.ascontiguousarray([c[0], c[0], c[0]], dtype=np.float32)
            col_const_ptr = c.ctypes.data
            col_ptr = 0
        elif c.shape == (3,):
            col_const_ptr = c.ctypes.data
            col_ptr = 0            
        else:
            c = _make_contiguous_3d(c, n=n_primitives, extend_scalars=True)
            assert c.shape == pos.shape, "Colors and data points shapes must be the same."
            if c is not None: col_ptr = c.ctypes.data
            else: col_ptr = 0
            col_const_ptr = 0

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

        elif geom == Geometry.ParticleSetTextured:
            if r is None:
                msg = "ParticleSetTextured setup failed, radii data is missing."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

            if (u is None) or (v is None):
                if r is None:
                    msg = "ParticleSetTextured setup failed, need U / V vectors or radii data."
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

        elif geom == Geometry.SegmentChain:
            if n_primitives < 2:
                msg = "SegmentChain requires at least 2 data points."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

            if c is None:
                msg = "SegmentChain setup failed, colors data is missing."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

            if r is None:
                msg = "SegmentChain setup failed, radii data is missing."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

        elif geom == Geometry.BSplineQuad:
            if n_primitives < 3:
                msg = "BSplineQuad requires at least 3 data points."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

            if c is None:
                msg = "BSplineQuad setup failed, colors data is missing."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

            if r is None:
                msg = "BSplineQuad setup failed, radii data is missing."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

        elif geom == Geometry.BSplineCubic:
            if n_primitives < 4:
                msg = "BSplineCubic requires at least 4 data points."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

            if c is None:
                msg = "BSplineCubic setup failed, colors data is missing."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                is_ok = False

            if r is None:
                msg = "BSplineCubic setup failed, radii data is missing."
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
                g_handle = self._optix.setup_geometry(geom.value, geom_attr.value, name, mat, rnd, n_primitives,
                                                      pos_ptr, col_const_ptr, col_ptr, radii_ptr, u_ptr, v_ptr, w_ptr)

                if g_handle > 0:
                    self._logger.info("...done, handle: %d", g_handle)
                    self.geometry_data[name] = GeometryMeta(name, g_handle, n_primitives)
                    self.geometry_names[g_handle] = name
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
                    mat: Optional[str] = None,
                    pos: Optional[Any] = None, c: Optional[Any] = None, r: Optional[Any] = None,
                    u: Optional[Any] = None, v: Optional[Any] = None, w: Optional[Any] = None) -> None:
        """Update data of an existing geometry.

        Note that on data size changes (``pos`` array size different than provided with :meth:`plotoptix.NpOptiX.set_data`)
        also other properties must be provided matching the new size, otherwise default values are used.

        Parameters
        ----------
        name : string
            Name of the geometry.
        mat : string, optional
            Material name.
        pos : array_like, optional
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
            U vector(s) of parallelograms / parallelepipeds / tetrahedrons / textured particles.
            Single vector sets const. value for all primitives.
        v : array_like, optional
            V vector(s) of parallelograms / parallelepipeds / tetrahedrons / textured particles.
            Single vector sets const. value for all primitives.
        w : array_like, optional
            W vector(s) of parallelepipeds / tetrahedrons. Single vector sets const.
            value for all primitives.
        """
        if name is None: raise ValueError()
        if not isinstance(name, str): name = str(name)

        if mat is None: mat = ""

        if not name in self.geometry_data:
            msg = "Geometry %s does not exists yet, use set_data() instead." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        n_primitives = self.geometry_data[name]._size
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
            size_changed = (n_primitives != self.geometry_data[name]._size)
            pos_ptr = pos.ctypes.data

        # Prepare colors data
        col_const_ptr = 0
        col_ptr = 0

        if size_changed and c is None:
            c = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32)
        elif c is not None:
            c = np.ascontiguousarray(c, dtype=np.float32)

        if c is not None:
            if c.shape == (1,):
                c = np.ascontiguousarray([c[0], c[0], c[0]], dtype=np.float32)
                col_const_ptr = c.ctypes.data
            elif c.shape == (3,):
                col_const_ptr = c.ctypes.data           
            else:
                c = _make_contiguous_3d(c, n=n_primitives, extend_scalars=True)
                if c is not None: col_ptr = c.ctypes.data

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
            g_handle = self._optix.update_geometry(name, mat, n_primitives,
                                                   pos_ptr, col_const_ptr, col_ptr, radii_ptr,
                                                   u_ptr, v_ptr, w_ptr)

            if (g_handle > 0) and (g_handle == self.geometry_data[name]._handle):
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_data[name]._size = n_primitives
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
                    r: Any = np.ascontiguousarray([0.05], dtype=np.float32),
                    c: Any = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32),
                    normals: Optional[Any] = None,
                    range_x: Optional[Tuple[float, float]] = None,
                    range_z: Optional[Tuple[float, float]] = None,
                    floor_y: Optional[float] = None,
                    floor_c: Optional[Any] = None,
                    geom: Union[Geometry, str] = Geometry.Mesh,
                    mat: str = "diffuse",
                    make_normals: bool = False) -> None:
        """Create new surface geometry for the 2D dataset.

        Data is provided as 2D array of :math:`z = f(x, y)` values, with the shape ``(n, m)``,
        where ``n`` and ``m`` are at least 2. Additional data features can be
        visualized with color (array of RGB values, shape ``(n, m, 3)``).
        
        Convention of vertical Y and horizontal XZ plane is adopted.

        Parameters
        ----------
        name : string
            Name of the new surface geometry.
        pos : array_like
            Z values of data points.
        r : Any, optional
            Radii of vertices for the :attr:`plotoptix.enums.Geometry.Graph` geometry,
            interpolated along the wireframe edges. Single value sets constant radius
            for all vertices.
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
        geom : Geometry enum or string, optional
            Geometry of the surface, only :attr:`plotoptix.enums.Geometry.Mesh` or
            :attr:`plotoptix.enums.Geometry.Graph` are supported.
        mat : string, optional
            Material name.
        make_normals : bool, optional
            Calculate normals for data points, if not provided with ``normals``
            argument. Normals of all triangles attached to the point are averaged.
        """
        if name is None: raise ValueError()
        if not isinstance(name, str): name = str(name)

        if name in self.geometry_data:
            msg = "Geometry %s already exists, use update_data_2d() instead." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if isinstance(geom, str): geom = Geometry[geom]
        if not geom in [Geometry.Mesh, Geometry.Graph]:
            msg = "Geometry type %s not supported by the surface plot." % geom.name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if not isinstance(pos, np.ndarray): pos = np.ascontiguousarray(pos, dtype=np.float32)
        assert len(pos.shape) == 2 and pos.shape[0] > 1 and pos.shape[1] > 1, "Required vertex data shape is (z,x), where z >= 2 and x >= 2."
        if pos.dtype != np.float32: pos = np.ascontiguousarray(pos, dtype=np.float32)
        if not pos.flags['C_CONTIGUOUS']: pos = np.ascontiguousarray(pos, dtype=np.float32)
        pos_ptr = pos.ctypes.data

        if r is not None and geom == Geometry.Graph:
            if not isinstance(r, np.ndarray): r = np.ascontiguousarray(r, dtype=np.float32)
            if r.dtype != np.float32: r = np.ascontiguousarray(r, dtype=np.float32)
            if len(r.shape) > 1 or r.shape[0] > 1:
                assert r.shape == pos.shape[:2], "Radii shape must be (v,u), with u and v matching the surface points shape."
            if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
        if r is not None and geom == Geometry.Graph:
            if r.shape[0] == 1:
                r = np.full(pos.shape[:2], r[0], dtype=np.float32)
                if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
            if r.shape != pos.shape[:2]:
                msg = "Radii (r) shape does not match the shape of preceding data arguments."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                return
            radii_ptr = r.ctypes.data
        else: radii_ptr = 0


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
            if c.shape == (3,):
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
                if floor_c.shape == (3,):
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
            g_handle = self._optix.setup_surface(geom.value, name, mat, pos.shape[1], pos.shape[0], pos_ptr, radii_ptr, n_ptr, c_ptr, cl_ptr,
                                                 range_x[0], range_x[1], range_z[0], range_z[1], floor_y, make_normals)

            if g_handle > 0:
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_data[name] = GeometryMeta(name, g_handle, pos.shape[0] * pos.shape[1])
                self.geometry_names[g_handle] = name
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
                       mat: Optional[str] = None,
                       pos: Optional[Any] = None,
                       r: Optional[Any] = None,
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
        mat : string, optional
            Material name.
        pos : array_like, optional
            Z values of data points.
        r : Any, optional
            Radii of vertices for the :attr:`plotoptix.enums.Geometry.Graph` geometry,
            interpolated along the wireframe edges. Single value sets constant radius
            for all vertices.
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

        if mat is None: mat = ""

        if not name in self.geometry_data:
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

        if size_changed and r is None:
            r = np.ascontiguousarray([0.05], dtype=np.float32)
        if r is not None:
            if not isinstance(r, np.ndarray): r = np.ascontiguousarray(r, dtype=np.float32)
            if r.dtype != np.float32: r = np.ascontiguousarray(r, dtype=np.float32)
            if len(r.shape) > 1 or r.shape[0] > 1:
                assert r.shape == size_xz, "Radii shape must be (x,z), with x and z matching the data points shape."
            if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
        radii_ptr = 0
        if r is not None:
            if r.shape[0] == 1:
                r = np.full(size_xz, r[0], dtype=np.float32)
                if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
            if size_xz != r.shape:
                msg = "Radii (r) shape does not match the number of data points."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                return
            radii_ptr = r.ctypes.data

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
            g_handle = self._optix.update_surface(name, mat, size_xz[1], size_xz[0],
                                                  pos_ptr, radii_ptr, n_ptr, c_ptr, cl_ptr,
                                                  range_x[0], range_x[1], range_z[0], range_z[1],
                                                  floor_y)

            if (g_handle > 0) and (g_handle == self.geometry_data[name]._handle):
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_data[name]._size = size_xz[0] * size_xz[1]
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
                    r: Any = np.ascontiguousarray([0.05], dtype=np.float32),
                    c: Any = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32),
                    normals: Optional[Any] = None,
                    geom: Union[Geometry, str] = Geometry.Mesh,
                    mat: str = "diffuse",
                    wrap_u: bool = False,
                    wrap_v: bool = False,
                    make_normals: bool = False) -> None:
        """Create new (parametric) surface geometry.

        Data is provided as 2D array of :math:`[x, y, z] = f(u, v)` values, with the shape
        ``(n, m, 3)``, where ``n`` and ``m`` are at least 2. Additional data features can be
        visualized with color (array of RGB values, shape ``(n, m, 3)``) or wireframe thickness
        if the :attr:`plotoptix.enums.Geometry.Graph` geometry is used.
        
        Parameters
        ----------
        name : string
            Name of the new surface geometry.
        pos : array_like
            XYZ values of surface points.
        r : Any, optional
            Radii of vertices for the :attr:`plotoptix.enums.Geometry.Graph` geometry,
            interpolated along the wireframe edges. Single value sets constant radius
            for all vertices.
        c : Any, optional
            Colors of surface points. Single value means a constant gray level.
            3-component array means a constant RGB color. Array of the shape
            ``(n, m, 3)`` will set individual color for each surface point,
            interpolated between points; ``n`` and ``m`` have to be the same
            as in the surface points shape.
        normals : array_like, optional
            Normal vectors at provided surface points. Array shape has to be ``(n, m, 3)``,
            with ``n`` and ``m`` the same as in the surface points shape.
        geom : Geometry enum or string, optional
            Geometry of the surface, only :attr:`plotoptix.enums.Geometry.Mesh` or
            :attr:`plotoptix.enums.Geometry.Graph` are supported.
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

        if isinstance(geom, str): geom = Geometry[geom]
        if not geom in [Geometry.Mesh, Geometry.Graph]:
            msg = "Geometry type %s not supported by the parametric surface." % geom.name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if name in self.geometry_data:
            msg = "Geometry %s already exists, use update_surface() instead." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if not isinstance(pos, np.ndarray): pos = np.ascontiguousarray(pos, dtype=np.float32)
        assert len(pos.shape) == 3 and pos.shape[0] > 1 and pos.shape[1] > 1 and pos.shape[2] == 3, "Required surface points shape is (v,u,3), where u >= 2 and v >= 2."
        if pos.dtype != np.float32: pos = np.ascontiguousarray(pos, dtype=np.float32)
        if not pos.flags['C_CONTIGUOUS']: pos = np.ascontiguousarray(pos, dtype=np.float32)
        pos_ptr = pos.ctypes.data

        if r is not None and geom == Geometry.Graph:
            if not isinstance(r, np.ndarray): r = np.ascontiguousarray(r, dtype=np.float32)
            if r.dtype != np.float32: r = np.ascontiguousarray(r, dtype=np.float32)
            if len(r.shape) > 1 or r.shape[0] > 1:
                assert r.shape == pos.shape[:2], "Radii shape must be (v,u), with u and v matching the surface points shape."
            if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
        if r is not None and geom == Geometry.Graph:
            if r.shape[0] == 1:
                r = np.full(pos.shape[:2], r[0], dtype=np.float32)
                if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
            if r.shape != pos.shape[:2]:
                msg = "Radii (r) shape does not match the shape of preceding data arguments."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                return
            radii_ptr = r.ctypes.data
        else: radii_ptr = 0

        n_ptr = 0
        if normals is not None and geom == Geometry.Mesh:
            if not isinstance(normals, np.ndarray): normals = np.ascontiguousarray(normals, dtype=np.float32)
            assert normals.shape == pos.shape, "Normals shape must be (v,u,3), with u and v matching the surface points shape."
            if normals.dtype != np.float32: normals = np.ascontiguousarray(normals, dtype=np.float32)
            if not normals.flags['C_CONTIGUOUS']: normals = np.ascontiguousarray(normals, dtype=np.float32)
            n_ptr = normals.ctypes.data
            make_normals = False

        c_ptr = 0
        c_const_ptr = 0
        if c is not None:
            if isinstance(c, float) or isinstance(c, int): c = np.full(3, c, dtype=np.float32)
            if not isinstance(c, np.ndarray): c = np.ascontiguousarray(c, dtype=np.float32)
            if c.dtype != np.float32: c = np.ascontiguousarray(c, dtype=np.float32)
            if not c.flags['C_CONTIGUOUS']: c = np.ascontiguousarray(c, dtype=np.float32)
            if c.shape == (3,):
                c_const_ptr = c.ctypes.data
            elif c.shape == pos.shape:
                c_ptr = c.ctypes.data
            else:
                msg = "Colors shape must be (3,) or (v,u,3), with u and v matching the surface points shape."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

        try:
            self._padlock.acquire()
            self._logger.info("Setup surface %s...", name)
            g_handle = self._optix.setup_psurface(geom.value, name, mat, pos.shape[1], pos.shape[0], pos_ptr, radii_ptr, n_ptr, c_const_ptr, c_ptr, wrap_u, wrap_v, make_normals)

            if g_handle > 0:
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_data[name] = GeometryMeta(name, g_handle, pos.shape[0] * pos.shape[1])
                self.geometry_names[g_handle] = name
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
                       mat: Optional[str] = None,
                       pos: Optional[Any] = None,
                       r: Optional[Any] = None,
                       c: Optional[Any] = None,
                       normals: Optional[Any] = None) -> None:
        """Update surface geometry data or properties.

        Parameters
        ----------
        name : string
            Name of the surface geometry.
        mat : string, optional
            Material name.
        pos : array_like, optional
            XYZ values of surface points.
        r : Any, optional
            Radii of vertices for the :attr:`plotoptix.enums.Geometry.Graph` geometry,
            interpolated along the edges. Single value sets constant radius for all vertices.
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

        if mat is None: mat = ""

        if not name in self.geometry_data:
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
        size_uv3 = (s_v.value, s_u.value, 3)
        size_uv1 = (s_v.value, s_u.value)
        size_changed = False

        pos_ptr = 0
        if pos is not None:
            if not isinstance(pos, np.ndarray): pos = np.ascontiguousarray(pos, dtype=np.float32)
            assert len(pos.shape) == 3 and pos.shape[0] > 1 and pos.shape[1] > 1 and pos.shape[2] == 3, "Required vertex data shape is (v,u,3), where u >= 2 and v >= 2."
            if pos.dtype != np.float32: pos = np.ascontiguousarray(pos, dtype=np.float32)
            if not pos.flags['C_CONTIGUOUS']: pos = np.ascontiguousarray(pos, dtype=np.float32)
            if pos.shape != size_uv3: size_changed = True
            size_uv3 = pos.shape
            pos_ptr = pos.ctypes.data

        if size_changed and r is None:
            r = np.ascontiguousarray([0.05], dtype=np.float32)
        if r is not None:
            if not isinstance(r, np.ndarray): r = np.ascontiguousarray(r, dtype=np.float32)
            if r.dtype != np.float32: r = np.ascontiguousarray(r, dtype=np.float32)
            if len(r.shape) > 1 or r.shape[0] > 1:
                assert r.shape == size_uv1, "Radii shape must be (v,u), with u and v matching the surface points shape."
            if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
        radii_ptr = 0
        if r is not None:
            if r.shape[0] == 1:
                r = np.full(size_uv1, r[0], dtype=np.float32)
                if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
            if size_uv1 != r.shape:
                msg = "Radii (r) shape does not match the number of surface points."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                return
            radii_ptr = r.ctypes.data

        c_ptr = 0
        c_const_ptr = 0
        if size_changed and c is None: c = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32)
        if c is not None:
            if isinstance(c, float) or isinstance(c, int): c = np.full(3, c, dtype=np.float32)
            if not isinstance(c, np.ndarray): c = np.ascontiguousarray(c, dtype=np.float32)
            if c.dtype != np.float32: c = np.ascontiguousarray(c, dtype=np.float32)
            if not c.flags['C_CONTIGUOUS']: c = np.ascontiguousarray(c, dtype=np.float32)
            if c.shape == (3,):
                c_const_ptr = c.ctypes.data
            elif c.shape == size_uv3:
                c_ptr = c.ctypes.data
            else:
                msg = "Colors shape must be (3,) or (v,u,3), with u and v matching the surface points shape."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

        n_ptr = 0
        if normals is not None:
            if not isinstance(normals, np.ndarray): normals = np.ascontiguousarray(normals, dtype=np.float32)
            assert normals.shape == size_uv3, "Normals shape must be (v,u,3), with u and v matching the surface points shape."
            if normals.dtype != np.float32: normals = np.ascontiguousarray(normals, dtype=np.float32)
            if not normals.flags['C_CONTIGUOUS']: normals = np.ascontiguousarray(normals, dtype=np.float32)
            n_ptr = normals.ctypes.data

        try:
            self._padlock.acquire()
            self._logger.info("Update surface %s, size (%d, %d)...", name, size_uv1[1], size_uv1[0])
            g_handle = self._optix.update_psurface(name, mat, size_uv1[1], size_uv1[0], pos_ptr, radii_ptr, n_ptr, c_const_ptr, c_ptr)

            if (g_handle > 0) and (g_handle == self.geometry_data[name]._handle):
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_data[name]._size = size_uv1[0] * size_uv1[1]
            else:
                msg = "Geometry update failed."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                
        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise
        finally:
            self._padlock.release()

    def set_graph(self, name: str, pos: Any, edges: Any,
                  r: Any = np.ascontiguousarray([0.05], dtype=np.float32),
                  c: Any = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32),
                  mat: str = "diffuse") -> None:
        """Create new graph (mesh wireframe) geometry.

        Data is provided as vertices :math:`[x, y, z]`, with the shape ``(n, 3)``, and edges
        (doublets of vertex indices), with the shape ``(n, 2)`` or ``(m)`` where :math:`m = 2*n`.
        Data features can be visualized with colors (array of RGB values assigned to the graph
        vertices, shape ``(n, 3)``) and/or vertex radii.

        Parameters
        ----------
        name : string
            Name of the new graph geometry.
        pos : array_like
            XYZ values of the graph vertices.
        edges : array_like
            Graph edges as indices (doublets) to vertices in the ``pos`` array.
        r : Any, optional
            Radii of vertices, interpolated along the edges. Single value sets constant
            radius for all vertices.
        c : Any, optional
            Colors of the graph vertices. Single value means a constant gray level.
            3-component array means a constant RGB color. Array of the shape
            ``(n, 3)`` will set individual color for each vertex, interpolated along
            the edges; ``n`` has to be equal to the vertex number in ``pos`` array.
        mat : string, optional
            Material name.
        """

        if name is None: raise ValueError()
        if not isinstance(name, str): name = str(name)

        if name in self.geometry_data:
            msg = "Geometry %s already exists, use update_graph() instead." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if not isinstance(pos, np.ndarray): pos = np.ascontiguousarray(pos, dtype=np.float32)
        assert len(pos.shape) == 2 and pos.shape[0] > 1 and pos.shape[1] == 3, "Required vertex data shape is (n,3), where n >= 2."
        if pos.dtype != np.float32: pos = np.ascontiguousarray(pos, dtype=np.float32)
        if not pos.flags['C_CONTIGUOUS']: pos = np.ascontiguousarray(pos, dtype=np.float32)
        pos_ptr = pos.ctypes.data
        n_vertices = pos.shape[0]

        if not isinstance(edges, np.ndarray): edges = np.ascontiguousarray(edges, dtype=np.int32)
        if edges.dtype != np.int32: edges = np.ascontiguousarray(edges, dtype=np.int32)
        if not edges.flags['C_CONTIGUOUS']: edges = np.ascontiguousarray(edges, dtype=np.int32)
        assert (len(edges.shape) == 2 and edges.shape[1] == 2) or (len(edges.shape) == 1 and (edges.shape[0] % 2 == 0)), "Required index shape is (n,2) or (m), where m is a multiple of 2."
        edges_ptr = edges.ctypes.data
        n_edges = edges.size // 2

        if r is not None:
            if not isinstance(r, np.ndarray): r = np.ascontiguousarray(r, dtype=np.float32)
            if r.dtype != np.float32: r = np.ascontiguousarray(r, dtype=np.float32)
            if len(r.shape) > 1: r = r.flatten()
            if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
        if r is not None:
            if r.shape[0] == 1:
                if n_vertices > 0: r = np.full(n_vertices, r[0], dtype=np.float32)
                else:
                    msg = "Cannot resolve proper radii (r) shape from preceding data arguments."
                    self._logger.error(msg)
                    if self._raise_on_error: raise ValueError(msg)
                    return
                if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
            if (n_vertices > 0) and (n_vertices != r.shape[0]):
                msg = "Radii (r) shape does not match the shape of preceding data arguments."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                return
            radii_ptr = r.ctypes.data
        else: radii_ptr = 0

        c = np.ascontiguousarray(c, dtype=np.float32)
        if c.shape == (1,):
            c = np.ascontiguousarray([c[0], c[0], c[0]], dtype=np.float32)
            col_const_ptr = c.ctypes.data
            col_ptr = 0
        elif c.shape == (3,):
            col_const_ptr = c.ctypes.data
            col_ptr = 0
        else:
            c = _make_contiguous_3d(c, n=n_vertices, extend_scalars=True)
            assert c.shape == pos.shape,  "Colors shape must be (n,3), with n matching the number of graph vertices."
            if c is not None: col_ptr = c.ctypes.data
            else: col_ptr = 0
            col_const_ptr = 0

        try:
            self._padlock.acquire()
            self._logger.info("Setup graph %s...", name)
            g_handle = self._optix.setup_graph(name, mat, n_vertices, n_edges, pos_ptr, radii_ptr, edges_ptr, col_const_ptr, col_ptr)

            if g_handle > 0:
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_data[name] = GeometryMeta(name, g_handle, n_vertices)
                self.geometry_names[g_handle] = name
            else:
                msg = "Graph setup failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise
        finally:
            self._padlock.release()

    def update_graph(self, name: str,
                     mat: Optional[str] = None,
                     pos: Optional[Any] = None,
                     edges: Optional[Any] = None,
                     r: Optional[Any] = None,
                     c: Optional[Any] = None) -> None:
        """Update data of an existing graph (mesh wireframe) geometry.

        All data or only selected arrays may be uptated. If vertices and edges are left
        unchanged then ``color`` and ``r`` array sizes should match the size of the graph,
        i.e. existing ``pos`` shape.
        
        Parameters
        ----------
        name : string
            Name of the graph geometry.
        mat : string, optional
            Material name.
        pos : array_like, optional
            XYZ values of the graph vertices.
        edges : array_like, optional
            Graph edges as indices (doublets) to the ``pos`` array.
        r : Any, optional
            Radii of vertices, interpolated along the edges. Single value sets
            constant radius for all vertices.
        c : Any, optional
            Colors of graph vertices. Single value means a constant gray level.
            3-component array means a constant RGB color. Array of the shape
            ``(n, 3)`` will set individual color for each vertex,
            interpolated along edges; ``n`` has to be equal to the vertex
            number in ``pos`` array.
        """

        if name is None: raise ValueError()
        if not isinstance(name, str): name = str(name)

        if mat is None: mat = ""

        if not name in self.geometry_data:
            msg = "Graph %s does not exists yet, use set_graph() instead." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        m_vertices = self._optix.get_geometry_size(name)
        #m_edges = self._optix.get_edges_count(name)
        size_changed = False

        pos_ptr = 0
        n_vertices = 0
        if pos is not None:
            if not isinstance(pos, np.ndarray): pos = np.ascontiguousarray(pos, dtype=np.float32)
            assert len(pos.shape) == 2 and pos.shape[0] > 1 and pos.shape[1] == 3, "Required vertex data shape is (n,3), where n >= 2."
            if pos.dtype != np.float32: pos = np.ascontiguousarray(pos, dtype=np.float32)
            if not pos.flags['C_CONTIGUOUS']: pos = np.ascontiguousarray(pos, dtype=np.float32)
            if pos.shape[0] != m_vertices: size_changed = True
            pos_ptr = pos.ctypes.data
            n_vertices = pos.shape[0]
            m_vertices = n_vertices

        edges_ptr = 0
        n_edges = 0
        if edges is not None:
            if not isinstance(edges, np.ndarray): edges = np.ascontiguousarray(edges, dtype=np.int32)
            if edges.dtype != np.int32: edges = np.ascontiguousarray(edges, dtype=np.int32)
            if not edges.flags['C_CONTIGUOUS']: edges = np.ascontiguousarray(edges, dtype=np.int32)
            assert (len(edges.shape) == 2 and edges.shape[1] == 3) or (len(edges.shape) == 1 and (edges.shape[0] % 2 == 0)), "Required index shape is (n,3) or (m), where m is a multiple of 2."
            edges_ptr = edges.ctypes.data
            n_edges = edges.size // 2
            #m_edges = n_edges

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
                r = np.full(m_vertices, r[0], dtype=np.float32)
                if not r.flags['C_CONTIGUOUS']: r = np.ascontiguousarray(r, dtype=np.float32)
            if m_vertices != r.shape[0]:
                msg = "Radii (r) shape does not match the number of graph vertices."
                self._logger.error(msg)
                if self._raise_on_error: raise ValueError(msg)
                return
            radii_ptr = r.ctypes.data

        c_ptr = 0
        c_const_ptr = 0
        if size_changed and c is None: c = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32)
        if c is not None:
            if isinstance(c, float) or isinstance(c, int): c = np.full(3, c, dtype=np.float32)
            if not isinstance(c, np.ndarray): c = np.ascontiguousarray(c, dtype=np.float32)
            if c.dtype != np.float32: c = np.ascontiguousarray(c, dtype=np.float32)
            if not c.flags['C_CONTIGUOUS']: c = np.ascontiguousarray(c, dtype=np.float32)
            if c.shape == (3,):
                c_const_ptr = c.ctypes.data
            elif c.shape == (m_vertices, 3):
                c_ptr = c.ctypes.data
            else:
                msg = "Colors shape must be (n,3), with n matching the number of graph vertices."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

        try:
            self._padlock.acquire()
            self._logger.info("Update graph %s...", name)
            g_handle = self._optix.update_graph(name, mat, m_vertices, n_edges, pos_ptr, radii_ptr, edges_ptr, c_const_ptr, c_ptr)

            if (g_handle > 0) and (g_handle == self.geometry_data[name]._handle):
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_data[name]._size = m_vertices
            else:
                msg = "Graph update failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise
        finally:
            self._padlock.release()


    def set_mesh(self, name: str, pos: Any, faces: Any,
                 c: Any = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32),
                 normals: Optional[Any] = None,
                 nidx: Optional[Any] = None,
                 uvmap: Optional[Any] = None,
                 uvidx: Optional[Any] = None,
                 mat: str = "diffuse",
                 make_normals: bool = False) -> None:
        """Create new mesh geometry.

        Data is provided as vertices :math:`[x, y, z]`, with the shape ``(n, 3)``, and faces
        (triplets of vertex indices), with the shape ``(n, 3)`` or ``(m)`` where :math:`m = 3*n`.
        Data features can be visualized with color (array of RGB values assigned to the mesh
        vertices, shape ``(n, 3)``).

        Mesh ``normals`` can be provided as an array of 3D vectors. Mappng of normals to
        faces can be provided as an array of ``nidx`` indexes. If mapping is not provided
        then face vertex data is used (requires same number of vertices and normal vectors).

        Smooth shading normals can be pre-calculated if ``make_normals=True`` and normals
        data is not provided.

        Texture UV mapping ``uvmap`` can be provided as an array of 2D vectors. Mappng of
        UV coordinates to faces can be provided as an array of ``uvidx`` indexes. If mapping
        is not provided then face vertex data is used (requires same number of vertices
        and UV points).
       
        Parameters
        ----------
        name : string
            Name of the new mesh geometry.
        pos : array_like
            XYZ values of the mesh vertices.
        faces : array_like
            Mesh faces as indices (triplets) to the ``pos`` array.
        c : Any, optional
            Colors of mesh vertices. Single value means a constant gray level.
            3-component array means a constant RGB color. Array of the shape
            ``(n, 3)`` will set individual color for each vertex,
            interpolated on face surfaces; ``n`` has to be equal to the vertex
            number in ``pos`` array.
        normals : array_like, optional
            Normal vectors.
        nidx : array_like, optional
            Normal to face mapping, ``faces`` is used if not provided.
        uvmap : array_like, optional
            Texture UV coordinates.
        uvidx : array_like, optional
            Texture UV to face mapping, ``faces`` is used if not provided.
        mat : string, optional
            Material name.
        make_normals : bool, optional
            Calculate smooth shading of the mesh, if ``normals`` are not provided.
            Normals of all triangles attached to the mesh vertex are averaged.
        """

        if name is None: raise ValueError()
        if not isinstance(name, str): name = str(name)

        if name in self.geometry_data:
            msg = "Geometry %s already exists, use update_mesh() instead." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        if not isinstance(pos, np.ndarray): pos = np.ascontiguousarray(pos, dtype=np.float32)
        assert len(pos.shape) == 2 and pos.shape[0] > 2 and pos.shape[1] == 3, "Required vertex data shape is (n,3), where n >= 3."
        if pos.dtype != np.float32: pos = np.ascontiguousarray(pos, dtype=np.float32)
        if not pos.flags['C_CONTIGUOUS']: pos = np.ascontiguousarray(pos, dtype=np.float32)
        pos_ptr = pos.ctypes.data
        n_vertices = pos.shape[0]

        if not isinstance(faces, np.ndarray): faces = np.ascontiguousarray(faces, dtype=np.int32)
        if faces.dtype != np.int32: faces = np.ascontiguousarray(faces, dtype=np.int32)
        if not faces.flags['C_CONTIGUOUS']: faces = np.ascontiguousarray(faces, dtype=np.int32)
        assert (len(faces.shape) == 2 and faces.shape[1] == 3) or (len(faces.shape) == 1 and (faces.shape[0] % 3 == 0)), "Required index shape is (n,3) or (m), where m is a multiple of 3."
        faces_ptr = faces.ctypes.data
        n_faces = faces.size // 3

        c = np.ascontiguousarray(c, dtype=np.float32)
        if c.shape == (1,):
            c = np.ascontiguousarray([c[0], c[0], c[0]], dtype=np.float32)
            col_const_ptr = c.ctypes.data
            col_ptr = 0
        elif c.shape == (3,):
            col_const_ptr = c.ctypes.data
            col_ptr = 0            
        else:
            c = _make_contiguous_3d(c, n=n_vertices, extend_scalars=True)
            assert c.shape == pos.shape,  "Colors shape must be (n,3), with n matching the number of mesh vertices."
            if c is not None: col_ptr = c.ctypes.data
            else: col_ptr = 0
            col_const_ptr = 0

        n_ptr = 0
        n_normals = 0
        if normals is not None:
            if not isinstance(normals, np.ndarray): normals = np.ascontiguousarray(normals, dtype=np.float32)
            if nidx is None:
                assert normals.shape == pos.shape, "If normal index data not provided, normals shape must be (n,3), with n matching the mesh vertex positions shape."
            else:
                assert len(normals.shape) == 2 and normals.shape[0] > 2 and normals.shape[1] == 3, "Required normals data shape is (n,3), where n >= 3."
            if normals.dtype != np.float32: normals = np.ascontiguousarray(normals, dtype=np.float32)
            if not normals.flags['C_CONTIGUOUS']: normals = np.ascontiguousarray(normals, dtype=np.float32)
            n_ptr = normals.ctypes.data
            n_normals = normals.shape[0]
            make_normals = False

        nidx_ptr = 0
        if nidx is not None:
            if not isinstance(nidx, np.ndarray): nidx = np.ascontiguousarray(nidx, dtype=np.int32)
            if nidx.dtype != np.int32: nidx = np.ascontiguousarray(nidx, dtype=np.int32)
            if not nidx.flags['C_CONTIGUOUS']: nidx = np.ascontiguousarray(nidx, dtype=np.int32)
            assert np.array_equal(nidx.shape, faces.shape), "Required same shape of normal index and face index arrays."
            nidx_ptr = nidx.ctypes.data

        uv_ptr = 0
        n_uv = 0
        if uvmap is not None:
            if not isinstance(uvmap, np.ndarray): uvmap = np.ascontiguousarray(uvmap, dtype=np.float32)
            if uvidx is None:
                assert uvmap.shape[0] == pos.shape[0], "If UV index data not provided, uvmap shape must be (n,2), with n matching the number of mesh vertices."
            else:
                assert len(uvmap.shape) == 2 and uvmap.shape[0] > 2 and uvmap.shape[1] == 2, "Required UV data shape is (n,2), where n >= 3."
            if uvmap.dtype != np.float32: uvmap = np.ascontiguousarray(uvmap, dtype=np.float32)
            if not uvmap.flags['C_CONTIGUOUS']: uvmap = np.ascontiguousarray(uvmap, dtype=np.float32)
            uv_ptr = uvmap.ctypes.data
            n_uv = uvmap.shape[0]

        uvidx_ptr = 0
        if uvidx is not None:
            if not isinstance(uvidx, np.ndarray): uvidx = np.ascontiguousarray(uvidx, dtype=np.int32)
            if uvidx.dtype != np.int32: uvidx = np.ascontiguousarray(uvidx, dtype=np.int32)
            if not uvidx.flags['C_CONTIGUOUS']: uvidx = np.ascontiguousarray(uvidx, dtype=np.int32)
            assert np.array_equal(uvidx.shape, faces.shape), "Required same shape of UV index and face index arrays."
            uvidx_ptr = uvidx.ctypes.data

        try:
            self._padlock.acquire()
            self._logger.info("Setup mesh %s...", name)
            g_handle = self._optix.setup_mesh(name, mat, n_vertices, n_faces, n_normals, n_uv, pos_ptr, faces_ptr, col_const_ptr, col_ptr, n_ptr, nidx_ptr, uv_ptr, uvidx_ptr, make_normals)

            if g_handle > 0:
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_data[name] = GeometryMeta(name, g_handle, n_vertices)
                self.geometry_names[g_handle] = name
            else:
                msg = "Mesh setup failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise
        finally:
            self._padlock.release()

    def update_mesh(self, name: str,
                    mat: Optional[str] = None,
                    pos: Optional[Any] = None,
                    faces: Optional[Any] = None,
                    c: Optional[Any] = None,
                    normals: Optional[Any] = None,
                    nidx: Optional[Any] = None,
                    uvmap: Optional[Any] = None,
                    uvidx: Optional[Any] = None) -> None:
        """Update data of an existing mesh geometry.

        All data or only some of arrays may be uptated. If vertices and faces are left
        unchanged then other arrays sizes should match the sizes of the mesh, i.e. ``c``
        shape should match existing ``pos`` shape, ``nidx`` and ``uvidx`` shapes should
        match ``faces`` shape or if index mapping is not provided then ``normals`` and
        ``uvmap`` shapes should match ``pos`` shape.
        
        Parameters
        ----------
        name : string
            Name of the mesh geometry.
        mat : string, optional
            Material name.
        pos : array_like, optional
            XYZ values of the mesh vertices.
        faces : array_like, optional
            Mesh faces as indices (triplets) to the ``pos`` array.
        c : Any, optional
            Colors of mesh vertices. Single value means a constant gray level.
            3-component array means a constant RGB color. Array of the shape
            ``(n, 3)`` will set individual color for each vertex,
            interpolated on face surfaces; ``n`` has to be equal to the vertex
            number in ``pos`` array.
        normals : array_like, optional
            Normal vectors.
        nidx : array_like, optional
            Normal to face mapping, existing mesh ``faces`` is used if not provided.
        uvmap : array_like, optional
            Texture UV coordinates.
        uvidx : array_like, optional
            Texture UV to face mapping, existing mesh ``faces`` is used if not provided.
        """

        if name is None: raise ValueError()
        if not isinstance(name, str): name = str(name)

        if mat is None: mat = ""

        if not name in self.geometry_data:
            msg = "Mesh %s does not exists yet, use set_mesh() instead." % name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        m_vertices = self._optix.get_geometry_size(name)
        m_faces = self._optix.get_faces_count(name)
        size_changed = False

        pos_ptr = 0
        n_vertices = 0
        if pos is not None:
            if not isinstance(pos, np.ndarray): pos = np.ascontiguousarray(pos, dtype=np.float32)
            assert len(pos.shape) == 2 and pos.shape[0] > 2 and pos.shape[1] == 3, "Required vertex data shape is (n,3), where n >= 3."
            if pos.dtype != np.float32: pos = np.ascontiguousarray(pos, dtype=np.float32)
            if not pos.flags['C_CONTIGUOUS']: pos = np.ascontiguousarray(pos, dtype=np.float32)
            if pos.shape[0] != m_vertices: size_changed = True
            pos_ptr = pos.ctypes.data
            n_vertices = pos.shape[0]
            m_vertices = n_vertices

        faces_ptr = 0
        n_faces = 0
        if faces is not None:
            if not isinstance(faces, np.ndarray): faces = np.ascontiguousarray(faces, dtype=np.int32)
            if faces.dtype != np.int32: faces = np.ascontiguousarray(faces, dtype=np.int32)
            if not faces.flags['C_CONTIGUOUS']: faces = np.ascontiguousarray(faces, dtype=np.int32)
            assert (len(faces.shape) == 2 and faces.shape[1] == 3) or (len(faces.shape) == 1 and (faces.shape[0] % 3 == 0)), "Required index shape is (n,3) or (m), where m is a multiple of 3."
            faces_ptr = faces.ctypes.data
            n_faces = faces.size // 3
            m_faces = n_faces

        c_ptr = 0
        c_const_ptr = 0
        if size_changed and c is None: c = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32)
        if c is not None:
            if isinstance(c, float) or isinstance(c, int): c = np.full(3, c, dtype=np.float32)
            if not isinstance(c, np.ndarray): c = np.ascontiguousarray(c, dtype=np.float32)
            if c.dtype != np.float32: c = np.ascontiguousarray(c, dtype=np.float32)
            if not c.flags['C_CONTIGUOUS']: c = np.ascontiguousarray(c, dtype=np.float32)
            if c.shape == (3,):
                c_const_ptr = c.ctypes.data
            elif c.shape == (m_vertices, 3):
                c_ptr = c.ctypes.data
            else:
                msg = "Colors shape must be (n,3), with n matching the number of mesh vertices."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

        n_ptr = 0
        n_normals = 0
        if normals is not None:
            if not isinstance(normals, np.ndarray): normals = np.ascontiguousarray(normals, dtype=np.float32)
            if nidx is None:
                assert normals.shape[0] == m_vertices, "If normal index data not provided, normals shape must be (n,3), with n matching the mesh vertex positions shape."
            else:
                assert len(normals.shape) == 2 and normals.shape[0] > 2 and normals.shape[1] == 3, "Required normals data shape is (n,3), where n >= 3."
            if normals.dtype != np.float32: normals = np.ascontiguousarray(normals, dtype=np.float32)
            if not normals.flags['C_CONTIGUOUS']: normals = np.ascontiguousarray(normals, dtype=np.float32)
            n_ptr = normals.ctypes.data
            n_normals = normals.shape[0]
            make_normals = False

        nidx_ptr = 0
        if nidx is not None:
            if not isinstance(nidx, np.ndarray): nidx = np.ascontiguousarray(nidx, dtype=np.int32)
            if nidx.dtype != np.int32: nidx = np.ascontiguousarray(nidx, dtype=np.int32)
            if not nidx.flags['C_CONTIGUOUS']: nidx = np.ascontiguousarray(nidx, dtype=np.int32)
            assert (len(nidx.shape) == 2 and nidx.shape[0] == m_faces) or (len(nidx.shape) == 1 and nidx.shape[0] == 3 * m_faces), "Required same shape of normal index and face index arrays."
            nidx_ptr = nidx.ctypes.data

        uv_ptr = 0
        n_uv = 0
        if uvmap is not None:
            if not isinstance(uvmap, np.ndarray): uvmap = np.ascontiguousarray(uvmap, dtype=np.float32)
            if uvidx is None:
                assert uvmap.shape[0] == m_vertices, "If UV index data not provided, uvmap shape must be (n,2), with n matching the number of mesh vertices."
            else:
                assert len(uvmap.shape) == 2 and uvmap.shape[0] > 2 and uvmap.shape[1] == 2, "Required UV data shape is (n,2), where n >= 3."
            if uvmap.dtype != np.float32: uvmap = np.ascontiguousarray(uvmap, dtype=np.float32)
            if not uvmap.flags['C_CONTIGUOUS']: uvmap = np.ascontiguousarray(uvmap, dtype=np.float32)
            uv_ptr = uvmap.ctypes.data
            n_uv = uvmap.shape[0]

        uvidx_ptr = 0
        if uvidx is not None:
            if not isinstance(uvidx, np.ndarray): uvidx = np.ascontiguousarray(uvidx, dtype=np.int32)
            if uvidx.dtype != np.int32: uvidx = np.ascontiguousarray(uvidx, dtype=np.int32)
            if not uvidx.flags['C_CONTIGUOUS']: uvidx = np.ascontiguousarray(uvidx, dtype=np.int32)
            assert (len(uvidx.shape) == 2 and uvidx.shape[0] == m_faces) or (len(uvidx.shape) == 1 and uvidx.shape[0] == 3 * m_faces), "Required same shape of UV index and face index arrays."
            uvidx_ptr = uvidx.ctypes.data

        try:
            self._padlock.acquire()
            self._logger.info("Update mesh %s...", name)
            g_handle = self._optix.update_mesh(name, mat, n_vertices, n_faces, n_normals, n_uv, pos_ptr, faces_ptr, c_const_ptr, c_ptr, n_ptr, nidx_ptr, uv_ptr, uvidx_ptr)

            if (g_handle > 0) and (g_handle == self.geometry_data[name]._handle):
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_data[name]._size = m_vertices
            else:
                msg = "Mesh update failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise
        finally:
            self._padlock.release()


    def load_mesh_obj(self, file_name: str, mesh_name: Optional[str] = None, parent: Optional[str] = None,
                      c: Any = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32),
                      mat: str = "diffuse",
                      make_normals: bool = False) -> None:
        """Load mesh geometry from Wavefront .obj file.

        Note: this method can read files with named objects only. Use :meth:`plotoptix.NpOptiX.load_merged_mesh_obj`
        for reading files with raw, unnamed mesh.

        Parameters
        ----------
        file_name : string
            File name (local file path or url) to read from.
        mesh_name : string, optional
            Name of the mesh to import from the file. All meshes are imported
            if ``None`` value or empty string is used.
        parent : string, optional
            Optional name of a mesh to set as a parent of all other meshes loaded from the file. All transformations
            applied to the parent will be applied to children meshes as well.
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
        if file_name is None: raise ValueError()

        if not isinstance(file_name, str): file_name = str(file_name)

        if mesh_name is None: mesh_name = ""

        if not isinstance(mesh_name, str): mesh_name = str(mesh_name)

        if parent is None: parent = ""

        if not isinstance(parent, str): parent = str(parent)

        if mesh_name in self.geometry_data:
            msg = "Geometry %s already exists, use update_mesh() instead." % mesh_name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        c = _make_contiguous_vector(c, n_dim=3)
        if c is not None: col_ptr = c.ctypes.data
        else: col_ptr = 0

        try:
            self._padlock.acquire()
            self._logger.info("Load mesh from file %s ...", file_name)
            s = self._optix.load_mesh_obj(file_name, mesh_name, parent, mat, col_ptr, make_normals)

            if len(s) > 2:
                meta = json.loads(s)
                for key, value in meta.items():
                    self.geometry_data[key] = GeometryMeta(key, value["Handle"], value["Size"])
                    self.geometry_names[value["Handle"]] = key
                    self._logger.info("...loaded: %s (%d vertices)", key, value["Size"])
            else:
                msg = "Mesh loading failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise
        finally:
            self._padlock.release()

    def load_multiple_mesh_obj(self, file_name: str, mat: dict, default: str = "diffuse",
                               parent: Optional[str] = None) -> None:
        """Load meshesh from Wavefront .obj file, assign materials from dictrionary.

        Note: this method can read files with named objects only. Use :meth:`plotoptix.NpOptiX.load_merged_mesh_obj`
        for reading files with raw, unnamed mesh.

        Parameters
        ----------
        file_name : string
            File name (local file path or url) to read from.
        mat : dict
            Mesh name to material name dictionary. All meshes with names starting with keys in ``dict`` will have
            corresponding material assigned.
        default : string, optional
            Default material name, assigned if mesh name not fount in ``mat``.
        parent : string, optional
            Optional full name of a mesh to set as a parent of all other meshes loaded from the file. All transformations
            applied to the parent will be applied to children meshes as well.
        """
        if file_name is None: raise ValueError()

        if not isinstance(file_name, str): file_name = str(file_name)

        if parent is None: parent = ""

        if not isinstance(parent, str): parent = str(parent)

        for n in mat:
            if not isinstance(mat[n], str):
                mat[n] = str(mat[n])

        try:
            self._padlock.acquire()
            self._logger.info("Load mesh from file %s ...", file_name)

            s = self._optix.load_multiple_mesh_obj(file_name, json.dumps(mat), default, parent)

            if len(s) > 2:
                meta = json.loads(s)
                for key, value in meta.items():
                    self.geometry_data[key] = GeometryMeta(key, value["Handle"], value["Size"])
                    self.geometry_names[value["Handle"]] = key
                    self._logger.info("...loaded: %s (%d vertices)", key, value["Size"])
            else:
                msg = "Mesh loading failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise
        finally:
            self._padlock.release()


    def load_merged_mesh_obj(self, file_name: str, mesh_name: str,
                             c: Any = np.ascontiguousarray([0.94, 0.94, 0.94], dtype=np.float32),
                             mat: str = "diffuse", make_normals: bool = False) -> None:
        """Load and merge mesh geometries from Wavefront .obj file.

        All objects are imported from file and merged in a single PlotOptiX mesh. This method
        can read files with no named objects specified.

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

        if mesh_name in self.geometry_data:
            msg = "Geometry %s already exists, use update_mesh() instead." % mesh_name
            self._logger.error(msg)
            if self._raise_on_error: raise ValueError(msg)
            return

        c = _make_contiguous_vector(c, n_dim=3)
        if c is not None: col_ptr = c.ctypes.data
        else: col_ptr = 0

        try:
            self._padlock.acquire()
            self._logger.info("Load and merge meshes from file %s ...", file_name)
            g_handle = self._optix.load_merged_mesh_obj(file_name, mesh_name, mat, col_ptr, make_normals)

            if g_handle > 0:
                self._logger.info("...done, handle: %d", g_handle)
                self.geometry_data[mesh_name] = GeometryMeta(mesh_name, g_handle, self._optix.get_geometry_size(mesh_name))
                self.geometry_names[g_handle] = mesh_name
            else:
                msg = "Mesh loading failed."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

        except Exception as e:
            self._logger.error(str(e))
            if self._raise_on_error: raise
        finally:
            self._padlock.release()


    def get_geometry_names(self) -> list:
        """Return list of geometries' names.
        """
        return list(self.geometry_data.keys())

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

    def scale_geometry(self, name: str, s: Union[float, Tuple[float, float, float]],
                       center: Optional[Tuple[float, float, float]] = None,
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
        s : float,  tuple (float, float, float)
            Scaling factor, single value or (x, y, z) scales.
        center : tuple (float, float, float), optional
            Scaling center. If not provided, scaling is made w.r.t. the primitive center.
        update : bool, optional
            Update GPU buffer.
        """
        if name is None: raise ValueError()

        if isinstance(s, float) or isinstance(s, int):
            s = float(s)
            if center is None:
                if not self._optix.scale_geometry(name, s, update):
                    msg = "Geometry scale by scalar failed."
                    self._logger.error(msg)
                    if self._raise_on_error: raise RuntimeError(msg)
            else:
                if not isinstance(center, tuple): center = tuple(center)
                if not self._optix.scale_geometry_c(name, s, center[0], center[1], center[2], update):
                    msg = "Geometry scale by scalar w.r.t. the center failed."
                    self._logger.error(msg)
                    if self._raise_on_error: raise RuntimeError(msg)
        else:
            if not isinstance(s, tuple): s = tuple(s)
            if center is None:
                if not self._optix.scale_geometry_xyz(name, s[0], s[1], s[2], update):
                    msg = "Geometry scale by vector failed."
                    self._logger.error(msg)
                    if self._raise_on_error: raise RuntimeError(msg)
            else:
                if not isinstance(center, tuple): center = tuple(center)
                if not self._optix.scale_geometry_xyz_c(name, s[0], s[1], s[2], center[0], center[1], center[2], update):
                    msg = "Geometry scale by vector w.r.t. the center failed."
                    self._logger.error(msg)
                    if self._raise_on_error: raise RuntimeError(msg)

    def scale_primitive(self, name: str, idx: int, s: Union[float, Tuple[float, float, float]],
                        center: Optional[Tuple[float, float, float]] = None,
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
        s : float,  tuple (float, float, float)
            Scaling factor, single value or (x, y, z) scales.
        center : tuple (float, float, float), optional
            Scaling center. If not provided, scaling is made w.r.t. the primitive center.
        update : bool, optional
            Update GPU buffer.
        """
        if name is None: raise ValueError()

        if isinstance(s, float) or isinstance(s, int):
            s = float(s)
            if center is None:
                if not self._optix.scale_primitive(name, idx, s, update):
                    msg = "Primitive scale by scalar failed."
                    self._logger.error(msg)
                    if self._raise_on_error: raise RuntimeError(msg)
            else:
                if not isinstance(center, tuple): center = tuple(center)
                if not self._optix.scale_primitive_c(name, idx, s, center[0], center[1], center[2], update):
                    msg = "Primitive scale by scalar w.r.t. the center failed."
                    self._logger.error(msg)
                    if self._raise_on_error: raise RuntimeError(msg)
        else:
            if not isinstance(s, tuple): s = tuple(s)
            if center is None:
                if not self._optix.scale_primitive_xyz(name, idx, s[0], s[1], s[2], update):
                    msg = "Primitive scale by vector failed."
                    self._logger.error(msg)
                    if self._raise_on_error: raise RuntimeError(msg)
            else:
                if not isinstance(center, tuple): center = tuple(center)
                if not self._optix.scale_primitive_xyz_c(name, idx, s[0], s[1], s[2], center[0], center[1], center[2], update):
                    msg = "Primitive scale by vector w.r.t. the center failed."
                    self._logger.error(msg)
                    if self._raise_on_error: raise RuntimeError(msg)

    def update_geom_buffers(self, name: str,
                            mask: Union[GeomBuffer, str] = GeomBuffer.All,
                            forced: bool = False) -> None:
        """Update geometry buffers.

        Update geometry buffers in GPU after modifications made with
        :meth:`plotoptix.NpOptiX.move_geometry`, :meth:`plotoptix.NpOptiX.move_primitive`,
        and similar methods.

        Parameters
        ----------
        name : string
            Name of the geometry.
        mask : GeomBuffer or string, optional
            Which buffers to update. All buffers if not specified.
        forced : bool, optional
            Update even if the object was not tagged as outdated. Operations like rotations,
            scaling, shifts, are setting "out of date" flag, but direct modifications of
            buffers memory performed with :class:`plotoptix.geometry.PinnedBuffer` require
            forced update.
        """
        if name is None: raise ValueError()

        if isinstance(mask, str): mask = GeomBuffer[mask]

        if not self._optix.update_geom_buffers(name, mask.value, forced):
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
