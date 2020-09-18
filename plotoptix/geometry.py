"""Geometry class for PlotOptiX raytracer.

Basic geometry properties and interface to underlaying data buffers.

https://github.com/rnd-team-dev/plotoptix/blob/master/LICENSE.txt

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
"""

from typing import Optional, Union
from ctypes import byref, c_ubyte, c_float, c_uint, c_int, c_longlong
import numpy as np

from plotoptix.enums import GeomBuffer
from plotoptix._load_lib import load_optix

class GeometryMeta:

    _name = None
    """Unique name for the geometry object.
    """

    _handle = None
    """Unique int handle for the geometry object.
    """

    _size = 0
    """Number of primitives or data points.
    """

    def __init__(self, name: str, handle: int, size: int) -> None:
        """Geometry metadata for all mesh-less mesh based objects in the scene.

        Basic geometry properties and an interface to underlaying data buffers.
        """

        self._optix = load_optix()
        self._name = name
        self._handle = handle
        self._size = size

    def _pin_buffer(self, buffer: Union[GeomBuffer, str]) -> Optional[np.ndarray]:

        if isinstance(buffer, str): buffer = GeomBuffer[buffer]

        c_buffer = c_longlong()
        c_shape = c_longlong()
        c_size = c_int()
        c_type = c_uint()
        if self._optix.pin_geometry_buffer(
                                self._name, buffer.value,
                                byref(c_buffer), byref(c_shape),
                                byref(c_size), byref(c_type)):

            if c_type.value == 4:
                elem = c_float
            elif c_type.value == 3:
                elem = c_uint
            elif c_type.value == 2:
                elem = c_int
            elif c_type.value == 1:
                elem = c_ubyte
            else:
                msg = "Data type not supported."
                self._logger.error(msg)
                if self._raise_on_error: raise RuntimeError(msg)

            shape_buf = (c_int * c_size.value).from_address(c_shape.value)
            shape = np.ctypeslib.as_array(shape_buf)

            for s in shape: elem = elem * s

            return elem.from_address(c_buffer.value)

        else:
            msg = "Buffer not pinned."
            raise RuntimeError(msg)

            return None

    def _release_buffer(self, buffer: GeomBuffer) -> None:

        if isinstance(buffer, str): buffer = GeomBuffer[buffer]



        if not self._optix.unpin_geometry_buffer(self._name, buffer.value):
            msg = "Buffer not released."
            raise RuntimeError(msg)

class PinnedBuffer:
    """Pins an internal buffer memory and exposes it as an ``np.ndarray``.

    Use only within the ``with`` block as in the provided example. The exposed
    array is not going out of scope nor is anyhow protected outside that expression
    due to the current limitations of the array interface; be careful and do not
    use the array outside ``with`` as memory can be reallocated.

    Parameters
    ----------
    geom : GeometryMeta
        Geometry metadata for the object, available in :attr:`plotoptix.NpOptiX.geometry_data`
        dictionary.
    buffer : GeomBuffer or string
        Buffer kind to pin.

    Returns
    -------
    out : ndarray
        Buffer data wrapped in ``np.ndarray``.

    Examples
    --------
    >>> rt = NpOptiX()
    >>> rt.set_data("plot", xyz=np.random.random((100, 3)), r=0.05)
    >>>
    >>> with PinnedBuffer(rt.geometry_data["plot"], "Positions") as b:
    >>>     print("internal data:", b.shape)
    >>>     print("b[:3])
    >>>
    >>>     b *= 1.5
    >>>     rt.update_geom_buffers("plot", "Positions", forced=True)
    """

    _buffer = None
    """Buffer kind.
    """

    _data = None
    """Buffer array.
    """

    _geometry = None
    """Geometry metadata.
    """

    def __init__(self, geom: GeometryMeta, buffer: GeomBuffer) -> None:
        """Constructor.
        """

        self._geometry = geom
        self._buffer = buffer

    def __enter__(self) -> Optional[np.ndarray]:
        """Pin memory, wrap it in ``np.ndarray``.
        """
        buf = self._geometry._pin_buffer(self._buffer)
        if buf is not None:
            self._data = np.ctypeslib.as_array(buf) 
        return self._data

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Free pinned memory.
        
        Note: array syrvives on the Python side, do not use it.
        """
        self._geometry._release_buffer(self._buffer)
        #print(self._data.__array_interface__)
