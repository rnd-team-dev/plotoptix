"""Utility and convenience functions.
"""

import functools, operator
import numpy as np

from ctypes import byref, c_int
from typing import Any, Optional, Tuple, Union
from matplotlib import cm

from plotoptix._load_lib import load_optix
from plotoptix.enums import GpuArchitecture, ChannelOrder

_optix = load_optix()


def get_gpu_architecture() -> Optional[GpuArchitecture]:
    """Get configured SM architecture.

    Returns effective configuration of the PTX selection and ``-arch`` option
    of the shader compilation. Can verify matched SM architecture after
    constructing objects of :py:mod:`plotoptix.NpOptiX` and derived classes.

    Returns
    -------
    out : GpuArchitecture or None
        SM architecture or ``None`` if not recognized.

    See Also
    --------
    :py:mod:`plotoptix.enums.GpuArchitecture`
    """
    cfg = _optix.get_gpu_architecture()
    if cfg >= 0: return GpuArchitecture(10 * cfg)
    else: return None

def set_gpu_architecture(arch: Union[GpuArchitecture, str]) -> None:
    """Set SM architecture.

    May be used to force pre-compiled PTX selection and ``-arch`` option
    of the shader compilation. Default value is
    :py:mod:`plotoptix.enums.GpuArchitecture.Auto`. Use before
    constructing objects of :py:mod:`plotoptix.NpOptiX` and derived
    classes. :py:mod:`plotoptix.NpOptiX` constructor  will fail if SM
    architecture higher than available is set.

    See Also
    --------
    :py:mod:`plotoptix.enums.GpuArchitecture`
    """
    if isinstance(arch, str): arch = GpuArchitecture[arch]
    _optix.set_gpu_architecture(arch.value)


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

def _make_contiguous_3d(a: Optional[Any], n: int = -1, extend_scalars: bool = False) -> Optional[np.ndarray]:
    if a is None: return None

    if not isinstance(a, np.ndarray): a = np.ascontiguousarray(a, dtype=np.float32)

    if a.dtype != np.float32: a = np.ascontiguousarray(a, dtype=np.float32)

    if len(a.shape) == 1:
        if a.shape[0] == 1: a = np.full((1, 3), a[0], dtype=np.float32)
        elif a.shape[0] == 3: a = np.reshape(a, (1, 3))
        elif a.shape[0] == n or n < 0: a = np.reshape(a, (a.shape[0], 1))
        else:
            raise ValueError("Input shape not matching single 3D vector nor desired array length.")

    if len(a.shape) > 2:
        m = functools.reduce(operator.mul, a.shape[:-1], 1)
        if (n >= 0) and (n != m):
            raise ValueError("Input shape not matching desired array length.")
        a = np.reshape(a, (m, a.shape[-1]))

    if n >= 0:
        if (a.shape[0] == 1) and (n != a.shape[0]):
            _a = np.zeros((n, a.shape[-1]), dtype=np.float32)
            _a[:] = a[0]
            a = _a
        if n != a.shape[0]:
            raise ValueError("Input shape not matching desired array length.")

    if a.shape[-1] != 3:
        _a = np.zeros((a.shape[0], 3), dtype=np.float32)
        if a.shape[-1] == 1:
            if extend_scalars:
                _a = np.repeat(a, 3).reshape((a.shape[0], 3))
            else:
                _a[:,0] = a[:,0]
        elif a.shape[-1] == 2: _a[:,[0,1]] = a[:,[0,1]]
        else: _a[:,[0,1,2]] = a[:,[0,1,2]]
        a = _a

    if not a.flags['C_CONTIGUOUS']: a = np.ascontiguousarray(a, dtype=np.float32)

    return a

def _make_contiguous_2x3d(a: Optional[Any], extend_scalars: bool = False) -> Optional[np.ndarray]:
    if a is None: return None

    if not isinstance(a, np.ndarray): a = np.ascontiguousarray(a, dtype=np.float32)

    if a.dtype != np.float32: a = np.ascontiguousarray(a, dtype=np.float32)

    if len(a.shape) > 3:
        raise ValueError("Input shape should be (n,m) or (n,m,3).")

    if len(a.shape) == 2:
        _a = np.zeros(a.shape + (3,), dtype=np.float32)
        if extend_scalars:
            _a = np.repeat(a, 3).reshape(a.shape + (3,))
        else:
            _a[...,0] = a[:]
        a = _a

    elif len(a.shape) == 3 and a.shape[-1] != 3:
        _a = np.zeros(a.shape[:2] + (3,), dtype=np.float32)
        if a.shape[-1] == 1:
            if extend_scalars:
                _a = np.repeat(a, 3).reshape(a.shape[:2] + (3,))
            else:
                _a[...,0] = a[:,0]
        elif a.shape[-1] == 2: _a[:,:,[0,1]] = a[:,:,[0,1]]
        else: _a[:,:,[0,1,2]] = a[:,:,[0,1,2]]
        a = _a

    if not a.flags['C_CONTIGUOUS']: a = np.ascontiguousarray(a, dtype=np.float32)

    return a

def make_color(c: Any,
               exposure: float = 1.0,
               gamma: float = 1.0,
               input_range: float = 1.0,
               extend_scalars: bool = True,
               channel_order: Union[ChannelOrder, str] = ChannelOrder.RGB) -> np.ndarray:
    """Prepare 1D array of colors to account for the postprocessing corrections.

    Colors of geometry objects or background in the ray traced image may
    look very different than expected from raw color values assigned at
    the scene initialization, if post-processing corrections are applied.
    This method applies inverse gamma and exposure corrections and returns
    RGB values resulting with desired colors in the image.

    Input values range may be specified, so RGB can be provided as 0-255
    values, for convenience.

    The output array shape is ``(n, 3)``, where n is deduced from the input
    array. Single scalar value and values in 1D arrays are treated as a gray
    levels if ``extend_scalars=True``.

    Parameters
    ----------
    c : Any
        Target color values, array_like of any shape or single scalar value.
    exposure : float, optional
        Exposure value applied in post-processing.
    gamma : float, optional
        Gamma value applied in post-processing.
    input_range : float, optional
        Range of the input color values.
    extend_scalars : bool, optional
        Convert single scalar and 1D arrays to gray values encoded as RGB.
    channel_order : ChannelOrder, optional
        Swap RGB to BGR and add aplha channel if necessary.

    Returns
    -------
    out : np.ndarray
        C-contiguous, float32 numpy array with RGB color values pre-calculated
        to account for post-processing corrections.
    """
    if isinstance(channel_order, str): channel_order = ChannelOrder[channel_order]

    if isinstance(c, np.ndarray):
        c = c.astype(np.float32)
    else:
        c = np.ascontiguousarray(c, dtype=np.float32)

    if input_range != 1.0: c *= (1 / input_range)
    if gamma != 1.0: c = np.power(c, gamma, dtype=np.float32)
    if exposure != 1.0: c *= (1 / exposure)
    c = _make_contiguous_3d(c, extend_scalars=extend_scalars)

    if channel_order == ChannelOrder.RGBA:
        _c = np.zeros((c.shape[0], 4), dtype=np.float32)
        _c[:,:-1] = c
        c = _c
        if not c.flags['C_CONTIGUOUS']: c = np.ascontiguousarray(c, dtype=np.float32)
    elif channel_order == ChannelOrder.BGRA:
        c[:,[0, 2]] = c[:,[2, 0]]
        _c = np.zeros((c.shape[0], 4), dtype=np.float32)
        _c[:,:-1] = c
        c = _c
        if not c.flags['C_CONTIGUOUS']: c = np.ascontiguousarray(c, dtype=np.float32)
    elif channel_order == ChannelOrder.BGR:
        c[:,[0, 2]] = c[:,[2, 0]]
        if not c.flags['C_CONTIGUOUS']: c = np.ascontiguousarray(c, dtype=np.float32)

    return c

def make_color_2d(c: Any,
                  exposure: float = 1.0,
                  gamma: float = 1.0,
                  input_range: float = 1.0,
                  extend_scalars: bool = True,
                  channel_order: Union[ChannelOrder, str] = ChannelOrder.RGB,
                  alpha: float = 1.0) -> np.ndarray:
    """Prepare 2D array of colors to account for the postprocessing corrections.

    Colors of geometry objects in the ray traced image may look very different
    than expected from raw color values assigned at objects initialization, if
    post-processing corrections are applied. This method applies inverse gamma
    and exposure corrections and returns RGB values resulting with desired colors
    in the image. Optionally, alpha channel may be added and filled with provided
    value.

    Input values range may be specified, so RGB can be provided as e.g. 0-255
    values, for convenience.

    Input array shape should be ``(n, m)``, ``(n, m, 1)``, or  ``(n, m, 3)``.
    The output array shape is ``(n, m, 3)`` or ``(n, m, 4)``. Single scalar values
    are treated as a gray levels if ``extend_scalars=True``.

    Parameters
    ----------
    c : Any
        Target color values, array_like of shape ``(n, m)``, ``(n, m, 1)``,
        or  ``(n, m, 3)``.
    exposure : float, optional
        Exposure value applied in post-processing.
    gamma : float, optional
        Gamma value applied in post-processing.
    input_range : float, optional
        Range of the input color values.
    extend_scalars : bool, optional
        Convert single scalar values to gray levels encoded as RGB.
    channel_order : ChannelOrder, optional
        Swap RGB to BGR and add aplha channel if necessary.
    alpha : float, optional
        Value used to fill alpha channel, if necessary.

    Returns
    -------
    out : np.ndarray
        C-contiguous, float32 numpy array with RGB color values pre-calculated
        to account for post-processing corrections.
    """
    if isinstance(channel_order, str): channel_order = ChannelOrder[channel_order]

    if not isinstance(c, np.ndarray): c = np.ascontiguousarray(c, dtype=np.float32)

    if isinstance(c, np.ndarray) and c.dtype != np.float32: c = c.astype(np.float32)

    if input_range != 1.0: c *= (1 / input_range)
    if gamma != 1.0: c = np.power(c, gamma, dtype=np.float32)
    if exposure != 1.0: c *= (1 / exposure)

    c = _make_contiguous_2x3d(c, extend_scalars=extend_scalars)

    if channel_order == ChannelOrder.RGBA:
        _c = np.zeros(c.shape[:2] + (4,), dtype=np.float32)
        _c[...,:-1] = c
        _c[...,-1] = alpha
        c = _c
        if not c.flags['C_CONTIGUOUS']: c = np.ascontiguousarray(c, dtype=np.float32)
    elif channel_order == ChannelOrder.BGRA:
        c[...,[0, 2]] = c[...,[2, 0]]
        _c = np.zeros(c.shape[:2] + (4,), dtype=np.float32)
        _c[...,:-1] = c
        _c[...,-1] = alpha
        c = _c
        if not c.flags['C_CONTIGUOUS']: c = np.ascontiguousarray(c, dtype=np.float32)
    elif channel_order == ChannelOrder.BGR:
        c[...,[0, 2]] = c[...,[2, 0]]
        if not c.flags['C_CONTIGUOUS']: c = np.ascontiguousarray(c, dtype=np.float32)

    return c


def map_to_colors(x: Any, cm_name: str) -> np.ndarray:
    """Map input variable to matplotlib color palette.

    Scale variable ``x`` to the range <0; 1> and map it to RGB colors from
    matplotlib's colormap with ``cm_name``.

    Parameters
    ----------
    x : array_like
        Input variable, array_like of any shape.
    cm_name : string
        Matplotlib colormap name.

    Returns
    -------
    out : np.ndarray
        Numpy array with RGB color values mapped from the input array values.
        The output shape is ``x.shape + (3,)``.
    """
    if x is None: raise ValueError()

    if not isinstance(x, np.ndarray): x = np.asarray(x)

    min_x = x.min()
    max_x = x.max()
    if min_x != max_x:
        x = (1 / (x.max() - min_x)) * (x - min_x)
    else:
        x = np.zeros(x.shape)

    c = cm.get_cmap(cm_name)(x)
    return np.delete(c, np.s_[-1], len(c.shape) - 1) # skip alpha

def get_image_meta(file_name: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Get image metadata from file.

    Read image file header and return width, height, samples per pixel
    and bits per sample values.

    Parameters
    ----------
    file_name : string
        Image file name.

    Returns
    -------
    out : tuple (int, int, int, int)
        Image width, height, samples per pixel, bits per sample.
    """
    c_width = c_int()
    c_height = c_int()
    c_spp = c_int()
    c_bps = c_int()
    if _optix.get_image_meta(file_name, byref(c_width), byref(c_height), byref(c_spp), byref(c_bps)):
        return c_width.value, c_height.value, c_spp.value, c_bps.value
    else:
        return None, None, None, None

def read_image(file_name: str, normalized: bool = False) -> Optional[np.ndarray]:
    """Read image from file.

    Read image file into numpy array. Array shape is ``(height, width, 3(4))`` for RGB(A) images
    and ``(height, width)`` for grayscale images. Image data type is preserved (``numpy.uint8``
    or ``numpy.uint16``) by default, or values are scaled to ``[0; 1]`` range and ``numpy.float32``
    type is returned, if ``normalized`` is set to ``True``. Color channel order is preserved.

    Tiff images are supported with size up to your memory limit (files >> GB are OK). Bmp, gif,
    jpg, and png formats are supported as well.

    Parameters
    ----------
    file_name : string
        Image file name.
    normalized : bool, optional
        Normalize values to ``[0; 1]`` range.

    Returns
    -------
    out : np.ndarray
        Image data.
    """
    c_width = c_int()
    c_height = c_int()
    c_spp = c_int()
    c_bps = c_int()
    if not _optix.get_image_meta(file_name, byref(c_width), byref(c_height), byref(c_spp), byref(c_bps)):
        return None

    if normalized:
        t = np.float32
    else:
        if c_bps.value == 8: t = np.uint8
        elif c_bps.value == 16: t = np.uint16
        elif c_bps.value == 32: t = np.float32
        else: raise ValueError("Image bits per sample value not supported.")

    if c_spp.value == 1: data = np.zeros((c_height.value, c_width.value), dtype=t)
    else: data = np.zeros((c_height.value, c_width.value, c_spp.value), dtype=t)

    if not data.flags['C_CONTIGUOUS']: data = np.ascontiguousarray(data, dtype=t)

    img = None
    if normalized:
        if _optix.read_image_normalized(file_name, data.ctypes.data, c_width.value, c_height.value, c_spp.value):
            img = data
    else:
        if _optix.read_image(file_name, data.ctypes.data, c_width.value, c_height.value, c_spp.value, c_bps.value):
            img = data
    return img

def simplex(pos: Any, noise: Optional[np.ndarray] = None) -> np.ndarray:
    """Generate simplex noise.

    Generate noise using open simplex algorithm. 2D, 3D or 4D algorithm is
    used depending on the ``pos.shape[-1]`` value. Output array shape is
    ``pos.shape[:-1]``. Noise can be generated 'in place' if optional
    ``noise`` array is provided (``pos.shape`` has to match ``pos.shape[:-1]``).


    Parameters
    ----------
    pos : array_like
        Noise inputs, array_like of shape ``(n, ..., d)``, where d is 2, 3, or 4.
    noise : np.ndarray, optional
        Array used to write the output noise (same as return value). New array
        is created if ``noise`` argument is not provided.

    Returns
    -------
    out : np.ndarray
        Noise generated from ``pos`` inputs.
    """
    if not isinstance(pos, np.ndarray): pos = np.ascontiguousarray(pos, dtype=np.float32)

    if len(pos.shape) < 2 or pos.shape[-1] not in [2, 3, 4]:
        raise ValueError("Positions array shape should be (n, ..., d), where d is 2, 3, or 4.")

    if pos.dtype != np.float32: pos = np.ascontiguousarray(pos, dtype=np.float32)
    if not pos.flags['C_CONTIGUOUS']: pos = np.ascontiguousarray(pos, dtype=np.float32)

    if noise is None: noise = np.zeros(pos.shape[:-1], dtype=np.float32)
    else:
        if noise.shape != pos.shape[:-1]:
            raise ValueError("Noise shape does not match positions array shape.")
        if noise.dtype != np.float32: noise = np.ascontiguousarray(noise, dtype=np.float32)

    if not noise.flags['C_CONTIGUOUS']: noise = np.ascontiguousarray(noise, dtype=np.float32)

    if pos.shape[-1] == 2:
        _optix.open_simplex_2d(noise.ctypes.data, pos.ctypes.data, np.prod(noise.shape))
    elif pos.shape[-1] == 3:
        _optix.open_simplex_3d(noise.ctypes.data, pos.ctypes.data, np.prod(noise.shape))
    elif pos.shape[-1] == 4:
        _optix.open_simplex_4d(noise.ctypes.data, pos.ctypes.data, np.prod(noise.shape))
        pass
    else: raise ValueError("Positions array shape should be (n, ..., d), where d is 2, 3, or 4.")

    return noise
