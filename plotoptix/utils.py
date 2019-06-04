"""Utility and convenience functions.
"""

import functools, operator
import numpy as np

from typing import Any, Optional
from matplotlib import cm

from plotoptix._load_lib import load_optix

_optix = load_optix()


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

def _make_contiguous_3d(a: Optional[Any], n: int = -1, extend_scalars = False) -> Optional[np.ndarray]:
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


def make_color(c: Any,
               exposure: float = 1.0,
               gamma: float = 1.0,
               input_range: float = 1.0,
               extend_scalars: bool = True) -> np.ndarray:
    """Prepare colors to account for the postprocessing corrections.

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

    Returns
    -------
    out : np.ndarray
        C-contiguous, float32 numpy array with RGB color values pre-calculated
        to account for post-processing corrections.
    """
    c = _make_contiguous_3d(c, extend_scalars=extend_scalars)
    return (1 / exposure) * np.power((1 / input_range) * c, gamma)


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
    x = (1 / (x.max() - min_x)) * (x - min_x)

    c = cm.get_cmap(cm_name)(x)
    return np.delete(c, np.s_[-1], len(c.shape) - 1) # skip alpha

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
