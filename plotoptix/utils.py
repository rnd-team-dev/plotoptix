"""Utility and convenience functions.
"""

import numpy as np

from typing import Any, Optional
from matplotlib import cm

from plotoptix._load_lib import load_optix

_optix = load_optix()

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
