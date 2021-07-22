"""
This file is a part of the fdrot package.

Authors: Paweł Ordyna
"""

import numpy as np
import numba
from numba import njit


# TODO hard copy signature


@njit(parallel=True, cache=True)
def numba_multiply_arrays(array1: np.ndarray,
                          array2: np.ndarray) -> np.ndarray:
    """ Multiply two numpy arrays """
    return array1 * array2


@njit((numba.float64[:, :, ::1], numba.float64[:, ::1]), parallel=True, cache=True)
def average_over_pulse(input_arr: np.ndarray, output_arr: np.ndarray) -> None:
    input_arr[:, :, :] = np.cos(input_arr)**2
    output_arr[:, :] = np.sum(input_arr, axis=2)
    output_arr[:, :] = np.arccos(np.sqrt(output_arr))
