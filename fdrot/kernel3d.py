"""
This file is a part of the fdrot package.

Authors: Paweł Ordyna
"""

import numpy as np
import numba
from numba import njit, prange

# TODO hard copy signature


@njit((numba.float64[::1], numba.float64[:, :, ::1], numba.float64[:, :, ::1],
       numba.uint32, numba.uint32, numba.int32, numba.int32),
      parallel=True)
def kernel3d(pulse: np.ndarray,
             input_arr: np.ndarray,
             output: np.ndarray,
             global_start: int, global_stop: int,
             leading_start: int, leading_stop: int) -> None:
    """ Calculates the Faraday Rotation from a 3D input.

    The rotation is integrated over a single time step and a given pulse
    shape along one of the simulation box axis.

    Args:
        pulse: Pulse over which it's integrated.
        input_arr: B field in the right direction, already multiplied
          with the electron density. Second dimension is  the first
          axis in the output, 3rd axis is the  second  one in the
          output; The effect is integrated  over the first axis.
        output: The obtained rotation is added to it.
        global_start: global boundary to the left,the cell from which
          the propagation begins (This cell is also included).
        global_stop:  global boundary to the right, the cell at which
          the propagation stops (This cell is not included).
        leading_start: Interval start for the leading slice of the pulse
          (in a single time step).
        leading_stop: Interval stop for the leading slice of the pulse
          (in a single time step).
    """

    pulse_len = pulse.shape[0]
    if leading_start < global_start:
        leading_start = global_start
    if leading_stop > global_stop + (pulse_len - 1):
        leading_stop = global_stop + (pulse_len - 1)
    duration = leading_stop - leading_start  # in cells
    # cell where the most right slice of the pulse is:
    pulse_head = leading_start
    # cell where the most left slice of the pulse is:
    pulse_tail = pulse_head - (pulse_len - 1)
    for tt in prange(duration):
        # Assigned variables in prange loops are private; pulse_head,
        # pulse_tail can only be read here, so they stay global and
        # available in the loop.
        # cell where the most right slice of the pulse is:
        pulse_head_private = pulse_head + tt
        # cell where the most left slice of the pulse is:
        pulse_tail_private = pulse_tail + tt
        # Let's cut pulse on the global boundaries.
        cut_at_tail = 0  # Cells cut at the pulse tail.
        cut_at_head = 0  # Cells cut at the pulse head.
        if pulse_tail_private < global_start:
            cut_at_tail, pulse_tail_private = (global_start -
                                               pulse_tail_private), \
                                              global_start
        if pulse_head_private > global_stop - 1:
            cut_at_head, pulse_head_private = (pulse_head_private
                                               - (global_stop - 1)),\
                                              global_stop - 1
        # Broadcasting arrays:
        if cut_at_head == 0:
            cut_pulse = pulse[cut_at_tail:]
        else:
            cut_pulse = pulse[cut_at_tail:-cut_at_head]
        field_chunk = input_arr[pulse_tail_private:pulse_head_private + 1
                                  :, :]  # Take all 'y' cut in 'x'.
        # Faraday Rotation originating from the time interval [tt, tt+1].
        # summed += ... : Shapes: (x,) * (x,y)  -> (y,) # prange reduction
        for ii in prange(cut_pulse.shape[0]):
            output[cut_at_tail+ii, :, :] = cut_pulse[ii] * field_chunk[ii, :, :]

