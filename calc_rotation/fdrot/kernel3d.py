import numpy as np


def kernel3d(pulse: np.ndarray,
             input_arr: np.ndarray,
             output: np.ndarray,
             global_start: int, global_stop: int,
             leading_start: int, leading_stop: int) -> None:
    """ Integrates the Faraday Rotation over a single time step and a given pulse shape.

    :param pulse: Pulse over which it's integrated.
    :param global_start: global boundary to the left,the cell from which the propagation begins
            (This cell is also included).
    :param global_stop:  global boundary to the right, the cell at which the propagation stops
            (This cell is not included).
    :param input_arr: B field in the right direction, already multiplied with the electron density.
            First dimension is also the first in the output, second dim. is second in the output,
            it's integrated over the last one.
    :param output: The obtained rotation is added to it.
    :param leading_start: Interval start for the leading slice of the pulse (in a single time step).
    :param leading_stop:  Interval stop for the leading slice of the pulse (in a single time step).
    """
    pulse_len = pulse.shape[0]
    if leading_start < global_start:
        leading_start = global_start
    if leading_stop > global_stop:
        leading_stop = global_stop
    duration = leading_stop - leading_start + (pulse_len - 1)  # in cells, +/- 1
    summed = np.zeros(input_arr.shape[1])  # y
    for zz in range(input_arr.shape[0]):
        pulse_head = leading_start  # cell where the most right slice of the pulse is. +/- 1
        pulse_tail = pulse_head - (pulse_len - 1)  # cell where the most left slice of the pulse is. +/- 1
        summed[:] = 0
        input_slice = input_arr[zz, :, :]  # 2D (y,x)
        for tt in range(duration):
            # Let's cut pulse on the global boundaries.
            cut_at_tail = 0  # Cells cut at the pulse tail.
            cut_at_head = 0  # Cells cut at the pulse head.
            if pulse_tail < global_start:
                cut_at_tail, pulse_tail = global_start - pulse_tail, global_start
            if pulse_head > global_stop - 1:
                cut_at_head, pulse_head = pulse_head - (global_stop - 1), global_stop - 1

            # Broadcasting arrays:
            if cut_at_head == 0:
                cut_pulse = pulse[cut_at_tail:]

            else:
                cut_pulse = pulse[cut_at_tail:-cut_at_head]
            slice_chunk = input_slice[:, pulse_tail:pulse_head + 1]  # Take all 'y' cut in 'x'.
            # Faraday Rotation originating from the time interval [tt, tt+1].
            summed += np.dot(slice_chunk, cut_pulse)  # Shapes: (y,x) * (x,) -> (y,)

            # propagate by one cell:
            pulse_head += 1  # cell where the most right slice of the pulse is.
            pulse_tail += 1  # cell where the most left slice of the pulse is.
        output[zz, :] += summed  # (y,) + (y,) # -zz -> zz
