cimport cython
import  numpy as np
from libc.math cimport sqrt
from typing import Tuple

# cdef inline double d_square(double x): # maybe use instead of x**2, should be faster
#   return x*x

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int kernel_2d_perp(double [:,::1] input_data, double [:,::1] output_data, Py_ssize_t interval_start,
                         Py_ssize_t interval_stop, double factor=1, bint interpolation=0, bint inc_sym=0):
    # array shape:
    cdef Py_ssize_t r_len, s_len
    r_len = input_data.shape[1]
    s_len = input_data.shape[0]
    cdef Py_ssize_t r_len_half = r_len // 2
    # iterators:
    cdef Py_ssize_t yy, xx ,rr, zz
    # 'up' and 'down' stand for the upper and lower parts of the distribution.
    cdef double radius
    cdef double summed_line_up, summed_line_down
    # linear interpolation:
    cdef double interpolated_value_up, interpolated_value_down
    cdef double value_diff_up, value_diff_down
    cdef double* value_1_down
    cdef double* value_1_up
    cdef double* value_2_down
    cdef double* value_2_up

    # When using without without slicing (just one step), multiply the outcome with 2,
    # to include the other, identical, part.
    if inc_sym:
        factor *= 2

    # Handling both odd and even number of cells in a slice.
    cdef Py_ssize_t offset, loop_start
    cdef float r_offset
    cdef bint middle_line_splitting = 0
    if  r_len%2 == 0:
        offset = 1
        loop_start = 0
        r_offset = 0.5
    else:
        offset = 0
        loop_start = 1
        r_offset = 0
        if interval_stop == 0:
            interval_stop = 1
            middle_line_splitting = 1
        # There is a middle line, in the odd case which doesn't belong to neither the lower
        # nor the upper distribution. It is summed here, and the main loop starts with y=1
        # (loop_start).
        summed_line_up = 0

        # The first loop is over slices.
        for zz in range(s_len):
            # second one goes over x. We start, where the X-ray enters the cylinder.
            # That is at the full radius, and we proceed towards the middle of the way
            # through the cylinder.
            for xx in range(interval_start , interval_stop , -1):
                # This is the radius at the middle point of the cell (point P), halfway through
                # a single step (xx +1) in the propagation direction (X-direction).
                # The r_offset is not applied to the radius, it self, yet.
                radius = sqrt((0.25)**2 + (xx - r_offset)**2)
                rr = int(radius)
                # Value at floor(radius).
                value_1_up = &input_data[zz,r_len_half + rr]
                if not interpolation or r_len_half - 1 == rr:
                    interpolated_value_up = value_1_up[0]
                else:
                    # Interpolation between two data points.
                    value_2_up = &input_data[zz,r_len_half + rr + 1]
                    value_diff_up = value_2_up[0] - value_1_up[0]
                    interpolated_value_up = (value_1_up[0] +
                                                 value_diff_up
                                                 * (radius - (rr + r_offset)))
                # Add to the integral.
                summed_line_up += (interpolated_value_up
                                   * (0.25)/radius)
            # Write the output.
            output_data[r_len_half, s_len -1 - zz] = factor * summed_line_up

    # Main part:
    # The first loop is over slices.
    for zz in range(s_len):
        # The second one is over the Y-Axis. That upwards, is away from the middle of the circle.
        # The X-Ray beam propagates along the X-Axis.
        for yy in range(loop_start, r_len_half):
            summed_line_up = 0
            summed_line_down = 0
            # In a case, that we integrate only over a part of the way, the first iteration of the
            # next loop has to be set extra.
            if r_len_half - yy <= interval_start:
                x_start  = r_len_half - yy
            else:
                x_start = interval_start
            # The most inner one goes over x. We start, where the X-ray enters the cylinder.
            # That is at the full radius, and we proceed towards the middle of the way
            # through the cylinder.
            for xx in range(x_start, interval_stop , -1):
                # This is the radius at the middle point of the cell (point P), halfway through
                # a single step (xx +1) in the propagation direction (X-direction).
                # The r_offset is not applied to the radius, it self, yet.
                radius = sqrt((yy + 0.5)**2 + (xx - r_offset)**2)
                # Value at floor(radius).
                rr = int(radius)
                value_1_up = &input_data[zz,r_len_half + rr]
                value_1_down = &input_data[zz, r_len_half - offset - rr]
                if not interpolation or r_len_half - 1 == rr:
                    interpolated_value_up = value_1_up[0]
                    interpolated_value_down = value_1_down[0]
                else:
                    # Interpolation between two data points.
                    value_2_up = &input_data[zz,r_len_half + rr +1]
                    value_2_down = &input_data[zz,r_len_half  - offset - rr - 1]

                    value_diff_up = value_2_up[0] - value_1_up[0]
                    value_diff_down = value_2_down[0] - value_1_down[0]

                    interpolated_value_up = (value_1_up[0] +
                                             value_diff_up
                                             * (radius - (rr + r_offset)))
                    interpolated_value_down = (value_1_down[0] +
                                               value_diff_down
                                               * (radius - (rr + r_offset)))
                # Add to the integrals.
                summed_line_up += (interpolated_value_up
                                   * (yy + 0.5)/radius)
                summed_line_down += (interpolated_value_down
                                     * (yy + 0.5)/radius)
            if middle_line_splitting:
                pass  # to be implemented

            # Write the output.
            output_data[r_len_half - offset -yy, s_len - 1 - zz] = factor * summed_line_up
            output_data[r_len_half +  yy,  s_len - 1 - zz]  = factor * summed_line_down

    return 0


def rotation_static_2d(input_data, interpolation=False):
    try:
        input_data.flags
    except AttributeError:
        print('input_data should be a numpy array!')
        raise
    if input_data.flags['F_CONTIGUOUS']:
        print('input_data should be stored in a raw major, C contiguous.')
        raise ValueError
    elif not input_data.flags['C_CONTIGUOUS']:
        print('input_data has to be a contiguous array.')
        raise ValueError
    if input_data.dtype != np.float64:
        print('input should be an ndarray of type np.float64.')
        print('Converting to float64 and continuing...')
        input_data = input_data.astype(np.float64)

    output = np.zeros(input_data.size, dtype=np.float64, order='C').reshape(
                    input_data.shape[1], input_data.shape[0])

    cdef double [:,::1] input_data_view = input_data
    cdef double [:,::1] output_view = output
    kernel_2d_perp(input_data_view, output_view, input_data.shape[1], 0, 1,
                   interpolation=interpolation, inc_sym=1)
    return output

def translator(start, stop, full_length):
     half = full_length / 2
    if full_length %2 == 0:
        modifier = ...

    if start <= half and stop <= stop:
        con_start = half - start
        stop = half - stop
    if start > half and stop < stop:
        con_start = stop - half
        con_stop = start - half
    if start <= half and stop > half:
        # split it !

    return

def  one_step_long_pulse(double [:,:,::1] output, double [:,::1] step, Py_ssize_t full_step_size,
                         Py_ssize_t x_start_glob, Py_ssize_t x_end_glob, Py_ssize_t leading_interval_start, float factor, bint interpolation) -> None:

    raise NotImplementedError
    cdef Py_ssize_t n, interval_start, interval_stop
    n = output.shape[0]
    for nn in range(n)
        interval_start = leading_interval_start - nn
        interval_stop = interval_start + full_step_size
        if interval_start < x_start_glob:
            interval_start = x_start_glob
        if interval_stop > x_end_glob:
            interval_stop = x_end_glob
        if interval_stop <= interval_start:
            break  #   Continue would be more clear, but for further nn it will also continue, so no point in that.
        rotation_slice(step, output[n -nn,:,:], interval_start, interval_stop, factor=factor, interpolation=interpolation)
